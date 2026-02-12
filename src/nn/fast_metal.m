#import <Metal/Metal.h>
#import <Foundation/Foundation.h>
#include "fast_metal.h"
#include <stdio.h>
#include <math.h>

// ===================================================================
// Metal buffer wrapper
// ===================================================================

struct MetalBuf {
    id<MTLBuffer> buf;
    size_t size;
};

// ===================================================================
// Global Metal state — single encoder pattern for minimal overhead
// ===================================================================

static id<MTLDevice> g_dev;
static id<MTLCommandQueue> g_queue;

// Pipeline states for all kernels
static id<MTLComputePipelineState> g_q8_matvec;
static id<MTLComputePipelineState> g_f16_matvec;
static id<MTLComputePipelineState> g_f16_batch_matvec;
static id<MTLComputePipelineState> g_rms_norm;
static id<MTLComputePipelineState> g_per_head_rms_norm;
static id<MTLComputePipelineState> g_rope;
static id<MTLComputePipelineState> g_kv_cache_store;
static id<MTLComputePipelineState> g_attention;
static id<MTLComputePipelineState> g_silu_mul;
static id<MTLComputePipelineState> g_residual_add;

// Training pipeline states
static id<MTLComputePipelineState> g_f16_matmul_ps = nil;
static id<MTLComputePipelineState> g_f16_matmul_nt_ps = nil;
static id<MTLComputePipelineState> g_transpose_fwd_ps = nil;
static id<MTLComputePipelineState> g_transpose_rev_ps = nil;
static id<MTLComputePipelineState> g_repeat_kv_ps = nil;
static id<MTLComputePipelineState> g_rope_train_ps = nil;
static id<MTLComputePipelineState> g_silu_mul_bwd_ps = nil;
static id<MTLComputePipelineState> g_rms_norm_train_bwd_ps = nil;
static id<MTLComputePipelineState> g_scale_ps = nil;
static id<MTLComputePipelineState> g_add_scaled_ps = nil;
static id<MTLComputePipelineState> g_softmax_ce_ps = nil;
static id<MTLComputePipelineState> g_attn_train_fwd_ps = nil;
static id<MTLComputePipelineState> g_attn_bwd_dq_ps = nil;
static id<MTLComputePipelineState> g_attn_bwd_dkv_ps = nil;
static id<MTLComputePipelineState> g_repeat_kv_bwd_ps = nil;
static id<MTLComputePipelineState> g_rope_train_bwd_ps = nil;
static id<MTLComputePipelineState> g_float_matmul_ps = nil;
static id<MTLComputePipelineState> g_float_matmul_tn_ps = nil;
static id<MTLComputePipelineState> g_float_matmul_nt_ps = nil;
static id<MTLComputePipelineState> g_buf_copy_ps = nil;
static id<MTLComputePipelineState> g_clamp_ps = nil;
static id<MTLComputePipelineState> g_gelu_mul = nil;
static id<MTLComputePipelineState> g_gelu_mul_bwd_ps = nil;
static id<MTLComputePipelineState> g_grpo_pg_ps = nil;
static id<MTLComputePipelineState> g_f16_sgd_fused_ps = nil;
static id<MTLComputePipelineState> g_f16_grad_accum_ps = nil;
static id<MTLComputePipelineState> g_f16_sgd_apply_ps = nil;

// Single command buffer + encoder reused across all dispatches per flush
static id<MTLCommandBuffer> g_cmdbuf;
static id<MTLComputeCommandEncoder> g_enc;

// Reference counting for safe multi-init (inference + SFT share Metal state)
static int g_init_refcount = 0;

// ===================================================================
// Init / Shutdown
// ===================================================================

int fast_metal_init(void) {
    if (g_init_refcount++ > 0) return 0;  // already initialized
    @autoreleasepool {
        g_dev = MTLCreateSystemDefaultDevice();
        if (!g_dev) {
            fprintf(stderr, "[FastMetal] No Metal device\n");
            return -1;
        }
        g_queue = [g_dev newCommandQueue];
        g_queue.label = @"FastInference";

        NSString *shaderPath = @"shaders/q8_kernels.metal";
        NSError *err = nil;
        NSString *src = [NSString stringWithContentsOfFile:shaderPath
                                  encoding:NSUTF8StringEncoding error:&err];
        if (!src) {
            shaderPath = @"build/q8_kernels.metal";
            src = [NSString stringWithContentsOfFile:shaderPath
                            encoding:NSUTF8StringEncoding error:&err];
        }
        if (!src) {
            fprintf(stderr, "[FastMetal] Cannot read shader file\n");
            return -1;
        }

        MTLCompileOptions *opts = [[MTLCompileOptions alloc] init];
        opts.mathMode = MTLMathModeFast;
        id<MTLLibrary> lib = [g_dev newLibraryWithSource:src options:opts error:&err];
        if (!lib) {
            fprintf(stderr, "[FastMetal] Shader error: %s\n",
                    err.localizedDescription.UTF8String);
            return -1;
        }

        // Create pipeline states for all kernels
        id<MTLFunction> fn;

        #define MAKE_PSO(var, name) \
            fn = [lib newFunctionWithName:@name]; \
            if (!fn) { fprintf(stderr, "[FastMetal] Function '%s' not found\n", name); return -1; } \
            var = [g_dev newComputePipelineStateWithFunction:fn error:&err]; \
            if (!var) { fprintf(stderr, "[FastMetal] Pipeline error for '%s'\n", name); return -1; }

        MAKE_PSO(g_q8_matvec,        "q8_matvec");
        MAKE_PSO(g_f16_matvec,       "f16_matvec");
        MAKE_PSO(g_f16_batch_matvec, "f16_batch_matvec");
        MAKE_PSO(g_rms_norm,         "rms_norm");
        MAKE_PSO(g_per_head_rms_norm,"per_head_rms_norm");
        MAKE_PSO(g_rope,             "rope");
        MAKE_PSO(g_kv_cache_store,   "kv_cache_store");
        MAKE_PSO(g_attention,        "attention");
        MAKE_PSO(g_silu_mul,         "silu_mul");
        MAKE_PSO(g_residual_add,     "residual_add");

        // Training kernels
        MAKE_PSO(g_f16_matmul_ps,        "f16_matmul");
        MAKE_PSO(g_f16_matmul_nt_ps,     "f16_matmul_nt");
        MAKE_PSO(g_transpose_fwd_ps,     "transpose_heads_fwd");
        MAKE_PSO(g_transpose_rev_ps,     "transpose_heads_rev");
        MAKE_PSO(g_repeat_kv_ps,         "repeat_kv");
        MAKE_PSO(g_rope_train_ps,        "rope_train");
        MAKE_PSO(g_silu_mul_bwd_ps,      "silu_mul_backward");
        MAKE_PSO(g_rms_norm_train_bwd_ps,"rms_norm_train_backward");
        MAKE_PSO(g_scale_ps,             "scale_tensor");
        MAKE_PSO(g_add_scaled_ps,        "add_scaled");
        MAKE_PSO(g_softmax_ce_ps,        "softmax_ce_train");
        MAKE_PSO(g_attn_train_fwd_ps,   "attn_train_fwd");
        MAKE_PSO(g_attn_bwd_dq_ps,      "attn_bwd_dq");
        MAKE_PSO(g_attn_bwd_dkv_ps,     "attn_bwd_dkv");
        MAKE_PSO(g_repeat_kv_bwd_ps,    "repeat_kv_bwd");
        MAKE_PSO(g_rope_train_bwd_ps,   "rope_train_bwd");
        MAKE_PSO(g_float_matmul_ps,     "float_matmul");
        MAKE_PSO(g_float_matmul_tn_ps,  "float_matmul_tn");
        MAKE_PSO(g_float_matmul_nt_ps,  "float_matmul_nt");
        MAKE_PSO(g_buf_copy_ps,         "buf_copy");
        MAKE_PSO(g_clamp_ps,            "clamp_tensor");
        MAKE_PSO(g_gelu_mul,            "gelu_mul");
        MAKE_PSO(g_gelu_mul_bwd_ps,     "gelu_mul_backward");
        MAKE_PSO(g_grpo_pg_ps,          "grpo_policy_grad");
        MAKE_PSO(g_f16_sgd_fused_ps,   "f16_sgd_fused");
        MAKE_PSO(g_f16_grad_accum_ps,  "f16_grad_accum");
        MAKE_PSO(g_f16_sgd_apply_ps,   "f16_sgd_apply");

        #undef MAKE_PSO

        g_cmdbuf = nil;
        g_enc = nil;

        printf("[FastMetal] GPU: %s (max threads/tg: %lu)\n",
               g_dev.name.UTF8String,
               (unsigned long)g_q8_matvec.maxTotalThreadsPerThreadgroup);
        return 0;
    }
}

void fast_metal_shutdown(void) {
    if (--g_init_refcount > 0) return;  // other users still active
    g_q8_matvec = nil; g_f16_matvec = nil; g_f16_batch_matvec = nil;
    g_rms_norm = nil; g_per_head_rms_norm = nil;
    g_rope = nil; g_kv_cache_store = nil; g_attention = nil;
    g_silu_mul = nil; g_residual_add = nil;
    // Training pipelines
    g_f16_matmul_ps = nil; g_f16_matmul_nt_ps = nil;
    g_transpose_fwd_ps = nil; g_transpose_rev_ps = nil;
    g_repeat_kv_ps = nil; g_rope_train_ps = nil;
    g_silu_mul_bwd_ps = nil; g_rms_norm_train_bwd_ps = nil;
    g_scale_ps = nil; g_add_scaled_ps = nil; g_softmax_ce_ps = nil;
    g_attn_train_fwd_ps = nil; g_attn_bwd_dq_ps = nil; g_attn_bwd_dkv_ps = nil;
    g_repeat_kv_bwd_ps = nil; g_rope_train_bwd_ps = nil;
    g_float_matmul_ps = nil; g_float_matmul_tn_ps = nil; g_float_matmul_nt_ps = nil;
    g_buf_copy_ps = nil; g_clamp_ps = nil; g_gelu_mul = nil; g_gelu_mul_bwd_ps = nil;
    g_grpo_pg_ps = nil;
    g_f16_sgd_fused_ps = nil;
    g_f16_grad_accum_ps = nil; g_f16_sgd_apply_ps = nil;
    g_enc = nil; g_cmdbuf = nil;
    g_queue = nil; g_dev = nil;
}

// ===================================================================
// Buffer management
// ===================================================================

MetalBuf *metal_buf_create(size_t size) {
    MetalBuf *b = malloc(sizeof(MetalBuf));
    b->size = size;
    b->buf = [g_dev newBufferWithLength:size options:MTLResourceStorageModeShared];
    return b;
}

MetalBuf *metal_buf_from_data(const void *data, size_t sz) {
    MetalBuf *b = malloc(sizeof(MetalBuf));
    b->size = sz;
    b->buf = [g_dev newBufferWithBytes:data length:sz options:MTLResourceStorageModeShared];
    return b;
}

void metal_buf_free(MetalBuf *b) {
    if (b) { b->buf = nil; free(b); }
}

void *metal_buf_ptr(MetalBuf *b) {
    return b->buf.contents;
}

// ===================================================================
// Single encoder + memory barrier between dispatches
// ===================================================================

static id<MTLComputeCommandEncoder> get_encoder(void) {
    if (!g_enc) {
        g_cmdbuf = [g_queue commandBuffer];
        g_enc = [g_cmdbuf computeCommandEncoder];
    }
    return g_enc;
}

static void barrier(void) {
    [g_enc memoryBarrierWithScope:MTLBarrierScopeBuffers];
}

void metal_flush(void) {
    @autoreleasepool {
        if (g_enc) {
            [g_enc endEncoding];
            g_enc = nil;
        }
        if (g_cmdbuf) {
            [g_cmdbuf commit];
            [g_cmdbuf waitUntilCompleted];
            g_cmdbuf = nil;
        }
    }
}

// ===================================================================
// Dispatch functions — all reuse single encoder
// ===================================================================

void metal_enqueue_q8_matvec(MetalBuf *W_buf, MetalBuf *x_buf, MetalBuf *y_buf,
                             int rows, int nb) {
    id<MTLComputeCommandEncoder> enc = get_encoder();
    [enc setComputePipelineState:g_q8_matvec];
    [enc setBuffer:W_buf->buf offset:0 atIndex:0];
    [enc setBuffer:x_buf->buf offset:0 atIndex:1];
    [enc setBuffer:y_buf->buf offset:0 atIndex:2];
    uint32_t nb32 = (uint32_t)nb;
    uint32_t rows32 = (uint32_t)rows;
    [enc setBytes:&nb32 length:sizeof(nb32) atIndex:3];
    [enc setBytes:&rows32 length:sizeof(rows32) atIndex:4];

    // 8 simdgroups × 32 lanes = 256 threads per threadgroup
    NSUInteger n_tg = ((NSUInteger)rows + 7) / 8;
    [enc dispatchThreadgroups:MTLSizeMake(n_tg, 1, 1)
        threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
    barrier();
}

void metal_enqueue_f16_matvec(MetalBuf *W_buf, MetalBuf *x_buf, MetalBuf *y_buf,
                              int rows, int cols) {
    id<MTLComputeCommandEncoder> enc = get_encoder();
    [enc setComputePipelineState:g_f16_matvec];
    [enc setBuffer:W_buf->buf offset:0 atIndex:0];
    [enc setBuffer:x_buf->buf offset:0 atIndex:1];
    [enc setBuffer:y_buf->buf offset:0 atIndex:2];
    uint32_t cols32 = (uint32_t)cols;
    uint32_t rows32 = (uint32_t)rows;
    [enc setBytes:&cols32 length:sizeof(cols32) atIndex:3];
    [enc setBytes:&rows32 length:sizeof(rows32) atIndex:4];

    // 8 simdgroups × 32 lanes = 256 threads per threadgroup
    NSUInteger n_tg = ((NSUInteger)rows + 7) / 8;
    [enc dispatchThreadgroups:MTLSizeMake(n_tg, 1, 1)
        threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
    barrier();
}

void metal_enqueue_f16_batch_matvec(MetalBuf *W_buf, MetalBuf *X_buf, MetalBuf *Y_buf,
                                     int rows, int cols, int B) {
    id<MTLComputeCommandEncoder> enc = get_encoder();
    [enc setComputePipelineState:g_f16_batch_matvec];
    [enc setBuffer:W_buf->buf offset:0 atIndex:0];
    [enc setBuffer:X_buf->buf offset:0 atIndex:1];
    [enc setBuffer:Y_buf->buf offset:0 atIndex:2];
    uint32_t cols32 = (uint32_t)cols, rows32 = (uint32_t)rows, B32 = (uint32_t)B;
    [enc setBytes:&cols32 length:sizeof(cols32) atIndex:3];
    [enc setBytes:&rows32 length:sizeof(rows32) atIndex:4];
    [enc setBytes:&B32 length:sizeof(B32) atIndex:5];

    NSUInteger n_tg = ((NSUInteger)rows + 7) / 8;
    [enc dispatchThreadgroups:MTLSizeMake(n_tg, 1, 1)
        threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
    barrier();
}

void metal_enqueue_rms_norm(MetalBuf *x_buf, MetalBuf *gamma_buf, MetalBuf *out_buf,
                            int dim, float eps) {
    id<MTLComputeCommandEncoder> enc = get_encoder();
    [enc setComputePipelineState:g_rms_norm];
    [enc setBuffer:x_buf->buf offset:0 atIndex:0];
    [enc setBuffer:gamma_buf->buf offset:0 atIndex:1];
    [enc setBuffer:out_buf->buf offset:0 atIndex:2];
    uint32_t dim32 = (uint32_t)dim;
    [enc setBytes:&dim32 length:sizeof(dim32) atIndex:3];
    [enc setBytes:&eps length:sizeof(eps) atIndex:4];

    NSUInteger tg = MIN(256, (NSUInteger)dim);
    [enc dispatchThreadgroups:MTLSizeMake(1, 1, 1)
        threadsPerThreadgroup:MTLSizeMake(tg, 1, 1)];
    barrier();
}

void metal_enqueue_rms_norm_batched(MetalBuf *x_buf, MetalBuf *gamma_buf, MetalBuf *out_buf,
                                     int dim, float eps, int B) {
    id<MTLComputeCommandEncoder> enc = get_encoder();
    [enc setComputePipelineState:g_rms_norm];
    [enc setBuffer:x_buf->buf offset:0 atIndex:0];
    [enc setBuffer:gamma_buf->buf offset:0 atIndex:1];
    [enc setBuffer:out_buf->buf offset:0 atIndex:2];
    uint32_t dim32 = (uint32_t)dim;
    [enc setBytes:&dim32 length:sizeof(dim32) atIndex:3];
    [enc setBytes:&eps length:sizeof(eps) atIndex:4];

    NSUInteger tg = MIN(256, (NSUInteger)dim);
    [enc dispatchThreadgroups:MTLSizeMake(B, 1, 1)
        threadsPerThreadgroup:MTLSizeMake(tg, 1, 1)];
    barrier();
}

void metal_enqueue_per_head_rms_norm(MetalBuf *vec_buf, MetalBuf *gamma_buf,
                                      int n_heads, int head_dim, float eps) {
    id<MTLComputeCommandEncoder> enc = get_encoder();
    [enc setComputePipelineState:g_per_head_rms_norm];
    [enc setBuffer:vec_buf->buf offset:0 atIndex:0];
    [enc setBuffer:gamma_buf->buf offset:0 atIndex:1];
    uint32_t hd32 = (uint32_t)head_dim;
    [enc setBytes:&hd32 length:sizeof(hd32) atIndex:2];
    [enc setBytes:&eps length:sizeof(eps) atIndex:3];

    NSUInteger tg = MIN(256, (NSUInteger)head_dim);
    [enc dispatchThreadgroups:MTLSizeMake(n_heads, 1, 1)
        threadsPerThreadgroup:MTLSizeMake(tg, 1, 1)];
    barrier();
}

void metal_enqueue_rope(MetalBuf *q_buf, MetalBuf *k_buf,
                        int Hq, int Hkv, int hd, int pos, float theta) {
    id<MTLComputeCommandEncoder> enc = get_encoder();
    [enc setComputePipelineState:g_rope];
    [enc setBuffer:q_buf->buf offset:0 atIndex:0];
    [enc setBuffer:k_buf->buf offset:0 atIndex:1];
    uint32_t Hq32 = (uint32_t)Hq, Hkv32 = (uint32_t)Hkv;
    uint32_t hd32 = (uint32_t)hd, pos32 = (uint32_t)pos;
    [enc setBytes:&Hq32 length:sizeof(Hq32) atIndex:2];
    [enc setBytes:&Hkv32 length:sizeof(Hkv32) atIndex:3];
    [enc setBytes:&hd32 length:sizeof(hd32) atIndex:4];
    [enc setBytes:&pos32 length:sizeof(pos32) atIndex:5];
    [enc setBytes:&theta length:sizeof(theta) atIndex:6];

    uint32_t half_hd = hd / 2;
    NSUInteger total = (NSUInteger)(Hq + Hkv) * half_hd;
    NSUInteger tg = MIN(256, total);
    [enc dispatchThreads:MTLSizeMake(total, 1, 1)
   threadsPerThreadgroup:MTLSizeMake(tg, 1, 1)];
    barrier();
}

void metal_enqueue_kv_cache_store(MetalBuf *k_buf, MetalBuf *v_buf,
                                   MetalBuf *k_cache, MetalBuf *v_cache,
                                   int Hkv, int max_seq, int hd, int pos) {
    id<MTLComputeCommandEncoder> enc = get_encoder();
    [enc setComputePipelineState:g_kv_cache_store];
    [enc setBuffer:k_buf->buf offset:0 atIndex:0];
    [enc setBuffer:v_buf->buf offset:0 atIndex:1];
    [enc setBuffer:k_cache->buf offset:0 atIndex:2];
    [enc setBuffer:v_cache->buf offset:0 atIndex:3];
    uint32_t Hkv32 = (uint32_t)Hkv, max_seq32 = (uint32_t)max_seq;
    uint32_t hd32 = (uint32_t)hd, pos32 = (uint32_t)pos;
    [enc setBytes:&Hkv32 length:sizeof(Hkv32) atIndex:4];
    [enc setBytes:&max_seq32 length:sizeof(max_seq32) atIndex:5];
    [enc setBytes:&hd32 length:sizeof(hd32) atIndex:6];
    [enc setBytes:&pos32 length:sizeof(pos32) atIndex:7];

    NSUInteger total = (NSUInteger)Hkv * hd;
    NSUInteger tg = MIN(256, total);
    [enc dispatchThreads:MTLSizeMake(total, 1, 1)
   threadsPerThreadgroup:MTLSizeMake(tg, 1, 1)];
    barrier();
}

void metal_enqueue_attention(MetalBuf *Q_buf, MetalBuf *Kc_buf, MetalBuf *Vc_buf,
                             MetalBuf *attn_out_buf, int n_attend, int hd,
                             int max_seq, int Hq, int group_ratio) {
    id<MTLComputeCommandEncoder> enc = get_encoder();
    [enc setComputePipelineState:g_attention];
    [enc setBuffer:Q_buf->buf offset:0 atIndex:0];
    [enc setBuffer:Kc_buf->buf offset:0 atIndex:1];
    [enc setBuffer:Vc_buf->buf offset:0 atIndex:2];
    [enc setBuffer:attn_out_buf->buf offset:0 atIndex:3];
    uint32_t n32 = (uint32_t)n_attend, hd32 = (uint32_t)hd;
    uint32_t ms32 = (uint32_t)max_seq, gr32 = (uint32_t)group_ratio;
    float inv_sqrt_hd = 1.0f / sqrtf((float)hd);
    [enc setBytes:&n32 length:sizeof(n32) atIndex:4];
    [enc setBytes:&hd32 length:sizeof(hd32) atIndex:5];
    [enc setBytes:&ms32 length:sizeof(ms32) atIndex:6];
    [enc setBytes:&gr32 length:sizeof(gr32) atIndex:7];
    [enc setBytes:&inv_sqrt_hd length:sizeof(inv_sqrt_hd) atIndex:8];

    NSUInteger tg = 256;
    [enc dispatchThreadgroups:MTLSizeMake(Hq, 1, 1)
        threadsPerThreadgroup:MTLSizeMake(tg, 1, 1)];
    barrier();
}

void metal_enqueue_silu_mul(MetalBuf *gate_buf, MetalBuf *up_buf, int n) {
    id<MTLComputeCommandEncoder> enc = get_encoder();
    [enc setComputePipelineState:g_silu_mul];
    [enc setBuffer:gate_buf->buf offset:0 atIndex:0];
    [enc setBuffer:up_buf->buf offset:0 atIndex:1];

    NSUInteger tg = MIN(256, (NSUInteger)n);
    [enc dispatchThreads:MTLSizeMake(n, 1, 1)
   threadsPerThreadgroup:MTLSizeMake(tg, 1, 1)];
    barrier();
}

void metal_enqueue_gelu_mul(MetalBuf *gate_buf, MetalBuf *up_buf, int n) {
    id<MTLComputeCommandEncoder> enc = get_encoder();
    [enc setComputePipelineState:g_gelu_mul];
    [enc setBuffer:gate_buf->buf offset:0 atIndex:0];
    [enc setBuffer:up_buf->buf offset:0 atIndex:1];

    NSUInteger tg = MIN(256, (NSUInteger)n);
    [enc dispatchThreads:MTLSizeMake(n, 1, 1)
   threadsPerThreadgroup:MTLSizeMake(tg, 1, 1)];
    barrier();
}

void metal_enqueue_residual_add(MetalBuf *x_buf, MetalBuf *y_buf, int n) {
    id<MTLComputeCommandEncoder> enc = get_encoder();
    [enc setComputePipelineState:g_residual_add];
    [enc setBuffer:x_buf->buf offset:0 atIndex:0];
    [enc setBuffer:y_buf->buf offset:0 atIndex:1];

    NSUInteger tg = MIN(256, (NSUInteger)n);
    [enc dispatchThreads:MTLSizeMake(n, 1, 1)
   threadsPerThreadgroup:MTLSizeMake(tg, 1, 1)];
    barrier();
}

// ===================================================================
// Offset-aware dispatch for batched inference
// ===================================================================

void metal_enqueue_per_head_rms_norm_off(MetalBuf *vec_buf, size_t vec_off,
                                          MetalBuf *gamma_buf,
                                          int n_heads, int head_dim, float eps) {
    id<MTLComputeCommandEncoder> enc = get_encoder();
    [enc setComputePipelineState:g_per_head_rms_norm];
    [enc setBuffer:vec_buf->buf offset:vec_off atIndex:0];
    [enc setBuffer:gamma_buf->buf offset:0 atIndex:1];
    uint32_t hd32 = (uint32_t)head_dim;
    [enc setBytes:&hd32 length:sizeof(hd32) atIndex:2];
    [enc setBytes:&eps length:sizeof(eps) atIndex:3];

    NSUInteger tg = MIN(256, (NSUInteger)head_dim);
    [enc dispatchThreadgroups:MTLSizeMake(n_heads, 1, 1)
        threadsPerThreadgroup:MTLSizeMake(tg, 1, 1)];
    barrier();
}

void metal_enqueue_rope_off(MetalBuf *q_buf, size_t q_off,
                             MetalBuf *k_buf, size_t k_off,
                             int Hq, int Hkv, int hd, int pos, float theta) {
    id<MTLComputeCommandEncoder> enc = get_encoder();
    [enc setComputePipelineState:g_rope];
    [enc setBuffer:q_buf->buf offset:q_off atIndex:0];
    [enc setBuffer:k_buf->buf offset:k_off atIndex:1];
    uint32_t Hq32 = (uint32_t)Hq, Hkv32 = (uint32_t)Hkv;
    uint32_t hd32 = (uint32_t)hd, pos32 = (uint32_t)pos;
    [enc setBytes:&Hq32 length:sizeof(Hq32) atIndex:2];
    [enc setBytes:&Hkv32 length:sizeof(Hkv32) atIndex:3];
    [enc setBytes:&hd32 length:sizeof(hd32) atIndex:4];
    [enc setBytes:&pos32 length:sizeof(pos32) atIndex:5];
    [enc setBytes:&theta length:sizeof(theta) atIndex:6];

    uint32_t half_hd = hd / 2;
    NSUInteger total = (NSUInteger)(Hq + Hkv) * half_hd;
    NSUInteger tg = MIN(256, total);
    [enc dispatchThreads:MTLSizeMake(total, 1, 1)
   threadsPerThreadgroup:MTLSizeMake(tg, 1, 1)];
    barrier();
}

void metal_enqueue_kv_cache_store_off(MetalBuf *k_buf, size_t k_off,
                                       MetalBuf *v_buf, size_t v_off,
                                       MetalBuf *k_cache, MetalBuf *v_cache,
                                       int Hkv, int max_seq, int hd, int pos) {
    id<MTLComputeCommandEncoder> enc = get_encoder();
    [enc setComputePipelineState:g_kv_cache_store];
    [enc setBuffer:k_buf->buf offset:k_off atIndex:0];
    [enc setBuffer:v_buf->buf offset:v_off atIndex:1];
    [enc setBuffer:k_cache->buf offset:0 atIndex:2];
    [enc setBuffer:v_cache->buf offset:0 atIndex:3];
    uint32_t Hkv32 = (uint32_t)Hkv, max_seq32 = (uint32_t)max_seq;
    uint32_t hd32 = (uint32_t)hd, pos32 = (uint32_t)pos;
    [enc setBytes:&Hkv32 length:sizeof(Hkv32) atIndex:4];
    [enc setBytes:&max_seq32 length:sizeof(max_seq32) atIndex:5];
    [enc setBytes:&hd32 length:sizeof(hd32) atIndex:6];
    [enc setBytes:&pos32 length:sizeof(pos32) atIndex:7];

    NSUInteger total = (NSUInteger)Hkv * hd;
    NSUInteger tg = MIN(256, total);
    [enc dispatchThreads:MTLSizeMake(total, 1, 1)
   threadsPerThreadgroup:MTLSizeMake(tg, 1, 1)];
    barrier();
}

void metal_enqueue_attention_off(MetalBuf *Q_buf, size_t q_off,
                                  MetalBuf *Kc_buf, MetalBuf *Vc_buf,
                                  MetalBuf *attn_out_buf, size_t out_off,
                                  int n_attend, int hd, int max_seq, int Hq,
                                  int group_ratio) {
    id<MTLComputeCommandEncoder> enc = get_encoder();
    [enc setComputePipelineState:g_attention];
    [enc setBuffer:Q_buf->buf offset:q_off atIndex:0];
    [enc setBuffer:Kc_buf->buf offset:0 atIndex:1];
    [enc setBuffer:Vc_buf->buf offset:0 atIndex:2];
    [enc setBuffer:attn_out_buf->buf offset:out_off atIndex:3];
    uint32_t n32 = (uint32_t)n_attend, hd32 = (uint32_t)hd;
    uint32_t ms32 = (uint32_t)max_seq, gr32 = (uint32_t)group_ratio;
    float inv_sqrt_hd = 1.0f / sqrtf((float)hd);
    [enc setBytes:&n32 length:sizeof(n32) atIndex:4];
    [enc setBytes:&hd32 length:sizeof(hd32) atIndex:5];
    [enc setBytes:&ms32 length:sizeof(ms32) atIndex:6];
    [enc setBytes:&gr32 length:sizeof(gr32) atIndex:7];
    [enc setBytes:&inv_sqrt_hd length:sizeof(inv_sqrt_hd) atIndex:8];

    NSUInteger tg = 256;
    [enc dispatchThreadgroups:MTLSizeMake(Hq, 1, 1)
        threadsPerThreadgroup:MTLSizeMake(tg, 1, 1)];
    barrier();
}

// ===================================================================
// Training dispatch functions
// ===================================================================

void metal_enqueue_f16_matmul(MetalBuf *W_buf, MetalBuf *A_buf, MetalBuf *C_buf,
                               int M, int N, int K) {
    id<MTLComputeCommandEncoder> enc = get_encoder();
    [enc setComputePipelineState:g_f16_matmul_ps];
    [enc setBuffer:W_buf->buf offset:0 atIndex:0];
    [enc setBuffer:A_buf->buf offset:0 atIndex:1];
    [enc setBuffer:C_buf->buf offset:0 atIndex:2];
    uint32_t M32 = (uint32_t)M, N32 = (uint32_t)N, K32 = (uint32_t)K;
    [enc setBytes:&M32 length:sizeof(M32) atIndex:3];
    [enc setBytes:&K32 length:sizeof(K32) atIndex:4];
    [enc setBytes:&N32 length:sizeof(N32) atIndex:5];

    [enc dispatchThreadgroups:MTLSizeMake((N + 31) / 32, (M + 31) / 32, 1)
        threadsPerThreadgroup:MTLSizeMake(32, 32, 1)];
    barrier();
}

void metal_enqueue_f16_matmul_nt(MetalBuf *W_buf, MetalBuf *A_buf, MetalBuf *C_buf,
                                  int M, int N, int K) {
    id<MTLComputeCommandEncoder> enc = get_encoder();
    [enc setComputePipelineState:g_f16_matmul_nt_ps];
    [enc setBuffer:W_buf->buf offset:0 atIndex:0];
    [enc setBuffer:A_buf->buf offset:0 atIndex:1];
    [enc setBuffer:C_buf->buf offset:0 atIndex:2];
    uint32_t M32 = (uint32_t)M, N32 = (uint32_t)N, K32 = (uint32_t)K;
    [enc setBytes:&M32 length:sizeof(M32) atIndex:3];
    [enc setBytes:&N32 length:sizeof(N32) atIndex:4];
    [enc setBytes:&K32 length:sizeof(K32) atIndex:5];

    [enc dispatchThreadgroups:MTLSizeMake((K + 31) / 32, (M + 31) / 32, 1)
        threadsPerThreadgroup:MTLSizeMake(32, 32, 1)];
    barrier();
}

void metal_enqueue_transpose_heads_fwd(MetalBuf *src, MetalBuf *dst,
                                        int seq_len, int n_heads, int head_dim) {
    id<MTLComputeCommandEncoder> enc = get_encoder();
    [enc setComputePipelineState:g_transpose_fwd_ps];
    [enc setBuffer:src->buf offset:0 atIndex:0];
    [enc setBuffer:dst->buf offset:0 atIndex:1];
    uint32_t sl32 = (uint32_t)seq_len, nh32 = (uint32_t)n_heads, hd32 = (uint32_t)head_dim;
    [enc setBytes:&sl32 length:sizeof(sl32) atIndex:2];
    [enc setBytes:&nh32 length:sizeof(nh32) atIndex:3];
    [enc setBytes:&hd32 length:sizeof(hd32) atIndex:4];

    NSUInteger total = (NSUInteger)seq_len * n_heads * head_dim;
    NSUInteger tpg = MIN(256, total);
    [enc dispatchThreadgroups:MTLSizeMake((total + 255) / 256, 1, 1)
        threadsPerThreadgroup:MTLSizeMake(tpg, 1, 1)];
    barrier();
}

void metal_enqueue_transpose_heads_rev(MetalBuf *src, MetalBuf *dst,
                                        int seq_len, int n_heads, int head_dim) {
    id<MTLComputeCommandEncoder> enc = get_encoder();
    [enc setComputePipelineState:g_transpose_rev_ps];
    [enc setBuffer:src->buf offset:0 atIndex:0];
    [enc setBuffer:dst->buf offset:0 atIndex:1];
    uint32_t sl32 = (uint32_t)seq_len, nh32 = (uint32_t)n_heads, hd32 = (uint32_t)head_dim;
    [enc setBytes:&sl32 length:sizeof(sl32) atIndex:2];
    [enc setBytes:&nh32 length:sizeof(nh32) atIndex:3];
    [enc setBytes:&hd32 length:sizeof(hd32) atIndex:4];

    NSUInteger total = (NSUInteger)seq_len * n_heads * head_dim;
    NSUInteger tpg = MIN(256, total);
    [enc dispatchThreadgroups:MTLSizeMake((total + 255) / 256, 1, 1)
        threadsPerThreadgroup:MTLSizeMake(tpg, 1, 1)];
    barrier();
}

void metal_enqueue_repeat_kv(MetalBuf *src, MetalBuf *dst,
                              int n_kv, int seq_len, int head_dim, int group_ratio) {
    id<MTLComputeCommandEncoder> enc = get_encoder();
    [enc setComputePipelineState:g_repeat_kv_ps];
    [enc setBuffer:src->buf offset:0 atIndex:0];
    [enc setBuffer:dst->buf offset:0 atIndex:1];
    uint32_t nkv32 = (uint32_t)n_kv, sl32 = (uint32_t)seq_len;
    uint32_t hd32 = (uint32_t)head_dim, gr32 = (uint32_t)group_ratio;
    [enc setBytes:&nkv32 length:sizeof(nkv32) atIndex:2];
    [enc setBytes:&sl32 length:sizeof(sl32) atIndex:3];
    [enc setBytes:&hd32 length:sizeof(hd32) atIndex:4];
    [enc setBytes:&gr32 length:sizeof(gr32) atIndex:5];

    NSUInteger total = (NSUInteger)n_kv * group_ratio * seq_len * head_dim;
    NSUInteger tpg = MIN(256, total);
    [enc dispatchThreadgroups:MTLSizeMake((total + 255) / 256, 1, 1)
        threadsPerThreadgroup:MTLSizeMake(tpg, 1, 1)];
    barrier();
}

void metal_enqueue_rope_train(MetalBuf *q_buf, MetalBuf *k_buf,
                               int Hq, int Hkv, int hd, int seq_len, float theta) {
    id<MTLComputeCommandEncoder> enc = get_encoder();
    [enc setComputePipelineState:g_rope_train_ps];
    [enc setBuffer:q_buf->buf offset:0 atIndex:0];
    [enc setBuffer:k_buf->buf offset:0 atIndex:1];
    uint32_t Hq32 = (uint32_t)Hq, Hkv32 = (uint32_t)Hkv;
    uint32_t hd32 = (uint32_t)hd, sl32 = (uint32_t)seq_len;
    [enc setBytes:&Hq32 length:sizeof(Hq32) atIndex:2];
    [enc setBytes:&Hkv32 length:sizeof(Hkv32) atIndex:3];
    [enc setBytes:&hd32 length:sizeof(hd32) atIndex:4];
    [enc setBytes:&sl32 length:sizeof(sl32) atIndex:5];
    [enc setBytes:&theta length:sizeof(theta) atIndex:6];

    uint32_t half_hd = hd / 2;
    NSUInteger total = (NSUInteger)(Hq + Hkv) * seq_len * half_hd;
    NSUInteger tpg = MIN(256, total);
    [enc dispatchThreadgroups:MTLSizeMake((total + 255) / 256, 1, 1)
        threadsPerThreadgroup:MTLSizeMake(tpg, 1, 1)];
    barrier();
}

void metal_enqueue_silu_mul_backward(MetalBuf *dout, MetalBuf *gate, MetalBuf *up,
                                      MetalBuf *dgate, MetalBuf *dup, int n) {
    id<MTLComputeCommandEncoder> enc = get_encoder();
    [enc setComputePipelineState:g_silu_mul_bwd_ps];
    [enc setBuffer:dout->buf offset:0 atIndex:0];
    [enc setBuffer:gate->buf offset:0 atIndex:1];
    [enc setBuffer:up->buf offset:0 atIndex:2];
    [enc setBuffer:dgate->buf offset:0 atIndex:3];
    [enc setBuffer:dup->buf offset:0 atIndex:4];

    [enc dispatchThreadgroups:MTLSizeMake((n + 255) / 256, 1, 1)
        threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
    barrier();
}

void metal_enqueue_gelu_mul_backward(MetalBuf *dout, MetalBuf *gate, MetalBuf *up,
                                      MetalBuf *dgate, MetalBuf *dup, int n) {
    id<MTLComputeCommandEncoder> enc = get_encoder();
    [enc setComputePipelineState:g_gelu_mul_bwd_ps];
    [enc setBuffer:dout->buf offset:0 atIndex:0];
    [enc setBuffer:gate->buf offset:0 atIndex:1];
    [enc setBuffer:up->buf offset:0 atIndex:2];
    [enc setBuffer:dgate->buf offset:0 atIndex:3];
    [enc setBuffer:dup->buf offset:0 atIndex:4];

    [enc dispatchThreadgroups:MTLSizeMake((n + 255) / 256, 1, 1)
        threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
    barrier();
}

void metal_enqueue_rms_norm_backward(MetalBuf *x, MetalBuf *dy, MetalBuf *gamma,
                                      MetalBuf *dx, int batch, int dim, float eps) {
    id<MTLComputeCommandEncoder> enc = get_encoder();
    [enc setComputePipelineState:g_rms_norm_train_bwd_ps];
    [enc setBuffer:x->buf offset:0 atIndex:0];
    [enc setBuffer:dy->buf offset:0 atIndex:1];
    [enc setBuffer:gamma->buf offset:0 atIndex:2];
    [enc setBuffer:dx->buf offset:0 atIndex:3];
    uint32_t dim32 = (uint32_t)dim;
    [enc setBytes:&dim32 length:sizeof(dim32) atIndex:4];
    [enc setBytes:&eps length:sizeof(eps) atIndex:5];

    NSUInteger tpg = MIN(256, (NSUInteger)dim);
    [enc dispatchThreadgroups:MTLSizeMake(batch, 1, 1)
        threadsPerThreadgroup:MTLSizeMake(tpg, 1, 1)];
    barrier();
}

void metal_enqueue_scale(MetalBuf *x, float alpha, int n) {
    id<MTLComputeCommandEncoder> enc = get_encoder();
    [enc setComputePipelineState:g_scale_ps];
    [enc setBuffer:x->buf offset:0 atIndex:0];
    [enc setBytes:&alpha length:sizeof(alpha) atIndex:1];

    [enc dispatchThreadgroups:MTLSizeMake((n + 255) / 256, 1, 1)
        threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
    barrier();
}

void metal_enqueue_add_scaled(MetalBuf *y, MetalBuf *x, float alpha, int n) {
    id<MTLComputeCommandEncoder> enc = get_encoder();
    [enc setComputePipelineState:g_add_scaled_ps];
    [enc setBuffer:y->buf offset:0 atIndex:0];
    [enc setBuffer:x->buf offset:0 atIndex:1];
    [enc setBytes:&alpha length:sizeof(alpha) atIndex:2];

    [enc dispatchThreadgroups:MTLSizeMake((n + 255) / 256, 1, 1)
        threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
    barrier();
}

void metal_enqueue_softmax_ce(MetalBuf *logits, MetalBuf *targets, MetalBuf *losses,
                               MetalBuf *dlogits, int batch, int vocab_size) {
    id<MTLComputeCommandEncoder> enc = get_encoder();
    [enc setComputePipelineState:g_softmax_ce_ps];
    [enc setBuffer:logits->buf offset:0 atIndex:0];
    [enc setBuffer:targets->buf offset:0 atIndex:1];
    [enc setBuffer:losses->buf offset:0 atIndex:2];
    [enc setBuffer:dlogits->buf offset:0 atIndex:3];
    uint32_t vs32 = (uint32_t)vocab_size;
    [enc setBytes:&vs32 length:sizeof(vs32) atIndex:4];

    NSUInteger tpg = MIN(256, (NSUInteger)vocab_size);
    [enc dispatchThreadgroups:MTLSizeMake(batch, 1, 1)
        threadsPerThreadgroup:MTLSizeMake(tpg, 1, 1)];
    barrier();
}

// ===================================================================
// Additional training dispatch functions (attention, LoRA, etc.)
// ===================================================================

void metal_enqueue_attn_train_fwd(MetalBuf *Q, MetalBuf *K, MetalBuf *V,
                                   MetalBuf *out, MetalBuf *probs,
                                   int H, int N, int D, float scale) {
    id<MTLComputeCommandEncoder> enc = get_encoder();
    [enc setComputePipelineState:g_attn_train_fwd_ps];
    [enc setBuffer:Q->buf offset:0 atIndex:0];
    [enc setBuffer:K->buf offset:0 atIndex:1];
    [enc setBuffer:V->buf offset:0 atIndex:2];
    [enc setBuffer:out->buf offset:0 atIndex:3];
    [enc setBuffer:probs->buf offset:0 atIndex:4];
    uint32_t N32 = (uint32_t)N, D32 = (uint32_t)D;
    [enc setBytes:&N32 length:sizeof(N32) atIndex:5];
    [enc setBytes:&D32 length:sizeof(D32) atIndex:6];
    [enc setBytes:&scale length:sizeof(scale) atIndex:7];

    NSUInteger tpg = MIN(256, (NSUInteger)N);
    [enc dispatchThreadgroups:MTLSizeMake(H, N, 1)
        threadsPerThreadgroup:MTLSizeMake(tpg, 1, 1)];
    barrier();
}

void metal_enqueue_attn_bwd_dq(MetalBuf *d_out, MetalBuf *probs, MetalBuf *V,
                                MetalBuf *K, MetalBuf *d_score, MetalBuf *dQ,
                                int H, int N, int D, float scale) {
    id<MTLComputeCommandEncoder> enc = get_encoder();
    [enc setComputePipelineState:g_attn_bwd_dq_ps];
    [enc setBuffer:d_out->buf offset:0 atIndex:0];
    [enc setBuffer:probs->buf offset:0 atIndex:1];
    [enc setBuffer:V->buf offset:0 atIndex:2];
    [enc setBuffer:K->buf offset:0 atIndex:3];
    [enc setBuffer:d_score->buf offset:0 atIndex:4];
    [enc setBuffer:dQ->buf offset:0 atIndex:5];
    uint32_t N32 = (uint32_t)N, D32 = (uint32_t)D;
    [enc setBytes:&N32 length:sizeof(N32) atIndex:6];
    [enc setBytes:&D32 length:sizeof(D32) atIndex:7];
    [enc setBytes:&scale length:sizeof(scale) atIndex:8];

    NSUInteger tpg = MIN(256, (NSUInteger)N);
    [enc dispatchThreadgroups:MTLSizeMake(H, N, 1)
        threadsPerThreadgroup:MTLSizeMake(tpg, 1, 1)];
    barrier();
}

void metal_enqueue_attn_bwd_dkv(MetalBuf *d_score, MetalBuf *Q, MetalBuf *probs,
                                 MetalBuf *d_out, MetalBuf *dK, MetalBuf *dV,
                                 int H, int N, int D) {
    id<MTLComputeCommandEncoder> enc = get_encoder();
    [enc setComputePipelineState:g_attn_bwd_dkv_ps];
    [enc setBuffer:d_score->buf offset:0 atIndex:0];
    [enc setBuffer:Q->buf offset:0 atIndex:1];
    [enc setBuffer:probs->buf offset:0 atIndex:2];
    [enc setBuffer:d_out->buf offset:0 atIndex:3];
    [enc setBuffer:dK->buf offset:0 atIndex:4];
    [enc setBuffer:dV->buf offset:0 atIndex:5];
    uint32_t N32 = (uint32_t)N, D32 = (uint32_t)D;
    [enc setBytes:&N32 length:sizeof(N32) atIndex:6];
    [enc setBytes:&D32 length:sizeof(D32) atIndex:7];

    NSUInteger tpg = MIN(256, (NSUInteger)D);
    [enc dispatchThreadgroups:MTLSizeMake(H, N, 1)
        threadsPerThreadgroup:MTLSizeMake(tpg, 1, 1)];
    barrier();
}

void metal_enqueue_repeat_kv_bwd(MetalBuf *d_expanded, MetalBuf *d_kv,
                                  int n_kv, int seq_len, int head_dim, int group_ratio) {
    id<MTLComputeCommandEncoder> enc = get_encoder();
    [enc setComputePipelineState:g_repeat_kv_bwd_ps];
    [enc setBuffer:d_expanded->buf offset:0 atIndex:0];
    [enc setBuffer:d_kv->buf offset:0 atIndex:1];
    uint32_t nkv32 = (uint32_t)n_kv, sl32 = (uint32_t)seq_len;
    uint32_t hd32 = (uint32_t)head_dim, gr32 = (uint32_t)group_ratio;
    [enc setBytes:&nkv32 length:sizeof(nkv32) atIndex:2];
    [enc setBytes:&sl32 length:sizeof(sl32) atIndex:3];
    [enc setBytes:&hd32 length:sizeof(hd32) atIndex:4];
    [enc setBytes:&gr32 length:sizeof(gr32) atIndex:5];

    NSUInteger total = (NSUInteger)n_kv * seq_len * head_dim;
    NSUInteger tpg = MIN(256, total);
    [enc dispatchThreadgroups:MTLSizeMake((total + 255) / 256, 1, 1)
        threadsPerThreadgroup:MTLSizeMake(tpg, 1, 1)];
    barrier();
}

void metal_enqueue_rope_train_bwd(MetalBuf *q_buf, MetalBuf *k_buf,
                                   int Hq, int Hkv, int hd, int seq_len, float theta) {
    id<MTLComputeCommandEncoder> enc = get_encoder();
    [enc setComputePipelineState:g_rope_train_bwd_ps];
    [enc setBuffer:q_buf->buf offset:0 atIndex:0];
    [enc setBuffer:k_buf->buf offset:0 atIndex:1];
    uint32_t Hq32 = (uint32_t)Hq, Hkv32 = (uint32_t)Hkv;
    uint32_t hd32 = (uint32_t)hd, sl32 = (uint32_t)seq_len;
    [enc setBytes:&Hq32 length:sizeof(Hq32) atIndex:2];
    [enc setBytes:&Hkv32 length:sizeof(Hkv32) atIndex:3];
    [enc setBytes:&hd32 length:sizeof(hd32) atIndex:4];
    [enc setBytes:&sl32 length:sizeof(sl32) atIndex:5];
    [enc setBytes:&theta length:sizeof(theta) atIndex:6];

    uint32_t half_hd = hd / 2;
    NSUInteger total = (NSUInteger)(Hq + Hkv) * seq_len * half_hd;
    NSUInteger tpg = MIN(256, total);
    [enc dispatchThreadgroups:MTLSizeMake((total + 255) / 256, 1, 1)
        threadsPerThreadgroup:MTLSizeMake(tpg, 1, 1)];
    barrier();
}

void metal_enqueue_float_matmul(MetalBuf *A, MetalBuf *W, MetalBuf *C,
                                 int M, int N, int K) {
    id<MTLComputeCommandEncoder> enc = get_encoder();
    [enc setComputePipelineState:g_float_matmul_ps];
    [enc setBuffer:A->buf offset:0 atIndex:0];
    [enc setBuffer:W->buf offset:0 atIndex:1];
    [enc setBuffer:C->buf offset:0 atIndex:2];
    uint32_t M32 = (uint32_t)M, K32 = (uint32_t)K, N32 = (uint32_t)N;
    [enc setBytes:&M32 length:sizeof(M32) atIndex:3];
    [enc setBytes:&K32 length:sizeof(K32) atIndex:4];
    [enc setBytes:&N32 length:sizeof(N32) atIndex:5];

    [enc dispatchThreadgroups:MTLSizeMake((N + 15) / 16, (M + 15) / 16, 1)
        threadsPerThreadgroup:MTLSizeMake(16, 16, 1)];
    barrier();
}

void metal_enqueue_float_matmul_tn(MetalBuf *A, MetalBuf *B, MetalBuf *C,
                                    int M, int N, int K) {
    id<MTLComputeCommandEncoder> enc = get_encoder();
    [enc setComputePipelineState:g_float_matmul_tn_ps];
    [enc setBuffer:A->buf offset:0 atIndex:0];
    [enc setBuffer:B->buf offset:0 atIndex:1];
    [enc setBuffer:C->buf offset:0 atIndex:2];
    uint32_t M32 = (uint32_t)M, K32 = (uint32_t)K, N32 = (uint32_t)N;
    [enc setBytes:&M32 length:sizeof(M32) atIndex:3];
    [enc setBytes:&K32 length:sizeof(K32) atIndex:4];
    [enc setBytes:&N32 length:sizeof(N32) atIndex:5];

    [enc dispatchThreadgroups:MTLSizeMake((N + 15) / 16, (M + 15) / 16, 1)
        threadsPerThreadgroup:MTLSizeMake(16, 16, 1)];
    barrier();
}

void metal_enqueue_float_matmul_nt(MetalBuf *A, MetalBuf *W, MetalBuf *C,
                                    int M, int N, int K) {
    id<MTLComputeCommandEncoder> enc = get_encoder();
    [enc setComputePipelineState:g_float_matmul_nt_ps];
    [enc setBuffer:A->buf offset:0 atIndex:0];
    [enc setBuffer:W->buf offset:0 atIndex:1];
    [enc setBuffer:C->buf offset:0 atIndex:2];
    uint32_t M32 = (uint32_t)M, N32 = (uint32_t)N, K32 = (uint32_t)K;
    [enc setBytes:&M32 length:sizeof(M32) atIndex:3];
    [enc setBytes:&N32 length:sizeof(N32) atIndex:4];
    [enc setBytes:&K32 length:sizeof(K32) atIndex:5];

    [enc dispatchThreadgroups:MTLSizeMake((K + 15) / 16, (M + 15) / 16, 1)
        threadsPerThreadgroup:MTLSizeMake(16, 16, 1)];
    barrier();
}

void metal_enqueue_copy(MetalBuf *src, MetalBuf *dst, int n) {
    id<MTLComputeCommandEncoder> enc = get_encoder();
    [enc setComputePipelineState:g_buf_copy_ps];
    [enc setBuffer:src->buf offset:0 atIndex:0];
    [enc setBuffer:dst->buf offset:0 atIndex:1];

    [enc dispatchThreadgroups:MTLSizeMake((n + 255) / 256, 1, 1)
        threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
    barrier();
}

void metal_enqueue_clamp(MetalBuf *x, float max_val, int n) {
    id<MTLComputeCommandEncoder> enc = get_encoder();
    [enc setComputePipelineState:g_clamp_ps];
    [enc setBuffer:x->buf offset:0 atIndex:0];
    [enc setBytes:&max_val length:sizeof(max_val) atIndex:1];

    [enc dispatchThreadgroups:MTLSizeMake((n + 255) / 256, 1, 1)
        threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
    barrier();
}

void metal_enqueue_f16_sgd_fused(MetalBuf *W, MetalBuf *dY, MetalBuf *X,
                                   int M, int K, int N, float lr) {
    id<MTLComputeCommandEncoder> enc = get_encoder();
    [enc setComputePipelineState:g_f16_sgd_fused_ps];
    [enc setBuffer:W->buf offset:0 atIndex:0];
    [enc setBuffer:dY->buf offset:0 atIndex:1];
    [enc setBuffer:X->buf offset:0 atIndex:2];
    uint32_t M32 = (uint32_t)M, K32 = (uint32_t)K, N32 = (uint32_t)N;
    [enc setBytes:&M32 length:sizeof(M32) atIndex:3];
    [enc setBytes:&K32 length:sizeof(K32) atIndex:4];
    [enc setBytes:&N32 length:sizeof(N32) atIndex:5];
    [enc setBytes:&lr length:sizeof(lr) atIndex:6];

    [enc dispatchThreadgroups:MTLSizeMake((N + 15) / 16, (M + 15) / 16, 1)
        threadsPerThreadgroup:MTLSizeMake(16, 16, 1)];
    barrier();
}

void metal_enqueue_grpo_policy_grad(MetalBuf *logits, MetalBuf *actions,
    MetalBuf *old_lp, MetalBuf *advs, MetalBuf *new_lp, MetalBuf *dlogits,
    int N, int V, float clip_eps) {
    id<MTLComputeCommandEncoder> enc = get_encoder();
    [enc setComputePipelineState:g_grpo_pg_ps];
    [enc setBuffer:logits->buf offset:0 atIndex:0];
    [enc setBuffer:actions->buf offset:0 atIndex:1];
    [enc setBuffer:old_lp->buf offset:0 atIndex:2];
    [enc setBuffer:advs->buf offset:0 atIndex:3];
    [enc setBuffer:new_lp->buf offset:0 atIndex:4];
    [enc setBuffer:dlogits->buf offset:0 atIndex:5];
    uint32_t V32 = (uint32_t)V;
    [enc setBytes:&V32 length:sizeof(V32) atIndex:6];
    [enc setBytes:&clip_eps length:sizeof(clip_eps) atIndex:7];

    [enc dispatchThreadgroups:MTLSizeMake(N, 1, 1)
        threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
    barrier();
}

void metal_enqueue_f16_grad_accum(MetalBuf *G, MetalBuf *dY, MetalBuf *X,
                                    int M, int K, int N) {
    id<MTLComputeCommandEncoder> enc = get_encoder();
    [enc setComputePipelineState:g_f16_grad_accum_ps];
    [enc setBuffer:G->buf offset:0 atIndex:0];
    [enc setBuffer:dY->buf offset:0 atIndex:1];
    [enc setBuffer:X->buf offset:0 atIndex:2];
    uint32_t M32 = (uint32_t)M, K32 = (uint32_t)K, N32 = (uint32_t)N;
    [enc setBytes:&M32 length:sizeof(M32) atIndex:3];
    [enc setBytes:&K32 length:sizeof(K32) atIndex:4];
    [enc setBytes:&N32 length:sizeof(N32) atIndex:5];

    [enc dispatchThreadgroups:MTLSizeMake((N + 15) / 16, (M + 15) / 16, 1)
        threadsPerThreadgroup:MTLSizeMake(16, 16, 1)];
    barrier();
}

void metal_enqueue_f16_sgd_apply(MetalBuf *W, MetalBuf *G, int n, float lr) {
    id<MTLComputeCommandEncoder> enc = get_encoder();
    [enc setComputePipelineState:g_f16_sgd_apply_ps];
    [enc setBuffer:W->buf offset:0 atIndex:0];
    [enc setBuffer:G->buf offset:0 atIndex:1];
    uint32_t n32 = (uint32_t)n;
    [enc setBytes:&n32 length:sizeof(n32) atIndex:2];
    [enc setBytes:&lr length:sizeof(lr) atIndex:3];

    [enc dispatchThreadgroups:MTLSizeMake((n + 255) / 256, 1, 1)
        threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
    barrier();
}
