#import <Metal/Metal.h>
#import <Foundation/Foundation.h>
#include "metal_backend.h"
#include <stdio.h>

// ------------------------------------------------------------------
// Metal state
// ------------------------------------------------------------------
static id<MTLDevice>       g_device       = nil;
static id<MTLCommandQueue> g_queue        = nil;
static id<MTLLibrary>      g_library      = nil;
static int                 g_initialized  = 0;

// Pipeline state cache
static NSMutableDictionary<NSString *, id<MTLComputePipelineState>> *g_pipelines = nil;

// Double-buffered async command pipeline
// Active buffer is being encoded; previous buffer may still be executing
static id<MTLCommandBuffer> g_cmd_buf = nil;      // active (being encoded)
static id<MTLCommandBuffer> g_prev_cmd_buf = nil;  // previous (may be in-flight)
static int g_encode_count = 0;                      // encodings since last commit
static const int MAX_ENCODINGS_PER_BUFFER = 64;     // auto-flush threshold

// ------------------------------------------------------------------
// Helpers
// ------------------------------------------------------------------
static id<MTLComputePipelineState> get_pipeline(NSString *name) {
    id<MTLComputePipelineState> ps = g_pipelines[name];
    if (ps) return ps;

    NSError *err = nil;
    id<MTLFunction> fn = [g_library newFunctionWithName:name];
    if (!fn) {
        fprintf(stderr, "[Metal] Function '%s' not found in library\n", [name UTF8String]);
        return nil;
    }
    ps = [g_device newComputePipelineStateWithFunction:fn error:&err];
    if (err) {
        fprintf(stderr, "[Metal] Pipeline error for '%s': %s\n",
                [name UTF8String], [[err localizedDescription] UTF8String]);
        return nil;
    }
    g_pipelines[name] = ps;
    return ps;
}

static id<MTLCommandBuffer> get_command_buffer(void) {
    if (!g_cmd_buf) {
        // Wait for previous buffer to finish before reusing
        if (g_prev_cmd_buf) {
            [g_prev_cmd_buf waitUntilCompleted];
            g_prev_cmd_buf = nil;
        }
        g_cmd_buf = [g_queue commandBuffer];
        g_encode_count = 0;
    }
    return g_cmd_buf;
}

// Commit current buffer and rotate (async — doesn't wait)
static void commit_command_buffer(void) {
    if (g_cmd_buf) {
        [g_cmd_buf commit];
        // Previous buffer should already be done (waited in get_command_buffer)
        g_prev_cmd_buf = g_cmd_buf;
        g_cmd_buf = nil;
        g_encode_count = 0;
    }
}

// Full synchronization: wait for all in-flight work
static void flush_command_buffer(void) {
    if (g_cmd_buf) {
        [g_cmd_buf commit];
        [g_cmd_buf waitUntilCompleted];
        g_cmd_buf = nil;
    }
    if (g_prev_cmd_buf) {
        [g_prev_cmd_buf waitUntilCompleted];
        g_prev_cmd_buf = nil;
    }
    g_encode_count = 0;
}

// Auto-flush: commit if too many encodings batched (keeps latency bounded)
static void maybe_auto_flush(void) {
    g_encode_count++;
    if (g_encode_count >= MAX_ENCODINGS_PER_BUFFER) {
        commit_command_buffer();
    }
}

// ------------------------------------------------------------------
// Init / cleanup
// ------------------------------------------------------------------
int metal_init(const char *shader_path) {
    @autoreleasepool {
        g_device = MTLCreateSystemDefaultDevice();
        if (!g_device) {
            fprintf(stderr, "[Metal] No Metal device found\n");
            return -1;
        }
        printf("[Metal] Device: %s\n", [[g_device name] UTF8String]);
        printf("[Metal] Unified memory: %s\n", g_device.hasUnifiedMemory ? "YES" : "NO");

        g_queue = [g_device newCommandQueue];
        g_pipelines = [NSMutableDictionary new];

        // Load shader library
        NSError *err = nil;
        NSString *path = [NSString stringWithUTF8String:shader_path];

        // Try .metallib first, then compile from .metal source
        if ([path hasSuffix:@".metallib"]) {
            NSURL *url = [NSURL fileURLWithPath:path];
            g_library = [g_device newLibraryWithURL:url error:&err];
        } else {
            NSString *src = [NSString stringWithContentsOfFile:path
                                                     encoding:NSUTF8StringEncoding
                                                        error:&err];
            if (!src) {
                fprintf(stderr, "[Metal] Cannot read shader file: %s\n", shader_path);
                return -1;
            }
            MTLCompileOptions *opts = [[MTLCompileOptions alloc] init];
            opts.mathMode = MTLMathModeFast;
            g_library = [g_device newLibraryWithSource:src options:opts error:&err];
        }
        if (err) {
            fprintf(stderr, "[Metal] Library error: %s\n", [[err localizedDescription] UTF8String]);
            if (!g_library) return -1;
        }

        g_initialized = 1;
        printf("[Metal] Initialized successfully\n");
        return 0;
    }
}

void metal_cleanup(void) {
    @autoreleasepool {
        flush_command_buffer();
        g_prev_cmd_buf = nil;
        g_pipelines = nil;
        g_library = nil;
        g_queue = nil;
        g_device = nil;
        g_initialized = 0;
    }
}

int metal_is_initialized(void) {
    return g_initialized;
}

// ------------------------------------------------------------------
// Buffer management
// ------------------------------------------------------------------
void *metal_create_shared_buffer(size_t bytes) {
    @autoreleasepool {
        if (bytes == 0) bytes = 16; // Metal doesn't like zero-size
        id<MTLBuffer> buf = [g_device newBufferWithLength:bytes
                                                  options:MTLResourceStorageModeShared];
        return (__bridge_retained void *)buf;
    }
}

void metal_free_buffer(void *buf) {
    if (!buf) return;
    @autoreleasepool {
        (void)(__bridge_transfer id<MTLBuffer>)buf;
    }
}

void *metal_buffer_contents(void *buf) {
    id<MTLBuffer> mtl_buf = (__bridge id<MTLBuffer>)buf;
    return [mtl_buf contents];
}

// ------------------------------------------------------------------
// Kernel dispatch helpers
// ------------------------------------------------------------------
typedef struct {
    void *bufs[16];
    int n_bufs;
    uint32_t params[32];
    int n_params;
} KernelArgs;

static void dispatch_kernel(NSString *name, KernelArgs *args,
                           MTLSize grid, MTLSize block) {
    @autoreleasepool {
        id<MTLComputePipelineState> ps = get_pipeline(name);
        if (!ps) return;

        id<MTLCommandBuffer> cmd = get_command_buffer();
        id<MTLComputeCommandEncoder> enc = [cmd computeCommandEncoder];

        [enc setComputePipelineState:ps];

        for (int i = 0; i < args->n_bufs; i++) {
            [enc setBuffer:(__bridge id<MTLBuffer>)args->bufs[i] offset:0 atIndex:i];
        }

        if (args->n_params > 0) {
            [enc setBytes:args->params
                   length:args->n_params * sizeof(uint32_t)
                  atIndex:args->n_bufs];
        }

        [enc dispatchThreadgroups:grid threadsPerThreadgroup:block];
        [enc endEncoding];
        maybe_auto_flush();
    }
}

// ------------------------------------------------------------------
// Matmul: C[M,N] = A[M,K] @ B[K,N] (with optional transposes)
// TILE_SIZE=32 threadgroup tiling with bank-conflict avoidance
// ------------------------------------------------------------------
void metal_matmul(void *buf_a, void *buf_b, void *buf_c,
                  int M, int K, int N, int transpose_a, int transpose_b) {
    KernelArgs args = {
        .bufs = {buf_a, buf_b, buf_c},
        .n_bufs = 3,
        .n_params = 5
    };
    args.params[0] = (uint32_t)M;
    args.params[1] = (uint32_t)K;
    args.params[2] = (uint32_t)N;
    args.params[3] = (uint32_t)transpose_a;
    args.params[4] = (uint32_t)transpose_b;

    MTLSize grid = MTLSizeMake((N + 31) / 32, (M + 31) / 32, 1);
    MTLSize block = MTLSizeMake(32, 32, 1);
    dispatch_kernel(@"matmul_tiled", &args, grid, block);
}

void metal_matmul_batched(void *buf_a, void *buf_b, void *buf_c,
                          int batch, int M, int K, int N,
                          int transpose_a, int transpose_b) {
    KernelArgs args = {
        .bufs = {buf_a, buf_b, buf_c},
        .n_bufs = 3,
        .n_params = 5
    };
    args.params[0] = (uint32_t)M;
    args.params[1] = (uint32_t)K;
    args.params[2] = (uint32_t)N;
    args.params[3] = (uint32_t)transpose_a;
    args.params[4] = (uint32_t)transpose_b;

    MTLSize grid = MTLSizeMake((N + 31) / 32, (M + 31) / 32, batch);
    MTLSize block = MTLSizeMake(32, 32, 1);
    dispatch_kernel(@"matmul_batched", &args, grid, block);
}

// ------------------------------------------------------------------
// Flash Attention (Block-tiled, BLOCK_Q=32)
// Q,K,V,O: [B*H, N, D]
// ------------------------------------------------------------------
void metal_flash_attention(void *buf_q, void *buf_k, void *buf_v, void *buf_o,
                           int B, int H, int N, int D, float scale, int is_causal) {
    @autoreleasepool {
        id<MTLComputePipelineState> ps = get_pipeline(@"flash_attention");
        if (!ps) return;

        id<MTLCommandBuffer> cmd = get_command_buffer();
        id<MTLComputeCommandEncoder> enc = [cmd computeCommandEncoder];
        [enc setComputePipelineState:ps];

        [enc setBuffer:(__bridge id<MTLBuffer>)buf_q offset:0 atIndex:0];
        [enc setBuffer:(__bridge id<MTLBuffer>)buf_k offset:0 atIndex:1];
        [enc setBuffer:(__bridge id<MTLBuffer>)buf_v offset:0 atIndex:2];
        [enc setBuffer:(__bridge id<MTLBuffer>)buf_o offset:0 atIndex:3];

        uint32_t params[] = {(uint32_t)N, (uint32_t)D, (uint32_t)is_causal};
        [enc setBytes:params length:sizeof(params) atIndex:4];
        float fparams[] = {scale};
        [enc setBytes:fparams length:sizeof(fparams) atIndex:5];

        // One threadgroup per (query_block, batch*head)
        // Each threadgroup has BLOCK_Q=32 threads, one per query row
        int n_q_blocks = (N + 32 - 1) / 32;
        MTLSize grid = MTLSizeMake(n_q_blocks, B * H, 1);
        MTLSize block = MTLSizeMake(32, 1, 1);
        [enc dispatchThreadgroups:grid threadsPerThreadgroup:block];
        [enc endEncoding];
    }
}

void metal_flash_attention_backward(void *buf_q, void *buf_k, void *buf_v,
                                    void *buf_o, void *buf_do,
                                    void *buf_dq, void *buf_dk, void *buf_dv,
                                    int B, int H, int N, int D, float scale, int is_causal) {
    @autoreleasepool {
        id<MTLComputePipelineState> ps = get_pipeline(@"flash_attention_backward");
        if (!ps) return;

        id<MTLCommandBuffer> cmd = get_command_buffer();
        id<MTLComputeCommandEncoder> enc = [cmd computeCommandEncoder];
        [enc setComputePipelineState:ps];

        [enc setBuffer:(__bridge id<MTLBuffer>)buf_q  offset:0 atIndex:0];
        [enc setBuffer:(__bridge id<MTLBuffer>)buf_k  offset:0 atIndex:1];
        [enc setBuffer:(__bridge id<MTLBuffer>)buf_v  offset:0 atIndex:2];
        [enc setBuffer:(__bridge id<MTLBuffer>)buf_o  offset:0 atIndex:3];
        [enc setBuffer:(__bridge id<MTLBuffer>)buf_do offset:0 atIndex:4];
        [enc setBuffer:(__bridge id<MTLBuffer>)buf_dq offset:0 atIndex:5];
        [enc setBuffer:(__bridge id<MTLBuffer>)buf_dk offset:0 atIndex:6];
        [enc setBuffer:(__bridge id<MTLBuffer>)buf_dv offset:0 atIndex:7];

        uint32_t params[] = {(uint32_t)N, (uint32_t)D, (uint32_t)is_causal};
        [enc setBytes:params length:sizeof(params) atIndex:8];
        float fparams[] = {scale};
        [enc setBytes:fparams length:sizeof(fparams) atIndex:9];

        int n_q_blocks = (N + 32 - 1) / 32;
        MTLSize grid = MTLSizeMake(n_q_blocks, B * H, 1);
        MTLSize block = MTLSizeMake(32, 1, 1);
        [enc dispatchThreadgroups:grid threadsPerThreadgroup:block];
        [enc endEncoding];
    }
}

// ------------------------------------------------------------------
// Layer Norm
// ------------------------------------------------------------------
void metal_layer_norm(void *buf_x, void *buf_out, void *buf_gamma, void *buf_beta,
                      int batch, int dim, float eps) {
    @autoreleasepool {
        id<MTLComputePipelineState> ps = get_pipeline(@"layer_norm");
        if (!ps) return;

        id<MTLCommandBuffer> cmd = get_command_buffer();
        id<MTLComputeCommandEncoder> enc = [cmd computeCommandEncoder];
        [enc setComputePipelineState:ps];

        [enc setBuffer:(__bridge id<MTLBuffer>)buf_x     offset:0 atIndex:0];
        [enc setBuffer:(__bridge id<MTLBuffer>)buf_out   offset:0 atIndex:1];
        [enc setBuffer:(__bridge id<MTLBuffer>)buf_gamma offset:0 atIndex:2];
        [enc setBuffer:(__bridge id<MTLBuffer>)buf_beta  offset:0 atIndex:3];

        uint32_t params[] = {(uint32_t)dim};
        [enc setBytes:params length:sizeof(params) atIndex:4];
        float fparams[] = {eps};
        [enc setBytes:fparams length:sizeof(fparams) atIndex:5];

        MTLSize grid = MTLSizeMake(batch, 1, 1);
        int tg_size = dim < 256 ? dim : 256;
        MTLSize block = MTLSizeMake(tg_size, 1, 1);
        [enc dispatchThreadgroups:grid threadsPerThreadgroup:block];
        [enc endEncoding];
    }
}

void metal_layer_norm_backward(void *buf_x, void *buf_dy, void *buf_gamma,
                               void *buf_dx, void *buf_dgamma, void *buf_dbeta,
                               int batch, int dim, float eps) {
    @autoreleasepool {
        id<MTLComputePipelineState> ps = get_pipeline(@"layer_norm_backward");
        if (!ps) return;

        id<MTLCommandBuffer> cmd = get_command_buffer();
        id<MTLComputeCommandEncoder> enc = [cmd computeCommandEncoder];
        [enc setComputePipelineState:ps];

        [enc setBuffer:(__bridge id<MTLBuffer>)buf_x      offset:0 atIndex:0];
        [enc setBuffer:(__bridge id<MTLBuffer>)buf_dy     offset:0 atIndex:1];
        [enc setBuffer:(__bridge id<MTLBuffer>)buf_gamma  offset:0 atIndex:2];
        [enc setBuffer:(__bridge id<MTLBuffer>)buf_dx     offset:0 atIndex:3];
        [enc setBuffer:(__bridge id<MTLBuffer>)buf_dgamma offset:0 atIndex:4];
        [enc setBuffer:(__bridge id<MTLBuffer>)buf_dbeta  offset:0 atIndex:5];

        uint32_t params[] = {(uint32_t)batch, (uint32_t)dim};
        [enc setBytes:params length:sizeof(params) atIndex:6];
        float fparams[] = {eps};
        [enc setBytes:fparams length:sizeof(fparams) atIndex:7];

        MTLSize grid = MTLSizeMake(batch, 1, 1);
        int tg_size = dim < 256 ? dim : 256;
        MTLSize block = MTLSizeMake(tg_size, 1, 1);
        [enc dispatchThreadgroups:grid threadsPerThreadgroup:block];
        [enc endEncoding];
    }
}

// ------------------------------------------------------------------
// Fused LayerNorm + Residual Add: out = LayerNorm(x + residual)
// ------------------------------------------------------------------
void metal_layer_norm_residual(void *buf_x, void *buf_residual, void *buf_out,
                               void *buf_gamma, void *buf_beta,
                               int batch, int dim, float eps) {
    @autoreleasepool {
        id<MTLComputePipelineState> ps = get_pipeline(@"layer_norm_residual");
        if (!ps) return;

        id<MTLCommandBuffer> cmd = get_command_buffer();
        id<MTLComputeCommandEncoder> enc = [cmd computeCommandEncoder];
        [enc setComputePipelineState:ps];

        [enc setBuffer:(__bridge id<MTLBuffer>)buf_x        offset:0 atIndex:0];
        [enc setBuffer:(__bridge id<MTLBuffer>)buf_residual  offset:0 atIndex:1];
        [enc setBuffer:(__bridge id<MTLBuffer>)buf_out       offset:0 atIndex:2];
        [enc setBuffer:(__bridge id<MTLBuffer>)buf_gamma     offset:0 atIndex:3];
        [enc setBuffer:(__bridge id<MTLBuffer>)buf_beta      offset:0 atIndex:4];

        uint32_t params[] = {(uint32_t)dim};
        [enc setBytes:params length:sizeof(params) atIndex:5];
        float fparams[] = {eps};
        [enc setBytes:fparams length:sizeof(fparams) atIndex:6];

        MTLSize grid = MTLSizeMake(batch, 1, 1);
        int tg_size = dim < 256 ? dim : 256;
        MTLSize block = MTLSizeMake(tg_size, 1, 1);
        [enc dispatchThreadgroups:grid threadsPerThreadgroup:block];
        [enc endEncoding];
    }
}

// ------------------------------------------------------------------
// GELU (float4 vectorized — each thread processes 4 elements)
// ------------------------------------------------------------------
void metal_gelu(void *buf_in, void *buf_out, int n) {
    KernelArgs args = {
        .bufs = {buf_in, buf_out},
        .n_bufs = 2,
        .n_params = 1
    };
    args.params[0] = (uint32_t)n;
    int n4 = (n + 3) / 4;  // number of threads needed (each does 4 elements)
    MTLSize grid = MTLSizeMake((n4 + 255) / 256, 1, 1);
    MTLSize block = MTLSizeMake(256, 1, 1);
    dispatch_kernel(@"gelu_forward", &args, grid, block);
}

void metal_gelu_backward(void *buf_in, void *buf_dy, void *buf_dx, int n) {
    KernelArgs args = {
        .bufs = {buf_in, buf_dy, buf_dx},
        .n_bufs = 3,
        .n_params = 1
    };
    args.params[0] = (uint32_t)n;
    int n4 = (n + 3) / 4;
    MTLSize grid = MTLSizeMake((n4 + 255) / 256, 1, 1);
    MTLSize block = MTLSizeMake(256, 1, 1);
    dispatch_kernel(@"gelu_backward", &args, grid, block);
}

// ------------------------------------------------------------------
// RoPE
// ------------------------------------------------------------------
void metal_rope(void *buf_x, void *buf_out, int B, int H, int N, int D, int offset) {
    @autoreleasepool {
        id<MTLComputePipelineState> ps = get_pipeline(@"rope_forward");
        if (!ps) return;

        id<MTLCommandBuffer> cmd = get_command_buffer();
        id<MTLComputeCommandEncoder> enc = [cmd computeCommandEncoder];
        [enc setComputePipelineState:ps];

        [enc setBuffer:(__bridge id<MTLBuffer>)buf_x   offset:0 atIndex:0];
        [enc setBuffer:(__bridge id<MTLBuffer>)buf_out offset:0 atIndex:1];

        uint32_t params[] = {(uint32_t)B, (uint32_t)H, (uint32_t)N,
                             (uint32_t)D, (uint32_t)offset};
        [enc setBytes:params length:sizeof(params) atIndex:2];

        MTLSize grid = MTLSizeMake((D/2 + 63) / 64, N, B * H);
        MTLSize block = MTLSizeMake(64, 1, 1);
        [enc dispatchThreadgroups:grid threadsPerThreadgroup:block];
        [enc endEncoding];
    }
}

void metal_rope_backward(void *buf_dy, void *buf_dx, int B, int H, int N, int D, int offset) {
    @autoreleasepool {
        id<MTLComputePipelineState> ps = get_pipeline(@"rope_backward");
        if (!ps) return;

        id<MTLCommandBuffer> cmd = get_command_buffer();
        id<MTLComputeCommandEncoder> enc = [cmd computeCommandEncoder];
        [enc setComputePipelineState:ps];

        [enc setBuffer:(__bridge id<MTLBuffer>)buf_dy offset:0 atIndex:0];
        [enc setBuffer:(__bridge id<MTLBuffer>)buf_dx offset:0 atIndex:1];

        uint32_t params[] = {(uint32_t)B, (uint32_t)H, (uint32_t)N,
                             (uint32_t)D, (uint32_t)offset};
        [enc setBytes:params length:sizeof(params) atIndex:2];

        MTLSize grid = MTLSizeMake((D/2 + 63) / 64, N, B * H);
        MTLSize block = MTLSizeMake(64, 1, 1);
        [enc dispatchThreadgroups:grid threadsPerThreadgroup:block];
        [enc endEncoding];
    }
}

// ------------------------------------------------------------------
// Embedding
// ------------------------------------------------------------------
void metal_embedding(void *buf_table, void *buf_indices, void *buf_out,
                     int vocab_size, int dim, int seq_len) {
    KernelArgs args = {
        .bufs = {buf_table, buf_indices, buf_out},
        .n_bufs = 3,
        .n_params = 3
    };
    args.params[0] = (uint32_t)vocab_size;
    args.params[1] = (uint32_t)dim;
    args.params[2] = (uint32_t)seq_len;

    MTLSize grid = MTLSizeMake((dim + 63) / 64, seq_len, 1);
    MTLSize block = MTLSizeMake(64, 1, 1);
    dispatch_kernel(@"embedding_lookup", &args, grid, block);
}

void metal_embedding_backward(void *buf_dy, void *buf_indices, void *buf_dtable,
                              int vocab_size, int dim, int seq_len) {
    KernelArgs args = {
        .bufs = {buf_dy, buf_indices, buf_dtable},
        .n_bufs = 3,
        .n_params = 3
    };
    args.params[0] = (uint32_t)vocab_size;
    args.params[1] = (uint32_t)dim;
    args.params[2] = (uint32_t)seq_len;

    MTLSize grid = MTLSizeMake((dim + 63) / 64, seq_len, 1);
    MTLSize block = MTLSizeMake(64, 1, 1);
    dispatch_kernel(@"embedding_backward", &args, grid, block);
}

// ------------------------------------------------------------------
// Softmax + Cross Entropy
// ------------------------------------------------------------------
void metal_softmax_ce_loss(void *buf_logits, void *buf_targets, void *buf_loss,
                           void *buf_dlogits, int batch, int vocab_size) {
    @autoreleasepool {
        id<MTLComputePipelineState> ps = get_pipeline(@"softmax_ce_loss");
        if (!ps) return;

        id<MTLCommandBuffer> cmd = get_command_buffer();
        id<MTLComputeCommandEncoder> enc = [cmd computeCommandEncoder];
        [enc setComputePipelineState:ps];

        [enc setBuffer:(__bridge id<MTLBuffer>)buf_logits  offset:0 atIndex:0];
        [enc setBuffer:(__bridge id<MTLBuffer>)buf_targets offset:0 atIndex:1];
        [enc setBuffer:(__bridge id<MTLBuffer>)buf_loss    offset:0 atIndex:2];
        [enc setBuffer:(__bridge id<MTLBuffer>)buf_dlogits offset:0 atIndex:3];

        uint32_t params[] = {(uint32_t)vocab_size};
        [enc setBytes:params length:sizeof(params) atIndex:4];

        MTLSize grid = MTLSizeMake(batch, 1, 1);
        int tg_size = vocab_size < 256 ? vocab_size : 256;
        MTLSize block = MTLSizeMake(tg_size, 1, 1);
        [enc dispatchThreadgroups:grid threadsPerThreadgroup:block];
        [enc endEncoding];
    }
}

// ------------------------------------------------------------------
// AdamW (float4 vectorized — each thread processes 4 elements)
// ------------------------------------------------------------------
void metal_adamw_update(void *buf_param, void *buf_grad,
                        void *buf_m, void *buf_v,
                        int n, float lr, float beta1, float beta2,
                        float eps, float weight_decay, int step) {
    @autoreleasepool {
        id<MTLComputePipelineState> ps = get_pipeline(@"adamw_update");
        if (!ps) return;

        id<MTLCommandBuffer> cmd = get_command_buffer();
        id<MTLComputeCommandEncoder> enc = [cmd computeCommandEncoder];
        [enc setComputePipelineState:ps];

        [enc setBuffer:(__bridge id<MTLBuffer>)buf_param offset:0 atIndex:0];
        [enc setBuffer:(__bridge id<MTLBuffer>)buf_grad  offset:0 atIndex:1];
        [enc setBuffer:(__bridge id<MTLBuffer>)buf_m     offset:0 atIndex:2];
        [enc setBuffer:(__bridge id<MTLBuffer>)buf_v     offset:0 atIndex:3];

        uint32_t iparams[] = {(uint32_t)n, (uint32_t)step};
        [enc setBytes:iparams length:sizeof(iparams) atIndex:4];
        float fparams[] = {lr, beta1, beta2, eps, weight_decay};
        [enc setBytes:fparams length:sizeof(fparams) atIndex:5];

        int n4 = (n + 3) / 4;
        MTLSize grid = MTLSizeMake((n4 + 255) / 256, 1, 1);
        MTLSize block = MTLSizeMake(256, 1, 1);
        [enc dispatchThreadgroups:grid threadsPerThreadgroup:block];
        [enc endEncoding];
    }
}

// ------------------------------------------------------------------
// Element-wise add (float4 vectorized)
// ------------------------------------------------------------------
void metal_add(void *buf_a, void *buf_b, void *buf_out, int n) {
    KernelArgs args = {
        .bufs = {buf_a, buf_b, buf_out},
        .n_bufs = 3,
        .n_params = 1
    };
    args.params[0] = (uint32_t)n;
    int n4 = (n + 3) / 4;
    MTLSize grid = MTLSizeMake((n4 + 255) / 256, 1, 1);
    MTLSize block = MTLSizeMake(256, 1, 1);
    dispatch_kernel(@"add_tensors", &args, grid, block);
}

// ------------------------------------------------------------------
// Element-wise multiply (float4 vectorized)
// ------------------------------------------------------------------
void metal_mul(void *buf_a, void *buf_b, void *buf_out, int n) {
    KernelArgs args = {
        .bufs = {buf_a, buf_b, buf_out},
        .n_bufs = 3,
        .n_params = 1
    };
    args.params[0] = (uint32_t)n;
    int n4 = (n + 3) / 4;
    MTLSize grid = MTLSizeMake((n4 + 255) / 256, 1, 1);
    MTLSize block = MTLSizeMake(256, 1, 1);
    dispatch_kernel(@"mul_tensors", &args, grid, block);
}

// ------------------------------------------------------------------
// RMSNorm
// ------------------------------------------------------------------
void metal_rms_norm(void *buf_x, void *buf_out, void *buf_gamma,
                    int batch, int dim, float eps) {
    @autoreleasepool {
        id<MTLComputePipelineState> ps = get_pipeline(@"rms_norm");
        if (!ps) return;

        id<MTLCommandBuffer> cmd = get_command_buffer();
        id<MTLComputeCommandEncoder> enc = [cmd computeCommandEncoder];
        [enc setComputePipelineState:ps];

        [enc setBuffer:(__bridge id<MTLBuffer>)buf_x     offset:0 atIndex:0];
        [enc setBuffer:(__bridge id<MTLBuffer>)buf_out   offset:0 atIndex:1];
        [enc setBuffer:(__bridge id<MTLBuffer>)buf_gamma offset:0 atIndex:2];

        uint32_t params[] = {(uint32_t)dim};
        [enc setBytes:params length:sizeof(params) atIndex:3];
        float fparams[] = {eps};
        [enc setBytes:fparams length:sizeof(fparams) atIndex:4];

        MTLSize grid = MTLSizeMake(batch, 1, 1);
        int tg_size = dim < 256 ? dim : 256;
        MTLSize block = MTLSizeMake(tg_size, 1, 1);
        [enc dispatchThreadgroups:grid threadsPerThreadgroup:block];
        [enc endEncoding];
    }
}

void metal_rms_norm_backward(void *buf_x, void *buf_dy, void *buf_gamma,
                             void *buf_dx, void *buf_dgamma,
                             int batch, int dim, float eps) {
    @autoreleasepool {
        id<MTLComputePipelineState> ps = get_pipeline(@"rms_norm_backward");
        if (!ps) return;

        id<MTLCommandBuffer> cmd = get_command_buffer();
        id<MTLComputeCommandEncoder> enc = [cmd computeCommandEncoder];
        [enc setComputePipelineState:ps];

        [enc setBuffer:(__bridge id<MTLBuffer>)buf_x      offset:0 atIndex:0];
        [enc setBuffer:(__bridge id<MTLBuffer>)buf_dy     offset:0 atIndex:1];
        [enc setBuffer:(__bridge id<MTLBuffer>)buf_gamma  offset:0 atIndex:2];
        [enc setBuffer:(__bridge id<MTLBuffer>)buf_dx     offset:0 atIndex:3];
        [enc setBuffer:(__bridge id<MTLBuffer>)buf_dgamma offset:0 atIndex:4];

        uint32_t params[] = {(uint32_t)batch, (uint32_t)dim};
        [enc setBytes:params length:sizeof(params) atIndex:5];
        float fparams[] = {eps};
        [enc setBytes:fparams length:sizeof(fparams) atIndex:6];

        MTLSize grid = MTLSizeMake(batch, 1, 1);
        int tg_size = dim < 256 ? dim : 256;
        MTLSize block = MTLSizeMake(tg_size, 1, 1);
        [enc dispatchThreadgroups:grid threadsPerThreadgroup:block];
        [enc endEncoding];
    }
}

// ------------------------------------------------------------------
// SiLU (float4 vectorized — each thread processes 4 elements)
// ------------------------------------------------------------------
void metal_silu(void *buf_in, void *buf_out, int n) {
    KernelArgs args = {
        .bufs = {buf_in, buf_out},
        .n_bufs = 2,
        .n_params = 1
    };
    args.params[0] = (uint32_t)n;
    int n4 = (n + 3) / 4;
    MTLSize grid = MTLSizeMake((n4 + 255) / 256, 1, 1);
    MTLSize block = MTLSizeMake(256, 1, 1);
    dispatch_kernel(@"silu_forward", &args, grid, block);
}

void metal_silu_backward(void *buf_in, void *buf_dy, void *buf_dx, int n) {
    KernelArgs args = {
        .bufs = {buf_in, buf_dy, buf_dx},
        .n_bufs = 3,
        .n_params = 1
    };
    args.params[0] = (uint32_t)n;
    int n4 = (n + 3) / 4;
    MTLSize grid = MTLSizeMake((n4 + 255) / 256, 1, 1);
    MTLSize block = MTLSizeMake(256, 1, 1);
    dispatch_kernel(@"silu_backward", &args, grid, block);
}

// ------------------------------------------------------------------
// LoRA forward: delta[M, out_f] = x[M, in_f] @ A^T[in_f, rank] @ B^T[rank, out_f]
// ------------------------------------------------------------------
void metal_lora_forward(void *buf_x, void *buf_a, void *buf_b, void *buf_delta,
                        int M, int in_features, int out_features, int rank) {
    KernelArgs args = {
        .bufs = {buf_x, buf_a, buf_b, buf_delta},
        .n_bufs = 4,
        .n_params = 4
    };
    args.params[0] = (uint32_t)M;
    args.params[1] = (uint32_t)in_features;
    args.params[2] = (uint32_t)out_features;
    args.params[3] = (uint32_t)rank;

    MTLSize grid = MTLSizeMake((out_features + 31) / 32, (M + 31) / 32, 1);
    MTLSize block = MTLSizeMake(32, 32, 1);
    dispatch_kernel(@"lora_forward", &args, grid, block);
}

// ------------------------------------------------------------------
// 4-bit Dequantization (QLoRA)
// Each thread unpacks 1 byte -> 2 x 4-bit values
// ------------------------------------------------------------------
void metal_dequantize_4bit(void *buf_packed, void *buf_scales, void *buf_zeros,
                           void *buf_output, int n_elements, int group_size) {
    @autoreleasepool {
        id<MTLComputePipelineState> ps = get_pipeline(@"dequantize_4bit");
        if (!ps) return;

        id<MTLCommandBuffer> cmd = get_command_buffer();
        id<MTLComputeCommandEncoder> enc = [cmd computeCommandEncoder];
        [enc setComputePipelineState:ps];

        [enc setBuffer:(__bridge id<MTLBuffer>)buf_packed offset:0 atIndex:0];
        [enc setBuffer:(__bridge id<MTLBuffer>)buf_scales offset:0 atIndex:1];
        [enc setBuffer:(__bridge id<MTLBuffer>)buf_zeros  offset:0 atIndex:2];
        [enc setBuffer:(__bridge id<MTLBuffer>)buf_output offset:0 atIndex:3];

        uint32_t params[] = {(uint32_t)n_elements, (uint32_t)group_size};
        [enc setBytes:params length:sizeof(params) atIndex:4];

        int n_bytes = (n_elements + 1) / 2;  // each byte holds 2 values
        MTLSize grid = MTLSizeMake((n_bytes + 255) / 256, 1, 1);
        MTLSize block = MTLSizeMake(256, 1, 1);
        [enc dispatchThreadgroups:grid threadsPerThreadgroup:block];
        [enc endEncoding];
    }
}

// ------------------------------------------------------------------
// Synchronize
// ------------------------------------------------------------------
void metal_synchronize(void) {
    flush_command_buffer();
}
