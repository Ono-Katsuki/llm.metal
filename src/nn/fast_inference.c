#include "fast_inference.h"
#include "fast_metal.h"
#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <math.h>
#include <time.h>

// ==================================================================
// Q8_0 Quantization — same format as GGML
// ==================================================================

#define QK8 32

typedef struct {
    uint16_t d;
    int8_t   qs[QK8];
} __attribute__((packed)) block_q8;

static void quantize_row_q8(const float *src, block_q8 *dst, int n_blocks) {
    for (int b = 0; b < n_blocks; b++) {
        const float *sp = src + b * QK8;
        block_q8 *blk = &dst[b];
        float amax = 0.0f;
        for (int j = 0; j < QK8; j++) {
            float a = fabsf(sp[j]);
            if (a > amax) amax = a;
        }
        float scale = amax / 127.0f;
        __fp16 h = (__fp16)scale;
        memcpy(&blk->d, &h, 2);
        if (scale != 0.0f) {
            float inv = 1.0f / scale;
            for (int j = 0; j < QK8; j++) {
                int v = (int)roundf(sp[j] * inv);
                blk->qs[j] = (int8_t)(v < -128 ? -128 : (v > 127 ? 127 : v));
            }
        } else {
            memset(blk->qs, 0, QK8);
        }
    }
}

// ==================================================================
// Weight matrix: Q8_0 or F16 data in Metal buffer
// ==================================================================

typedef struct {
    MetalBuf *mbuf;
    int rows, cols, nb;  // nb used only for Q8_0 (cols/32)
} WMat;

static WMat *wmat_create_q8(const float *src, int rows, int cols) {
    WMat *w = calloc(1, sizeof(WMat));
    w->rows = rows;
    w->cols = cols;
    w->nb = cols / QK8;
    size_t data_bytes = (size_t)rows * w->nb * sizeof(block_q8);
    block_q8 *data = malloc(data_bytes);

    for (int i = 0; i < rows; i++)
        quantize_row_q8(src + (size_t)i * cols, &data[(size_t)i * w->nb], w->nb);

    w->mbuf = metal_buf_from_data(data, data_bytes);
    free(data);
    return w;
}

static WMat *wmat_create_f16(const float *src, int rows, int cols) {
    WMat *w = calloc(1, sizeof(WMat));
    w->rows = rows;
    w->cols = cols;
    w->nb = 0;
    size_t n = (size_t)rows * cols;
    __fp16 *data = malloc(n * sizeof(__fp16));
    for (size_t i = 0; i < n; i++)
        data[i] = (__fp16)src[i];
    w->mbuf = metal_buf_from_data(data, n * sizeof(__fp16));
    free(data);
    return w;
}

static void wmat_free(WMat *w) {
    if (!w) return;
    metal_buf_free(w->mbuf);
    free(w);
}

// ==================================================================
// Per-layer weight storage
// ==================================================================

typedef struct {
    WMat *q_proj, *k_proj, *v_proj, *o_proj;
    WMat *gate_proj, *up_proj, *down_proj;
    MetalBuf *input_norm_g, *post_attn_norm_g;
    MetalBuf *q_norm_g, *k_norm_g;
} LayerW;

typedef struct {
    int d_model, n_q_heads, n_kv_heads, head_dim, intermediate_size, vocab_size;
    int n_layers, max_seq_len;
    float rope_theta;
    int use_fp16;

    // Metal scratch buffers
    MetalBuf *mb_x, *mb_x2, *mb_ln_out;
    MetalBuf *mb_q, *mb_k, *mb_v, *mb_attn_out;
    MetalBuf *mb_gate, *mb_up, *mb_ff_out, *mb_logits;

    // CPU pointers into shared buffers (embedding write + logits read)
    float *x, *logits;

    // KV cache on GPU (per-layer Metal buffers)
    MetalBuf **mb_k_cache, **mb_v_cache;
    int cur_len;

    // Weights
    LayerW *layers;
    WMat *lm_head;
    MetalBuf *mb_final_norm_g;
    const float *token_emb;
} InferenceState;

static MetalBuf *make_buf(size_t bytes, float **cpu_ptr) {
    MetalBuf *mb = metal_buf_create(bytes);
    if (cpu_ptr) *cpu_ptr = (float *)metal_buf_ptr(mb);
    return mb;
}

// Dispatch appropriate matvec kernel based on precision mode
static inline void dispatch_matvec(InferenceState *s, WMat *w,
                                    MetalBuf *x_buf, MetalBuf *y_buf) {
    if (s->use_fp16)
        metal_enqueue_f16_matvec(w->mbuf, x_buf, y_buf, w->rows, w->cols);
    else
        metal_enqueue_q8_matvec(w->mbuf, x_buf, y_buf, w->rows, w->nb);
}

static InferenceState *state_create(Qwen3Model *m, int max_seq_len, int use_fp16) {
    if (max_seq_len > 4096) {
        printf("[FastGen] Warning: clamping max_seq_len to 4096 (kernel limit)\n");
        max_seq_len = 4096;
    }

    InferenceState *s = calloc(1, sizeof(InferenceState));
    s->d_model = m->d_model;
    s->n_q_heads = m->n_q_heads;
    s->n_kv_heads = m->n_kv_heads;
    s->head_dim = m->head_dim;
    s->intermediate_size = m->intermediate_size;
    s->vocab_size = m->vocab_size;
    s->n_layers = m->n_layers;
    s->max_seq_len = max_seq_len;
    s->rope_theta = m->rope_theta;
    s->use_fp16 = use_fp16;

    int D = m->d_model;
    int Hq_hd = m->n_q_heads * m->head_dim;
    int Hkv_hd = m->n_kv_heads * m->head_dim;
    int IS = m->intermediate_size;

    // Scratch buffers (only x and logits need CPU access)
    s->mb_x       = make_buf(D * sizeof(float), &s->x);
    s->mb_x2      = make_buf(D * sizeof(float), NULL);
    s->mb_ln_out  = make_buf(D * sizeof(float), NULL);
    s->mb_q       = make_buf(Hq_hd * sizeof(float), NULL);
    s->mb_k       = make_buf(Hkv_hd * sizeof(float), NULL);
    s->mb_v       = make_buf(Hkv_hd * sizeof(float), NULL);
    s->mb_attn_out = make_buf(Hq_hd * sizeof(float), NULL);
    s->mb_gate    = make_buf(IS * sizeof(float), NULL);
    s->mb_up      = make_buf(IS * sizeof(float), NULL);
    s->mb_ff_out  = make_buf(D * sizeof(float), NULL);
    s->mb_logits  = make_buf(m->vocab_size * sizeof(float), &s->logits);

    // GPU KV cache (per layer)
    size_t cache_bytes = (size_t)m->n_kv_heads * max_seq_len * m->head_dim * sizeof(float);
    s->mb_k_cache = calloc(m->n_layers, sizeof(MetalBuf *));
    s->mb_v_cache = calloc(m->n_layers, sizeof(MetalBuf *));
    for (int i = 0; i < m->n_layers; i++) {
        s->mb_k_cache[i] = metal_buf_create(cache_bytes);
        s->mb_v_cache[i] = metal_buf_create(cache_bytes);
    }
    s->cur_len = 0;

    // Convert and upload weights
    WMat *(*create_fn)(const float *, int, int) = use_fp16 ? wmat_create_f16 : wmat_create_q8;
    const char *mode_str = use_fp16 ? "F16" : "Q8_0";
    printf("[FastGen] Converting weights to %s + uploading to GPU...\n", mode_str);
    struct timespec t0, t1;
    clock_gettime(CLOCK_MONOTONIC, &t0);

    s->layers = calloc(m->n_layers, sizeof(LayerW));
    for (int i = 0; i < m->n_layers; i++) {
        Qwen3Block *blk = m->blocks[i];
        LayerW *lw = &s->layers[i];

        lw->q_proj = create_fn(blk->attn->q_proj->weight->data, Hq_hd, D);
        lw->k_proj = create_fn(blk->attn->k_proj->weight->data, Hkv_hd, D);
        lw->v_proj = create_fn(blk->attn->v_proj->weight->data, Hkv_hd, D);
        lw->o_proj = create_fn(blk->attn->o_proj->weight->data, D, Hq_hd);
        lw->gate_proj = create_fn(blk->gate_proj->weight->data, IS, D);
        lw->up_proj = create_fn(blk->up_proj->weight->data, IS, D);
        lw->down_proj = create_fn(blk->down_proj->weight->data, D, IS);

        lw->input_norm_g = metal_buf_from_data(
            blk->input_norm->gamma->data, D * sizeof(float));
        lw->post_attn_norm_g = metal_buf_from_data(
            blk->post_attn_norm->gamma->data, D * sizeof(float));
        lw->q_norm_g = metal_buf_from_data(
            blk->attn->q_norm->gamma->data, m->head_dim * sizeof(float));
        lw->k_norm_g = metal_buf_from_data(
            blk->attn->k_norm->gamma->data, m->head_dim * sizeof(float));

        if ((i + 1) % 6 == 0)
            printf("[FastGen]   Converted layers 0-%d\n", i);
    }

    s->lm_head = create_fn(m->lm_head->weight->data, m->vocab_size, D);
    s->mb_final_norm_g = metal_buf_from_data(
        m->final_norm->gamma->data, D * sizeof(float));
    s->token_emb = m->token_emb->weight->data;

    clock_gettime(CLOCK_MONOTONIC, &t1);
    double ms = (t1.tv_sec - t0.tv_sec) * 1000.0 + (t1.tv_nsec - t0.tv_nsec) / 1e6;
    printf("[FastGen] %s + GPU upload: %.0f ms\n", mode_str, ms);

    // Print weight memory usage
    double weight_mb;
    if (use_fp16) {
        size_t total_params = 0;
        for (int i = 0; i < m->n_layers; i++) {
            LayerW *lw = &s->layers[i];
            total_params += (size_t)lw->q_proj->rows * lw->q_proj->cols;
            total_params += (size_t)lw->k_proj->rows * lw->k_proj->cols;
            total_params += (size_t)lw->v_proj->rows * lw->v_proj->cols;
            total_params += (size_t)lw->o_proj->rows * lw->o_proj->cols;
            total_params += (size_t)lw->gate_proj->rows * lw->gate_proj->cols;
            total_params += (size_t)lw->up_proj->rows * lw->up_proj->cols;
            total_params += (size_t)lw->down_proj->rows * lw->down_proj->cols;
        }
        total_params += (size_t)s->lm_head->rows * s->lm_head->cols;
        weight_mb = (double)(total_params * 2) / (1024.0 * 1024.0);
    } else {
        weight_mb = 0; // Q8_0 size can be computed but not critical
    }
    if (use_fp16)
        printf("[FastGen] F16 weight memory: %.1f MB (%.2f GB)\n",
               weight_mb, weight_mb / 1024.0);

    return s;
}

static void state_free(InferenceState *s) {
    if (!s) return;
    metal_buf_free(s->mb_x); metal_buf_free(s->mb_x2); metal_buf_free(s->mb_ln_out);
    metal_buf_free(s->mb_q); metal_buf_free(s->mb_k); metal_buf_free(s->mb_v);
    metal_buf_free(s->mb_attn_out);
    metal_buf_free(s->mb_gate); metal_buf_free(s->mb_up);
    metal_buf_free(s->mb_ff_out); metal_buf_free(s->mb_logits);
    metal_buf_free(s->mb_final_norm_g);
    for (int i = 0; i < s->n_layers; i++) {
        metal_buf_free(s->mb_k_cache[i]);
        metal_buf_free(s->mb_v_cache[i]);
        LayerW *lq = &s->layers[i];
        wmat_free(lq->q_proj); wmat_free(lq->k_proj);
        wmat_free(lq->v_proj); wmat_free(lq->o_proj);
        wmat_free(lq->gate_proj); wmat_free(lq->up_proj);
        wmat_free(lq->down_proj);
        metal_buf_free(lq->input_norm_g);
        metal_buf_free(lq->post_attn_norm_g);
        metal_buf_free(lq->q_norm_g);
        metal_buf_free(lq->k_norm_g);
    }
    free(s->mb_k_cache); free(s->mb_v_cache);
    free(s->layers);
    wmat_free(s->lm_head);
    free(s);
}

// ==================================================================
// Full GPU forward pass — single encoder, single flush per token
// ==================================================================

static void forward_token(InferenceState *s, int token, int pos) {
    int D = s->d_model;
    int Hq = s->n_q_heads;
    int Hkv = s->n_kv_heads;
    int hd = s->head_dim;
    int group_ratio = Hq / Hkv;
    int n_attend = pos + 1;
    float eps = 1e-6f;

    // Embedding lookup (CPU → shared GPU buffer)
    memcpy(s->x, &s->token_emb[token * D], D * sizeof(float));

    // All layers — entirely on GPU, single encoder
    for (int layer = 0; layer < s->n_layers; layer++) {
        LayerW *lw = &s->layers[layer];

        // --- Attention block ---
        metal_enqueue_rms_norm(s->mb_x, lw->input_norm_g, s->mb_ln_out, D, eps);
        dispatch_matvec(s, lw->q_proj, s->mb_ln_out, s->mb_q);
        dispatch_matvec(s, lw->k_proj, s->mb_ln_out, s->mb_k);
        dispatch_matvec(s, lw->v_proj, s->mb_ln_out, s->mb_v);
        metal_enqueue_per_head_rms_norm(s->mb_q, lw->q_norm_g, Hq, hd, eps);
        metal_enqueue_per_head_rms_norm(s->mb_k, lw->k_norm_g, Hkv, hd, eps);
        metal_enqueue_rope(s->mb_q, s->mb_k, Hq, Hkv, hd, pos, s->rope_theta);
        metal_enqueue_kv_cache_store(s->mb_k, s->mb_v,
                                      s->mb_k_cache[layer], s->mb_v_cache[layer],
                                      Hkv, s->max_seq_len, hd, pos);
        metal_enqueue_attention(s->mb_q, s->mb_k_cache[layer], s->mb_v_cache[layer],
                                s->mb_attn_out, n_attend, hd,
                                s->max_seq_len, Hq, group_ratio);
        dispatch_matvec(s, lw->o_proj, s->mb_attn_out, s->mb_x2);
        metal_enqueue_residual_add(s->mb_x, s->mb_x2, D);

        // --- FFN block ---
        metal_enqueue_rms_norm(s->mb_x, lw->post_attn_norm_g, s->mb_ln_out, D, eps);
        dispatch_matvec(s, lw->gate_proj, s->mb_ln_out, s->mb_gate);
        dispatch_matvec(s, lw->up_proj, s->mb_ln_out, s->mb_up);
        metal_enqueue_silu_mul(s->mb_gate, s->mb_up, s->intermediate_size);
        dispatch_matvec(s, lw->down_proj, s->mb_gate, s->mb_ff_out);
        metal_enqueue_residual_add(s->mb_x, s->mb_ff_out, D);
    }

    // Final norm + LM head
    metal_enqueue_rms_norm(s->mb_x, s->mb_final_norm_g, s->mb_ln_out, D, eps);
    dispatch_matvec(s, s->lm_head, s->mb_ln_out, s->mb_logits);
    metal_flush();  // Single GPU submission for entire token
}

static int argmax(const float *v, int n) {
    int best = 0;
    float best_val = v[0];
    for (int i = 1; i < n; i++) {
        if (v[i] > best_val) { best_val = v[i]; best = i; }
    }
    return best;
}

// ==================================================================
// Main generate function
// ==================================================================

int qwen3_generate_fast(Qwen3Model *model, const uint32_t *prompt_tokens,
                        int prompt_len, uint32_t *output_tokens,
                        int max_gen_len, int max_seq_len, int use_fp16) {
    if (fast_metal_init() != 0) {
        fprintf(stderr, "[FastGen] Metal init failed\n");
        return 0;
    }

    InferenceState *s = state_create(model, max_seq_len, use_fp16);

    struct timespec t0, t1;

    // Prefill
    printf("[FastGen] Prefilling %d tokens...\n", prompt_len);
    clock_gettime(CLOCK_MONOTONIC, &t0);

    for (int i = 0; i < prompt_len; i++) {
        forward_token(s, (int)prompt_tokens[i], i);
    }
    s->cur_len = prompt_len;

    clock_gettime(CLOCK_MONOTONIC, &t1);
    double prefill_ms = (t1.tv_sec - t0.tv_sec) * 1000.0 + (t1.tv_nsec - t0.tv_nsec) / 1e6;
    printf("[FastGen] Prefill: %.0f ms (%.1f tok/s)\n",
           prefill_ms, prompt_len * 1000.0 / prefill_ms);

    // First generated token
    int next_token = argmax(s->logits, s->vocab_size);
    output_tokens[0] = (uint32_t)next_token;
    int n_gen = 1;

    // Decode
    printf("[FastGen] Decoding...\n");
    clock_gettime(CLOCK_MONOTONIC, &t0);

    for (int gen = 1; gen < max_gen_len && s->cur_len < max_seq_len; gen++) {
        if (next_token == 151645 || next_token == 151643 || next_token == 0) break;

        forward_token(s, next_token, s->cur_len);
        s->cur_len++;

        next_token = argmax(s->logits, s->vocab_size);
        output_tokens[n_gen++] = (uint32_t)next_token;

        if (n_gen % 10 == 0) {
            clock_gettime(CLOCK_MONOTONIC, &t1);
            double elapsed = (t1.tv_sec - t0.tv_sec) * 1000.0 + (t1.tv_nsec - t0.tv_nsec) / 1e6;
            printf("[FastGen]   %d tokens, %.1f tok/s\n",
                   n_gen - 1, (n_gen - 1) * 1000.0 / elapsed);
        }
    }

    clock_gettime(CLOCK_MONOTONIC, &t1);
    double decode_ms = (t1.tv_sec - t0.tv_sec) * 1000.0 + (t1.tv_nsec - t0.tv_nsec) / 1e6;
    int n_decoded = n_gen - 1;
    if (n_decoded > 0) {
        printf("[FastGen] Decode: %d tokens in %.0f ms (%.1f tok/s, %.0f ms/tok)\n",
               n_decoded, decode_ms, n_decoded * 1000.0 / decode_ms,
               decode_ms / n_decoded);
    }

    state_free(s);
    fast_metal_shutdown();
    return n_gen;
}

// ==================================================================
// Batched inference — B sequences processed simultaneously (F16 only)
// Reads weight matrix ONCE for all B items → B× throughput
// ==================================================================

typedef struct {
    int d_model, n_q_heads, n_kv_heads, head_dim, intermediate_size, vocab_size;
    int n_layers, max_seq_len, B;
    float rope_theta;

    // Scratch buffers: [B * dim] contiguous
    MetalBuf *mb_x, *mb_x2, *mb_ln_out;
    MetalBuf *mb_q, *mb_k, *mb_v, *mb_attn_out;
    MetalBuf *mb_gate, *mb_up, *mb_ff_out, *mb_logits;
    float *x, *logits;

    // Per-batch-item KV caches: [n_layers][B]
    MetalBuf ***mb_k_cache, ***mb_v_cache;
    int cur_len;

    // Weights (shared across batch items)
    LayerW *layers;
    WMat *lm_head;
    MetalBuf *mb_final_norm_g;
    const float *token_emb;
} BatchState;

static BatchState *batch_state_create(Qwen3Model *m, int max_seq_len, int B) {
    if (max_seq_len > 4096) max_seq_len = 4096;

    BatchState *s = calloc(1, sizeof(BatchState));
    s->d_model = m->d_model;
    s->n_q_heads = m->n_q_heads;
    s->n_kv_heads = m->n_kv_heads;
    s->head_dim = m->head_dim;
    s->intermediate_size = m->intermediate_size;
    s->vocab_size = m->vocab_size;
    s->n_layers = m->n_layers;
    s->max_seq_len = max_seq_len;
    s->rope_theta = m->rope_theta;
    s->B = B;

    int D = m->d_model;
    int Hq_hd = m->n_q_heads * m->head_dim;
    int Hkv_hd = m->n_kv_heads * m->head_dim;
    int IS = m->intermediate_size;

    // B× scratch buffers (contiguous for batched matvec)
    s->mb_x       = metal_buf_create(B * D * sizeof(float));
    s->x          = (float *)metal_buf_ptr(s->mb_x);
    s->mb_x2      = metal_buf_create(B * D * sizeof(float));
    s->mb_ln_out  = metal_buf_create(B * D * sizeof(float));
    s->mb_q       = metal_buf_create(B * Hq_hd * sizeof(float));
    s->mb_k       = metal_buf_create(B * Hkv_hd * sizeof(float));
    s->mb_v       = metal_buf_create(B * Hkv_hd * sizeof(float));
    s->mb_attn_out = metal_buf_create(B * Hq_hd * sizeof(float));
    s->mb_gate    = metal_buf_create(B * IS * sizeof(float));
    s->mb_up      = metal_buf_create(B * IS * sizeof(float));
    s->mb_ff_out  = metal_buf_create(B * D * sizeof(float));
    s->mb_logits  = metal_buf_create(B * m->vocab_size * sizeof(float));
    s->logits     = (float *)metal_buf_ptr(s->mb_logits);

    // Per-item KV caches
    size_t cache_bytes = (size_t)m->n_kv_heads * max_seq_len * m->head_dim * sizeof(float);
    s->mb_k_cache = calloc(m->n_layers, sizeof(MetalBuf **));
    s->mb_v_cache = calloc(m->n_layers, sizeof(MetalBuf **));
    for (int l = 0; l < m->n_layers; l++) {
        s->mb_k_cache[l] = calloc(B, sizeof(MetalBuf *));
        s->mb_v_cache[l] = calloc(B, sizeof(MetalBuf *));
        for (int b = 0; b < B; b++) {
            s->mb_k_cache[l][b] = metal_buf_create(cache_bytes);
            s->mb_v_cache[l][b] = metal_buf_create(cache_bytes);
        }
    }
    s->cur_len = 0;

    // Convert weights to F16
    printf("[BatchGen] Converting weights to F16 (batch=%d)...\n", B);
    struct timespec t0, t1;
    clock_gettime(CLOCK_MONOTONIC, &t0);

    s->layers = calloc(m->n_layers, sizeof(LayerW));
    for (int i = 0; i < m->n_layers; i++) {
        Qwen3Block *blk = m->blocks[i];
        LayerW *lw = &s->layers[i];
        lw->q_proj = wmat_create_f16(blk->attn->q_proj->weight->data, Hq_hd, D);
        lw->k_proj = wmat_create_f16(blk->attn->k_proj->weight->data, Hkv_hd, D);
        lw->v_proj = wmat_create_f16(blk->attn->v_proj->weight->data, Hkv_hd, D);
        lw->o_proj = wmat_create_f16(blk->attn->o_proj->weight->data, D, Hq_hd);
        lw->gate_proj = wmat_create_f16(blk->gate_proj->weight->data, IS, D);
        lw->up_proj = wmat_create_f16(blk->up_proj->weight->data, IS, D);
        lw->down_proj = wmat_create_f16(blk->down_proj->weight->data, D, IS);
        lw->input_norm_g = metal_buf_from_data(blk->input_norm->gamma->data, D * sizeof(float));
        lw->post_attn_norm_g = metal_buf_from_data(blk->post_attn_norm->gamma->data, D * sizeof(float));
        lw->q_norm_g = metal_buf_from_data(blk->attn->q_norm->gamma->data, m->head_dim * sizeof(float));
        lw->k_norm_g = metal_buf_from_data(blk->attn->k_norm->gamma->data, m->head_dim * sizeof(float));
        if ((i + 1) % 6 == 0) printf("[BatchGen]   Converted layers 0-%d\n", i);
    }
    s->lm_head = wmat_create_f16(m->lm_head->weight->data, m->vocab_size, D);
    s->mb_final_norm_g = metal_buf_from_data(m->final_norm->gamma->data, D * sizeof(float));
    s->token_emb = m->token_emb->weight->data;

    clock_gettime(CLOCK_MONOTONIC, &t1);
    double ms = (t1.tv_sec - t0.tv_sec) * 1000.0 + (t1.tv_nsec - t0.tv_nsec) / 1e6;
    printf("[BatchGen] F16 + GPU upload: %.0f ms\n", ms);
    return s;
}

static void batch_state_free(BatchState *s) {
    if (!s) return;
    metal_buf_free(s->mb_x); metal_buf_free(s->mb_x2); metal_buf_free(s->mb_ln_out);
    metal_buf_free(s->mb_q); metal_buf_free(s->mb_k); metal_buf_free(s->mb_v);
    metal_buf_free(s->mb_attn_out);
    metal_buf_free(s->mb_gate); metal_buf_free(s->mb_up);
    metal_buf_free(s->mb_ff_out); metal_buf_free(s->mb_logits);
    metal_buf_free(s->mb_final_norm_g);
    for (int l = 0; l < s->n_layers; l++) {
        for (int b = 0; b < s->B; b++) {
            metal_buf_free(s->mb_k_cache[l][b]);
            metal_buf_free(s->mb_v_cache[l][b]);
        }
        free(s->mb_k_cache[l]); free(s->mb_v_cache[l]);
        LayerW *lw = &s->layers[l];
        wmat_free(lw->q_proj); wmat_free(lw->k_proj);
        wmat_free(lw->v_proj); wmat_free(lw->o_proj);
        wmat_free(lw->gate_proj); wmat_free(lw->up_proj);
        wmat_free(lw->down_proj);
        metal_buf_free(lw->input_norm_g); metal_buf_free(lw->post_attn_norm_g);
        metal_buf_free(lw->q_norm_g); metal_buf_free(lw->k_norm_g);
    }
    free(s->mb_k_cache); free(s->mb_v_cache);
    free(s->layers);
    wmat_free(s->lm_head);
    free(s);
}

// Copy KV cache from batch item src to batch item dst (for prefill sharing)
__attribute__((unused))
static void copy_kv_cache(BatchState *s, int dst, int src) {
    size_t cache_bytes = (size_t)s->n_kv_heads * s->max_seq_len * s->head_dim * sizeof(float);
    for (int l = 0; l < s->n_layers; l++) {
        memcpy(metal_buf_ptr(s->mb_k_cache[l][dst]),
               metal_buf_ptr(s->mb_k_cache[l][src]), cache_bytes);
        memcpy(metal_buf_ptr(s->mb_v_cache[l][dst]),
               metal_buf_ptr(s->mb_v_cache[l][src]), cache_bytes);
    }
}

// Batched forward: process B tokens at the same position
static void forward_token_batch(BatchState *s, const int *tokens, int pos) {
    int D = s->d_model;
    int Hq = s->n_q_heads, Hkv = s->n_kv_heads, hd = s->head_dim;
    int Hq_hd = Hq * hd, Hkv_hd = Hkv * hd;
    int group_ratio = Hq / Hkv;
    int n_attend = pos + 1;
    int IS = s->intermediate_size;
    float eps = 1e-6f;
    int B = s->B;

    // Embedding lookup for all B items
    for (int b = 0; b < B; b++)
        memcpy(s->x + b * D, &s->token_emb[tokens[b] * D], D * sizeof(float));

    for (int layer = 0; layer < s->n_layers; layer++) {
        LayerW *lw = &s->layers[layer];

        // RMSNorm (batched: B threadgroups)
        metal_enqueue_rms_norm_batched(s->mb_x, lw->input_norm_g, s->mb_ln_out, D, eps, B);

        // QKV matvec (batched: read W once for all B)
        metal_enqueue_f16_batch_matvec(lw->q_proj->mbuf, s->mb_ln_out, s->mb_q,
                                        lw->q_proj->rows, lw->q_proj->cols, B);
        metal_enqueue_f16_batch_matvec(lw->k_proj->mbuf, s->mb_ln_out, s->mb_k,
                                        lw->k_proj->rows, lw->k_proj->cols, B);
        metal_enqueue_f16_batch_matvec(lw->v_proj->mbuf, s->mb_ln_out, s->mb_v,
                                        lw->v_proj->rows, lw->v_proj->cols, B);

        // Per-item: QK norm, RoPE, KV cache store, attention
        for (int b = 0; b < B; b++) {
            size_t q_off = (size_t)b * Hq_hd * sizeof(float);
            size_t k_off = (size_t)b * Hkv_hd * sizeof(float);
            size_t attn_off = q_off;

            metal_enqueue_per_head_rms_norm_off(s->mb_q, q_off, lw->q_norm_g, Hq, hd, eps);
            metal_enqueue_per_head_rms_norm_off(s->mb_k, k_off, lw->k_norm_g, Hkv, hd, eps);
            metal_enqueue_rope_off(s->mb_q, q_off, s->mb_k, k_off,
                                    Hq, Hkv, hd, pos, s->rope_theta);
            metal_enqueue_kv_cache_store_off(s->mb_k, k_off, s->mb_v, k_off,
                                              s->mb_k_cache[layer][b], s->mb_v_cache[layer][b],
                                              Hkv, s->max_seq_len, hd, pos);
            metal_enqueue_attention_off(s->mb_q, q_off,
                                         s->mb_k_cache[layer][b], s->mb_v_cache[layer][b],
                                         s->mb_attn_out, attn_off,
                                         n_attend, hd, s->max_seq_len, Hq, group_ratio);
        }

        // O proj (batched)
        metal_enqueue_f16_batch_matvec(lw->o_proj->mbuf, s->mb_attn_out, s->mb_x2,
                                        lw->o_proj->rows, lw->o_proj->cols, B);
        metal_enqueue_residual_add(s->mb_x, s->mb_x2, B * D);

        // FFN (batched matvecs, element-wise ops scale with B)
        metal_enqueue_rms_norm_batched(s->mb_x, lw->post_attn_norm_g, s->mb_ln_out, D, eps, B);
        metal_enqueue_f16_batch_matvec(lw->gate_proj->mbuf, s->mb_ln_out, s->mb_gate,
                                        lw->gate_proj->rows, lw->gate_proj->cols, B);
        metal_enqueue_f16_batch_matvec(lw->up_proj->mbuf, s->mb_ln_out, s->mb_up,
                                        lw->up_proj->rows, lw->up_proj->cols, B);
        metal_enqueue_silu_mul(s->mb_gate, s->mb_up, B * IS);
        metal_enqueue_f16_batch_matvec(lw->down_proj->mbuf, s->mb_gate, s->mb_ff_out,
                                        lw->down_proj->rows, lw->down_proj->cols, B);
        metal_enqueue_residual_add(s->mb_x, s->mb_ff_out, B * D);
    }

    // Final norm + LM head (batched)
    metal_enqueue_rms_norm_batched(s->mb_x, s->mb_final_norm_g, s->mb_ln_out, D, eps, B);
    metal_enqueue_f16_batch_matvec(s->lm_head->mbuf, s->mb_ln_out, s->mb_logits,
                                    s->lm_head->rows, s->lm_head->cols, B);
    metal_flush();
}

int qwen3_generate_fast_batch(Qwen3Model *model, const uint32_t *prompt_tokens,
                               int prompt_len, uint32_t *output_tokens,
                               int max_gen_len, int max_seq_len, int batch_size) {
    if (fast_metal_init() != 0) {
        fprintf(stderr, "[BatchGen] Metal init failed\n");
        return 0;
    }

    int B = batch_size;
    BatchState *s = batch_state_create(model, max_seq_len, B);
    struct timespec t0, t1;

    // Prefill (single sequence, then copy KV cache to all B items)
    printf("[BatchGen] Prefilling %d tokens...\n", prompt_len);
    clock_gettime(CLOCK_MONOTONIC, &t0);

    int V = s->vocab_size;
    for (int i = 0; i < prompt_len; i++) {
        int tok = (int)prompt_tokens[i];
        int tokens_b[32];
        for (int b = 0; b < B; b++) tokens_b[b] = tok;
        forward_token_batch(s, tokens_b, i);
    }
    // All B items now have identical KV caches from prefill
    s->cur_len = prompt_len;

    clock_gettime(CLOCK_MONOTONIC, &t1);
    double prefill_ms = (t1.tv_sec - t0.tv_sec) * 1000.0 + (t1.tv_nsec - t0.tv_nsec) / 1e6;
    printf("[BatchGen] Prefill: %.0f ms (%.1f tok/s)\n",
           prefill_ms, prompt_len * 1000.0 / prefill_ms);

    // First generated token (same for all B since same prompt)
    int next_tokens[32];
    for (int b = 0; b < B; b++) {
        next_tokens[b] = argmax(s->logits + b * V, V);
        output_tokens[b * max_gen_len] = (uint32_t)next_tokens[b];
    }
    int n_gen = 1;
    int active[32];
    for (int b = 0; b < B; b++) active[b] = 1;

    // Decode loop — B tokens per step
    printf("[BatchGen] Decoding (batch=%d)...\n", B);
    clock_gettime(CLOCK_MONOTONIC, &t0);

    for (int gen = 1; gen < max_gen_len && s->cur_len < max_seq_len; gen++) {
        // Check if all sequences have ended
        int any_active = 0;
        for (int b = 0; b < B; b++) {
            if (active[b] && (next_tokens[b] == 151645 ||
                              next_tokens[b] == 151643 ||
                              next_tokens[b] == 0))
                active[b] = 0;
            any_active |= active[b];
        }
        if (!any_active) break;

        forward_token_batch(s, next_tokens, s->cur_len);
        s->cur_len++;

        for (int b = 0; b < B; b++) {
            next_tokens[b] = argmax(s->logits + b * V, V);
            output_tokens[b * max_gen_len + n_gen] = (uint32_t)next_tokens[b];
        }
        n_gen++;

        if (n_gen % 10 == 0) {
            clock_gettime(CLOCK_MONOTONIC, &t1);
            double elapsed = (t1.tv_sec - t0.tv_sec) * 1000.0 + (t1.tv_nsec - t0.tv_nsec) / 1e6;
            int total_tok = (n_gen - 1) * B;
            printf("[BatchGen]   %d steps, %d total tok, %.1f tok/s (%.1f ms/step)\n",
                   n_gen - 1, total_tok, total_tok * 1000.0 / elapsed,
                   elapsed / (n_gen - 1));
        }
    }

    clock_gettime(CLOCK_MONOTONIC, &t1);
    double decode_ms = (t1.tv_sec - t0.tv_sec) * 1000.0 + (t1.tv_nsec - t0.tv_nsec) / 1e6;
    int n_decoded = n_gen - 1;
    if (n_decoded > 0) {
        int total_tok = n_decoded * B;
        printf("[BatchGen] Decode: %d steps × %d batch = %d tokens in %.0f ms\n",
               n_decoded, B, total_tok, decode_ms);
        printf("[BatchGen]   Throughput: %.1f tok/s\n", total_tok * 1000.0 / decode_ms);
        printf("[BatchGen]   Latency: %.0f ms/step\n", decode_ms / n_decoded);
    }

    batch_state_free(s);
    fast_metal_shutdown();
    return n_gen;
}
