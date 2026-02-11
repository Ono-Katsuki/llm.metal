#include "fast_sft.h"
#include "fast_metal.h"
#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <math.h>
#include <time.h>

// ================================================================
// Types
// ================================================================

typedef struct { MetalBuf *mbuf; int rows, cols; } WMat;

typedef struct {
    WMat *q_proj, *k_proj, *v_proj, *o_proj;
    WMat *gate_proj, *up_proj, *down_proj;
    MetalBuf *input_norm_g, *post_attn_norm_g;
    MetalBuf *q_norm_g, *k_norm_g;
} LayerW;

typedef struct {
    MetalBuf *A_q, *B_q, *A_v, *B_v;       // weights on GPU (float)
    MetalBuf *dA_q, *dB_q, *dA_v, *dB_v;   // gradients on GPU
    int Aq_size, Bq_size, Av_size, Bv_size; // element counts
    // AdamW state (CPU)
    float *m_Aq, *v_Aq, *m_Bq, *v_Bq;
    float *m_Av, *v_Av, *m_Bv, *v_Bv;
} LoRAW;

typedef struct {
    MetalBuf *x_in, *ln1_out;
    MetalBuf *q_pre_norm, *k_pre_norm;
    MetalBuf *q_final, *k_exp, *v_exp;
    MetalBuf *probs;
    MetalBuf *x_mid, *ln2_out;
    MetalBuf *gate_pre, *up_val;
    MetalBuf *lora_q_mid, *lora_v_mid;
} LayerSave;

struct SFTState {
    int D, Hq, Hkv, hd, IS, V, NL, N;
    int group_ratio;
    float rope_theta, eps, lora_scaling;
    int lora_rank;
    int step_count;

    LayerW *layers;
    WMat *lm_head;
    MetalBuf *final_norm_g;
    const float *token_emb;

    LoRAW *lora;
    LayerSave *saves;
    MetalBuf *save_x_final, *save_ln_final;

    // Forward scratch
    MetalBuf *mb_x, *mb_ln, *mb_q, *mb_k, *mb_v;
    MetalBuf *mb_q_t, *mb_k_t, *mb_v_t;
    MetalBuf *mb_k_exp, *mb_v_exp;
    MetalBuf *mb_attn_out, *mb_attn_flat, *mb_x2;
    MetalBuf *mb_gate, *mb_up, *mb_ff;
    MetalBuf *mb_logits;
    MetalBuf *mb_lora_tmp, *mb_lora_out;

    // Loss
    MetalBuf *mb_targets, *mb_losses, *mb_dlogits;

    // Backward scratch (reuse forward where possible)
    MetalBuf *mb_dx, *mb_dx2, *mb_d_score;

    float *x_cpu, *losses_cpu;
};

// ================================================================
// Helpers
// ================================================================

static WMat *wmat_f16(const float *src, int rows, int cols) {
    WMat *w = calloc(1, sizeof(WMat));
    w->rows = rows; w->cols = cols;
    size_t n = (size_t)rows * cols;
    __fp16 *data = malloc(n * sizeof(__fp16));
    for (size_t i = 0; i < n; i++) data[i] = (__fp16)src[i];
    w->mbuf = metal_buf_from_data(data, n * sizeof(__fp16));
    free(data);
    return w;
}

static void wmat_free(WMat *w) {
    if (!w) return;
    metal_buf_free(w->mbuf); free(w);
}

static MetalBuf *mbuf_f(int n) { return metal_buf_create((size_t)n * 4); }
static MetalBuf *mbuf_data(const float *d, int n) {
    return metal_buf_from_data(d, (size_t)n * 4);
}

// ================================================================
// State creation
// ================================================================

SFTState *sft_state_create(Qwen3Model *model, int seq_len,
                           int lora_rank, float lora_alpha) {
    if (fast_metal_init() != 0) {
        fprintf(stderr, "[FastSFT] Metal init failed\n");
        return NULL;
    }

    SFTState *s = calloc(1, sizeof(SFTState));
    s->D = model->d_model;
    s->Hq = model->n_q_heads;
    s->Hkv = model->n_kv_heads;
    s->hd = model->head_dim;
    s->IS = model->intermediate_size;
    s->V = model->vocab_size;
    s->NL = model->n_layers;
    s->N = seq_len;
    s->group_ratio = s->Hq / s->Hkv;
    s->rope_theta = model->rope_theta;
    s->eps = 1e-6f;
    s->lora_rank = lora_rank;
    s->lora_scaling = lora_alpha / (float)lora_rank;
    s->step_count = 0;

    int BN = seq_len;
    int Hq_hd = s->Hq * s->hd;
    int Hkv_hd = s->Hkv * s->hd;
    int Hq_ND = s->Hq * BN * s->hd;
    int Hkv_ND = s->Hkv * BN * s->hd;
    int R = lora_rank;

    printf("[FastSFT] Converting weights to F16...\n");
    struct timespec t0, t1;
    clock_gettime(CLOCK_MONOTONIC, &t0);

    // Convert frozen weights to F16
    s->layers = calloc(s->NL, sizeof(LayerW));
    for (int i = 0; i < s->NL; i++) {
        Qwen3Block *blk = model->blocks[i];
        LayerW *lw = &s->layers[i];
        lw->q_proj = wmat_f16(blk->attn->q_proj->weight->data, Hq_hd, s->D);
        lw->k_proj = wmat_f16(blk->attn->k_proj->weight->data, Hkv_hd, s->D);
        lw->v_proj = wmat_f16(blk->attn->v_proj->weight->data, Hkv_hd, s->D);
        lw->o_proj = wmat_f16(blk->attn->o_proj->weight->data, s->D, Hq_hd);
        lw->gate_proj = wmat_f16(blk->gate_proj->weight->data, s->IS, s->D);
        lw->up_proj = wmat_f16(blk->up_proj->weight->data, s->IS, s->D);
        lw->down_proj = wmat_f16(blk->down_proj->weight->data, s->D, s->IS);
        lw->input_norm_g = mbuf_data(blk->input_norm->gamma->data, s->D);
        lw->post_attn_norm_g = mbuf_data(blk->post_attn_norm->gamma->data, s->D);
        lw->q_norm_g = mbuf_data(blk->attn->q_norm->gamma->data, s->hd);
        lw->k_norm_g = mbuf_data(blk->attn->k_norm->gamma->data, s->hd);
        if ((i + 1) % 6 == 0)
            printf("[FastSFT]   Converted layers 0-%d\n", i);
    }
    s->lm_head = wmat_f16(model->lm_head->weight->data, s->V, s->D);
    s->final_norm_g = mbuf_data(model->final_norm->gamma->data, s->D);
    s->token_emb = model->token_emb->weight->data;

    // Initialize LoRA weights on GPU
    s->lora = calloc(s->NL, sizeof(LoRAW));
    for (int i = 0; i < s->NL; i++) {
        LoRAW *lr = &s->lora[i];
        int Aq_n = R * s->D, Bq_n = Hq_hd * R;
        int Av_n = R * s->D, Bv_n = Hkv_hd * R;
        lr->Aq_size = Aq_n; lr->Bq_size = Bq_n;
        lr->Av_size = Av_n; lr->Bv_size = Bv_n;

        // Copy from model if LoRA already attached, else init
        if (model->q_loras && model->q_loras[i]) {
            lr->A_q = mbuf_data(model->q_loras[i]->lora_A->data, Aq_n);
            lr->B_q = mbuf_data(model->q_loras[i]->lora_B->data, Bq_n);
        } else {
            // Kaiming init for A, zero init for B
            float *tmp = malloc(Aq_n * sizeof(float));
            float std_a = 1.0f / sqrtf((float)s->D);
            for (int j = 0; j < Aq_n; j++)
                tmp[j] = ((float)rand() / RAND_MAX - 0.5f) * 2.0f * std_a;
            lr->A_q = mbuf_data(tmp, Aq_n);
            free(tmp);
            tmp = calloc(Bq_n, sizeof(float));
            lr->B_q = mbuf_data(tmp, Bq_n);
            free(tmp);
        }
        if (model->v_loras && model->v_loras[i]) {
            lr->A_v = mbuf_data(model->v_loras[i]->lora_A->data, Av_n);
            lr->B_v = mbuf_data(model->v_loras[i]->lora_B->data, Bv_n);
        } else {
            float *tmp = malloc(Av_n * sizeof(float));
            float std_a = 1.0f / sqrtf((float)s->D);
            for (int j = 0; j < Av_n; j++)
                tmp[j] = ((float)rand() / RAND_MAX - 0.5f) * 2.0f * std_a;
            lr->A_v = mbuf_data(tmp, Av_n);
            free(tmp);
            tmp = calloc(Bv_n, sizeof(float));
            lr->B_v = mbuf_data(tmp, Bv_n);
            free(tmp);
        }
        lr->dA_q = mbuf_f(Aq_n); lr->dB_q = mbuf_f(Bq_n);
        lr->dA_v = mbuf_f(Av_n); lr->dB_v = mbuf_f(Bv_n);

        // AdamW state
        lr->m_Aq = calloc(Aq_n, sizeof(float)); lr->v_Aq = calloc(Aq_n, sizeof(float));
        lr->m_Bq = calloc(Bq_n, sizeof(float)); lr->v_Bq = calloc(Bq_n, sizeof(float));
        lr->m_Av = calloc(Av_n, sizeof(float)); lr->v_Av = calloc(Av_n, sizeof(float));
        lr->m_Bv = calloc(Bv_n, sizeof(float)); lr->v_Bv = calloc(Bv_n, sizeof(float));
    }

    // Allocate saved activations per layer
    s->saves = calloc(s->NL, sizeof(LayerSave));
    for (int i = 0; i < s->NL; i++) {
        LayerSave *sv = &s->saves[i];
        sv->x_in = mbuf_f(BN * s->D);
        sv->ln1_out = mbuf_f(BN * s->D);
        sv->q_pre_norm = mbuf_f(Hq_ND);
        sv->k_pre_norm = mbuf_f(Hkv_ND);
        sv->q_final = mbuf_f(Hq_ND);
        sv->k_exp = mbuf_f(Hq_ND);
        sv->v_exp = mbuf_f(Hq_ND);
        sv->probs = mbuf_f(s->Hq * BN * BN);
        sv->x_mid = mbuf_f(BN * s->D);
        sv->ln2_out = mbuf_f(BN * s->D);
        sv->gate_pre = mbuf_f(BN * s->IS);
        sv->up_val = mbuf_f(BN * s->IS);
        sv->lora_q_mid = mbuf_f(BN * R);
        sv->lora_v_mid = mbuf_f(BN * R);
    }
    s->save_x_final = mbuf_f(BN * s->D);
    s->save_ln_final = mbuf_f(BN * s->D);

    // Forward scratch
    s->mb_x = metal_buf_create(BN * s->D * sizeof(float));
    s->x_cpu = (float *)metal_buf_ptr(s->mb_x);
    s->mb_ln = mbuf_f(BN * s->D);
    s->mb_q = mbuf_f(BN * Hq_hd);
    s->mb_k = mbuf_f(BN * Hkv_hd);
    s->mb_v = mbuf_f(BN * Hkv_hd);
    s->mb_q_t = mbuf_f(Hq_ND);
    s->mb_k_t = mbuf_f(Hkv_ND);
    s->mb_v_t = mbuf_f(Hkv_ND);
    s->mb_k_exp = mbuf_f(Hq_ND);
    s->mb_v_exp = mbuf_f(Hq_ND);
    s->mb_attn_out = mbuf_f(Hq_ND);
    s->mb_attn_flat = mbuf_f(BN * Hq_hd);
    s->mb_x2 = mbuf_f(BN * s->D);
    s->mb_gate = mbuf_f(BN * s->IS);
    s->mb_up = mbuf_f(BN * s->IS);
    s->mb_ff = mbuf_f(BN * s->D);
    s->mb_logits = metal_buf_create((size_t)BN * s->V * sizeof(float));
    s->mb_lora_tmp = mbuf_f(BN * R);
    s->mb_lora_out = mbuf_f(BN * Hq_hd); // max(Hq*hd, Hkv*hd)

    // Loss
    s->mb_targets = metal_buf_create(BN * sizeof(uint32_t));
    s->mb_losses = metal_buf_create(BN * sizeof(float));
    s->losses_cpu = (float *)metal_buf_ptr(s->mb_losses);
    s->mb_dlogits = metal_buf_create((size_t)BN * s->V * sizeof(float));

    // Backward scratch
    s->mb_dx = mbuf_f(BN * s->D);
    s->mb_dx2 = mbuf_f(BN * s->D);
    s->mb_d_score = mbuf_f(s->Hq * BN * BN);

    clock_gettime(CLOCK_MONOTONIC, &t1);
    double ms = (t1.tv_sec - t0.tv_sec) * 1000.0 + (t1.tv_nsec - t0.tv_nsec) / 1e6;
    printf("[FastSFT] Init: %.0f ms, seq=%d, lora_rank=%d\n", ms, seq_len, lora_rank);
    return s;
}

// ================================================================
// Forward pass (GPU, single encoder)
// ================================================================

static void sft_forward(SFTState *s) {
    int BN = s->N, D = s->D, Hq = s->Hq, Hkv = s->Hkv, hd = s->hd;
    int IS = s->IS, R = s->lora_rank;
    int Hq_hd = Hq * hd, Hkv_hd = Hkv * hd;
    float eps = s->eps, sc = s->lora_scaling;
    float attn_scale = 1.0f / sqrtf((float)hd);

    for (int L = 0; L < s->NL; L++) {
        LayerW *lw = &s->layers[L];
        LoRAW *lr = &s->lora[L];
        LayerSave *sv = &s->saves[L];

        // Save input
        metal_enqueue_copy(s->mb_x, sv->x_in, BN * D);

        // Input RMSNorm
        metal_enqueue_rms_norm_batched(s->mb_x, lw->input_norm_g, s->mb_ln, D, eps, BN);
        metal_enqueue_copy(s->mb_ln, sv->ln1_out, BN * D);

        // Q projection: q = ln @ W_q^T
        metal_enqueue_f16_matmul(s->mb_ln, lw->q_proj->mbuf, s->mb_q, BN, Hq_hd, D);
        // LoRA Q: q += sc * ln @ A_q^T @ B_q^T
        metal_enqueue_float_matmul(s->mb_ln, lr->A_q, s->mb_lora_tmp, BN, R, D);
        metal_enqueue_copy(s->mb_lora_tmp, sv->lora_q_mid, BN * R);
        metal_enqueue_float_matmul(s->mb_lora_tmp, lr->B_q, s->mb_lora_out, BN, Hq_hd, R);
        metal_enqueue_add_scaled(s->mb_q, s->mb_lora_out, sc, BN * Hq_hd);

        // K projection
        metal_enqueue_f16_matmul(s->mb_ln, lw->k_proj->mbuf, s->mb_k, BN, Hkv_hd, D);

        // V projection + LoRA V
        metal_enqueue_f16_matmul(s->mb_ln, lw->v_proj->mbuf, s->mb_v, BN, Hkv_hd, D);
        metal_enqueue_float_matmul(s->mb_ln, lr->A_v, s->mb_lora_tmp, BN, R, D);
        metal_enqueue_copy(s->mb_lora_tmp, sv->lora_v_mid, BN * R);
        metal_enqueue_float_matmul(s->mb_lora_tmp, lr->B_v, s->mb_lora_out, BN, Hkv_hd, R);
        metal_enqueue_add_scaled(s->mb_v, s->mb_lora_out, sc, BN * Hkv_hd);

        // Transpose: [BN, H*D] -> [H, N, D]
        metal_enqueue_transpose_heads_fwd(s->mb_q, s->mb_q_t, BN, Hq, hd);
        metal_enqueue_transpose_heads_fwd(s->mb_k, s->mb_k_t, BN, Hkv, hd);
        metal_enqueue_transpose_heads_fwd(s->mb_v, s->mb_v_t, BN, Hkv, hd);

        // Save pre-norm Q, K
        metal_enqueue_copy(s->mb_q_t, sv->q_pre_norm, Hq * BN * hd);
        metal_enqueue_copy(s->mb_k_t, sv->k_pre_norm, Hkv * BN * hd);

        // QK norm (in-place, treating [H, N, D] as batch of H*N vectors of dim D)
        metal_enqueue_rms_norm_batched(s->mb_q_t, lw->q_norm_g, s->mb_q_t, hd, eps, Hq * BN);
        metal_enqueue_rms_norm_batched(s->mb_k_t, lw->k_norm_g, s->mb_k_t, hd, eps, Hkv * BN);

        // RoPE (in-place)
        metal_enqueue_rope_train(s->mb_q_t, s->mb_k_t, Hq, Hkv, hd, BN, s->rope_theta);

        // Save Q after norm+RoPE
        metal_enqueue_copy(s->mb_q_t, sv->q_final, Hq * BN * hd);

        // Repeat KV: [Hkv, N, D] -> [Hq, N, D]
        metal_enqueue_repeat_kv(s->mb_k_t, s->mb_k_exp, Hkv, BN, hd, s->group_ratio);
        metal_enqueue_repeat_kv(s->mb_v_t, s->mb_v_exp, Hkv, BN, hd, s->group_ratio);
        metal_enqueue_copy(s->mb_k_exp, sv->k_exp, Hq * BN * hd);
        metal_enqueue_copy(s->mb_v_exp, sv->v_exp, Hq * BN * hd);

        // Causal attention
        metal_enqueue_attn_train_fwd(s->mb_q_t, s->mb_k_exp, s->mb_v_exp,
                                      s->mb_attn_out, sv->probs,
                                      Hq, BN, hd, attn_scale);

        // Transpose back: [Hq, N, D] -> [BN, Hq*D]
        metal_enqueue_transpose_heads_rev(s->mb_attn_out, s->mb_attn_flat, BN, Hq, hd);

        // O projection
        metal_enqueue_f16_matmul(s->mb_attn_flat, lw->o_proj->mbuf, s->mb_x2, BN, D, Hq_hd);

        // Residual add
        metal_enqueue_residual_add(s->mb_x, s->mb_x2, BN * D);

        // Save x_mid (after attention residual)
        metal_enqueue_copy(s->mb_x, sv->x_mid, BN * D);

        // Post-attention RMSNorm
        metal_enqueue_rms_norm_batched(s->mb_x, lw->post_attn_norm_g, s->mb_ln, D, eps, BN);
        metal_enqueue_copy(s->mb_ln, sv->ln2_out, BN * D);

        // SwiGLU FFN
        metal_enqueue_f16_matmul(s->mb_ln, lw->gate_proj->mbuf, s->mb_gate, BN, IS, D);
        metal_enqueue_copy(s->mb_gate, sv->gate_pre, BN * IS);
        metal_enqueue_f16_matmul(s->mb_ln, lw->up_proj->mbuf, s->mb_up, BN, IS, D);
        metal_enqueue_copy(s->mb_up, sv->up_val, BN * IS);
        metal_enqueue_silu_mul(s->mb_gate, s->mb_up, BN * IS);
        metal_enqueue_f16_matmul(s->mb_gate, lw->down_proj->mbuf, s->mb_ff, BN, D, IS);

        // Residual add
        metal_enqueue_residual_add(s->mb_x, s->mb_ff, BN * D);
    }

    // Save x before final norm
    metal_enqueue_copy(s->mb_x, s->save_x_final, BN * D);

    // Final RMSNorm
    metal_enqueue_rms_norm_batched(s->mb_x, s->final_norm_g, s->mb_ln, D, eps, BN);
    metal_enqueue_copy(s->mb_ln, s->save_ln_final, BN * D);

    // LM head
    metal_enqueue_f16_matmul(s->mb_ln, s->lm_head->mbuf, s->mb_logits, BN, s->V, D);

    // Softmax CE loss + gradient
    metal_enqueue_softmax_ce(s->mb_logits, s->mb_targets, s->mb_losses,
                              s->mb_dlogits, BN, s->V);
    // Scale dlogits by 1/N for mean loss gradient
    metal_enqueue_scale(s->mb_dlogits, 1.0f / (float)BN, BN * s->V);
}

// ================================================================
// Backward pass (GPU, single encoder, continued from forward)
// ================================================================

static void sft_backward(SFTState *s) {
    int BN = s->N, D = s->D, Hq = s->Hq, Hkv = s->Hkv, hd = s->hd;
    int IS = s->IS, R = s->lora_rank;
    int Hq_hd = Hq * hd, Hkv_hd = Hkv * hd;
    float eps = s->eps, sc = s->lora_scaling;
    float attn_scale = 1.0f / sqrtf((float)hd);
    float grad_clip = 1e6f;  // only prevents float32 overflow, not gradient explosion

    // Backward through LM head: dx = dlogits @ W_lm_head
    metal_enqueue_f16_matmul_nt(s->mb_dlogits, s->lm_head->mbuf, s->mb_dx, BN, s->V, D);

    // Backward through final RMSNorm
    metal_enqueue_rms_norm_backward(s->save_x_final, s->mb_dx, s->final_norm_g,
                                     s->mb_dx2, BN, D, eps);
    metal_enqueue_copy(s->mb_dx2, s->mb_dx, BN * D);

    for (int L = s->NL - 1; L >= 0; L--) {
        LayerW *lw = &s->layers[L];
        LoRAW *lr = &s->lora[L];
        LayerSave *sv = &s->saves[L];

        // === FFN Backward ===
        metal_enqueue_f16_matmul_nt(s->mb_dx, lw->down_proj->mbuf, s->mb_gate, BN, D, IS);
        metal_enqueue_silu_mul_backward(s->mb_gate, sv->gate_pre, sv->up_val,
                                         s->mb_gate, s->mb_up, BN * IS);
        metal_enqueue_f16_matmul_nt(s->mb_gate, lw->gate_proj->mbuf, s->mb_ln, BN, IS, D);
        metal_enqueue_f16_matmul_nt(s->mb_up, lw->up_proj->mbuf, s->mb_x2, BN, IS, D);
        metal_enqueue_residual_add(s->mb_ln, s->mb_x2, BN * D);
        metal_enqueue_rms_norm_backward(sv->x_mid, s->mb_ln, lw->post_attn_norm_g,
                                         s->mb_dx2, BN, D, eps);
        metal_enqueue_residual_add(s->mb_dx, s->mb_dx2, BN * D);

        // === Attention Backward ===
        metal_enqueue_f16_matmul_nt(s->mb_dx, lw->o_proj->mbuf, s->mb_attn_flat, BN, D, Hq_hd);
        metal_enqueue_transpose_heads_fwd(s->mb_attn_flat, s->mb_attn_out, BN, Hq, hd);
        metal_enqueue_attn_bwd_dq(s->mb_attn_out, sv->probs, sv->v_exp,
                                   sv->k_exp, s->mb_d_score, s->mb_q_t,
                                   Hq, BN, hd, attn_scale);
        metal_enqueue_attn_bwd_dkv(s->mb_d_score, sv->q_final, sv->probs,
                                    s->mb_attn_out, s->mb_k_exp, s->mb_v_exp,
                                    Hq, BN, hd);
        metal_enqueue_repeat_kv_bwd(s->mb_k_exp, s->mb_k_t, Hkv, BN, hd, s->group_ratio);
        metal_enqueue_repeat_kv_bwd(s->mb_v_exp, s->mb_v_t, Hkv, BN, hd, s->group_ratio);
        metal_enqueue_rope_train_bwd(s->mb_q_t, s->mb_k_t, Hq, Hkv, hd, BN, s->rope_theta);
        metal_enqueue_rms_norm_backward(sv->q_pre_norm, s->mb_q_t, lw->q_norm_g,
                                         s->mb_q_t, Hq * BN, hd, eps);
        metal_enqueue_rms_norm_backward(sv->k_pre_norm, s->mb_k_t, lw->k_norm_g,
                                         s->mb_k_t, Hkv * BN, hd, eps);
        metal_enqueue_transpose_heads_rev(s->mb_q_t, s->mb_q, BN, Hq, hd);
        metal_enqueue_transpose_heads_rev(s->mb_k_t, s->mb_k, BN, Hkv, hd);
        metal_enqueue_transpose_heads_rev(s->mb_v_t, s->mb_v, BN, Hkv, hd);

        // === LoRA Q backward ===
        metal_enqueue_float_matmul_tn(s->mb_q, sv->lora_q_mid, lr->dB_q, Hq_hd, R, BN);
        metal_enqueue_scale(lr->dB_q, sc, Hq_hd * R);
        metal_enqueue_float_matmul_nt(s->mb_q, lr->B_q, s->mb_lora_tmp, BN, Hq_hd, R);
        metal_enqueue_scale(s->mb_lora_tmp, sc, BN * R);
        metal_enqueue_float_matmul_tn(s->mb_lora_tmp, sv->ln1_out, lr->dA_q, R, D, BN);
        metal_enqueue_float_matmul_nt(s->mb_lora_tmp, lr->A_q, s->mb_lora_out, BN, R, D);

        // === Base projections backward ===
        metal_enqueue_f16_matmul_nt(s->mb_q, lw->q_proj->mbuf, s->mb_ln, BN, Hq_hd, D);
        metal_enqueue_residual_add(s->mb_ln, s->mb_lora_out, BN * D);
        metal_enqueue_f16_matmul_nt(s->mb_k, lw->k_proj->mbuf, s->mb_x2, BN, Hkv_hd, D);
        metal_enqueue_residual_add(s->mb_ln, s->mb_x2, BN * D);

        // === LoRA V backward ===
        metal_enqueue_float_matmul_tn(s->mb_v, sv->lora_v_mid, lr->dB_v, Hkv_hd, R, BN);
        metal_enqueue_scale(lr->dB_v, sc, Hkv_hd * R);
        metal_enqueue_float_matmul_nt(s->mb_v, lr->B_v, s->mb_lora_tmp, BN, Hkv_hd, R);
        metal_enqueue_scale(s->mb_lora_tmp, sc, BN * R);
        metal_enqueue_float_matmul_tn(s->mb_lora_tmp, sv->ln1_out, lr->dA_v, R, D, BN);
        metal_enqueue_float_matmul_nt(s->mb_lora_tmp, lr->A_v, s->mb_lora_out, BN, R, D);

        metal_enqueue_f16_matmul_nt(s->mb_v, lw->v_proj->mbuf, s->mb_x2, BN, Hkv_hd, D);
        metal_enqueue_residual_add(s->mb_ln, s->mb_x2, BN * D);
        metal_enqueue_residual_add(s->mb_ln, s->mb_lora_out, BN * D);

        // === Input norm backward + residual ===
        metal_enqueue_rms_norm_backward(sv->x_in, s->mb_ln, lw->input_norm_g,
                                         s->mb_dx2, BN, D, eps);
        metal_enqueue_residual_add(s->mb_dx, s->mb_dx2, BN * D);

        // Gradient clipping to prevent explosion through deep layers
        metal_enqueue_clamp(s->mb_dx, grad_clip, BN * D);
    }
}

// ================================================================
// AdamW update for LoRA params (CPU, after flush)
// ================================================================

static void adamw_update(float *param, float *grad, float *m, float *v,
                          int n, float lr, float beta1, float beta2,
                          float eps, float wd, int step) {
    float bc1 = 1.0f - powf(beta1, (float)step);
    float bc2 = 1.0f - powf(beta2, (float)step);
    for (int i = 0; i < n; i++) {
        float g = grad[i];
        m[i] = beta1 * m[i] + (1 - beta1) * g;
        v[i] = beta2 * v[i] + (1 - beta2) * g * g;
        float m_hat = m[i] / bc1;
        float v_hat = v[i] / bc2;
        param[i] -= lr * (m_hat / (sqrtf(v_hat) + eps) + wd * param[i]);
    }
}

static void clip_grad_norm(SFTState *s, float max_norm) {
    // Compute global gradient L2 norm across all LoRA parameters
    double total_sq = 0;
    for (int L = 0; L < s->NL; L++) {
        LoRAW *lr = &s->lora[L];
        float *bufs[] = { metal_buf_ptr(lr->dA_q), metal_buf_ptr(lr->dB_q),
                          metal_buf_ptr(lr->dA_v), metal_buf_ptr(lr->dB_v) };
        int sizes[] = { lr->Aq_size, lr->Bq_size, lr->Av_size, lr->Bv_size };
        for (int b = 0; b < 4; b++) {
            float *g = bufs[b];
            for (int i = 0; i < sizes[b]; i++)
                total_sq += (double)g[i] * g[i];
        }
    }
    float grad_norm = (float)sqrt(total_sq);
    if (grad_norm > max_norm) {
        float scale = max_norm / grad_norm;
        for (int L = 0; L < s->NL; L++) {
            LoRAW *lr = &s->lora[L];
            float *bufs[] = { metal_buf_ptr(lr->dA_q), metal_buf_ptr(lr->dB_q),
                              metal_buf_ptr(lr->dA_v), metal_buf_ptr(lr->dB_v) };
            int sizes[] = { lr->Aq_size, lr->Bq_size, lr->Av_size, lr->Bv_size };
            for (int b = 0; b < 4; b++) {
                float *g = bufs[b];
                for (int i = 0; i < sizes[b]; i++)
                    g[i] *= scale;
            }
        }
    }
}

static void lora_update(SFTState *s, float lr) {
    float beta1 = 0.9f, beta2 = 0.999f, eps = 1e-8f, wd = 0.01f;
    int step = s->step_count;

    // Global gradient norm clipping (max_grad_norm = 1.0)
    clip_grad_norm(s, 1.0f);

    for (int L = 0; L < s->NL; L++) {
        LoRAW *lr_w = &s->lora[L];
        adamw_update(metal_buf_ptr(lr_w->A_q), metal_buf_ptr(lr_w->dA_q),
                     lr_w->m_Aq, lr_w->v_Aq, lr_w->Aq_size, lr, beta1, beta2, eps, wd, step);
        adamw_update(metal_buf_ptr(lr_w->B_q), metal_buf_ptr(lr_w->dB_q),
                     lr_w->m_Bq, lr_w->v_Bq, lr_w->Bq_size, lr, beta1, beta2, eps, wd, step);
        adamw_update(metal_buf_ptr(lr_w->A_v), metal_buf_ptr(lr_w->dA_v),
                     lr_w->m_Av, lr_w->v_Av, lr_w->Av_size, lr, beta1, beta2, eps, wd, step);
        adamw_update(metal_buf_ptr(lr_w->B_v), metal_buf_ptr(lr_w->dB_v),
                     lr_w->m_Bv, lr_w->v_Bv, lr_w->Bv_size, lr, beta1, beta2, eps, wd, step);
    }
}

// ================================================================
// Train step
// ================================================================

float sft_train_step(SFTState *s, const uint32_t *input_tokens,
                     const uint32_t *target_tokens, float lr) {
    int BN = s->N;
    int D = s->D;

    // Embedding lookup (CPU → shared GPU buffer)
    for (int i = 0; i < BN; i++)
        memcpy(s->x_cpu + i * D, &s->token_emb[input_tokens[i] * D], D * sizeof(float));

    // Copy targets to GPU
    memcpy(metal_buf_ptr(s->mb_targets), target_tokens, BN * sizeof(uint32_t));

    // Forward + loss (GPU, single encoder)
    sft_forward(s);

    // Backward (GPU, same encoder as forward)
    sft_backward(s);

    // Submit all GPU work and wait
    metal_flush();

    // Compute average loss on CPU
    float total_loss = 0;
    for (int i = 0; i < BN; i++) total_loss += s->losses_cpu[i];
    float avg_loss = total_loss / (float)BN;

    // Update LoRA weights on CPU
    s->step_count++;
    lora_update(s, lr);

    return avg_loss;
}

// ================================================================
// Sync LoRA to model
// ================================================================

void sft_sync_lora_to_model(SFTState *s, Qwen3Model *model) {
    if (!model->q_loras) {
        qwen3_attach_lora(model, s->lora_rank, s->lora_scaling * s->lora_rank);
    }
    for (int L = 0; L < s->NL; L++) {
        LoRAW *lr = &s->lora[L];
        memcpy(model->q_loras[L]->lora_A->data, metal_buf_ptr(lr->A_q),
               lr->Aq_size * sizeof(float));
        memcpy(model->q_loras[L]->lora_B->data, metal_buf_ptr(lr->B_q),
               lr->Bq_size * sizeof(float));
        memcpy(model->v_loras[L]->lora_A->data, metal_buf_ptr(lr->A_v),
               lr->Av_size * sizeof(float));
        memcpy(model->v_loras[L]->lora_B->data, metal_buf_ptr(lr->B_v),
               lr->Bv_size * sizeof(float));
    }
}

// ================================================================
// Cleanup
// ================================================================

void sft_state_free(SFTState *s) {
    if (!s) return;
    for (int i = 0; i < s->NL; i++) {
        LayerW *lw = &s->layers[i];
        wmat_free(lw->q_proj); wmat_free(lw->k_proj);
        wmat_free(lw->v_proj); wmat_free(lw->o_proj);
        wmat_free(lw->gate_proj); wmat_free(lw->up_proj);
        wmat_free(lw->down_proj);
        metal_buf_free(lw->input_norm_g); metal_buf_free(lw->post_attn_norm_g);
        metal_buf_free(lw->q_norm_g); metal_buf_free(lw->k_norm_g);

        LoRAW *lr = &s->lora[i];
        metal_buf_free(lr->A_q); metal_buf_free(lr->B_q);
        metal_buf_free(lr->A_v); metal_buf_free(lr->B_v);
        metal_buf_free(lr->dA_q); metal_buf_free(lr->dB_q);
        metal_buf_free(lr->dA_v); metal_buf_free(lr->dB_v);
        free(lr->m_Aq); free(lr->v_Aq); free(lr->m_Bq); free(lr->v_Bq);
        free(lr->m_Av); free(lr->v_Av); free(lr->m_Bv); free(lr->v_Bv);

        LayerSave *sv = &s->saves[i];
        metal_buf_free(sv->x_in); metal_buf_free(sv->ln1_out);
        metal_buf_free(sv->q_pre_norm); metal_buf_free(sv->k_pre_norm);
        metal_buf_free(sv->q_final); metal_buf_free(sv->k_exp);
        metal_buf_free(sv->v_exp); metal_buf_free(sv->probs);
        metal_buf_free(sv->x_mid); metal_buf_free(sv->ln2_out);
        metal_buf_free(sv->gate_pre); metal_buf_free(sv->up_val);
        metal_buf_free(sv->lora_q_mid); metal_buf_free(sv->lora_v_mid);
    }
    free(s->layers); free(s->lora); free(s->saves);
    wmat_free(s->lm_head);
    metal_buf_free(s->final_norm_g);
    metal_buf_free(s->save_x_final); metal_buf_free(s->save_ln_final);

    metal_buf_free(s->mb_x); metal_buf_free(s->mb_ln);
    metal_buf_free(s->mb_q); metal_buf_free(s->mb_k); metal_buf_free(s->mb_v);
    metal_buf_free(s->mb_q_t); metal_buf_free(s->mb_k_t); metal_buf_free(s->mb_v_t);
    metal_buf_free(s->mb_k_exp); metal_buf_free(s->mb_v_exp);
    metal_buf_free(s->mb_attn_out); metal_buf_free(s->mb_attn_flat);
    metal_buf_free(s->mb_x2); metal_buf_free(s->mb_gate); metal_buf_free(s->mb_up);
    metal_buf_free(s->mb_ff); metal_buf_free(s->mb_logits);
    metal_buf_free(s->mb_lora_tmp); metal_buf_free(s->mb_lora_out);

    metal_buf_free(s->mb_targets); metal_buf_free(s->mb_losses);
    metal_buf_free(s->mb_dlogits);
    metal_buf_free(s->mb_dx); metal_buf_free(s->mb_dx2);
    metal_buf_free(s->mb_d_score);

    fast_metal_shutdown();
    free(s);
}
