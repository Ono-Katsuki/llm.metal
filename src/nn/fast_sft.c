#include "fast_sft.h"
#include "fast_metal.h"
#include "wmat.h"
#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <math.h>
#include <time.h>

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
    MetalBuf *o_proj_out;     // Gemma3: post_attn_norm backward input
    MetalBuf *ff_raw_out;     // Gemma3: post_ff_norm backward input
    MetalBuf *save_attn_flat; // full-param: o_proj forward input [BN, Hq*hd]
    MetalBuf *save_gate_act;  // full-param: down_proj forward input (post-activation) [BN, IS]
} LayerSave;

typedef struct {
    MetalBuf *q, *k, *v, *o, *gate, *up, *down;
} GradAccumLayer;

struct SFTState {
    int D, Hq, Hkv, hd, IS, V, NL, N;
    int group_ratio;
    float *rope_thetas;       // per-layer rope theta
    float eps, lora_scaling;
    float emb_scale;          // Gemma3: sqrt(d_model), Qwen3: 1.0
    ModelType model_type;
    int lora_rank;
    int step_count;
    int full_param;           // 1 = full-param SGD, 0 = LoRA

    GradAccumLayer *grad_accum;     // [NL], NULL if not in accum mode
    MetalBuf *grad_accum_lm_head;   // lm_head grad accum, NULL if not in accum mode

    LayerW *layers;
    WMat *lm_head;
    MetalBuf *final_norm_g;
    const float *token_emb;
    float *token_emb_owned;   // owned copy (allows freeing model struct)

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

static MetalBuf *mbuf_f(int n) { return metal_buf_create((size_t)n * 4); }
static MetalBuf *mbuf_data(const float *d, int n) {
    return metal_buf_from_data(d, (size_t)n * 4);
}

// Initialize LoRA weights and AdamW state for one layer
static void init_lora_layer(LoRAW *lr, int R, int D, int Hq_hd, int Hkv_hd,
                             LoRALinear **q_loras, LoRALinear **v_loras, int i) {
    int Aq_n = R * D, Bq_n = Hq_hd * R;
    int Av_n = R * D, Bv_n = Hkv_hd * R;
    lr->Aq_size = Aq_n; lr->Bq_size = Bq_n;
    lr->Av_size = Av_n; lr->Bv_size = Bv_n;

    if (q_loras && q_loras[i]) {
        lr->A_q = mbuf_data(q_loras[i]->lora_A->data, Aq_n);
        lr->B_q = mbuf_data(q_loras[i]->lora_B->data, Bq_n);
    } else {
        float *tmp = malloc(Aq_n * sizeof(float));
        float std_a = 1.0f / sqrtf((float)D);
        for (int j = 0; j < Aq_n; j++)
            tmp[j] = ((float)rand() / RAND_MAX - 0.5f) * 2.0f * std_a;
        lr->A_q = mbuf_data(tmp, Aq_n);
        free(tmp);
        tmp = calloc(Bq_n, sizeof(float));
        lr->B_q = mbuf_data(tmp, Bq_n);
        free(tmp);
    }
    if (v_loras && v_loras[i]) {
        lr->A_v = mbuf_data(v_loras[i]->lora_A->data, Av_n);
        lr->B_v = mbuf_data(v_loras[i]->lora_B->data, Bv_n);
    } else {
        float *tmp = malloc(Av_n * sizeof(float));
        float std_a = 1.0f / sqrtf((float)D);
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

    lr->m_Aq = calloc(Aq_n, sizeof(float)); lr->v_Aq = calloc(Aq_n, sizeof(float));
    lr->m_Bq = calloc(Bq_n, sizeof(float)); lr->v_Bq = calloc(Bq_n, sizeof(float));
    lr->m_Av = calloc(Av_n, sizeof(float)); lr->v_Av = calloc(Av_n, sizeof(float));
    lr->m_Bv = calloc(Bv_n, sizeof(float)); lr->v_Bv = calloc(Bv_n, sizeof(float));
}

// Allocate scratch buffers shared between Qwen3 and Gemma3
static void alloc_scratch_buffers(SFTState *s, int BN, int Hq_hd, int Hkv_hd,
                                   int Hq_ND, int Hkv_ND, int R) {
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
    s->mb_lora_tmp = R > 0 ? mbuf_f(BN * R) : NULL;
    s->mb_lora_out = R > 0 ? mbuf_f(BN * Hq_hd) : NULL;

    s->mb_targets = metal_buf_create(BN * sizeof(uint32_t));
    s->mb_losses = metal_buf_create(BN * sizeof(float));
    s->losses_cpu = (float *)metal_buf_ptr(s->mb_losses);
    s->mb_dlogits = metal_buf_create((size_t)BN * s->V * sizeof(float));

    s->mb_dx = mbuf_f(BN * s->D);
    s->mb_dx2 = mbuf_f(BN * s->D);
    s->mb_d_score = mbuf_f(s->Hq * BN * BN);
}

// ================================================================
// State creation — Qwen3
// ================================================================

SFTState *sft_state_create(Qwen3Model *model, int seq_len,
                           int lora_rank, float lora_alpha) {
    if (fast_metal_init() != 0) {
        fprintf(stderr, "[FastSFT] Metal init failed\n");
        return NULL;
    }

    SFTState *s = calloc(1, sizeof(SFTState));
    s->model_type = MODEL_QWEN3;
    s->D = model->d_model;
    s->Hq = model->n_q_heads;
    s->Hkv = model->n_kv_heads;
    s->hd = model->head_dim;
    s->IS = model->intermediate_size;
    s->V = model->vocab_size;
    s->NL = model->n_layers;
    s->N = seq_len;
    s->group_ratio = s->Hq / s->Hkv;
    s->emb_scale = 1.0f;
    s->eps = 1e-6f;
    s->lora_rank = lora_rank;
    s->lora_scaling = lora_alpha / (float)lora_rank;
    s->step_count = 0;

    // Per-layer rope thetas (all same for Qwen3)
    s->rope_thetas = malloc(s->NL * sizeof(float));
    for (int i = 0; i < s->NL; i++)
        s->rope_thetas[i] = model->rope_theta;

    int BN = seq_len;
    int Hq_hd = s->Hq * s->hd;
    int Hkv_hd = s->Hkv * s->hd;
    int Hq_ND = s->Hq * BN * s->hd;
    int Hkv_ND = s->Hkv * BN * s->hd;
    int R = lora_rank;

    printf("[FastSFT] Converting weights to F16...\n");
    struct timespec t0, t1;
    clock_gettime(CLOCK_MONOTONIC, &t0);

    // Build LayerWeightRef array from Qwen3Model
    LayerWeightRef *refs = calloc(s->NL, sizeof(LayerWeightRef));
    for (int i = 0; i < s->NL; i++) {
        Qwen3Block *blk = model->blocks[i];
        LayerWeightRef *r = &refs[i];
        r->q_proj = (WeightRef){ blk->attn->q_proj->weight->data, Hq_hd, s->D };
        r->k_proj = (WeightRef){ blk->attn->k_proj->weight->data, Hkv_hd, s->D };
        r->v_proj = (WeightRef){ blk->attn->v_proj->weight->data, Hkv_hd, s->D };
        r->o_proj = (WeightRef){ blk->attn->o_proj->weight->data, s->D, Hq_hd };
        r->gate_proj = (WeightRef){ blk->gate_proj->weight->data, s->IS, s->D };
        r->up_proj = (WeightRef){ blk->up_proj->weight->data, s->IS, s->D };
        r->down_proj = (WeightRef){ blk->down_proj->weight->data, s->D, s->IS };
        r->input_norm_g = blk->input_norm->gamma->data;
        r->post_attn_norm_g = blk->post_attn_norm->gamma->data;
        r->q_norm_g = blk->attn->q_norm->gamma->data;
        r->k_norm_g = blk->attn->k_norm->gamma->data;
        r->d_model = s->D;
        r->head_dim = s->hd;
    }

    s->layers = layers_convert_upload(refs, s->NL, 1);
    free(refs);
    s->lm_head = wmat_convert(model->lm_head->weight->data, model->vocab_size, model->d_model, 1);
    s->final_norm_g = norm_upload(model->final_norm->gamma->data, model->d_model);
    // Own copy of token_emb (allows freeing model struct after init)
    size_t emb_bytes = (size_t)s->V * s->D * sizeof(float);
    s->token_emb_owned = malloc(emb_bytes);
    memcpy(s->token_emb_owned, model->token_emb->weight->data, emb_bytes);
    s->token_emb = s->token_emb_owned;

    // Initialize LoRA
    s->lora = calloc(s->NL, sizeof(LoRAW));
    for (int i = 0; i < s->NL; i++)
        init_lora_layer(&s->lora[i], R, s->D, Hq_hd, Hkv_hd,
                        model->q_loras, model->v_loras, i);

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
        // Qwen3: not needed
        sv->o_proj_out = NULL;
        sv->ff_raw_out = NULL;
    }
    s->save_x_final = mbuf_f(BN * s->D);
    s->save_ln_final = mbuf_f(BN * s->D);

    alloc_scratch_buffers(s, BN, Hq_hd, Hkv_hd, Hq_ND, Hkv_ND, R);

    clock_gettime(CLOCK_MONOTONIC, &t1);
    double ms = (t1.tv_sec - t0.tv_sec) * 1000.0 + (t1.tv_nsec - t0.tv_nsec) / 1e6;
    printf("[FastSFT] Init: %.0f ms, seq=%d, lora_rank=%d\n", ms, seq_len, lora_rank);
    return s;
}

// ================================================================
// State creation — Gemma3
// ================================================================

SFTState *sft_state_create_gemma3(Gemma3Model *model, int seq_len,
                                   int lora_rank, float lora_alpha) {
    if (fast_metal_init() != 0) {
        fprintf(stderr, "[FastSFT] Metal init failed\n");
        return NULL;
    }

    SFTState *s = calloc(1, sizeof(SFTState));
    s->model_type = MODEL_GEMMA3;
    s->D = model->d_model;
    s->Hq = model->n_q_heads;
    s->Hkv = model->n_kv_heads;
    s->hd = model->head_dim;
    s->IS = model->intermediate_size;
    s->V = model->vocab_size;
    s->NL = model->n_layers;
    s->N = seq_len;
    s->group_ratio = s->Hq / s->Hkv;
    s->emb_scale = sqrtf((float)model->d_model);
    s->eps = 1e-6f;
    s->lora_rank = lora_rank;
    s->lora_scaling = lora_alpha / (float)lora_rank;
    s->step_count = 0;

    // Per-layer rope thetas (local vs global)
    s->rope_thetas = malloc(s->NL * sizeof(float));
    for (int i = 0; i < s->NL; i++) {
        int is_sliding = (i % 6) != 0;
        s->rope_thetas[i] = is_sliding ? model->local_rope_theta : model->global_rope_theta;
    }

    int BN = seq_len;
    int Hq_hd = s->Hq * s->hd;
    int Hkv_hd = s->Hkv * s->hd;
    int Hq_ND = s->Hq * BN * s->hd;
    int Hkv_ND = s->Hkv * BN * s->hd;
    int R = lora_rank;

    printf("[FastSFT] Converting Gemma3 weights to F16...\n");
    struct timespec t0, t1;
    clock_gettime(CLOCK_MONOTONIC, &t0);

    // Build LayerWeightRef array from Gemma3Model
    LayerWeightRef *refs = calloc(s->NL, sizeof(LayerWeightRef));
    for (int i = 0; i < s->NL; i++) {
        Gemma3Block *blk = model->blocks[i];
        LayerWeightRef *r = &refs[i];
        r->q_proj = (WeightRef){ blk->attn->q_proj->weight->data, Hq_hd, s->D };
        r->k_proj = (WeightRef){ blk->attn->k_proj->weight->data, Hkv_hd, s->D };
        r->v_proj = (WeightRef){ blk->attn->v_proj->weight->data, Hkv_hd, s->D };
        r->o_proj = (WeightRef){ blk->attn->o_proj->weight->data, s->D, Hq_hd };
        r->gate_proj = (WeightRef){ blk->gate_proj->weight->data, s->IS, s->D };
        r->up_proj = (WeightRef){ blk->up_proj->weight->data, s->IS, s->D };
        r->down_proj = (WeightRef){ blk->down_proj->weight->data, s->D, s->IS };
        r->input_norm_g = blk->input_norm->gamma->data;
        r->post_attn_norm_g = blk->post_attn_norm->gamma->data;
        r->pre_ff_norm_g = blk->pre_ff_norm->gamma->data;
        r->post_ff_norm_g = blk->post_ff_norm->gamma->data;
        r->q_norm_g = blk->attn->q_norm->gamma->data;
        r->k_norm_g = blk->attn->k_norm->gamma->data;
        r->d_model = s->D;
        r->head_dim = s->hd;
    }

    s->layers = layers_convert_upload(refs, s->NL, 1);
    free(refs);
    // Gemma3: lm_head is tied with token_emb
    s->lm_head = wmat_convert(model->token_emb->weight->data, model->vocab_size, model->d_model, 1);
    s->final_norm_g = norm_upload(model->final_norm->gamma->data, model->d_model);
    // Own copy of token_emb (allows freeing model struct after init)
    size_t emb_bytes = (size_t)s->V * s->D * sizeof(float);
    s->token_emb_owned = malloc(emb_bytes);
    memcpy(s->token_emb_owned, model->token_emb->weight->data, emb_bytes);
    s->token_emb = s->token_emb_owned;

    // Initialize LoRA
    s->lora = calloc(s->NL, sizeof(LoRAW));
    for (int i = 0; i < s->NL; i++)
        init_lora_layer(&s->lora[i], R, s->D, Hq_hd, Hkv_hd,
                        model->q_loras, model->v_loras, i);

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
        // Gemma3: extra saved activations for 4-norm structure
        sv->o_proj_out = mbuf_f(BN * s->D);
        sv->ff_raw_out = mbuf_f(BN * s->D);
    }
    s->save_x_final = mbuf_f(BN * s->D);
    s->save_ln_final = mbuf_f(BN * s->D);

    alloc_scratch_buffers(s, BN, Hq_hd, Hkv_hd, Hq_ND, Hkv_ND, R);

    clock_gettime(CLOCK_MONOTONIC, &t1);
    double ms = (t1.tv_sec - t0.tv_sec) * 1000.0 + (t1.tv_nsec - t0.tv_nsec) / 1e6;
    printf("[FastSFT] Gemma3 Init: %.0f ms, seq=%d, lora_rank=%d\n", ms, seq_len, lora_rank);
    return s;
}

// ================================================================
// State creation — Qwen3 full-param
// ================================================================

SFTState *sft_state_create_full(Qwen3Model *model, int seq_len) {
    if (fast_metal_init() != 0) {
        fprintf(stderr, "[FastSFT] Metal init failed\n");
        return NULL;
    }

    SFTState *s = calloc(1, sizeof(SFTState));
    s->model_type = MODEL_QWEN3;
    s->full_param = 1;
    s->D = model->d_model;
    s->Hq = model->n_q_heads;
    s->Hkv = model->n_kv_heads;
    s->hd = model->head_dim;
    s->IS = model->intermediate_size;
    s->V = model->vocab_size;
    s->NL = model->n_layers;
    s->N = seq_len;
    s->group_ratio = s->Hq / s->Hkv;
    s->emb_scale = 1.0f;
    s->eps = 1e-6f;
    s->lora_rank = 0;
    s->lora_scaling = 0;
    s->step_count = 0;

    s->rope_thetas = malloc(s->NL * sizeof(float));
    for (int i = 0; i < s->NL; i++)
        s->rope_thetas[i] = model->rope_theta;

    int BN = seq_len;
    int Hq_hd = s->Hq * s->hd;
    int Hkv_hd = s->Hkv * s->hd;
    int Hq_ND = s->Hq * BN * s->hd;
    int Hkv_ND = s->Hkv * BN * s->hd;

    printf("[FastSFT] Converting Qwen3 weights to F16 (full-param)...\n");
    struct timespec t0, t1;
    clock_gettime(CLOCK_MONOTONIC, &t0);

    LayerWeightRef *refs = calloc(s->NL, sizeof(LayerWeightRef));
    for (int i = 0; i < s->NL; i++) {
        Qwen3Block *blk = model->blocks[i];
        LayerWeightRef *r = &refs[i];
        r->q_proj = (WeightRef){ blk->attn->q_proj->weight->data, Hq_hd, s->D };
        r->k_proj = (WeightRef){ blk->attn->k_proj->weight->data, Hkv_hd, s->D };
        r->v_proj = (WeightRef){ blk->attn->v_proj->weight->data, Hkv_hd, s->D };
        r->o_proj = (WeightRef){ blk->attn->o_proj->weight->data, s->D, Hq_hd };
        r->gate_proj = (WeightRef){ blk->gate_proj->weight->data, s->IS, s->D };
        r->up_proj = (WeightRef){ blk->up_proj->weight->data, s->IS, s->D };
        r->down_proj = (WeightRef){ blk->down_proj->weight->data, s->D, s->IS };
        r->input_norm_g = blk->input_norm->gamma->data;
        r->post_attn_norm_g = blk->post_attn_norm->gamma->data;
        r->q_norm_g = blk->attn->q_norm->gamma->data;
        r->k_norm_g = blk->attn->k_norm->gamma->data;
        r->d_model = s->D;
        r->head_dim = s->hd;
    }

    s->layers = layers_convert_upload(refs, s->NL, 1);
    free(refs);
    s->lm_head = wmat_convert(model->lm_head->weight->data, model->vocab_size, model->d_model, 1);
    s->final_norm_g = norm_upload(model->final_norm->gamma->data, model->d_model);
    size_t emb_bytes = (size_t)s->V * s->D * sizeof(float);
    s->token_emb_owned = malloc(emb_bytes);
    memcpy(s->token_emb_owned, model->token_emb->weight->data, emb_bytes);
    s->token_emb = s->token_emb_owned;

    s->lora = NULL;

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
        sv->lora_q_mid = NULL;
        sv->lora_v_mid = NULL;
        sv->o_proj_out = NULL;
        sv->ff_raw_out = NULL;
        sv->save_attn_flat = mbuf_f(BN * Hq_hd);
        sv->save_gate_act = mbuf_f(BN * s->IS);
    }
    s->save_x_final = mbuf_f(BN * s->D);
    s->save_ln_final = mbuf_f(BN * s->D);

    alloc_scratch_buffers(s, BN, Hq_hd, Hkv_hd, Hq_ND, Hkv_ND, 0);

    clock_gettime(CLOCK_MONOTONIC, &t1);
    double ms = (t1.tv_sec - t0.tv_sec) * 1000.0 + (t1.tv_nsec - t0.tv_nsec) / 1e6;
    printf("[FastSFT] Qwen3 Full-param Init: %.0f ms, seq=%d\n", ms, seq_len);
    return s;
}

// ================================================================
// State creation — Gemma3 full-param
// ================================================================

SFTState *sft_state_create_gemma3_full(Gemma3Model *model, int seq_len) {
    if (fast_metal_init() != 0) {
        fprintf(stderr, "[FastSFT] Metal init failed\n");
        return NULL;
    }

    SFTState *s = calloc(1, sizeof(SFTState));
    s->model_type = MODEL_GEMMA3;
    s->full_param = 1;
    s->D = model->d_model;
    s->Hq = model->n_q_heads;
    s->Hkv = model->n_kv_heads;
    s->hd = model->head_dim;
    s->IS = model->intermediate_size;
    s->V = model->vocab_size;
    s->NL = model->n_layers;
    s->N = seq_len;
    s->group_ratio = s->Hq / s->Hkv;
    s->emb_scale = sqrtf((float)model->d_model);
    s->eps = 1e-6f;
    s->lora_rank = 0;
    s->lora_scaling = 0;
    s->step_count = 0;

    s->rope_thetas = malloc(s->NL * sizeof(float));
    for (int i = 0; i < s->NL; i++) {
        int is_sliding = (i % 6) != 0;
        s->rope_thetas[i] = is_sliding ? model->local_rope_theta : model->global_rope_theta;
    }

    int BN = seq_len;
    int Hq_hd = s->Hq * s->hd;
    int Hkv_hd = s->Hkv * s->hd;
    int Hq_ND = s->Hq * BN * s->hd;
    int Hkv_ND = s->Hkv * BN * s->hd;

    printf("[FastSFT] Converting Gemma3 weights to F16 (full-param)...\n");
    struct timespec t0, t1;
    clock_gettime(CLOCK_MONOTONIC, &t0);

    LayerWeightRef *refs = calloc(s->NL, sizeof(LayerWeightRef));
    for (int i = 0; i < s->NL; i++) {
        Gemma3Block *blk = model->blocks[i];
        LayerWeightRef *r = &refs[i];
        r->q_proj = (WeightRef){ blk->attn->q_proj->weight->data, Hq_hd, s->D };
        r->k_proj = (WeightRef){ blk->attn->k_proj->weight->data, Hkv_hd, s->D };
        r->v_proj = (WeightRef){ blk->attn->v_proj->weight->data, Hkv_hd, s->D };
        r->o_proj = (WeightRef){ blk->attn->o_proj->weight->data, s->D, Hq_hd };
        r->gate_proj = (WeightRef){ blk->gate_proj->weight->data, s->IS, s->D };
        r->up_proj = (WeightRef){ blk->up_proj->weight->data, s->IS, s->D };
        r->down_proj = (WeightRef){ blk->down_proj->weight->data, s->D, s->IS };
        r->input_norm_g = blk->input_norm->gamma->data;
        r->post_attn_norm_g = blk->post_attn_norm->gamma->data;
        r->pre_ff_norm_g = blk->pre_ff_norm->gamma->data;
        r->post_ff_norm_g = blk->post_ff_norm->gamma->data;
        r->q_norm_g = blk->attn->q_norm->gamma->data;
        r->k_norm_g = blk->attn->k_norm->gamma->data;
        r->d_model = s->D;
        r->head_dim = s->hd;
    }

    s->layers = layers_convert_upload(refs, s->NL, 1);
    free(refs);
    s->lm_head = wmat_convert(model->token_emb->weight->data, model->vocab_size, model->d_model, 1);
    s->final_norm_g = norm_upload(model->final_norm->gamma->data, model->d_model);
    size_t emb_bytes = (size_t)s->V * s->D * sizeof(float);
    s->token_emb_owned = malloc(emb_bytes);
    memcpy(s->token_emb_owned, model->token_emb->weight->data, emb_bytes);
    s->token_emb = s->token_emb_owned;

    s->lora = NULL;

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
        sv->lora_q_mid = NULL;
        sv->lora_v_mid = NULL;
        sv->o_proj_out = mbuf_f(BN * s->D);
        sv->ff_raw_out = mbuf_f(BN * s->D);
        sv->save_attn_flat = mbuf_f(BN * Hq_hd);
        sv->save_gate_act = mbuf_f(BN * s->IS);
    }
    s->save_x_final = mbuf_f(BN * s->D);
    s->save_ln_final = mbuf_f(BN * s->D);

    alloc_scratch_buffers(s, BN, Hq_hd, Hkv_hd, Hq_ND, Hkv_ND, 0);

    clock_gettime(CLOCK_MONOTONIC, &t1);
    double ms = (t1.tv_sec - t0.tv_sec) * 1000.0 + (t1.tv_nsec - t0.tv_nsec) / 1e6;
    printf("[FastSFT] Gemma3 Full-param Init: %.0f ms, seq=%d\n", ms, seq_len);
    return s;
}

// ================================================================
// Forward pass (GPU, single encoder)
// ================================================================

void forward_to_logits(SFTState *s) {
    int BN = s->N, D = s->D, Hq = s->Hq, Hkv = s->Hkv, hd = s->hd;
    int IS = s->IS, R = s->lora_rank;
    int Hq_hd = Hq * hd, Hkv_hd = Hkv * hd;
    float eps = s->eps, sc = s->lora_scaling;
    float attn_scale = 1.0f / sqrtf((float)hd);
    int is_gemma3 = (s->model_type == MODEL_GEMMA3);
    int full_param = s->full_param;

    for (int L = 0; L < s->NL; L++) {
        LayerW *lw = &s->layers[L];
        LoRAW *lr = full_param ? NULL : &s->lora[L];
        LayerSave *sv = &s->saves[L];

        // Save input
        metal_enqueue_copy(s->mb_x, sv->x_in, BN * D);

        // Input RMSNorm
        metal_enqueue_rms_norm_batched(s->mb_x, lw->input_norm_g, s->mb_ln, D, eps, BN);
        metal_enqueue_copy(s->mb_ln, sv->ln1_out, BN * D);

        // Q projection
        metal_enqueue_f16_matmul(s->mb_ln, lw->q_proj->mbuf, s->mb_q, BN, Hq_hd, D);
        if (!full_param) {
            metal_enqueue_float_matmul(s->mb_ln, lr->A_q, s->mb_lora_tmp, BN, R, D);
            metal_enqueue_copy(s->mb_lora_tmp, sv->lora_q_mid, BN * R);
            metal_enqueue_float_matmul(s->mb_lora_tmp, lr->B_q, s->mb_lora_out, BN, Hq_hd, R);
            metal_enqueue_add_scaled(s->mb_q, s->mb_lora_out, sc, BN * Hq_hd);
        }

        // K projection
        metal_enqueue_f16_matmul(s->mb_ln, lw->k_proj->mbuf, s->mb_k, BN, Hkv_hd, D);

        // V projection
        metal_enqueue_f16_matmul(s->mb_ln, lw->v_proj->mbuf, s->mb_v, BN, Hkv_hd, D);
        if (!full_param) {
            metal_enqueue_float_matmul(s->mb_ln, lr->A_v, s->mb_lora_tmp, BN, R, D);
            metal_enqueue_copy(s->mb_lora_tmp, sv->lora_v_mid, BN * R);
            metal_enqueue_float_matmul(s->mb_lora_tmp, lr->B_v, s->mb_lora_out, BN, Hkv_hd, R);
            metal_enqueue_add_scaled(s->mb_v, s->mb_lora_out, sc, BN * Hkv_hd);
        }

        // Transpose: [BN, H*D] -> [H, N, D]
        metal_enqueue_transpose_heads_fwd(s->mb_q, s->mb_q_t, BN, Hq, hd);
        metal_enqueue_transpose_heads_fwd(s->mb_k, s->mb_k_t, BN, Hkv, hd);
        metal_enqueue_transpose_heads_fwd(s->mb_v, s->mb_v_t, BN, Hkv, hd);

        // Save pre-norm Q, K
        metal_enqueue_copy(s->mb_q_t, sv->q_pre_norm, Hq * BN * hd);
        metal_enqueue_copy(s->mb_k_t, sv->k_pre_norm, Hkv * BN * hd);

        // QK norm
        metal_enqueue_rms_norm_batched(s->mb_q_t, lw->q_norm_g, s->mb_q_t, hd, eps, Hq * BN);
        metal_enqueue_rms_norm_batched(s->mb_k_t, lw->k_norm_g, s->mb_k_t, hd, eps, Hkv * BN);

        // RoPE (per-layer theta)
        metal_enqueue_rope_train(s->mb_q_t, s->mb_k_t, Hq, Hkv, hd, BN, s->rope_thetas[L]);

        // Save Q after norm+RoPE
        metal_enqueue_copy(s->mb_q_t, sv->q_final, Hq * BN * hd);

        // Repeat KV
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

        // Save attn_flat for full-param backward (o_proj input)
        if (full_param)
            metal_enqueue_copy(s->mb_attn_flat, sv->save_attn_flat, BN * Hq_hd);

        // O projection
        metal_enqueue_f16_matmul(s->mb_attn_flat, lw->o_proj->mbuf, s->mb_x2, BN, D, Hq_hd);

        if (is_gemma3) {
            // Gemma3: o_proj → post_attn_norm → residual_add → pre_ff_norm → FFN
            metal_enqueue_copy(s->mb_x2, sv->o_proj_out, BN * D);
            metal_enqueue_rms_norm_batched(s->mb_x2, lw->post_attn_norm_g, s->mb_x2, D, eps, BN);
            metal_enqueue_residual_add(s->mb_x, s->mb_x2, BN * D);
            metal_enqueue_copy(s->mb_x, sv->x_mid, BN * D);
            metal_enqueue_rms_norm_batched(s->mb_x, lw->pre_ff_norm_g, s->mb_ln, D, eps, BN);
            metal_enqueue_copy(s->mb_ln, sv->ln2_out, BN * D);
        } else {
            // Qwen3: o_proj → residual_add → post_attn_norm → FFN
            metal_enqueue_residual_add(s->mb_x, s->mb_x2, BN * D);
            metal_enqueue_copy(s->mb_x, sv->x_mid, BN * D);
            metal_enqueue_rms_norm_batched(s->mb_x, lw->post_attn_norm_g, s->mb_ln, D, eps, BN);
            metal_enqueue_copy(s->mb_ln, sv->ln2_out, BN * D);
        }

        // FFN
        metal_enqueue_f16_matmul(s->mb_ln, lw->gate_proj->mbuf, s->mb_gate, BN, IS, D);
        metal_enqueue_copy(s->mb_gate, sv->gate_pre, BN * IS);
        metal_enqueue_f16_matmul(s->mb_ln, lw->up_proj->mbuf, s->mb_up, BN, IS, D);
        metal_enqueue_copy(s->mb_up, sv->up_val, BN * IS);

        if (is_gemma3) {
            metal_enqueue_gelu_mul(s->mb_gate, s->mb_up, BN * IS);
        } else {
            metal_enqueue_silu_mul(s->mb_gate, s->mb_up, BN * IS);
        }

        // Save gate activation for full-param backward (down_proj input)
        if (full_param)
            metal_enqueue_copy(s->mb_gate, sv->save_gate_act, BN * IS);

        metal_enqueue_f16_matmul(s->mb_gate, lw->down_proj->mbuf, s->mb_ff, BN, D, IS);

        if (is_gemma3) {
            // Gemma3: down_proj → post_ff_norm → residual_add
            metal_enqueue_copy(s->mb_ff, sv->ff_raw_out, BN * D);
            metal_enqueue_rms_norm_batched(s->mb_ff, lw->post_ff_norm_g, s->mb_ff, D, eps, BN);
            metal_enqueue_residual_add(s->mb_x, s->mb_ff, BN * D);
        } else {
            // Qwen3: down_proj → residual_add
            metal_enqueue_residual_add(s->mb_x, s->mb_ff, BN * D);
        }
    }

    // Save x before final norm
    metal_enqueue_copy(s->mb_x, s->save_x_final, BN * D);

    // Final RMSNorm
    metal_enqueue_rms_norm_batched(s->mb_x, s->final_norm_g, s->mb_ln, D, eps, BN);
    metal_enqueue_copy(s->mb_ln, s->save_ln_final, BN * D);

    // LM head
    metal_enqueue_f16_matmul(s->mb_ln, s->lm_head->mbuf, s->mb_logits, BN, s->V, D);
}

static void sft_apply_ce_loss(SFTState *s) {
    int BN = s->N;
    // Softmax CE loss + gradient
    metal_enqueue_softmax_ce(s->mb_logits, s->mb_targets, s->mb_losses,
                              s->mb_dlogits, BN, s->V);
    metal_enqueue_scale(s->mb_dlogits, 1.0f / (float)BN, BN * s->V);
}

// ================================================================
// Backward pass (GPU, single encoder, continued from forward)
// ================================================================

static void sft_backward_ex(SFTState *s, float lr) {
    int BN = s->N, D = s->D, Hq = s->Hq, Hkv = s->Hkv, hd = s->hd;
    int IS = s->IS, R = s->lora_rank;
    int Hq_hd = Hq * hd, Hkv_hd = Hkv * hd;
    float eps = s->eps, sc = s->lora_scaling;
    float attn_scale = 1.0f / sqrtf((float)hd);
    float grad_clip = 1e6f;
    int is_gemma3 = (s->model_type == MODEL_GEMMA3);
    int full_param = s->full_param;

    // Backward through LM head
    metal_enqueue_f16_matmul_nt(s->mb_dlogits, s->lm_head->mbuf, s->mb_dx, BN, s->V, D);
    if (full_param) {
        if (s->grad_accum)
            metal_enqueue_f16_grad_accum(s->grad_accum_lm_head, s->mb_dlogits, s->save_ln_final,
                                          s->V, BN, D);
        else
            metal_enqueue_f16_sgd_fused(s->lm_head->mbuf, s->mb_dlogits, s->save_ln_final,
                                         s->V, BN, D, lr);
    }

    // Backward through final RMSNorm
    metal_enqueue_rms_norm_backward(s->save_x_final, s->mb_dx, s->final_norm_g,
                                     s->mb_dx2, BN, D, eps);
    metal_enqueue_copy(s->mb_dx2, s->mb_dx, BN * D);

    for (int L = s->NL - 1; L >= 0; L--) {
        LayerW *lw = &s->layers[L];
        LoRAW *lora = full_param ? NULL : &s->lora[L];
        LayerSave *sv = &s->saves[L];

        // === FFN Backward ===
        if (is_gemma3) {
            // Gemma3: dx → rms_norm_bwd(ff_raw_out, dx, post_ff_norm_g) → dx2
            metal_enqueue_rms_norm_backward(sv->ff_raw_out, s->mb_dx, lw->post_ff_norm_g,
                                             s->mb_dx2, BN, D, eps);
            // dx2 → down_proj^T → gate_grad
            metal_enqueue_f16_matmul_nt(s->mb_dx2, lw->down_proj->mbuf, s->mb_gate, BN, D, IS);
            if (full_param) {
                if (s->grad_accum)
                    metal_enqueue_f16_grad_accum(s->grad_accum[L].down, s->mb_dx2, sv->save_gate_act,
                                                  D, BN, IS);
                else
                    metal_enqueue_f16_sgd_fused(lw->down_proj->mbuf, s->mb_dx2, sv->save_gate_act,
                                                 D, BN, IS, lr);
            }
            // gate_grad → gelu_mul_backward
            metal_enqueue_gelu_mul_backward(s->mb_gate, sv->gate_pre, sv->up_val,
                                             s->mb_gate, s->mb_up, BN * IS);
            // gate/up → gate_proj^T, up_proj^T → d_pre_ff_norm_out
            metal_enqueue_f16_matmul_nt(s->mb_gate, lw->gate_proj->mbuf, s->mb_ln, BN, IS, D);
            if (full_param) {
                if (s->grad_accum)
                    metal_enqueue_f16_grad_accum(s->grad_accum[L].gate, s->mb_gate, sv->ln2_out,
                                                  IS, BN, D);
                else
                    metal_enqueue_f16_sgd_fused(lw->gate_proj->mbuf, s->mb_gate, sv->ln2_out,
                                                 IS, BN, D, lr);
            }
            metal_enqueue_f16_matmul_nt(s->mb_up, lw->up_proj->mbuf, s->mb_x2, BN, IS, D);
            if (full_param) {
                if (s->grad_accum)
                    metal_enqueue_f16_grad_accum(s->grad_accum[L].up, s->mb_up, sv->ln2_out,
                                                  IS, BN, D);
                else
                    metal_enqueue_f16_sgd_fused(lw->up_proj->mbuf, s->mb_up, sv->ln2_out,
                                                 IS, BN, D, lr);
            }
            metal_enqueue_residual_add(s->mb_ln, s->mb_x2, BN * D);
            // d_pre_ff_norm_out → rms_norm_bwd(x_mid, ln, pre_ff_norm_g) → dx2
            metal_enqueue_rms_norm_backward(sv->x_mid, s->mb_ln, lw->pre_ff_norm_g,
                                             s->mb_dx2, BN, D, eps);
            metal_enqueue_residual_add(s->mb_dx, s->mb_dx2, BN * D);
        } else {
            // Qwen3: original path
            metal_enqueue_f16_matmul_nt(s->mb_dx, lw->down_proj->mbuf, s->mb_gate, BN, D, IS);
            if (full_param) {
                if (s->grad_accum)
                    metal_enqueue_f16_grad_accum(s->grad_accum[L].down, s->mb_dx, sv->save_gate_act,
                                                  D, BN, IS);
                else
                    metal_enqueue_f16_sgd_fused(lw->down_proj->mbuf, s->mb_dx, sv->save_gate_act,
                                                 D, BN, IS, lr);
            }
            metal_enqueue_silu_mul_backward(s->mb_gate, sv->gate_pre, sv->up_val,
                                             s->mb_gate, s->mb_up, BN * IS);
            metal_enqueue_f16_matmul_nt(s->mb_gate, lw->gate_proj->mbuf, s->mb_ln, BN, IS, D);
            if (full_param) {
                if (s->grad_accum)
                    metal_enqueue_f16_grad_accum(s->grad_accum[L].gate, s->mb_gate, sv->ln2_out,
                                                  IS, BN, D);
                else
                    metal_enqueue_f16_sgd_fused(lw->gate_proj->mbuf, s->mb_gate, sv->ln2_out,
                                                 IS, BN, D, lr);
            }
            metal_enqueue_f16_matmul_nt(s->mb_up, lw->up_proj->mbuf, s->mb_x2, BN, IS, D);
            if (full_param) {
                if (s->grad_accum)
                    metal_enqueue_f16_grad_accum(s->grad_accum[L].up, s->mb_up, sv->ln2_out,
                                                  IS, BN, D);
                else
                    metal_enqueue_f16_sgd_fused(lw->up_proj->mbuf, s->mb_up, sv->ln2_out,
                                                 IS, BN, D, lr);
            }
            metal_enqueue_residual_add(s->mb_ln, s->mb_x2, BN * D);
            metal_enqueue_rms_norm_backward(sv->x_mid, s->mb_ln, lw->post_attn_norm_g,
                                             s->mb_dx2, BN, D, eps);
            metal_enqueue_residual_add(s->mb_dx, s->mb_dx2, BN * D);
        }

        // === Attention Backward ===
        if (is_gemma3) {
            // Gemma3: dx → rms_norm_bwd(o_proj_out, dx, post_attn_norm_g) → dx2
            metal_enqueue_rms_norm_backward(sv->o_proj_out, s->mb_dx, lw->post_attn_norm_g,
                                             s->mb_dx2, BN, D, eps);
            // dx2 → o_proj^T → attn_flat
            metal_enqueue_f16_matmul_nt(s->mb_dx2, lw->o_proj->mbuf, s->mb_attn_flat, BN, D, Hq_hd);
            if (full_param) {
                if (s->grad_accum)
                    metal_enqueue_f16_grad_accum(s->grad_accum[L].o, s->mb_dx2, sv->save_attn_flat,
                                                  D, BN, Hq_hd);
                else
                    metal_enqueue_f16_sgd_fused(lw->o_proj->mbuf, s->mb_dx2, sv->save_attn_flat,
                                                 D, BN, Hq_hd, lr);
            }
        } else {
            // Qwen3: dx → o_proj^T → attn_flat
            metal_enqueue_f16_matmul_nt(s->mb_dx, lw->o_proj->mbuf, s->mb_attn_flat, BN, D, Hq_hd);
            if (full_param) {
                if (s->grad_accum)
                    metal_enqueue_f16_grad_accum(s->grad_accum[L].o, s->mb_dx, sv->save_attn_flat,
                                                  D, BN, Hq_hd);
                else
                    metal_enqueue_f16_sgd_fused(lw->o_proj->mbuf, s->mb_dx, sv->save_attn_flat,
                                                 D, BN, Hq_hd, lr);
            }
        }

        metal_enqueue_transpose_heads_fwd(s->mb_attn_flat, s->mb_attn_out, BN, Hq, hd);
        metal_enqueue_attn_bwd_dq(s->mb_attn_out, sv->probs, sv->v_exp,
                                   sv->k_exp, s->mb_d_score, s->mb_q_t,
                                   Hq, BN, hd, attn_scale);
        metal_enqueue_attn_bwd_dkv(s->mb_d_score, sv->q_final, sv->probs,
                                    s->mb_attn_out, s->mb_k_exp, s->mb_v_exp,
                                    Hq, BN, hd);
        metal_enqueue_repeat_kv_bwd(s->mb_k_exp, s->mb_k_t, Hkv, BN, hd, s->group_ratio);
        metal_enqueue_repeat_kv_bwd(s->mb_v_exp, s->mb_v_t, Hkv, BN, hd, s->group_ratio);
        metal_enqueue_rope_train_bwd(s->mb_q_t, s->mb_k_t, Hq, Hkv, hd, BN, s->rope_thetas[L]);
        metal_enqueue_rms_norm_backward(sv->q_pre_norm, s->mb_q_t, lw->q_norm_g,
                                         s->mb_q_t, Hq * BN, hd, eps);
        metal_enqueue_rms_norm_backward(sv->k_pre_norm, s->mb_k_t, lw->k_norm_g,
                                         s->mb_k_t, Hkv * BN, hd, eps);
        metal_enqueue_transpose_heads_rev(s->mb_q_t, s->mb_q, BN, Hq, hd);
        metal_enqueue_transpose_heads_rev(s->mb_k_t, s->mb_k, BN, Hkv, hd);
        metal_enqueue_transpose_heads_rev(s->mb_v_t, s->mb_v, BN, Hkv, hd);

        if (full_param) {
            // === Full-param SGD for Q, K, V projections ===
            if (s->grad_accum) {
                metal_enqueue_f16_grad_accum(s->grad_accum[L].q, s->mb_q, sv->ln1_out,
                                              Hq_hd, BN, D);
                metal_enqueue_f16_grad_accum(s->grad_accum[L].k, s->mb_k, sv->ln1_out,
                                              Hkv_hd, BN, D);
                metal_enqueue_f16_grad_accum(s->grad_accum[L].v, s->mb_v, sv->ln1_out,
                                              Hkv_hd, BN, D);
            } else {
                metal_enqueue_f16_sgd_fused(lw->q_proj->mbuf, s->mb_q, sv->ln1_out,
                                             Hq_hd, BN, D, lr);
                metal_enqueue_f16_sgd_fused(lw->k_proj->mbuf, s->mb_k, sv->ln1_out,
                                             Hkv_hd, BN, D, lr);
                metal_enqueue_f16_sgd_fused(lw->v_proj->mbuf, s->mb_v, sv->ln1_out,
                                             Hkv_hd, BN, D, lr);
            }

            // === Base projections backward (dX through Q, K, V) ===
            metal_enqueue_f16_matmul_nt(s->mb_q, lw->q_proj->mbuf, s->mb_ln, BN, Hq_hd, D);
            metal_enqueue_f16_matmul_nt(s->mb_k, lw->k_proj->mbuf, s->mb_x2, BN, Hkv_hd, D);
            metal_enqueue_residual_add(s->mb_ln, s->mb_x2, BN * D);
            metal_enqueue_f16_matmul_nt(s->mb_v, lw->v_proj->mbuf, s->mb_x2, BN, Hkv_hd, D);
            metal_enqueue_residual_add(s->mb_ln, s->mb_x2, BN * D);
        } else {
            // === LoRA Q backward ===
            metal_enqueue_float_matmul_tn(s->mb_q, sv->lora_q_mid, lora->dB_q, Hq_hd, R, BN);
            metal_enqueue_scale(lora->dB_q, sc, Hq_hd * R);
            metal_enqueue_float_matmul_nt(s->mb_q, lora->B_q, s->mb_lora_tmp, BN, Hq_hd, R);
            metal_enqueue_scale(s->mb_lora_tmp, sc, BN * R);
            metal_enqueue_float_matmul_tn(s->mb_lora_tmp, sv->ln1_out, lora->dA_q, R, D, BN);
            metal_enqueue_float_matmul_nt(s->mb_lora_tmp, lora->A_q, s->mb_lora_out, BN, R, D);

            // === Base projections backward ===
            metal_enqueue_f16_matmul_nt(s->mb_q, lw->q_proj->mbuf, s->mb_ln, BN, Hq_hd, D);
            metal_enqueue_residual_add(s->mb_ln, s->mb_lora_out, BN * D);
            metal_enqueue_f16_matmul_nt(s->mb_k, lw->k_proj->mbuf, s->mb_x2, BN, Hkv_hd, D);
            metal_enqueue_residual_add(s->mb_ln, s->mb_x2, BN * D);

            // === LoRA V backward ===
            metal_enqueue_float_matmul_tn(s->mb_v, sv->lora_v_mid, lora->dB_v, Hkv_hd, R, BN);
            metal_enqueue_scale(lora->dB_v, sc, Hkv_hd * R);
            metal_enqueue_float_matmul_nt(s->mb_v, lora->B_v, s->mb_lora_tmp, BN, Hkv_hd, R);
            metal_enqueue_scale(s->mb_lora_tmp, sc, BN * R);
            metal_enqueue_float_matmul_tn(s->mb_lora_tmp, sv->ln1_out, lora->dA_v, R, D, BN);
            metal_enqueue_float_matmul_nt(s->mb_lora_tmp, lora->A_v, s->mb_lora_out, BN, R, D);

            metal_enqueue_f16_matmul_nt(s->mb_v, lw->v_proj->mbuf, s->mb_x2, BN, Hkv_hd, D);
            metal_enqueue_residual_add(s->mb_ln, s->mb_x2, BN * D);
            metal_enqueue_residual_add(s->mb_ln, s->mb_lora_out, BN * D);
        }

        // === Input norm backward + residual ===
        metal_enqueue_rms_norm_backward(sv->x_in, s->mb_ln, lw->input_norm_g,
                                         s->mb_dx2, BN, D, eps);
        metal_enqueue_residual_add(s->mb_dx, s->mb_dx2, BN * D);

        // Gradient clipping
        metal_enqueue_clamp(s->mb_dx, grad_clip, BN * D);
    }
}

void sft_backward(SFTState *s) { sft_backward_ex(s, 0); }
void sft_backward_sgd(SFTState *s, float lr) { sft_backward_ex(s, lr); }
void sft_backward_accum(SFTState *s) { sft_backward_ex(s, 0); }
int sft_is_full_param(SFTState *s) { return s->full_param; }

// Helper to create a zero-initialized F16 MetalBuf
static MetalBuf *mbuf_h_zero(int n) {
    MetalBuf *b = metal_buf_create((size_t)n * 2);
    memset(metal_buf_ptr(b), 0, (size_t)n * 2);
    return b;
}

void sft_alloc_grad_accum(SFTState *s) {
    if (s->grad_accum) return;  // already allocated
    int Hq_hd = s->Hq * s->hd;
    int Hkv_hd = s->Hkv * s->hd;

    s->grad_accum = calloc(s->NL, sizeof(GradAccumLayer));
    for (int L = 0; L < s->NL; L++) {
        GradAccumLayer *ga = &s->grad_accum[L];
        ga->q    = mbuf_h_zero(Hq_hd * s->D);
        ga->k    = mbuf_h_zero(Hkv_hd * s->D);
        ga->v    = mbuf_h_zero(Hkv_hd * s->D);
        ga->o    = mbuf_h_zero(s->D * Hq_hd);
        ga->gate = mbuf_h_zero(s->IS * s->D);
        ga->up   = mbuf_h_zero(s->IS * s->D);
        ga->down = mbuf_h_zero(s->D * s->IS);
    }
    s->grad_accum_lm_head = mbuf_h_zero(s->V * s->D);

    // Compute memory usage
    size_t per_layer = (size_t)(Hq_hd * s->D + Hkv_hd * s->D + Hkv_hd * s->D +
                                s->D * Hq_hd + s->IS * s->D + s->IS * s->D + s->D * s->IS) * 2;
    size_t total = per_layer * s->NL + (size_t)s->V * s->D * 2;
    printf("[FastSFT] Grad accum buffers allocated: %.2f MB\n", (float)total / (1024.0f * 1024.0f));
}

void sft_apply_grad_sgd(SFTState *s, float lr) {
    if (!s->grad_accum) return;
    int Hq_hd = s->Hq * s->hd;
    int Hkv_hd = s->Hkv * s->hd;

    for (int L = 0; L < s->NL; L++) {
        LayerW *lw = &s->layers[L];
        GradAccumLayer *ga = &s->grad_accum[L];
        metal_enqueue_f16_sgd_apply(lw->q_proj->mbuf,    ga->q,    Hq_hd * s->D,  lr);
        metal_enqueue_f16_sgd_apply(lw->k_proj->mbuf,    ga->k,    Hkv_hd * s->D, lr);
        metal_enqueue_f16_sgd_apply(lw->v_proj->mbuf,    ga->v,    Hkv_hd * s->D, lr);
        metal_enqueue_f16_sgd_apply(lw->o_proj->mbuf,    ga->o,    s->D * Hq_hd,  lr);
        metal_enqueue_f16_sgd_apply(lw->gate_proj->mbuf, ga->gate, s->IS * s->D,  lr);
        metal_enqueue_f16_sgd_apply(lw->up_proj->mbuf,   ga->up,   s->IS * s->D,  lr);
        metal_enqueue_f16_sgd_apply(lw->down_proj->mbuf, ga->down, s->D * s->IS,  lr);
    }
    metal_enqueue_f16_sgd_apply(s->lm_head->mbuf, s->grad_accum_lm_head, s->V * s->D, lr);
    metal_flush();
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

    // Embedding lookup (CPU -> shared GPU buffer)
    for (int i = 0; i < BN; i++)
        memcpy(s->x_cpu + i * D, &s->token_emb[input_tokens[i] * D], D * sizeof(float));

    // Gemma3: scale embeddings by sqrt(d_model)
    if (s->model_type == MODEL_GEMMA3) {
        for (int i = 0; i < BN * D; i++)
            s->x_cpu[i] *= s->emb_scale;
    }

    // Copy targets to GPU
    memcpy(metal_buf_ptr(s->mb_targets), target_tokens, BN * sizeof(uint32_t));

    // Forward + loss (GPU, single encoder)
    forward_to_logits(s);
    sft_apply_ce_loss(s);

    // Backward (GPU, same encoder as forward)
    if (s->full_param) {
        sft_backward_ex(s, lr);
    } else {
        sft_backward(s);
    }

    // Submit all GPU work and wait
    metal_flush();

    // Compute average loss on CPU
    float total_loss = 0;
    for (int i = 0; i < BN; i++) total_loss += s->losses_cpu[i];
    float avg_loss = total_loss / (float)BN;

    // Update LoRA weights on CPU (only for LoRA mode)
    if (!s->full_param) {
        s->step_count++;
        lora_update(s, lr);
    }

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

void sft_sync_lora_to_gemma3(SFTState *s, Gemma3Model *model) {
    if (!model->q_loras) {
        gemma3_attach_lora(model, s->lora_rank, s->lora_scaling * s->lora_rank);
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
// Accessors for GRPO integration
// ================================================================

MetalBuf *sft_get_logits(SFTState *s)    { return s->mb_logits; }
MetalBuf *sft_get_dlogits(SFTState *s)   { return s->mb_dlogits; }
float *sft_get_x_cpu(SFTState *s)        { return s->x_cpu; }
float *sft_get_losses_cpu(SFTState *s)   { return s->losses_cpu; }
int sft_get_seq_len(SFTState *s)         { return s->N; }
int sft_get_vocab_size(SFTState *s)      { return s->V; }
int sft_get_d_model(SFTState *s)         { return s->D; }
int sft_get_lora_rank(SFTState *s)       { return s->lora_rank; }
int sft_get_n_layers(SFTState *s)        { return s->NL; }
float sft_get_emb_scale(SFTState *s)     { return s->emb_scale; }
const float *sft_get_token_emb(SFTState *s) { return s->token_emb; }
int sft_get_step_count(SFTState *s)      { return s->step_count; }
void sft_set_step_count(SFTState *s, int step) { s->step_count = step; }

int sft_get_n_lora_grad_bufs(SFTState *s) { return s->NL * 4; }

float *sft_get_lora_grad_ptr(SFTState *s, int idx, int *out_size) {
    int L = idx / 4;
    int sub = idx % 4;
    LoRAW *lr = &s->lora[L];
    switch (sub) {
        case 0: if (out_size) *out_size = lr->Aq_size; return (float *)metal_buf_ptr(lr->dA_q);
        case 1: if (out_size) *out_size = lr->Bq_size; return (float *)metal_buf_ptr(lr->dB_q);
        case 2: if (out_size) *out_size = lr->Av_size; return (float *)metal_buf_ptr(lr->dA_v);
        case 3: if (out_size) *out_size = lr->Bv_size; return (float *)metal_buf_ptr(lr->dB_v);
    }
    return NULL;
}

void sft_set_lora_grad(SFTState *s, int idx, const float *data, int size) {
    int sz = 0;
    float *ptr = sft_get_lora_grad_ptr(s, idx, &sz);
    if (ptr && size <= sz)
        memcpy(ptr, data, size * sizeof(float));
}

void sft_lora_update(SFTState *s, float lr) {
    clip_grad_norm(s, 1.0f);
    lora_update(s, lr);
}

MetalBuf *sft_get_lora_weight_buf(SFTState *s, int layer, int sub) {
    LoRAW *lr = &s->lora[layer];
    switch (sub) {
        case 0: return lr->A_q;
        case 1: return lr->B_q;
        case 2: return lr->A_v;
        case 3: return lr->B_v;
    }
    return NULL;
}

float sft_get_lora_scaling(SFTState *s) { return s->lora_scaling; }

void sft_save_lora(SFTState *s, const char *path) {
    FILE *fp = fopen(path, "wb");
    if (!fp) {
        fprintf(stderr, "[FastSFT] Failed to open %s for writing\n", path);
        return;
    }
    // Header: magic, n_layers, rank
    uint32_t magic = 0x4C4F5241;  // "LORA"
    int32_t n_layers = s->NL;
    int32_t rank = s->lora_rank;
    fwrite(&magic, 4, 1, fp);
    fwrite(&n_layers, 4, 1, fp);
    fwrite(&rank, 4, 1, fp);
    // Per-layer: A_q, B_q, A_v, B_v (float32 from GPU shared memory)
    for (int L = 0; L < s->NL; L++) {
        LoRAW *lr = &s->lora[L];
        fwrite(metal_buf_ptr(lr->A_q), sizeof(float), lr->Aq_size, fp);
        fwrite(metal_buf_ptr(lr->B_q), sizeof(float), lr->Bq_size, fp);
        fwrite(metal_buf_ptr(lr->A_v), sizeof(float), lr->Av_size, fp);
        fwrite(metal_buf_ptr(lr->B_v), sizeof(float), lr->Bv_size, fp);
    }
    fclose(fp);
    // Compute total LoRA params
    size_t total = 0;
    for (int L = 0; L < s->NL; L++) {
        LoRAW *lr = &s->lora[L];
        total += lr->Aq_size + lr->Bq_size + lr->Av_size + lr->Bv_size;
    }
    printf("[FastSFT] LoRA saved to %s (rank=%d, %.2f MB)\n",
           path, rank, (float)(total * sizeof(float)) / (1024.0f * 1024.0f));
}

void sft_save_full_checkpoint(SFTState *s, const char *path) {
    FILE *fp = fopen(path, "wb");
    if (!fp) {
        fprintf(stderr, "[FastSFT] Failed to open %s for writing\n", path);
        return;
    }
    uint32_t magic = 0x46554C4C;  // "FULL"
    uint32_t model_type = (uint32_t)s->model_type;
    uint32_t n_layers = (uint32_t)s->NL;
    uint32_t d_model = (uint32_t)s->D;
    uint32_t n_q_heads = (uint32_t)s->Hq;
    uint32_t n_kv_heads = (uint32_t)s->Hkv;
    uint32_t head_dim = (uint32_t)s->hd;
    uint32_t intermediate_size = (uint32_t)s->IS;
    uint32_t vocab_size = (uint32_t)s->V;
    fwrite(&magic, 4, 1, fp);
    fwrite(&model_type, 4, 1, fp);
    fwrite(&n_layers, 4, 1, fp);
    fwrite(&d_model, 4, 1, fp);
    fwrite(&n_q_heads, 4, 1, fp);
    fwrite(&n_kv_heads, 4, 1, fp);
    fwrite(&head_dim, 4, 1, fp);
    fwrite(&intermediate_size, 4, 1, fp);
    fwrite(&vocab_size, 4, 1, fp);

    int Hq_hd = s->Hq * s->hd;
    int Hkv_hd = s->Hkv * s->hd;
    size_t total_bytes = 0;

    // Per layer: 7 weight matrices as F16
    for (int L = 0; L < s->NL; L++) {
        LayerW *lw = &s->layers[L];
        struct { WMat *w; size_t n; } mats[] = {
            { lw->q_proj,    (size_t)Hq_hd * s->D },
            { lw->k_proj,    (size_t)Hkv_hd * s->D },
            { lw->v_proj,    (size_t)Hkv_hd * s->D },
            { lw->o_proj,    (size_t)s->D * Hq_hd },
            { lw->gate_proj, (size_t)s->IS * s->D },
            { lw->up_proj,   (size_t)s->IS * s->D },
            { lw->down_proj, (size_t)s->D * s->IS },
        };
        for (int m = 0; m < 7; m++) {
            size_t bytes = mats[m].n * sizeof(uint16_t);
            fwrite(metal_buf_ptr(mats[m].w->mbuf), 1, bytes, fp);
            total_bytes += bytes;
        }
    }

    // lm_head as F16
    size_t lm_bytes = (size_t)s->V * s->D * sizeof(uint16_t);
    fwrite(metal_buf_ptr(s->lm_head->mbuf), 1, lm_bytes, fp);
    total_bytes += lm_bytes;

    fclose(fp);
    printf("[FastSFT] Full checkpoint saved to %s (%.2f MB)\n",
           path, (float)total_bytes / (1024.0f * 1024.0f));
}

// Weight sharing accessors
LayerW *sft_get_layers(SFTState *s)          { return s->layers; }
WMat *sft_get_lm_head(SFTState *s)           { return s->lm_head; }
MetalBuf *sft_get_final_norm_g(SFTState *s)  { return s->final_norm_g; }
ModelType sft_get_model_type(SFTState *s)     { return s->model_type; }
int sft_get_n_q_heads(SFTState *s)           { return s->Hq; }
int sft_get_n_kv_heads(SFTState *s)          { return s->Hkv; }
int sft_get_head_dim(SFTState *s)            { return s->hd; }
int sft_get_intermediate_size(SFTState *s)   { return s->IS; }
const float *sft_get_rope_thetas(SFTState *s) { return s->rope_thetas; }

// ================================================================
// Cleanup
// ================================================================

void sft_state_free(SFTState *s) {
    if (!s) return;
    if (s->grad_accum) {
        for (int i = 0; i < s->NL; i++) {
            GradAccumLayer *ga = &s->grad_accum[i];
            metal_buf_free(ga->q); metal_buf_free(ga->k); metal_buf_free(ga->v);
            metal_buf_free(ga->o); metal_buf_free(ga->gate); metal_buf_free(ga->up);
            metal_buf_free(ga->down);
        }
        free(s->grad_accum);
        metal_buf_free(s->grad_accum_lm_head);
    }
    for (int i = 0; i < s->NL; i++) {
        if (s->lora) {
            LoRAW *lr = &s->lora[i];
            metal_buf_free(lr->A_q); metal_buf_free(lr->B_q);
            metal_buf_free(lr->A_v); metal_buf_free(lr->B_v);
            metal_buf_free(lr->dA_q); metal_buf_free(lr->dB_q);
            metal_buf_free(lr->dA_v); metal_buf_free(lr->dB_v);
            free(lr->m_Aq); free(lr->v_Aq); free(lr->m_Bq); free(lr->v_Bq);
            free(lr->m_Av); free(lr->v_Av); free(lr->m_Bv); free(lr->v_Bv);
        }

        LayerSave *sv = &s->saves[i];
        metal_buf_free(sv->x_in); metal_buf_free(sv->ln1_out);
        metal_buf_free(sv->q_pre_norm); metal_buf_free(sv->k_pre_norm);
        metal_buf_free(sv->q_final); metal_buf_free(sv->k_exp);
        metal_buf_free(sv->v_exp); metal_buf_free(sv->probs);
        metal_buf_free(sv->x_mid); metal_buf_free(sv->ln2_out);
        metal_buf_free(sv->gate_pre); metal_buf_free(sv->up_val);
        if (sv->lora_q_mid) metal_buf_free(sv->lora_q_mid);
        if (sv->lora_v_mid) metal_buf_free(sv->lora_v_mid);
        if (sv->o_proj_out) metal_buf_free(sv->o_proj_out);
        if (sv->ff_raw_out) metal_buf_free(sv->ff_raw_out);
        if (sv->save_attn_flat) metal_buf_free(sv->save_attn_flat);
        if (sv->save_gate_act) metal_buf_free(sv->save_gate_act);
    }
    layers_free(s->layers, s->NL);
    free(s->lora); free(s->saves);
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
    if (s->mb_lora_tmp) metal_buf_free(s->mb_lora_tmp);
    if (s->mb_lora_out) metal_buf_free(s->mb_lora_out);

    metal_buf_free(s->mb_targets); metal_buf_free(s->mb_losses);
    metal_buf_free(s->mb_dlogits);
    metal_buf_free(s->mb_dx); metal_buf_free(s->mb_dx2);
    metal_buf_free(s->mb_d_score);

    free(s->rope_thetas);
    free(s->token_emb_owned);
    fast_metal_shutdown();
    free(s);
}
