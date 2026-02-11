#include "attention.h"
#include "lora.h"
#include "../core/metal_backend.h"
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <assert.h>
#include <stdio.h>

MultiHeadAttention *mha_create(int d_model, int n_heads) {
    MultiHeadAttention *mha = calloc(1, sizeof(MultiHeadAttention));
    mha->d_model = d_model;
    mha->n_heads = n_heads;
    mha->head_dim = d_model / n_heads;
    mha->scale = 1.0f / sqrtf((float)mha->head_dim);

    mha->q_proj = linear_create(d_model, d_model, 0);
    mha->k_proj = linear_create(d_model, d_model, 0);
    mha->v_proj = linear_create(d_model, d_model, 0);
    mha->o_proj = linear_create(d_model, d_model, 0);
    return mha;
}

void mha_free(MultiHeadAttention *mha) {
    if (!mha) return;
    linear_free(mha->q_proj);
    linear_free(mha->k_proj);
    linear_free(mha->v_proj);
    linear_free(mha->o_proj);
    free(mha);
}

// Saved data for backward pass — includes tensors that must be freed
typedef struct {
    MultiHeadAttention *mha;
    Tensor *q;       // [B*H, N, D]
    Tensor *k;       // [B*H, N, D]
    Tensor *v;       // [B*H, N, D]
    Tensor *attn_out; // [B*H, N, D]
    int batch;
    int seq_len;
} MHASavedData;

// Cleanup function for MHASavedData: frees saved tensors
static void mha_saved_cleanup(void *data) {
    MHASavedData *sd = (MHASavedData *)data;
    if (sd->q) tensor_free(sd->q);
    if (sd->k) tensor_free(sd->k);
    if (sd->v) tensor_free(sd->v);
    if (sd->attn_out) tensor_free(sd->attn_out);
    free(sd);
}

static void mha_attention_backward(GraphNode *node) {
    MHASavedData *sd = (MHASavedData *)node->saved_data;
    Tensor *attn_flat = node->output;  // [BN, d_model]
    int B = sd->batch;
    int H = sd->mha->n_heads;
    int N = sd->seq_len;
    int D = sd->mha->head_dim;
    int d_model = H * D;

    if (metal_is_initialized() && attn_flat->grad) {
        // Reshape gradient from [BN, d_model] -> [B*H, N, D] (inverse of forward transpose)
        int bhnd_shape[] = {B * H, N, D};
        Tensor *dO_reshaped = tensor_zeros(bhnd_shape, 3, DTYPE_FP32, 0);
        float *grad_src = attn_flat->grad->data;
        float *grad_dst = dO_reshaped->data;
        for (int b = 0; b < B; b++) {
            for (int n = 0; n < N; n++) {
                for (int h = 0; h < H; h++) {
                    int src_off = (b * N + n) * d_model + h * D;
                    int dst_off = (b * H + h) * N * D + n * D;
                    memcpy(&grad_dst[dst_off], &grad_src[src_off], D * sizeof(float));
                }
            }
        }

        Tensor *dQ = tensor_zeros(bhnd_shape, 3, DTYPE_FP32, 0);
        Tensor *dK = tensor_zeros(bhnd_shape, 3, DTYPE_FP32, 0);
        Tensor *dV = tensor_zeros(bhnd_shape, 3, DTYPE_FP32, 0);

        metal_flash_attention_backward(
            sd->q->metal_buf, sd->k->metal_buf, sd->v->metal_buf,
            sd->attn_out->metal_buf, dO_reshaped->metal_buf,
            dQ->metal_buf, dK->metal_buf, dV->metal_buf,
            1, B * H, N, D, sd->mha->scale, 1);
        metal_synchronize();

        if (sd->q->grad) tensor_add_inplace(sd->q->grad, dQ);
        if (sd->k->grad) tensor_add_inplace(sd->k->grad, dK);
        if (sd->v->grad) tensor_add_inplace(sd->v->grad, dV);

        tensor_free(dO_reshaped);
        tensor_free(dQ);
        tensor_free(dK);
        tensor_free(dV);
    }
}

// CPU RoPE: Apply rotary position embeddings
static void cpu_rope(float *x, float *out, int BH, int N, int D) {
    int half_D = D / 2;
    for (int bh = 0; bh < BH; bh++) {
        for (int pos = 0; pos < N; pos++) {
            int base = bh * N * D + pos * D;
            for (int d = 0; d < half_D; d++) {
                float freq = 1.0f / powf(10000.0f, (float)(2 * d) / (float)D);
                float angle = (float)pos * freq;
                float cos_a = cosf(angle);
                float sin_a = sinf(angle);

                float x0 = x[base + d];
                float x1 = x[base + d + half_D];

                out[base + d]          = x0 * cos_a - x1 * sin_a;
                out[base + d + half_D] = x0 * sin_a + x1 * cos_a;
            }
        }
    }
}

Tensor *mha_forward(MultiHeadAttention *mha, Tensor *x,
                    int batch, int seq_len, ComputeGraph *g) {
    int d = mha->d_model;
    int H = mha->n_heads;
    int D = mha->head_dim;
    int BN = batch * seq_len;

    // Project Q, K, V: [BN, d] -> [BN, d]
    Tensor *q_flat = linear_forward(mha->q_proj, x, g);
    Tensor *k_flat = linear_forward(mha->k_proj, x, g);
    Tensor *v_flat = linear_forward(mha->v_proj, x, g);

    // Reshape to [B*H, N, D]
    int bhnd_shape[] = {batch * H, seq_len, D};
    Tensor *Q = tensor_zeros(bhnd_shape, 3, DTYPE_FP32, 1);
    Tensor *K = tensor_zeros(bhnd_shape, 3, DTYPE_FP32, 1);
    Tensor *V = tensor_zeros(bhnd_shape, 3, DTYPE_FP32, 1);

    // Transpose: [B, N, H, D] -> [B, H, N, D]
    float *qf = q_flat->data, *kf = k_flat->data, *vf = v_flat->data;
    float *Qd = Q->data, *Kd = K->data, *Vd = V->data;
    for (int b = 0; b < batch; b++) {
        for (int n = 0; n < seq_len; n++) {
            for (int h = 0; h < H; h++) {
                int src_off = (b * seq_len + n) * d + h * D;
                int dst_off = (b * H + h) * seq_len * D + n * D;
                memcpy(&Qd[dst_off], &qf[src_off], D * sizeof(float));
                memcpy(&Kd[dst_off], &kf[src_off], D * sizeof(float));
                memcpy(&Vd[dst_off], &vf[src_off], D * sizeof(float));
            }
        }
    }

    // Apply RoPE to Q and K
    int rope_shape[] = {batch * H, seq_len, D};
    Tensor *Q_rope = tensor_zeros(rope_shape, 3, DTYPE_FP32, 1);
    Tensor *K_rope = tensor_zeros(rope_shape, 3, DTYPE_FP32, 1);

    if (metal_is_initialized()) {
        metal_rope(Q->metal_buf, Q_rope->metal_buf, 1, batch * H, seq_len, D, 0);
        metal_rope(K->metal_buf, K_rope->metal_buf, 1, batch * H, seq_len, D, 0);
        metal_synchronize();
    } else {
        // CPU RoPE
        cpu_rope(Q->data, Q_rope->data, batch * H, seq_len, D);
        cpu_rope(K->data, K_rope->data, batch * H, seq_len, D);
    }

    // Flash Attention (or CPU equivalent)
    Tensor *attn_out = tensor_zeros(bhnd_shape, 3, DTYPE_FP32, 1);

    if (metal_is_initialized()) {
        metal_flash_attention(Q_rope->metal_buf, K_rope->metal_buf, V->metal_buf,
                              attn_out->metal_buf,
                              1, batch * H, seq_len, D, mha->scale, 1);
        metal_synchronize();
    } else {
        // CPU attention with causal mask
        int BH = batch * H;
        for (int bh = 0; bh < BH; bh++) {
            float *qp = &Q_rope->data[bh * seq_len * D];
            float *kp = &K_rope->data[bh * seq_len * D];
            float *vp = &V->data[bh * seq_len * D];
            float *op = &attn_out->data[bh * seq_len * D];

            for (int i = 0; i < seq_len; i++) {
                float max_score = -1e30f;
                float *scores = calloc(seq_len, sizeof(float));
                for (int j = 0; j <= i; j++) { // causal
                    float score = 0;
                    for (int dd = 0; dd < D; dd++) {
                        score += qp[i * D + dd] * kp[j * D + dd];
                    }
                    score *= mha->scale;
                    scores[j] = score;
                    if (score > max_score) max_score = score;
                }
                float sum = 0;
                for (int j = 0; j <= i; j++) {
                    scores[j] = expf(scores[j] - max_score);
                    sum += scores[j];
                }
                for (int j = 0; j <= i; j++) scores[j] /= sum;
                for (int dd = 0; dd < D; dd++) {
                    float val = 0;
                    for (int j = 0; j <= i; j++) {
                        val += scores[j] * vp[j * D + dd];
                    }
                    op[i * D + dd] = val;
                }
                free(scores);
            }
        }
    }

    // Transpose back: [B, H, N, D] -> [B, N, H, D] = [BN, d]
    int out_shape[] = {BN, d};
    Tensor *attn_flat = tensor_zeros(out_shape, 2, DTYPE_FP32, 1);
    float *af = attn_out->data;
    float *of = attn_flat->data;
    for (int b2 = 0; b2 < batch; b2++) {
        for (int n = 0; n < seq_len; n++) {
            for (int h = 0; h < H; h++) {
                int src_off = (b2 * H + h) * seq_len * D + n * D;
                int dst_off = (b2 * seq_len + n) * d + h * D;
                memcpy(&of[dst_off], &af[src_off], D * sizeof(float));
            }
        }
    }

    // Output projection
    Tensor *output = linear_forward(mha->o_proj, attn_flat, g);

    // Register attention in graph for backward
    if (g) {
        MHASavedData *sd = calloc(1, sizeof(MHASavedData));
        sd->mha = mha;
        sd->q = Q_rope;
        sd->k = K_rope;
        sd->v = V;
        sd->attn_out = attn_out;
        sd->batch = batch;
        sd->seq_len = seq_len;

        Tensor *inputs[] = {Q_rope, K_rope, V};
        GraphNode *node = graph_add_node(g, attn_flat, inputs, 3, mha_attention_backward);
        node->saved_data = sd;
        node->cleanup_fn = mha_saved_cleanup;
    } else {
        // Eval mode: free tensors not needed
        tensor_free(Q_rope);
        tensor_free(K_rope);
        tensor_free(V);
        tensor_free(attn_out);
    }

    // Free intermediate tensors not in the computation graph
    tensor_free(Q);
    tensor_free(K);
    if (!g) {
        // Eval mode: free everything
        tensor_free(attn_flat);
        tensor_free(q_flat);
        tensor_free(k_flat);
        tensor_free(v_flat);
    }
    // When g != NULL: q_flat, k_flat, v_flat are linear_forward outputs (freed by graph_reset)
    //                 attn_flat is the attention node output (freed by graph_reset)

    return output;
}

void mha_collect_params(MultiHeadAttention *mha, ParamList *pl) {
    param_list_add(pl, mha->q_proj->weight);
    param_list_add(pl, mha->k_proj->weight);
    param_list_add(pl, mha->v_proj->weight);
    param_list_add(pl, mha->o_proj->weight);
}

// ==================================================================
// Grouped Query Attention (GQA)
// ==================================================================

GroupedQueryAttention *gqa_create(int d_model, int n_q_heads, int n_kv_heads,
                                  int head_dim, float rope_theta) {
    GroupedQueryAttention *gqa = calloc(1, sizeof(GroupedQueryAttention));
    gqa->d_model = d_model;
    gqa->n_q_heads = n_q_heads;
    gqa->n_kv_heads = n_kv_heads;
    gqa->head_dim = head_dim;
    gqa->scale = 1.0f / sqrtf((float)head_dim);
    gqa->rope_theta = rope_theta;

    int q_dim = n_q_heads * head_dim;
    int kv_dim = n_kv_heads * head_dim;

    gqa->q_proj = linear_create(d_model, q_dim, 0);
    gqa->k_proj = linear_create(d_model, kv_dim, 0);
    gqa->v_proj = linear_create(d_model, kv_dim, 0);
    gqa->o_proj = linear_create(q_dim, d_model, 0);
    gqa->q_norm = rms_norm_create(head_dim, 1e-6f);
    gqa->k_norm = rms_norm_create(head_dim, 1e-6f);
    return gqa;
}

void gqa_free(GroupedQueryAttention *gqa) {
    if (!gqa) return;
    linear_free(gqa->q_proj);
    linear_free(gqa->k_proj);
    linear_free(gqa->v_proj);
    linear_free(gqa->o_proj);
    rms_norm_free(gqa->q_norm);
    rms_norm_free(gqa->k_norm);
    free(gqa);
}

// Saved data for GQA backward
typedef struct {
    GroupedQueryAttention *gqa;
    Tensor *q;        // [B*n_q_heads, N, D]
    Tensor *k;        // [B*n_q_heads, N, D] (after repeat)
    Tensor *v;        // [B*n_q_heads, N, D] (after repeat)
    Tensor *attn_out; // [B*n_q_heads, N, D]
    int batch;
    int seq_len;
} GQASavedData;

static void gqa_saved_cleanup(void *data) {
    GQASavedData *sd = (GQASavedData *)data;
    if (sd->q) tensor_free(sd->q);
    if (sd->k) tensor_free(sd->k);
    if (sd->v) tensor_free(sd->v);
    if (sd->attn_out) tensor_free(sd->attn_out);
    free(sd);
}

static void gqa_attention_backward(GraphNode *node) {
    GQASavedData *sd = (GQASavedData *)node->saved_data;
    Tensor *attn_flat = node->output;  // [BN, d_model]
    int B = sd->batch;
    int Hq = sd->gqa->n_q_heads;
    int N = sd->seq_len;
    int D = sd->gqa->head_dim;
    int d_model = Hq * D;

    if (metal_is_initialized() && attn_flat->grad) {
        int bhnd_shape[] = {B * Hq, N, D};
        Tensor *dO_reshaped = tensor_zeros(bhnd_shape, 3, DTYPE_FP32, 0);
        float *grad_src = attn_flat->grad->data;
        float *grad_dst = dO_reshaped->data;
        for (int b = 0; b < B; b++) {
            for (int n = 0; n < N; n++) {
                for (int h = 0; h < Hq; h++) {
                    int src_off = (b * N + n) * d_model + h * D;
                    int dst_off = (b * Hq + h) * N * D + n * D;
                    memcpy(&grad_dst[dst_off], &grad_src[src_off], D * sizeof(float));
                }
            }
        }

        Tensor *dQ = tensor_zeros(bhnd_shape, 3, DTYPE_FP32, 0);
        Tensor *dK = tensor_zeros(bhnd_shape, 3, DTYPE_FP32, 0);
        Tensor *dV = tensor_zeros(bhnd_shape, 3, DTYPE_FP32, 0);

        metal_flash_attention_backward(
            sd->q->metal_buf, sd->k->metal_buf, sd->v->metal_buf,
            sd->attn_out->metal_buf, dO_reshaped->metal_buf,
            dQ->metal_buf, dK->metal_buf, dV->metal_buf,
            1, B * Hq, N, D, sd->gqa->scale, 1);
        metal_synchronize();

        if (sd->q->grad) tensor_add_inplace(sd->q->grad, dQ);
        if (sd->k->grad) tensor_add_inplace(sd->k->grad, dK);
        if (sd->v->grad) tensor_add_inplace(sd->v->grad, dV);

        tensor_free(dO_reshaped);
        tensor_free(dQ);
        tensor_free(dK);
        tensor_free(dV);
    }
}

// CPU RoPE with configurable theta
static void cpu_rope_theta(float *x, float *out, int BH, int N, int D, float theta) {
    int half_D = D / 2;
    for (int bh = 0; bh < BH; bh++) {
        for (int pos = 0; pos < N; pos++) {
            int base = bh * N * D + pos * D;
            for (int d = 0; d < half_D; d++) {
                float freq = 1.0f / powf(theta, (float)(2 * d) / (float)D);
                float angle = (float)pos * freq;
                float cos_a = cosf(angle);
                float sin_a = sinf(angle);
                float x0 = x[base + d];
                float x1 = x[base + d + half_D];
                out[base + d]          = x0 * cos_a - x1 * sin_a;
                out[base + d + half_D] = x0 * sin_a + x1 * cos_a;
            }
        }
    }
}

// Internal implementation with optional LoRA
static Tensor *gqa_forward_impl(GroupedQueryAttention *gqa, Tensor *x,
                                int batch, int seq_len, ComputeGraph *g,
                                LoRALinear *q_lora, LoRALinear *v_lora) {
    int Hq = gqa->n_q_heads;
    int Hkv = gqa->n_kv_heads;
    int D = gqa->head_dim;
    int BN = batch * seq_len;
    int group_ratio = Hq / Hkv;

    // Project Q, K, V — use LoRA when provided
    Tensor *q_flat = q_lora ? lora_forward(q_lora, x, g) : linear_forward(gqa->q_proj, x, g);
    Tensor *k_flat = linear_forward(gqa->k_proj, x, g);  // [BN, Hkv*D]
    Tensor *v_flat = v_lora ? lora_forward(v_lora, x, g) : linear_forward(gqa->v_proj, x, g);

    // Reshape Q to [B*Hq, N, D]
    int q_shape[] = {batch * Hq, seq_len, D};
    Tensor *Q = tensor_zeros(q_shape, 3, DTYPE_FP32, 1);
    float *qf = q_flat->data;
    float *Qd = Q->data;
    for (int b = 0; b < batch; b++) {
        for (int n = 0; n < seq_len; n++) {
            for (int h = 0; h < Hq; h++) {
                int src_off = (b * seq_len + n) * (Hq * D) + h * D;
                int dst_off = (b * Hq + h) * seq_len * D + n * D;
                memcpy(&Qd[dst_off], &qf[src_off], D * sizeof(float));
            }
        }
    }

    // Reshape K, V to [B*Hkv, N, D], then repeat to [B*Hq, N, D]
    int kv_shape[] = {batch * Hkv, seq_len, D};
    Tensor *K_small = tensor_zeros(kv_shape, 3, DTYPE_FP32, 0);
    Tensor *V_small = tensor_zeros(kv_shape, 3, DTYPE_FP32, 0);
    float *kf = k_flat->data;
    float *vf = v_flat->data;
    float *Ks = K_small->data;
    float *Vs = V_small->data;
    for (int b = 0; b < batch; b++) {
        for (int n = 0; n < seq_len; n++) {
            for (int h = 0; h < Hkv; h++) {
                int src_off = (b * seq_len + n) * (Hkv * D) + h * D;
                int dst_off = (b * Hkv + h) * seq_len * D + n * D;
                memcpy(&Ks[dst_off], &kf[src_off], D * sizeof(float));
                memcpy(&Vs[dst_off], &vf[src_off], D * sizeof(float));
            }
        }
    }

    // KV repeat: [B*Hkv, N, D] -> [B*Hq, N, D]
    Tensor *K = tensor_zeros(q_shape, 3, DTYPE_FP32, 1);
    Tensor *V = tensor_zeros(q_shape, 3, DTYPE_FP32, 1);
    float *Kd = K->data;
    float *Vd = V->data;
    for (int b = 0; b < batch; b++) {
        for (int hkv = 0; hkv < Hkv; hkv++) {
            int kv_off = (b * Hkv + hkv) * seq_len * D;
            for (int r = 0; r < group_ratio; r++) {
                int q_off = (b * Hq + hkv * group_ratio + r) * seq_len * D;
                memcpy(&Kd[q_off], &Ks[kv_off], seq_len * D * sizeof(float));
                memcpy(&Vd[q_off], &Vs[kv_off], seq_len * D * sizeof(float));
            }
        }
    }
    tensor_free(K_small);
    tensor_free(V_small);

    // Apply per-head QK RMSNorm: treat [B*Hq, N, D] as [B*Hq*N, D]
    {
        int qn_shape[] = {batch * Hq * seq_len, D};
        Tensor *Q_flat_norm = tensor_create(qn_shape, 2, DTYPE_FP32, 0);
        memcpy(Q_flat_norm->data, Q->data, batch * Hq * seq_len * D * sizeof(float));
        Tensor *Q_normed = rms_norm_forward(gqa->q_norm, Q_flat_norm, g);
        metal_synchronize();
        memcpy(Q->data, Q_normed->data, batch * Hq * seq_len * D * sizeof(float));
        if (!g) { tensor_free(Q_flat_norm); tensor_free(Q_normed); }
    }
    {
        int kn_shape[] = {batch * Hq * seq_len, D};
        Tensor *K_flat_norm = tensor_create(kn_shape, 2, DTYPE_FP32, 0);
        memcpy(K_flat_norm->data, K->data, batch * Hq * seq_len * D * sizeof(float));
        Tensor *K_normed = rms_norm_forward(gqa->k_norm, K_flat_norm, g);
        metal_synchronize();
        memcpy(K->data, K_normed->data, batch * Hq * seq_len * D * sizeof(float));
        if (!g) { tensor_free(K_flat_norm); tensor_free(K_normed); }
    }

    // Apply RoPE to Q and K
    Tensor *Q_rope = tensor_zeros(q_shape, 3, DTYPE_FP32, 1);
    Tensor *K_rope = tensor_zeros(q_shape, 3, DTYPE_FP32, 1);

    cpu_rope_theta(Q->data, Q_rope->data, batch * Hq, seq_len, D, gqa->rope_theta);
    cpu_rope_theta(K->data, K_rope->data, batch * Hq, seq_len, D, gqa->rope_theta);

    // Flash Attention
    int attn_shape[] = {batch * Hq, seq_len, D};
    Tensor *attn_out = tensor_zeros(attn_shape, 3, DTYPE_FP32, 1);

    if (metal_is_initialized()) {
        metal_flash_attention(Q_rope->metal_buf, K_rope->metal_buf, V->metal_buf,
                              attn_out->metal_buf,
                              1, batch * Hq, seq_len, D, gqa->scale, 1);
        metal_synchronize();
    } else {
        int BH = batch * Hq;
        for (int bh = 0; bh < BH; bh++) {
            float *qp = &Q_rope->data[bh * seq_len * D];
            float *kp = &K_rope->data[bh * seq_len * D];
            float *vp = &V->data[bh * seq_len * D];
            float *op = &attn_out->data[bh * seq_len * D];

            for (int i = 0; i < seq_len; i++) {
                float max_score = -1e30f;
                float *scores = calloc(seq_len, sizeof(float));
                for (int j = 0; j <= i; j++) {
                    float score = 0;
                    for (int dd = 0; dd < D; dd++)
                        score += qp[i * D + dd] * kp[j * D + dd];
                    score *= gqa->scale;
                    scores[j] = score;
                    if (score > max_score) max_score = score;
                }
                float sum = 0;
                for (int j = 0; j <= i; j++) {
                    scores[j] = expf(scores[j] - max_score);
                    sum += scores[j];
                }
                for (int j = 0; j <= i; j++) scores[j] /= sum;
                for (int dd = 0; dd < D; dd++) {
                    float val = 0;
                    for (int j = 0; j <= i; j++)
                        val += scores[j] * vp[j * D + dd];
                    op[i * D + dd] = val;
                }
                free(scores);
            }
        }
    }

    // Transpose back: [B, Hq, N, D] -> [B, N, Hq, D] = [BN, Hq*D]
    int out_shape[] = {BN, Hq * D};
    Tensor *attn_flat = tensor_zeros(out_shape, 2, DTYPE_FP32, 1);
    float *af = attn_out->data;
    float *of = attn_flat->data;
    for (int b2 = 0; b2 < batch; b2++) {
        for (int n = 0; n < seq_len; n++) {
            for (int h = 0; h < Hq; h++) {
                int src_off = (b2 * Hq + h) * seq_len * D + n * D;
                int dst_off = (b2 * seq_len + n) * (Hq * D) + h * D;
                memcpy(&of[dst_off], &af[src_off], D * sizeof(float));
            }
        }
    }

    // Output projection
    Tensor *output = linear_forward(gqa->o_proj, attn_flat, g);

    // Register in graph for backward
    if (g) {
        GQASavedData *sd = calloc(1, sizeof(GQASavedData));
        sd->gqa = gqa;
        sd->q = Q_rope;
        sd->k = K_rope;
        sd->v = V;
        sd->attn_out = attn_out;
        sd->batch = batch;
        sd->seq_len = seq_len;

        Tensor *inputs[] = {Q_rope, K_rope, V};
        GraphNode *node = graph_add_node(g, attn_flat, inputs, 3, gqa_attention_backward);
        node->saved_data = sd;
        node->cleanup_fn = gqa_saved_cleanup;
    } else {
        tensor_free(Q_rope);
        tensor_free(K_rope);
        tensor_free(V);
        tensor_free(attn_out);
    }

    tensor_free(Q);
    tensor_free(K);
    if (!g) {
        tensor_free(attn_flat);
        tensor_free(q_flat);
        tensor_free(k_flat);
        tensor_free(v_flat);
    }

    return output;
}

// Public API: no LoRA
Tensor *gqa_forward(GroupedQueryAttention *gqa, Tensor *x,
                    int batch, int seq_len, ComputeGraph *g) {
    return gqa_forward_impl(gqa, x, batch, seq_len, g, NULL, NULL);
}

// Public API: with optional LoRA on Q and V
Tensor *gqa_forward_lora(GroupedQueryAttention *gqa, Tensor *x,
                         int batch, int seq_len, ComputeGraph *g,
                         LoRALinear *q_lora, LoRALinear *v_lora) {
    return gqa_forward_impl(gqa, x, batch, seq_len, g, q_lora, v_lora);
}

void gqa_collect_params(GroupedQueryAttention *gqa, ParamList *pl) {
    param_list_add(pl, gqa->q_proj->weight);
    param_list_add(pl, gqa->k_proj->weight);
    param_list_add(pl, gqa->v_proj->weight);
    param_list_add(pl, gqa->o_proj->weight);
    param_list_add(pl, gqa->q_norm->gamma);
    param_list_add(pl, gqa->k_norm->gamma);
}

// ==================================================================
// KV Cache
// ==================================================================

KVCache *kv_cache_create(int n_layers, int n_kv_heads, int head_dim, int max_seq_len) {
    KVCache *c = calloc(1, sizeof(KVCache));
    c->n_layers = n_layers;
    c->n_kv_heads = n_kv_heads;
    c->head_dim = head_dim;
    c->max_seq_len = max_seq_len;
    c->cur_len = 0;
    c->k_cache = calloc(n_layers, sizeof(float *));
    c->v_cache = calloc(n_layers, sizeof(float *));
    size_t layer_size = (size_t)n_kv_heads * max_seq_len * head_dim;
    for (int i = 0; i < n_layers; i++) {
        c->k_cache[i] = calloc(layer_size, sizeof(float));
        c->v_cache[i] = calloc(layer_size, sizeof(float));
    }
    return c;
}

void kv_cache_free(KVCache *cache) {
    if (!cache) return;
    for (int i = 0; i < cache->n_layers; i++) {
        free(cache->k_cache[i]);
        free(cache->v_cache[i]);
    }
    free(cache->k_cache);
    free(cache->v_cache);
    free(cache);
}

void kv_cache_reset(KVCache *cache) {
    cache->cur_len = 0;
}

// ==================================================================
// Incremental GQA Forward (with KV Cache)
// ==================================================================
// x: [batch * n_new, d_model]  — only the NEW token(s)
// Returns: [batch * n_new, d_model]
// Appends new K/V to cache, attends over full cached sequence
Tensor *gqa_forward_cached(GroupedQueryAttention *gqa, Tensor *x,
                           int batch, int n_new, int layer_idx,
                           KVCache *cache) {
    int Hq = gqa->n_q_heads;
    int Hkv = gqa->n_kv_heads;
    int D = gqa->head_dim;
    int group_ratio = Hq / Hkv;
    int BN_new = batch * n_new;

    // Project Q, K, V for new tokens only
    Tensor *q_flat = linear_forward(gqa->q_proj, x, NULL);  // [BN_new, Hq*D]
    Tensor *k_flat = linear_forward(gqa->k_proj, x, NULL);  // [BN_new, Hkv*D]
    Tensor *v_flat = linear_forward(gqa->v_proj, x, NULL);  // [BN_new, Hkv*D]
    metal_synchronize();

    int prev_len = cache->cur_len;
    int total_len = prev_len + n_new;

    // --- Write new K/V into cache (per KV head layout) ---
    // Cache layout: [Hkv][max_seq_len][D]
    float *kc = cache->k_cache[layer_idx];
    float *vc = cache->v_cache[layer_idx];
    for (int b = 0; b < batch; b++) {
        for (int t = 0; t < n_new; t++) {
            int src_row = b * n_new + t;
            int dst_pos = prev_len + t;
            for (int h = 0; h < Hkv; h++) {
                int src_off = src_row * (Hkv * D) + h * D;
                int dst_off = h * cache->max_seq_len * D + dst_pos * D;
                memcpy(&kc[dst_off], &k_flat->data[src_off], D * sizeof(float));
                memcpy(&vc[dst_off], &v_flat->data[src_off], D * sizeof(float));
            }
        }
    }

    // --- Apply QK norm to new Q tokens ---
    // Q: reshape [BN_new, Hq*D] -> [BN_new*Hq, D], apply RMSNorm
    {
        int qn_shape[] = {BN_new * Hq, D};
        Tensor *Q_for_norm = tensor_create(qn_shape, 2, DTYPE_FP32, 0);
        // Reorder from [BN_new, Hq*D] to [BN_new*Hq, D]
        for (int bn = 0; bn < BN_new; bn++) {
            for (int h = 0; h < Hq; h++) {
                int src = bn * Hq * D + h * D;
                int dst = (bn * Hq + h) * D;
                memcpy(&Q_for_norm->data[dst], &q_flat->data[src], D * sizeof(float));
            }
        }
        Tensor *Q_normed = rms_norm_forward(gqa->q_norm, Q_for_norm, NULL);
        metal_synchronize();
        // Write back
        for (int bn = 0; bn < BN_new; bn++) {
            for (int h = 0; h < Hq; h++) {
                int src = (bn * Hq + h) * D;
                int dst = bn * Hq * D + h * D;
                memcpy(&q_flat->data[dst], &Q_normed->data[src], D * sizeof(float));
            }
        }
        tensor_free(Q_for_norm);
        tensor_free(Q_normed);
    }

    // --- Apply QK norm to new K tokens in cache ---
    {
        int kn_shape[] = {batch * n_new * Hkv, D};
        Tensor *K_for_norm = tensor_create(kn_shape, 2, DTYPE_FP32, 0);
        for (int b = 0; b < batch; b++) {
            for (int t = 0; t < n_new; t++) {
                int pos = prev_len + t;
                for (int h = 0; h < Hkv; h++) {
                    int cache_off = h * cache->max_seq_len * D + pos * D;
                    int norm_off = ((b * n_new + t) * Hkv + h) * D;
                    memcpy(&K_for_norm->data[norm_off], &kc[cache_off], D * sizeof(float));
                }
            }
        }
        Tensor *K_normed = rms_norm_forward(gqa->k_norm, K_for_norm, NULL);
        metal_synchronize();
        // Write normalized K back to cache
        for (int b = 0; b < batch; b++) {
            for (int t = 0; t < n_new; t++) {
                int pos = prev_len + t;
                for (int h = 0; h < Hkv; h++) {
                    int cache_off = h * cache->max_seq_len * D + pos * D;
                    int norm_off = ((b * n_new + t) * Hkv + h) * D;
                    memcpy(&kc[cache_off], &K_normed->data[norm_off], D * sizeof(float));
                }
            }
        }
        tensor_free(K_for_norm);
        tensor_free(K_normed);
    }

    // --- Apply RoPE to new Q and new K (in cache) ---
    // RoPE for Q: [BN_new, Hq*D] with positions [prev_len .. prev_len+n_new-1]
    for (int bn = 0; bn < BN_new; bn++) {
        int pos = prev_len + bn;  // position of this new token
        for (int h = 0; h < Hq; h++) {
            int base = bn * Hq * D + h * D;
            int half_D = D / 2;
            for (int d = 0; d < half_D; d++) {
                float freq = 1.0f / powf(gqa->rope_theta, (float)(2 * d) / (float)D);
                float angle = (float)pos * freq;
                float cos_a = cosf(angle);
                float sin_a = sinf(angle);
                float x0 = q_flat->data[base + d];
                float x1 = q_flat->data[base + d + half_D];
                q_flat->data[base + d]          = x0 * cos_a - x1 * sin_a;
                q_flat->data[base + d + half_D] = x0 * sin_a + x1 * cos_a;
            }
        }
    }

    // RoPE for new K positions in cache
    for (int b = 0; b < batch; b++) {
        for (int t = 0; t < n_new; t++) {
            int pos = prev_len + t;
            for (int h = 0; h < Hkv; h++) {
                int base = h * cache->max_seq_len * D + pos * D;
                int half_D = D / 2;
                for (int d = 0; d < half_D; d++) {
                    float freq = 1.0f / powf(gqa->rope_theta, (float)(2 * d) / (float)D);
                    float angle = (float)pos * freq;
                    float cos_a = cosf(angle);
                    float sin_a = sinf(angle);
                    float x0 = kc[base + d];
                    float x1 = kc[base + d + half_D];
                    kc[base + d]          = x0 * cos_a - x1 * sin_a;
                    kc[base + d + half_D] = x0 * sin_a + x1 * cos_a;
                }
            }
        }
    }

    cache->cur_len = total_len;

    // --- Attention: Q_new attends to full cached K/V ---
    // For each new query position, compute attention over all total_len positions
    // Q: [BN_new, Hq*D], K_cache: [Hkv, total_len, D], V_cache: [Hkv, total_len, D]
    int out_flat_shape[] = {BN_new, Hq * D};
    Tensor *attn_flat = tensor_create(out_flat_shape, 2, DTYPE_FP32, 0);

    for (int bn = 0; bn < BN_new; bn++) {
        int q_pos = prev_len + bn;  // absolute position of this query
        for (int hq = 0; hq < Hq; hq++) {
            int hkv = hq / group_ratio;
            float *qp = &q_flat->data[bn * Hq * D + hq * D];
            float *kp = &kc[hkv * cache->max_seq_len * D];
            float *vp = &vc[hkv * cache->max_seq_len * D];

            // Compute attention scores for positions [0, q_pos] (causal)
            int n_attend = q_pos + 1;
            float max_score = -1e30f;
            float *scores = (float *)malloc(n_attend * sizeof(float));

            for (int j = 0; j < n_attend; j++) {
                float score = 0;
                for (int d = 0; d < D; d++) {
                    score += qp[d] * kp[j * D + d];
                }
                score *= gqa->scale;
                scores[j] = score;
                if (score > max_score) max_score = score;
            }

            // Softmax
            float sum = 0;
            for (int j = 0; j < n_attend; j++) {
                scores[j] = expf(scores[j] - max_score);
                sum += scores[j];
            }
            float inv_sum = (sum > 0) ? 1.0f / sum : 0.0f;

            // Weighted sum of values
            float *out = &attn_flat->data[bn * Hq * D + hq * D];
            for (int d = 0; d < D; d++) {
                float val = 0;
                for (int j = 0; j < n_attend; j++) {
                    val += scores[j] * inv_sum * vp[j * D + d];
                }
                out[d] = val;
            }
            free(scores);
        }
    }

    // Output projection
    Tensor *output = linear_forward(gqa->o_proj, attn_flat, NULL);
    metal_synchronize();

    tensor_free(q_flat);
    tensor_free(k_flat);
    tensor_free(v_flat);
    tensor_free(attn_flat);

    return output;
}
