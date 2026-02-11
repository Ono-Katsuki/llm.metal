#include "fast_inference.h"
#include "fast_metal.h"
#include "wmat.h"
#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <math.h>
#include <time.h>
#include <sys/stat.h>

// Parse eos_token_id from config.json — supports int or array of ints.
// Returns count of EOS tokens found (0 if not found).
static int parse_eos_from_config(const char *model_path, int *eos_out, int max_eos) {
    char config_path[600];
    struct stat st;
    if (stat(model_path, &st) == 0 && S_ISDIR(st.st_mode))
        snprintf(config_path, sizeof(config_path), "%s/config.json", model_path);
    else {
        const char *sl = strrchr(model_path, '/');
        if (sl) snprintf(config_path, sizeof(config_path), "%.*s/config.json", (int)(sl - model_path), model_path);
        else    snprintf(config_path, sizeof(config_path), "config.json");
    }
    FILE *fp = fopen(config_path, "r");
    if (!fp) return 0;
    char buf[4096];
    size_t n = fread(buf, 1, sizeof(buf) - 1, fp);
    buf[n] = '\0';
    fclose(fp);

    const char *key = strstr(buf, "\"eos_token_id\"");
    if (!key) return 0;
    const char *p = key + strlen("\"eos_token_id\"");
    while (*p == ' ' || *p == ':' || *p == '\t') p++;

    int count = 0;
    if (*p == '[') {
        // Array: [1, 106]
        p++;
        while (*p && *p != ']' && count < max_eos) {
            while (*p == ' ' || *p == ',' || *p == '\n') p++;
            if (*p == ']') break;
            eos_out[count++] = (int)strtol(p, (char **)&p, 10);
        }
    } else {
        // Single int
        eos_out[count++] = (int)strtol(p, NULL, 10);
    }
    return count;
}

struct InferenceState {
    int d_model, n_q_heads, n_kv_heads, head_dim, intermediate_size, vocab_size;
    int n_layers, max_seq_len;
    int use_fp16;
    ModelType model_type;

    // Per-layer RoPE theta (Gemma3 hybrid: local vs global)
    float *rope_thetas;  // [n_layers]

    // EOS tokens (read from config.json)
    int eos_tokens[4];
    int n_eos;

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
};

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

// Build LayerWeightRef array from Qwen3Model
static LayerWeightRef *build_qwen3_refs(Qwen3Model *m) {
    int D = m->d_model;
    int Hq_hd = m->n_q_heads * m->head_dim;
    int Hkv_hd = m->n_kv_heads * m->head_dim;
    int IS = m->intermediate_size;

    LayerWeightRef *refs = calloc(m->n_layers, sizeof(LayerWeightRef));
    for (int i = 0; i < m->n_layers; i++) {
        Qwen3Block *blk = m->blocks[i];
        LayerWeightRef *r = &refs[i];
        r->q_proj = (WeightRef){ blk->attn->q_proj->weight->data, Hq_hd, D };
        r->k_proj = (WeightRef){ blk->attn->k_proj->weight->data, Hkv_hd, D };
        r->v_proj = (WeightRef){ blk->attn->v_proj->weight->data, Hkv_hd, D };
        r->o_proj = (WeightRef){ blk->attn->o_proj->weight->data, D, Hq_hd };
        r->gate_proj = (WeightRef){ blk->gate_proj->weight->data, IS, D };
        r->up_proj = (WeightRef){ blk->up_proj->weight->data, IS, D };
        r->down_proj = (WeightRef){ blk->down_proj->weight->data, D, IS };
        r->input_norm_g = blk->input_norm->gamma->data;
        r->post_attn_norm_g = blk->post_attn_norm->gamma->data;
        r->q_norm_g = blk->attn->q_norm->gamma->data;
        r->k_norm_g = blk->attn->k_norm->gamma->data;
        r->d_model = D;
        r->head_dim = m->head_dim;
    }
    return refs;
}

// Build LayerWeightRef array from Gemma3Model
static LayerWeightRef *build_gemma3_refs(Gemma3Model *m) {
    int D = m->d_model;
    int Hq_hd = m->n_q_heads * m->head_dim;
    int Hkv_hd = m->n_kv_heads * m->head_dim;
    int IS = m->intermediate_size;

    LayerWeightRef *refs = calloc(m->n_layers, sizeof(LayerWeightRef));
    for (int i = 0; i < m->n_layers; i++) {
        Gemma3Block *blk = m->blocks[i];
        LayerWeightRef *r = &refs[i];
        r->q_proj = (WeightRef){ blk->attn->q_proj->weight->data, Hq_hd, D };
        r->k_proj = (WeightRef){ blk->attn->k_proj->weight->data, Hkv_hd, D };
        r->v_proj = (WeightRef){ blk->attn->v_proj->weight->data, Hkv_hd, D };
        r->o_proj = (WeightRef){ blk->attn->o_proj->weight->data, D, Hq_hd };
        r->gate_proj = (WeightRef){ blk->gate_proj->weight->data, IS, D };
        r->up_proj = (WeightRef){ blk->up_proj->weight->data, IS, D };
        r->down_proj = (WeightRef){ blk->down_proj->weight->data, D, IS };
        r->input_norm_g = blk->input_norm->gamma->data;
        r->post_attn_norm_g = blk->post_attn_norm->gamma->data;
        r->pre_ff_norm_g = blk->pre_ff_norm->gamma->data;
        r->post_ff_norm_g = blk->post_ff_norm->gamma->data;
        r->q_norm_g = blk->attn->q_norm->gamma->data;
        r->k_norm_g = blk->attn->k_norm->gamma->data;
        r->d_model = D;
        r->head_dim = m->head_dim;
    }
    return refs;
}

// Common init logic shared by both model constructors
static InferenceState *inference_state_init_common(
    int d_model, int n_q_heads, int n_kv_heads, int head_dim,
    int intermediate_size, int vocab_size, int n_layers,
    int max_seq_len, int use_fp16, ModelType model_type,
    const LayerWeightRef *refs, const float *lm_head_data,
    const float *final_norm_gamma, const float *token_emb_data,
    const float *rope_thetas_src)
{
    if (fast_metal_init() != 0) {
        fprintf(stderr, "[FastGen] Metal init failed\n");
        return NULL;
    }
    if (max_seq_len > 4096) {
        fprintf(stderr, "[FastGen] Warning: clamping max_seq_len to 4096 (kernel limit)\n");
        max_seq_len = 4096;
    }

    InferenceState *s = calloc(1, sizeof(InferenceState));
    s->d_model = d_model;
    s->n_q_heads = n_q_heads;
    s->n_kv_heads = n_kv_heads;
    s->head_dim = head_dim;
    s->intermediate_size = intermediate_size;
    s->vocab_size = vocab_size;
    s->n_layers = n_layers;
    s->max_seq_len = max_seq_len;
    s->use_fp16 = use_fp16;
    s->model_type = model_type;

    // Default EOS (overridden by inference_state_load_eos)
    if (model_type == MODEL_GEMMA3) {
        s->eos_tokens[0] = 1;    s->eos_tokens[1] = 106;  s->n_eos = 2;
    } else {
        s->eos_tokens[0] = 151645; s->eos_tokens[1] = 151643; s->n_eos = 2;
    }

    // Per-layer rope thetas
    s->rope_thetas = malloc(n_layers * sizeof(float));
    memcpy(s->rope_thetas, rope_thetas_src, n_layers * sizeof(float));

    int D = d_model;
    int Hq_hd = n_q_heads * head_dim;
    int Hkv_hd = n_kv_heads * head_dim;
    int IS = intermediate_size;

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
    s->mb_logits  = make_buf(vocab_size * sizeof(float), &s->logits);

    // GPU KV cache (per layer)
    size_t cache_bytes = (size_t)n_kv_heads * max_seq_len * head_dim * sizeof(float);
    s->mb_k_cache = calloc(n_layers, sizeof(MetalBuf *));
    s->mb_v_cache = calloc(n_layers, sizeof(MetalBuf *));
    for (int i = 0; i < n_layers; i++) {
        s->mb_k_cache[i] = metal_buf_create(cache_bytes);
        s->mb_v_cache[i] = metal_buf_create(cache_bytes);
    }
    s->cur_len = 0;

    // Convert and upload weights
    const char *mode_str = use_fp16 ? "F16" : "Q8_0";
    fprintf(stderr, "[FastGen] Converting weights to %s + uploading to GPU...\n", mode_str);
    struct timespec t0, t1;
    clock_gettime(CLOCK_MONOTONIC, &t0);

    s->layers = layers_convert_upload(refs, n_layers, use_fp16);
    s->lm_head = wmat_convert(lm_head_data, vocab_size, d_model, use_fp16);
    s->mb_final_norm_g = norm_upload(final_norm_gamma, d_model);
    s->token_emb = token_emb_data;

    clock_gettime(CLOCK_MONOTONIC, &t1);
    double ms = (t1.tv_sec - t0.tv_sec) * 1000.0 + (t1.tv_nsec - t0.tv_nsec) / 1e6;
    fprintf(stderr, "[FastGen] %s + GPU upload: %.0f ms\n", mode_str, ms);

    // Print weight memory usage
    if (use_fp16) {
        size_t total_params = 0;
        for (int i = 0; i < n_layers; i++) {
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
        double weight_mb = (double)(total_params * 2) / (1024.0 * 1024.0);
        fprintf(stderr, "[FastGen] F16 weight memory: %.1f MB (%.2f GB)\n",
               weight_mb, weight_mb / 1024.0);
    }

    return s;
}

InferenceState *inference_state_create(Qwen3Model *m, int max_seq_len, int use_fp16) {
    LayerWeightRef *refs = build_qwen3_refs(m);

    // Qwen3: all layers use the same rope_theta
    float *thetas = malloc(m->n_layers * sizeof(float));
    for (int i = 0; i < m->n_layers; i++)
        thetas[i] = m->rope_theta;

    InferenceState *s = inference_state_init_common(
        m->d_model, m->n_q_heads, m->n_kv_heads, m->head_dim,
        m->intermediate_size, m->vocab_size, m->n_layers,
        max_seq_len, use_fp16, MODEL_QWEN3,
        refs, m->lm_head->weight->data,
        m->final_norm->gamma->data, m->token_emb->weight->data,
        thetas);

    free(refs);
    free(thetas);
    return s;
}

InferenceState *inference_state_create_gemma3(Gemma3Model *m, int max_seq_len, int use_fp16) {
    LayerWeightRef *refs = build_gemma3_refs(m);

    // Gemma3: per-layer rope theta (local vs global)
    float *thetas = malloc(m->n_layers * sizeof(float));
    for (int i = 0; i < m->n_layers; i++)
        thetas[i] = m->blocks[i]->is_sliding ? m->local_rope_theta : m->global_rope_theta;

    // Gemma3 uses tied embeddings for lm_head
    InferenceState *s = inference_state_init_common(
        m->d_model, m->n_q_heads, m->n_kv_heads, m->head_dim,
        m->intermediate_size, m->vocab_size, m->n_layers,
        max_seq_len, use_fp16, MODEL_GEMMA3,
        refs, m->token_emb->weight->data,  // tied lm_head
        m->final_norm->gamma->data, m->token_emb->weight->data,
        thetas);

    free(refs);
    free(thetas);
    return s;
}

ModelType inference_state_model_type(InferenceState *state) {
    return state->model_type;
}

void inference_state_load_eos(InferenceState *state, const char *model_path) {
    state->n_eos = parse_eos_from_config(model_path, state->eos_tokens, 4);
    if (state->n_eos == 0) {
        // Fallback defaults
        if (state->model_type == MODEL_GEMMA3) {
            state->eos_tokens[0] = 1;    // <eos>
            state->eos_tokens[1] = 106;  // <end_of_turn>
            state->n_eos = 2;
        } else {
            state->eos_tokens[0] = 151645;  // <|endoftext|>
            state->eos_tokens[1] = 151643;  // <|im_end|>
            state->n_eos = 2;
        }
    }
    fprintf(stderr, "[FastGen] EOS tokens (%d):", state->n_eos);
    for (int i = 0; i < state->n_eos; i++)
        fprintf(stderr, " %d", state->eos_tokens[i]);
    fprintf(stderr, "\n");
}

void inference_state_free(InferenceState *s) {
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
    }
    free(s->mb_k_cache); free(s->mb_v_cache);
    free(s->rope_thetas);
    layers_free(s->layers, s->n_layers);
    wmat_free(s->lm_head);
    fast_metal_shutdown();
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

    // Embedding lookup (CPU -> shared GPU buffer)
    memcpy(s->x, &s->token_emb[token * D], D * sizeof(float));

    // Gemma3: scale embeddings by sqrt(d_model)
    if (s->model_type == MODEL_GEMMA3) {
        float scale = sqrtf((float)D);
        for (int i = 0; i < D; i++) s->x[i] *= scale;
    }

    // All layers — entirely on GPU, single encoder
    for (int layer = 0; layer < s->n_layers; layer++) {
        LayerW *lw = &s->layers[layer];
        float layer_theta = s->rope_thetas[layer];

        // --- Attention block ---
        metal_enqueue_rms_norm(s->mb_x, lw->input_norm_g, s->mb_ln_out, D, eps);
        dispatch_matvec(s, lw->q_proj, s->mb_ln_out, s->mb_q);
        dispatch_matvec(s, lw->k_proj, s->mb_ln_out, s->mb_k);
        dispatch_matvec(s, lw->v_proj, s->mb_ln_out, s->mb_v);
        if (lw->q_norm_g)
            metal_enqueue_per_head_rms_norm(s->mb_q, lw->q_norm_g, Hq, hd, eps);
        if (lw->k_norm_g)
            metal_enqueue_per_head_rms_norm(s->mb_k, lw->k_norm_g, Hkv, hd, eps);
        metal_enqueue_rope(s->mb_q, s->mb_k, Hq, Hkv, hd, pos, layer_theta);
        metal_enqueue_kv_cache_store(s->mb_k, s->mb_v,
                                      s->mb_k_cache[layer], s->mb_v_cache[layer],
                                      Hkv, s->max_seq_len, hd, pos);
        metal_enqueue_attention(s->mb_q, s->mb_k_cache[layer], s->mb_v_cache[layer],
                                s->mb_attn_out, n_attend, hd,
                                s->max_seq_len, Hq, group_ratio);
        dispatch_matvec(s, lw->o_proj, s->mb_attn_out, s->mb_x2);
        // Gemma3: post-attention norm before residual add (use ln_out as temp)
        if (lw->post_attn_norm_g && lw->pre_ff_norm_g) {
            metal_enqueue_rms_norm(s->mb_x2, lw->post_attn_norm_g, s->mb_ln_out, D, eps);
            metal_enqueue_residual_add(s->mb_x, s->mb_ln_out, D);
        } else {
            metal_enqueue_residual_add(s->mb_x, s->mb_x2, D);
        }

        // --- FFN block ---
        // Gemma3: use pre_ff_norm; Qwen3: use post_attn_norm (which is really pre-FFN)
        MetalBuf *ff_norm = lw->pre_ff_norm_g ? lw->pre_ff_norm_g : lw->post_attn_norm_g;
        metal_enqueue_rms_norm(s->mb_x, ff_norm, s->mb_ln_out, D, eps);
        dispatch_matvec(s, lw->gate_proj, s->mb_ln_out, s->mb_gate);
        dispatch_matvec(s, lw->up_proj, s->mb_ln_out, s->mb_up);
        if (s->model_type == MODEL_GEMMA3)
            metal_enqueue_gelu_mul(s->mb_gate, s->mb_up, s->intermediate_size);
        else
            metal_enqueue_silu_mul(s->mb_gate, s->mb_up, s->intermediate_size);
        dispatch_matvec(s, lw->down_proj, s->mb_gate, s->mb_ff_out);
        // Gemma3: post-FFN norm before residual add (use x2 as temp)
        if (lw->post_ff_norm_g) {
            metal_enqueue_rms_norm(s->mb_ff_out, lw->post_ff_norm_g, s->mb_x2, D, eps);
            metal_enqueue_residual_add(s->mb_x, s->mb_x2, D);
        } else {
            metal_enqueue_residual_add(s->mb_x, s->mb_ff_out, D);
        }
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
// inference_generate: run one request on a persistent state
// ==================================================================

int inference_generate(InferenceState *state,
                       const uint32_t *prompt_tokens, int prompt_len,
                       uint32_t *output_tokens, int max_gen_len) {
    InferenceState *s = state;
    s->cur_len = 0;  // reset KV cache position

    struct timespec t0, t1;

    // Prefill
    fprintf(stderr, "[FastGen] Prefilling %d tokens...\n", prompt_len);
    clock_gettime(CLOCK_MONOTONIC, &t0);

    for (int i = 0; i < prompt_len; i++) {
        forward_token(s, (int)prompt_tokens[i], i);
    }
    s->cur_len = prompt_len;

    clock_gettime(CLOCK_MONOTONIC, &t1);
    double prefill_ms = (t1.tv_sec - t0.tv_sec) * 1000.0 + (t1.tv_nsec - t0.tv_nsec) / 1e6;
    fprintf(stderr, "[FastGen] Prefill: %.0f ms (%.1f tok/s)\n",
           prefill_ms, prompt_len * 1000.0 / prefill_ms);

    // First generated token
    int next_token = argmax(s->logits, s->vocab_size);
    output_tokens[0] = (uint32_t)next_token;
    int n_gen = 1;

    // Decode
    fprintf(stderr, "[FastGen] Decoding...\n");
    clock_gettime(CLOCK_MONOTONIC, &t0);

    for (int gen = 1; gen < max_gen_len && s->cur_len < s->max_seq_len; gen++) {
        int is_eos = 0;
        for (int e = 0; e < s->n_eos; e++)
            if (next_token == s->eos_tokens[e]) { is_eos = 1; break; }
        if (is_eos) break;

        forward_token(s, next_token, s->cur_len);
        s->cur_len++;

        next_token = argmax(s->logits, s->vocab_size);
        output_tokens[n_gen++] = (uint32_t)next_token;

        if (n_gen % 10 == 0) {
            clock_gettime(CLOCK_MONOTONIC, &t1);
            double elapsed = (t1.tv_sec - t0.tv_sec) * 1000.0 + (t1.tv_nsec - t0.tv_nsec) / 1e6;
            fprintf(stderr, "[FastGen]   %d tokens, %.1f tok/s\n",
                   n_gen - 1, (n_gen - 1) * 1000.0 / elapsed);
        }
    }

    clock_gettime(CLOCK_MONOTONIC, &t1);
    double decode_ms = (t1.tv_sec - t0.tv_sec) * 1000.0 + (t1.tv_nsec - t0.tv_nsec) / 1e6;
    int n_decoded = n_gen - 1;
    if (n_decoded > 0) {
        fprintf(stderr, "[FastGen] Decode: %d tokens in %.0f ms (%.1f tok/s, %.0f ms/tok)\n",
               n_decoded, decode_ms, n_decoded * 1000.0 / decode_ms,
               decode_ms / n_decoded);
    }

    return n_gen;
}

// ==================================================================
// One-shot convenience wrapper
// ==================================================================

int qwen3_generate_fast(Qwen3Model *model, const uint32_t *prompt_tokens,
                        int prompt_len, uint32_t *output_tokens,
                        int max_gen_len, int max_seq_len, int use_fp16) {
    InferenceState *s = inference_state_create(model, max_seq_len, use_fp16);
    if (!s) return 0;

    int n_gen = inference_generate(s, prompt_tokens, prompt_len,
                                   output_tokens, max_gen_len);

    inference_state_free(s);
    return n_gen;
}

// ==================================================================
// Batched inference — B sequences processed simultaneously (F16 only)
// Reads weight matrix ONCE for all B items -> B x throughput
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

    // B x scratch buffers (contiguous for batched matvec)
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
    fprintf(stderr, "[BatchGen] Converting weights to F16 (batch=%d)...\n", B);
    struct timespec t0, t1;
    clock_gettime(CLOCK_MONOTONIC, &t0);

    LayerWeightRef *refs = build_qwen3_refs(m);
    s->layers = layers_convert_upload(refs, m->n_layers, 1);
    free(refs);
    s->lm_head = wmat_convert(m->lm_head->weight->data, m->vocab_size, m->d_model, 1);
    s->mb_final_norm_g = norm_upload(m->final_norm->gamma->data, m->d_model);
    s->token_emb = m->token_emb->weight->data;

    clock_gettime(CLOCK_MONOTONIC, &t1);
    double ms = (t1.tv_sec - t0.tv_sec) * 1000.0 + (t1.tv_nsec - t0.tv_nsec) / 1e6;
    fprintf(stderr, "[BatchGen] F16 + GPU upload: %.0f ms\n", ms);
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
    }
    free(s->mb_k_cache); free(s->mb_v_cache);
    layers_free(s->layers, s->n_layers);
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

            if (lw->q_norm_g)
                metal_enqueue_per_head_rms_norm_off(s->mb_q, q_off, lw->q_norm_g, Hq, hd, eps);
            if (lw->k_norm_g)
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
    fprintf(stderr, "[BatchGen] Prefilling %d tokens...\n", prompt_len);
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
    fprintf(stderr, "[BatchGen] Prefill: %.0f ms (%.1f tok/s)\n",
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
    fprintf(stderr, "[BatchGen] Decoding (batch=%d)...\n", B);
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
            fprintf(stderr, "[BatchGen]   %d steps, %d total tok, %.1f tok/s (%.1f ms/step)\n",
                   n_gen - 1, total_tok, total_tok * 1000.0 / elapsed,
                   elapsed / (n_gen - 1));
        }
    }

    clock_gettime(CLOCK_MONOTONIC, &t1);
    double decode_ms = (t1.tv_sec - t0.tv_sec) * 1000.0 + (t1.tv_nsec - t0.tv_nsec) / 1e6;
    int n_decoded = n_gen - 1;
    if (n_decoded > 0) {
        int total_tok = n_decoded * B;
        fprintf(stderr, "[BatchGen] Decode: %d steps x %d batch = %d tokens in %.0f ms\n",
               n_decoded, B, total_tok, decode_ms);
        fprintf(stderr, "[BatchGen]   Throughput: %.1f tok/s\n", total_tok * 1000.0 / decode_ms);
        fprintf(stderr, "[BatchGen]   Latency: %.0f ms/step\n", decode_ms / n_decoded);
    }

    batch_state_free(s);
    fast_metal_shutdown();
    return n_gen;
}
