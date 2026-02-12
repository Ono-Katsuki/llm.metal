#include "fast_grpo.h"
#include "fast_sft.h"
#include "fast_inference.h"
#include "fast_metal.h"
#include "wmat.h"
#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <math.h>
#include <time.h>

// ================================================================
// GRPOState
// ================================================================

struct GRPOState {
    GRPOConfig cfg;
    InferenceState *inf;     // for generation (autoregressive)
    SFTState *sft;           // for forward/backward (batched)

    // Generation results (CPU)
    uint32_t **gen_tokens;   // [G][max_gen_len]
    float **gen_logprobs;    // [G][max_gen_len]
    int *gen_lengths;        // [G]
    float *rewards;          // [G]
    float *advantages;       // [G]

    // GRPO kernel GPU buffers
    MetalBuf *mb_actions;    // [seq_len] uint32
    MetalBuf *mb_old_lp;     // [seq_len] float
    MetalBuf *mb_advs;       // [seq_len] float
    MetalBuf *mb_new_lp;     // [seq_len] float

    // LoRA gradient accumulation (CPU)
    int n_grad_bufs;
    int *grad_buf_sizes;     // element count per buffer
    float **accum_grads;     // [n_grad_bufs][size]

    int step_count;
    ModelType model_type;
};

// ================================================================
// Temperature sampling
// ================================================================

typedef struct {
    int token;
    float log_prob;
} SampleResult;

static SampleResult sample_token(const float *logits, int V, float temperature) {
    // Find max for numerical stability
    float max_val = logits[0];
    for (int i = 1; i < V; i++)
        if (logits[i] > max_val) max_val = logits[i];

    // Compute scaled logits and sum for multinomial sampling
    float inv_t = 1.0f / temperature;
    float scaled_max = max_val * inv_t;

    float sum = 0;
    for (int i = 0; i < V; i++)
        sum += expf(logits[i] * inv_t - scaled_max);

    // Multinomial sampling
    float r = (float)rand() / ((float)RAND_MAX + 1.0f) * sum;
    float cumsum = 0;
    int sampled = V - 1;
    for (int i = 0; i < V; i++) {
        cumsum += expf(logits[i] * inv_t - scaled_max);
        if (cumsum >= r) { sampled = i; break; }
    }

    // log p_T(i) = logit_i / T - logsumexp(logits / T)
    float log_prob = logits[sampled] * inv_t - (logf(sum) + scaled_max);

    return (SampleResult){ .token = sampled, .log_prob = log_prob };
}

// ================================================================
// Reward functions
// ================================================================

typedef float (*RewardFn)(const uint32_t *tokens, int len);

static float reward_repetition(const uint32_t *tokens, int len) {
    if (len < 2) return 1.0f;
    int n_bigrams = len - 1;
    int unique = 0;
    for (int i = 0; i < n_bigrams; i++) {
        int is_dup = 0;
        for (int j = 0; j < i; j++) {
            if (tokens[j] == tokens[i] && tokens[j+1] == tokens[i+1]) {
                is_dup = 1;
                break;
            }
        }
        if (!is_dup) unique++;
    }
    return (float)unique / (float)n_bigrams;
}

static float reward_length(const uint32_t *tokens, int len) {
    float target = 32.0f;
    float diff = fabsf((float)len - target);
    return fmaxf(0.0f, 1.0f - diff / target);
}

static float reward_combined(const uint32_t *tokens, int len) {
    return 0.5f * reward_repetition(tokens, len) + 0.5f * reward_length(tokens, len);
}

static RewardFn get_reward_fn(const char *name) {
    if (name && strcmp(name, "repetition") == 0) return reward_repetition;
    if (name && strcmp(name, "length") == 0) return reward_length;
    return reward_combined;
}

// ================================================================
// State creation
// ================================================================

static GRPOState *grpo_state_init(GRPOConfig cfg, InferenceState *inf,
                                   SFTState *sft, ModelType mtype) {
    int G = cfg.group_size;
    int max_gen = cfg.max_gen_len;

    GRPOState *g = calloc(1, sizeof(GRPOState));
    g->cfg = cfg;
    g->inf = inf;
    g->sft = sft;
    g->model_type = mtype;
    g->step_count = 0;

    // Allocate generation buffers
    g->gen_tokens = calloc(G, sizeof(uint32_t *));
    g->gen_logprobs = calloc(G, sizeof(float *));
    g->gen_lengths = calloc(G, sizeof(int));
    g->rewards = calloc(G, sizeof(float));
    g->advantages = calloc(G, sizeof(float));
    for (int i = 0; i < G; i++) {
        g->gen_tokens[i] = calloc(max_gen, sizeof(uint32_t));
        g->gen_logprobs[i] = calloc(max_gen, sizeof(float));
    }

    // GPU buffers for GRPO kernel
    int seq_len = sft_get_seq_len(sft);
    g->mb_actions = metal_buf_create(seq_len * sizeof(uint32_t));
    g->mb_old_lp = metal_buf_create(seq_len * sizeof(float));
    g->mb_advs = metal_buf_create(seq_len * sizeof(float));
    g->mb_new_lp = metal_buf_create(seq_len * sizeof(float));

    // Allocate F16 grad accum buffers for accurate full-param mode
    if (sft_is_full_param(sft) && cfg.accurate)
        sft_alloc_grad_accum(sft);

    // LoRA gradient accumulation buffers (skip for full-param)
    if (sft_get_lora_rank(sft) > 0) {
        g->n_grad_bufs = sft_get_n_lora_grad_bufs(sft);
        g->grad_buf_sizes = calloc(g->n_grad_bufs, sizeof(int));
        g->accum_grads = calloc(g->n_grad_bufs, sizeof(float *));
        for (int i = 0; i < g->n_grad_bufs; i++) {
            int sz = 0;
            sft_get_lora_grad_ptr(sft, i, &sz);
            g->grad_buf_sizes[i] = sz;
            g->accum_grads[i] = calloc(sz, sizeof(float));
        }
    } else {
        g->n_grad_bufs = 0;
        g->grad_buf_sizes = NULL;
        g->accum_grads = NULL;
    }

    return g;
}

GRPOState *grpo_state_create_gemma3(Gemma3Model *model, GRPOConfig cfg,
                                     int seq_len, int lora_rank, float lora_alpha,
                                     const char *model_path) {
    int max_seq = seq_len + cfg.max_gen_len + 16;
    if (max_seq > 4096) max_seq = 4096;

    // Create SFT first (uploads weights once)
    SFTState *sft = sft_state_create_gemma3(model, seq_len, lora_rank, lora_alpha);
    if (!sft) {
        fprintf(stderr, "[GRPO] Failed to create SFT state\n");
        return NULL;
    }

    // Create inference sharing SFT's weights + LoRA (no duplicate upload)
    InferenceState *inf = inference_state_create_from_sft(sft, max_seq);
    if (!inf) {
        fprintf(stderr, "[GRPO] Failed to create inference state\n");
        sft_state_free(sft);
        return NULL;
    }
    inference_state_load_eos(inf, model_path ? model_path : "");

    return grpo_state_init(cfg, inf, sft, MODEL_GEMMA3);
}

GRPOState *grpo_state_create(Qwen3Model *model, GRPOConfig cfg,
                              int seq_len, int lora_rank, float lora_alpha,
                              const char *model_path) {
    int max_seq = seq_len + cfg.max_gen_len + 16;
    if (max_seq > 4096) max_seq = 4096;

    // Create SFT first (uploads weights once)
    SFTState *sft = sft_state_create(model, seq_len, lora_rank, lora_alpha);
    if (!sft) {
        fprintf(stderr, "[GRPO] Failed to create SFT state\n");
        return NULL;
    }

    // Create inference sharing SFT's weights + LoRA (no duplicate upload)
    InferenceState *inf = inference_state_create_from_sft(sft, max_seq);
    if (!inf) {
        fprintf(stderr, "[GRPO] Failed to create inference state\n");
        sft_state_free(sft);
        return NULL;
    }
    inference_state_load_eos(inf, model_path ? model_path : "");

    return grpo_state_init(cfg, inf, sft, MODEL_QWEN3);
}

GRPOState *grpo_state_create_full(Qwen3Model *model, GRPOConfig cfg,
                                    int seq_len, const char *model_path) {
    int max_seq = seq_len + cfg.max_gen_len + 16;
    if (max_seq > 4096) max_seq = 4096;

    SFTState *sft = sft_state_create_full(model, seq_len);
    if (!sft) {
        fprintf(stderr, "[GRPO] Failed to create full-param SFT state\n");
        return NULL;
    }

    InferenceState *inf = inference_state_create_from_sft(sft, max_seq);
    if (!inf) {
        fprintf(stderr, "[GRPO] Failed to create inference state\n");
        sft_state_free(sft);
        return NULL;
    }
    inference_state_load_eos(inf, model_path ? model_path : "");

    return grpo_state_init(cfg, inf, sft, MODEL_QWEN3);
}

GRPOState *grpo_state_create_gemma3_full(Gemma3Model *model, GRPOConfig cfg,
                                           int seq_len, const char *model_path) {
    int max_seq = seq_len + cfg.max_gen_len + 16;
    if (max_seq > 4096) max_seq = 4096;

    SFTState *sft = sft_state_create_gemma3_full(model, seq_len);
    if (!sft) {
        fprintf(stderr, "[GRPO] Failed to create full-param SFT state\n");
        return NULL;
    }

    InferenceState *inf = inference_state_create_from_sft(sft, max_seq);
    if (!inf) {
        fprintf(stderr, "[GRPO] Failed to create inference state\n");
        sft_state_free(sft);
        return NULL;
    }
    inference_state_load_eos(inf, model_path ? model_path : "");

    return grpo_state_init(cfg, inf, sft, MODEL_GEMMA3);
}

void grpo_state_free(GRPOState *g) {
    if (!g) return;
    int G = g->cfg.group_size;
    for (int i = 0; i < G; i++) {
        free(g->gen_tokens[i]);
        free(g->gen_logprobs[i]);
    }
    free(g->gen_tokens);
    free(g->gen_logprobs);
    free(g->gen_lengths);
    free(g->rewards);
    free(g->advantages);

    metal_buf_free(g->mb_actions);
    metal_buf_free(g->mb_old_lp);
    metal_buf_free(g->mb_advs);
    metal_buf_free(g->mb_new_lp);

    for (int i = 0; i < g->n_grad_bufs; i++)
        free(g->accum_grads[i]);
    free(g->accum_grads);
    free(g->grad_buf_sizes);

    inference_state_free(g->inf);
    sft_state_free(g->sft);
    free(g);
}

// ================================================================
// GRPO train step
// ================================================================

float grpo_train_step(GRPOState *g, const uint32_t *prompt, int prompt_len, float lr) {
    int G = g->cfg.group_size;
    int max_gen = g->cfg.max_gen_len;
    float temperature = g->cfg.temperature;
    float clip_eps = g->cfg.clip_eps;
    int V = inference_state_vocab_size(g->inf);
    RewardFn reward_fn = get_reward_fn(g->cfg.reward);

    int seq_len = sft_get_seq_len(g->sft);
    int D = sft_get_d_model(g->sft);
    float emb_scale = sft_get_emb_scale(g->sft);
    const float *token_emb = sft_get_token_emb(g->sft);

    // ----------------------------------------------------------------
    // Phase 1: Generate G completions with temperature sampling
    // ----------------------------------------------------------------
    for (int i = 0; i < G; i++) {
        inference_state_reset(g->inf);

        // Prefill prompt tokens
        const float *logits = NULL;
        for (int t = 0; t < prompt_len; t++)
            logits = inference_forward_token(g->inf, (int)prompt[t]);

        // Sample completion tokens
        int gen_len = 0;
        for (int t = 0; t < max_gen; t++) {
            SampleResult sr = sample_token(logits, V, temperature);
            g->gen_tokens[i][t] = (uint32_t)sr.token;
            g->gen_logprobs[i][t] = sr.log_prob;
            gen_len++;

            // EOS check
            if (sr.token == 1 || sr.token == 106 ||
                sr.token == 151645 || sr.token == 151643)
                break;

            if (t < max_gen - 1)
                logits = inference_forward_token(g->inf, sr.token);
        }
        g->gen_lengths[i] = gen_len;
    }

    // ----------------------------------------------------------------
    // Phase 2: Compute rewards and group-relative advantages
    // ----------------------------------------------------------------
    float mean_reward = 0;
    for (int i = 0; i < G; i++) {
        g->rewards[i] = reward_fn(g->gen_tokens[i], g->gen_lengths[i]);
        mean_reward += g->rewards[i];
    }
    mean_reward /= (float)G;

    float std_reward = 0;
    for (int i = 0; i < G; i++) {
        float d = g->rewards[i] - mean_reward;
        std_reward += d * d;
    }
    std_reward = sqrtf(std_reward / (float)G);

    for (int i = 0; i < G; i++)
        g->advantages[i] = (g->rewards[i] - mean_reward) / (std_reward + 1e-8f);

    fprintf(stderr, "[GRPO] rewards: [");
    for (int i = 0; i < G; i++)
        fprintf(stderr, "%.3f%s", g->rewards[i], i < G-1 ? ", " : "");
    fprintf(stderr, "] mean=%.3f std=%.3f\n", mean_reward, std_reward);

    // ----------------------------------------------------------------
    // Phase 3: Accumulate GRPO gradients across G completions
    // ----------------------------------------------------------------

    int full_param = sft_is_full_param(g->sft);
    int accurate = full_param && g->cfg.accurate;
    float per_comp_lr = lr / (float)G;

    // Zero accumulation buffers (LoRA only)
    for (int b = 0; b < g->n_grad_bufs; b++)
        memset(g->accum_grads[b], 0, g->grad_buf_sizes[b] * sizeof(float));

    for (int i = 0; i < G; i++) {
        int gen_len = g->gen_lengths[i];
        int comp_len = gen_len;
        if (comp_len > seq_len) comp_len = seq_len;

        // Build input sequence: [prompt_tail | completion]
        int prompt_use = seq_len - comp_len;
        if (prompt_use > prompt_len) prompt_use = prompt_len;
        if (prompt_use < 0) prompt_use = 0;
        int total = prompt_use + comp_len;
        if (total > seq_len) {
            comp_len = seq_len - prompt_use;
            total = seq_len;
        }

        // Load embeddings into SFT x buffer
        float *x_cpu = sft_get_x_cpu(g->sft);
        int prompt_start = prompt_len - prompt_use;

        for (int t = 0; t < prompt_use; t++) {
            int tok = (int)prompt[prompt_start + t];
            memcpy(x_cpu + t * D, &token_emb[tok * D], D * sizeof(float));
        }
        for (int t = 0; t < comp_len; t++) {
            int tok = (int)g->gen_tokens[i][t];
            memcpy(x_cpu + (prompt_use + t) * D, &token_emb[tok * D], D * sizeof(float));
        }

        // Embedding scale (Gemma3: sqrt(d_model))
        if (emb_scale != 1.0f) {
            for (int j = 0; j < total * D; j++)
                x_cpu[j] *= emb_scale;
        }

        // Zero-pad if total < seq_len
        if (total < seq_len)
            memset(x_cpu + total * D, 0, (seq_len - total) * D * sizeof(float));

        // Forward to logits
        forward_to_logits(g->sft);

        // Prepare GRPO kernel inputs
        uint32_t *actions_cpu = (uint32_t *)metal_buf_ptr(g->mb_actions);
        float *old_lp_cpu = (float *)metal_buf_ptr(g->mb_old_lp);
        float *advs_cpu = (float *)metal_buf_ptr(g->mb_advs);

        memset(actions_cpu, 0, seq_len * sizeof(uint32_t));
        memset(old_lp_cpu, 0, seq_len * sizeof(float));
        memset(advs_cpu, 0, seq_len * sizeof(float));

        // Fill completion positions (prompt positions get adv=0 → no gradient)
        for (int t = 0; t < comp_len; t++) {
            int pos = prompt_use + t;
            actions_cpu[pos] = g->gen_tokens[i][t];
            old_lp_cpu[pos] = g->gen_logprobs[i][t];
            advs_cpu[pos] = g->advantages[i];
        }

        // GRPO policy gradient kernel
        MetalBuf *mb_logits = sft_get_logits(g->sft);
        MetalBuf *mb_dlogits = sft_get_dlogits(g->sft);

        metal_enqueue_grpo_policy_grad(mb_logits, g->mb_actions,
                                        g->mb_old_lp, g->mb_advs,
                                        g->mb_new_lp, mb_dlogits,
                                        seq_len, sft_get_vocab_size(g->sft),
                                        clip_eps);

        // Scale gradient by 1/G for averaging
        metal_enqueue_scale(mb_dlogits, 1.0f / (float)G,
                           seq_len * sft_get_vocab_size(g->sft));

        if (accurate) {
            // Accurate: accumulate F16 gradients, no weight update yet
            sft_backward_accum(g->sft);
            metal_flush();
        } else if (full_param) {
            // Online: backward with fused SGD, update weights immediately
            sft_backward_sgd(g->sft, per_comp_lr);
            metal_flush();
        } else {
            // LoRA: backward without SGD, accumulate gradients on CPU
            sft_backward(g->sft);
            metal_flush();

            for (int b = 0; b < g->n_grad_bufs; b++) {
                int sz = g->grad_buf_sizes[b];
                float *gpu_grad = sft_get_lora_grad_ptr(g->sft, b, NULL);
                float *accum = g->accum_grads[b];
                for (int j = 0; j < sz; j++)
                    accum[j] += gpu_grad[j];
            }
        }
    }

    // ----------------------------------------------------------------
    // Phase 4: Apply accumulated gradients
    // ----------------------------------------------------------------
    if (accurate) {
        // Accurate full-param: apply accumulated F16 gradients once
        sft_apply_grad_sgd(g->sft, lr);
    } else if (!full_param) {
        // LoRA: write accumulated grads back and update
        for (int b = 0; b < g->n_grad_bufs; b++)
            sft_set_lora_grad(g->sft, b, g->accum_grads[b], g->grad_buf_sizes[b]);

        g->step_count++;
        sft_set_step_count(g->sft, g->step_count);
        sft_lora_update(g->sft, lr);
    }

    return mean_reward;
}

void grpo_save_lora(GRPOState *g, const char *path) {
    sft_save_lora(g->sft, path);
}

void grpo_save_full_checkpoint(GRPOState *g, const char *path) {
    sft_save_full_checkpoint(g->sft, path);
}

void grpo_sync_lora_to_gemma3(GRPOState *g, Gemma3Model *model) {
    sft_sync_lora_to_gemma3(g->sft, model);
}

void grpo_sync_lora_to_model(GRPOState *g, Qwen3Model *model) {
    sft_sync_lora_to_model(g->sft, model);
}
