#ifndef FAST_GRPO_H
#define FAST_GRPO_H

#include "qwen3.h"
#include "gemma3.h"

typedef struct {
    int group_size;       // G: completions per prompt (default: 4)
    int max_gen_len;      // max tokens per completion (default: 64)
    float temperature;    // sampling temperature (default: 0.7)
    float clip_eps;       // PPO clip epsilon (default: 0.2)
    const char *reward;   // "length", "repetition", "combined" (default: "combined")
    int accurate;         // 1 = grad accumulation (paper-correct), 0 = online SGD
} GRPOConfig;

typedef struct GRPOState GRPOState;

// Create GRPO state from Gemma3 model (F16 inference + SFT backward).
// model_path: path to model directory (for EOS token detection).
GRPOState *grpo_state_create_gemma3(Gemma3Model *model, GRPOConfig cfg,
                                     int seq_len, int lora_rank, float lora_alpha,
                                     const char *model_path);

// Create GRPO state from Qwen3 model.
GRPOState *grpo_state_create(Qwen3Model *model, GRPOConfig cfg,
                              int seq_len, int lora_rank, float lora_alpha,
                              const char *model_path);

// Create full-param GRPO state (SGD, no LoRA).
GRPOState *grpo_state_create_full(Qwen3Model *model, GRPOConfig cfg,
                                    int seq_len, const char *model_path);
GRPOState *grpo_state_create_gemma3_full(Gemma3Model *model, GRPOConfig cfg,
                                           int seq_len, const char *model_path);

void grpo_state_free(GRPOState *state);

// Run one GRPO training step on a single prompt.
// Returns mean reward across the group.
float grpo_train_step(GRPOState *state, const uint32_t *prompt, int prompt_len, float lr);

// Save LoRA weights directly from GPU buffers (no model struct needed).
void grpo_save_lora(GRPOState *state, const char *path);

// Save full checkpoint (all F16 weights) directly from GPU.
void grpo_save_full_checkpoint(GRPOState *state, const char *path);

// Sync LoRA weights back to model for saving (legacy, requires model alive).
void grpo_sync_lora_to_gemma3(GRPOState *state, Gemma3Model *model);
void grpo_sync_lora_to_model(GRPOState *state, Qwen3Model *model);

#endif // FAST_GRPO_H
