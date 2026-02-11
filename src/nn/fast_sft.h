#ifndef FAST_SFT_H
#define FAST_SFT_H

#include "qwen3.h"

// Fast SFT training with Metal GPU acceleration.
// Forward + backward pass in single GPU submission (no CPU-GPU sync overhead).
typedef struct SFTState SFTState;

// Create SFT training state with pre-allocated GPU buffers.
// All model weights are converted to F16 and uploaded to GPU.
// LoRA A/B are stored as float on GPU.
SFTState *sft_state_create(Qwen3Model *model, int seq_len,
                           int lora_rank, float lora_alpha);
void sft_state_free(SFTState *state);

// Run one SFT training step (forward + loss + backward + weight update).
// input_tokens: [seq_len] uint32 input token IDs
// target_tokens: [seq_len] uint32 target token IDs
// lr: current learning rate
// Returns average cross-entropy loss.
float sft_train_step(SFTState *state, const uint32_t *input_tokens,
                     const uint32_t *target_tokens, float lr);

// Sync LoRA weights from GPU back to model (for saving adapter).
void sft_sync_lora_to_model(SFTState *state, Qwen3Model *model);

#endif // FAST_SFT_H
