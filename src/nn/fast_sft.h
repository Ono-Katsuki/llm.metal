#ifndef FAST_SFT_H
#define FAST_SFT_H

#include "qwen3.h"
#include "gemma3.h"

// Fast SFT training with Metal GPU acceleration.
// Forward + backward pass in single GPU submission (no CPU-GPU sync overhead).
typedef struct SFTState SFTState;

// Create SFT training state with pre-allocated GPU buffers.
// All model weights are converted to F16 and uploaded to GPU.
// LoRA A/B are stored as float on GPU.
SFTState *sft_state_create(Qwen3Model *model, int seq_len,
                           int lora_rank, float lora_alpha);

// Create SFT training state for Gemma3 model.
SFTState *sft_state_create_gemma3(Gemma3Model *model, int seq_len,
                                   int lora_rank, float lora_alpha);

// Create full-param SFT state (SGD, no LoRA). All weights updated via fused SGD on GPU.
SFTState *sft_state_create_full(Qwen3Model *model, int seq_len);
SFTState *sft_state_create_gemma3_full(Gemma3Model *model, int seq_len);

// Save full checkpoint (all F16 weights) directly from GPU.
void sft_save_full_checkpoint(SFTState *state, const char *path);

// Backward pass with fused SGD weight update (for GRPO full-param).
void sft_backward_sgd(SFTState *state, float lr);

// Allocate F16 gradient accumulation buffers (for accurate full-param mode).
void sft_alloc_grad_accum(SFTState *state);

// Backward pass accumulating gradients into F16 buffers (no weight update).
void sft_backward_accum(SFTState *state);

// Apply accumulated gradients with SGD + zero clear.
void sft_apply_grad_sgd(SFTState *state, float lr);

// Query whether state is full-param mode.
int sft_is_full_param(SFTState *state);

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
void sft_sync_lora_to_gemma3(SFTState *state, Gemma3Model *model);

// === Internal API (used by GRPO) ===

// Forward through all layers + LM head, producing logits.
// Embeddings must already be loaded into the SFT state's mb_x buffer.
void forward_to_logits(SFTState *state);

// Backward pass (assuming dlogits has been populated).
void sft_backward(SFTState *state);

// Access SFT internal buffers for GRPO integration.
#include "fast_metal.h"
MetalBuf *sft_get_logits(SFTState *state);
MetalBuf *sft_get_dlogits(SFTState *state);
float *sft_get_x_cpu(SFTState *state);
float *sft_get_losses_cpu(SFTState *state);
int sft_get_seq_len(SFTState *state);
int sft_get_vocab_size(SFTState *state);
int sft_get_d_model(SFTState *state);
int sft_get_lora_rank(SFTState *state);
int sft_get_n_layers(SFTState *state);
float sft_get_emb_scale(SFTState *state);
const float *sft_get_token_emb(SFTState *state);
int sft_get_step_count(SFTState *state);
void sft_set_step_count(SFTState *state, int step);

// LoRA gradient access for GRPO accumulation.
// Returns number of LoRA gradient buffers (4 per layer: dA_q, dB_q, dA_v, dB_v).
int sft_get_n_lora_grad_bufs(SFTState *state);
// Get pointer and element count for LoRA grad buffer at index.
float *sft_get_lora_grad_ptr(SFTState *state, int idx, int *out_size);
// Write accumulated gradient back to LoRA grad buffer.
void sft_set_lora_grad(SFTState *state, int idx, const float *data, int size);

// Run AdamW update on LoRA weights (call after gradients are populated).
void sft_lora_update(SFTState *state, float lr);

// Save LoRA weights directly from GPU buffers (no model struct needed).
void sft_save_lora(SFTState *state, const char *path);

// Access LoRA weight buffers for inference sharing.
// Returns GPU MetalBuf pointers for layer L. sub: 0=A_q, 1=B_q, 2=A_v, 3=B_v.
MetalBuf *sft_get_lora_weight_buf(SFTState *state, int layer, int sub);
float sft_get_lora_scaling(SFTState *state);

// Weight sharing accessors (for inference_state_create_from_sft).
#include "wmat.h"
LayerW *sft_get_layers(SFTState *state);
WMat *sft_get_lm_head(SFTState *state);
MetalBuf *sft_get_final_norm_g(SFTState *state);
ModelType sft_get_model_type(SFTState *state);
int sft_get_n_q_heads(SFTState *state);
int sft_get_n_kv_heads(SFTState *state);
int sft_get_head_dim(SFTState *state);
int sft_get_intermediate_size(SFTState *state);
const float *sft_get_rope_thetas(SFTState *state);

#endif // FAST_SFT_H
