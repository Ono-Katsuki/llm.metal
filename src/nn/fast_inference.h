#ifndef FAST_INFERENCE_H
#define FAST_INFERENCE_H

#include "qwen3.h"
#include "gemma3.h"
#include "wmat.h"

// ==================================================================
// Persistent inference state (for serve mode)
// ==================================================================

typedef struct InferenceState InferenceState;

// Create inference state from Qwen3 model.
InferenceState *inference_state_create(Qwen3Model *model, int max_seq_len, int use_fp16);

// Create inference state from Gemma3 model.
InferenceState *inference_state_create_gemma3(Gemma3Model *model, int max_seq_len, int use_fp16);

// Free inference state and call fast_metal_shutdown().
void inference_state_free(InferenceState *state);

// Run a single generation request. Resets KV cache internally.
// Returns number of generated tokens written to output_tokens.
int inference_generate(InferenceState *state,
                       const uint32_t *prompt_tokens, int prompt_len,
                       uint32_t *output_tokens, int max_gen_len);

// Get model type of inference state.
ModelType inference_state_model_type(InferenceState *state);

// ==================================================================
// One-shot convenience wrappers
// ==================================================================

// Fast text generation with Metal GPU acceleration.
// use_fp16: 0 = Q8_0 quantized weights, 1 = F16 half-precision weights
// Returns number of generated tokens
int qwen3_generate_fast(Qwen3Model *model, const uint32_t *prompt_tokens,
                        int prompt_len, uint32_t *output_tokens,
                        int max_gen_len, int max_seq_len, int use_fp16);

// Batched generation: B sequences from same prompt, F16 only.
// output_tokens: [B * max_gen_len], filled row-major.
// Returns number of generated tokens per sequence.
int qwen3_generate_fast_batch(Qwen3Model *model, const uint32_t *prompt_tokens,
                               int prompt_len, uint32_t *output_tokens,
                               int max_gen_len, int max_seq_len, int batch_size);

#endif // FAST_INFERENCE_H
