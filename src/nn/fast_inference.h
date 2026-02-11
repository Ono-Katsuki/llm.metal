#ifndef FAST_INFERENCE_H
#define FAST_INFERENCE_H

#include "qwen3.h"

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
