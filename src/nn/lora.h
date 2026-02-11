#ifndef LORA_H
#define LORA_H

#include "../core/tensor.h"
#include "../core/autograd.h"
#include "layers.h"

// ==================================================================
// LoRA (Low-Rank Adaptation)
// y = x @ W_frozen^T + alpha/rank * (x @ A^T @ B^T)
// Only A and B are trainable; W_frozen is kept fixed.
// ==================================================================

typedef struct LoRALinear {
    Linear *base;           // frozen base linear layer (W_frozen)
    Tensor *lora_A;         // [rank, in_features] — trainable
    Tensor *lora_B;         // [out_features, rank] — trainable
    int in_features;
    int out_features;
    int rank;
    float alpha;            // scaling factor
    float scaling;          // alpha / rank
    int enabled;            // 1 = apply LoRA, 0 = base only
} LoRALinear;

// Create a LoRA adapter wrapping an existing Linear layer.
// The base layer's weight is frozen (requires_grad set to 0).
LoRALinear *lora_create(Linear *base, int rank, float alpha);
void        lora_free(LoRALinear *l);

// Forward: y = base_forward(x) + scaling * (x @ A^T @ B^T)
Tensor *lora_forward(LoRALinear *l, Tensor *input, ComputeGraph *g);

// Merge LoRA weights into base: W = W + scaling * B @ A
void lora_merge(LoRALinear *l);

// Collect only trainable params (A and B)
void lora_collect_params(LoRALinear *l, ParamList *pl);

// ==================================================================
// QLoRA: 4-bit quantized base weights + LoRA adapters
// ==================================================================

typedef struct {
    uint8_t *packed_weights; // [n_bytes] — 4-bit packed
    float   *scales;         // [n_groups]
    float   *zeros;          // [n_groups]
    void    *metal_packed;   // Metal buffer for packed weights
    void    *metal_scales;   // Metal buffer for scales
    void    *metal_zeros;    // Metal buffer for zeros
    int      n_elements;     // total weight elements
    int      group_size;     // quantization group size (typically 64 or 128)
    int      n_groups;
    int      in_features;
    int      out_features;

    Tensor  *lora_A;         // [rank, in_features] — trainable
    Tensor  *lora_B;         // [out_features, rank] — trainable
    int      rank;
    float    alpha;
    float    scaling;
    int      enabled;
} QLoRALinear;

// Create QLoRA from a pre-quantized weight buffer
QLoRALinear *qlora_create(int in_features, int out_features,
                          const uint8_t *packed, const float *scales,
                          const float *zeros, int group_size,
                          int rank, float alpha);
void         qlora_free(QLoRALinear *l);

// Forward: dequantize base weights on-the-fly + LoRA delta
Tensor *qlora_forward(QLoRALinear *l, Tensor *input, ComputeGraph *g);

// Collect trainable params (A and B only)
void qlora_collect_params(QLoRALinear *l, ParamList *pl);

#endif // LORA_H
