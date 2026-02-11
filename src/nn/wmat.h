#ifndef WMAT_H
#define WMAT_H

#include "fast_metal.h"

// ==================================================================
// Model type enum (shared across the codebase)
// ==================================================================

typedef enum {
    MODEL_QWEN3,
    MODEL_GEMMA3,
} ModelType;

// ==================================================================
// Weight matrix: Q8_0 or F16 data in Metal buffer
// ==================================================================

typedef struct WMat {
    MetalBuf *mbuf;
    int rows, cols, nb;  // nb used only for Q8_0 (cols/32)
} WMat;

// ==================================================================
// Per-layer weight storage
// ==================================================================

typedef struct {
    WMat *q_proj, *k_proj, *v_proj, *o_proj;
    WMat *gate_proj, *up_proj, *down_proj;
    MetalBuf *input_norm_g, *post_attn_norm_g;
    MetalBuf *pre_ff_norm_g, *post_ff_norm_g;  // Gemma3 only (NULL for Qwen3)
    MetalBuf *q_norm_g, *k_norm_g;
} LayerW;

// ==================================================================
// Model-independent weight references (for bulk conversion)
// ==================================================================

typedef struct {
    const float *data;
    int rows, cols;
} WeightRef;

typedef struct {
    WeightRef q_proj, k_proj, v_proj, o_proj;
    WeightRef gate_proj, up_proj, down_proj;
    const float *input_norm_g;
    const float *post_attn_norm_g;
    const float *pre_ff_norm_g;   // Gemma3 only (NULL for Qwen3)
    const float *post_ff_norm_g;  // Gemma3 only (NULL for Qwen3)
    const float *q_norm_g;  // NULL if model has no QK norm
    const float *k_norm_g;  // NULL if model has no QK norm
    int d_model;
    int head_dim;
} LayerWeightRef;

// ==================================================================
// Weight creation / destruction
// ==================================================================

WMat *wmat_create_f16(const float *src, int rows, int cols);
WMat *wmat_create_q8(const float *src, int rows, int cols);
void  wmat_free(WMat *w);

// ==================================================================
// Bulk weight conversion + upload (model-independent)
// ==================================================================

// Convert all layer weights and upload to GPU.
// use_fp16: 1 = F16, 0 = Q8_0
// Returns calloc'd array of n_layers LayerW.
LayerW *layers_convert_upload(const LayerWeightRef *refs, int n_layers, int use_fp16);

// Free all WMat and MetalBuf in layer array.
void layers_free(LayerW *layers, int n_layers);

// Convert arbitrary weight matrix and upload.
WMat *wmat_convert(const float *data, int rows, int cols, int use_fp16);

// Upload RMSNorm gamma to GPU.
MetalBuf *norm_upload(const float *gamma, int dim);

#endif // WMAT_H
