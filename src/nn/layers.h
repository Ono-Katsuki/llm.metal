#ifndef LAYERS_H
#define LAYERS_H

#include "../core/tensor.h"
#include "../core/autograd.h"

// ==================================================================
// Linear layer: y = xW^T + b
// ==================================================================
typedef struct {
    Tensor *weight;   // [out_features, in_features]
    Tensor *bias;     // [out_features] or NULL
    int in_features;
    int out_features;
} Linear;

Linear *linear_create(int in_features, int out_features, int use_bias);
void    linear_free(Linear *l);
Tensor *linear_forward(Linear *l, Tensor *input, ComputeGraph *g);

// ==================================================================
// Layer Normalization
// ==================================================================
typedef struct {
    Tensor *gamma;    // [dim]
    Tensor *beta;     // [dim]
    int dim;
    float eps;
} LayerNorm;

LayerNorm *layer_norm_create(int dim, float eps);
void       layer_norm_free(LayerNorm *ln);
Tensor    *layer_norm_forward(LayerNorm *ln, Tensor *input, ComputeGraph *g);

// ==================================================================
// Embedding layer
// ==================================================================
typedef struct {
    Tensor *weight;   // [vocab_size, dim]
    int vocab_size;
    int dim;
} Embedding;

Embedding *embedding_create(int vocab_size, int dim);
void       embedding_free(Embedding *e);
Tensor    *embedding_forward(Embedding *e, Tensor *indices, ComputeGraph *g);

// ==================================================================
// RMS Normalization (no mean subtraction, no beta)
// ==================================================================
typedef struct {
    Tensor *gamma;    // [dim]
    int dim;
    float eps;
} RMSNorm;

RMSNorm *rms_norm_create(int dim, float eps);
void     rms_norm_free(RMSNorm *rn);
Tensor  *rms_norm_forward(RMSNorm *rn, Tensor *input, ComputeGraph *g);

// ==================================================================
// GELU activation
// ==================================================================
Tensor *gelu_forward(Tensor *input, ComputeGraph *g);

// ==================================================================
// SiLU activation: x * sigmoid(x)
// ==================================================================
Tensor *silu_forward(Tensor *input, ComputeGraph *g);

// ==================================================================
// Residual add
// ==================================================================
Tensor *residual_add(Tensor *x, Tensor *residual, ComputeGraph *g);

// ==================================================================
// Element-wise multiply
// ==================================================================
Tensor *elementwise_mul(Tensor *a, Tensor *b, ComputeGraph *g);

// Parameter collection helpers
typedef struct {
    Tensor **params;
    int      n_params;
    int      capacity;
} ParamList;

ParamList *param_list_create(int capacity);
void       param_list_add(ParamList *pl, Tensor *param);
void       param_list_free(ParamList *pl);

#endif // LAYERS_H
