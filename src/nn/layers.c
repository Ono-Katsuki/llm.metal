#include "layers.h"
#include "../core/metal_backend.h"
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <assert.h>

// ==================================================================
// Linear
// ==================================================================

// Backward for linear: y = x @ W^T + b
static void linear_backward(GraphNode *node) {
    Tensor *output = node->output;    // [M, K]  (M=batch*seq, K=out_features)
    Tensor *input  = node->inputs[0]; // [M, N]  (N=in_features)
    Tensor *weight = node->inputs[1]; // [K, N]
    Tensor *bias   = node->n_inputs > 2 ? node->inputs[2] : NULL;

    int M = output->shape[0]; // batch*seq
    int N = weight->shape[1]; // in_features
    int K = weight->shape[0]; // out_features

    if (metal_is_initialized()) {
        // dx[M,N] = dout[M,K] @ W[K,N]
        if (input->grad) {
            metal_matmul(output->grad->metal_buf, weight->metal_buf,
                         input->grad->metal_buf, M, K, N, 0, 0);
        }

        // dW[K,N] = dout^T[K,M] @ x[M,N]
        // dout is stored as [M,K], use transpose_a=1
        if (weight->grad) {
            metal_matmul(output->grad->metal_buf, input->metal_buf,
                         weight->grad->metal_buf, K, M, N, 1, 0);
        }
        metal_synchronize();
    }

    // CPU fallback (also handles bias gradient)
    if (!metal_is_initialized()) {
        float *dout = output->grad->data;
        float *x = input->data;
        float *w = weight->data;

        // dx[M,N] = dout[M,K] @ W[K,N]
        if (input->grad) {
            float *dx = input->grad->data;
            for (int m = 0; m < M; m++) {
                for (int n = 0; n < N; n++) {
                    float sum = 0.0f;
                    for (int k = 0; k < K; k++) {
                        sum += dout[m * K + k] * w[k * N + n];
                    }
                    dx[m * N + n] += sum;
                }
            }
        }

        // dW[K,N] = dout^T[K,M] @ x[M,N]
        if (weight->grad) {
            float *dw = weight->grad->data;
            for (int k = 0; k < K; k++) {
                for (int n = 0; n < N; n++) {
                    float sum = 0.0f;
                    for (int m = 0; m < M; m++) {
                        sum += dout[m * K + k] * x[m * N + n];
                    }
                    dw[k * N + n] += sum;
                }
            }
        }
    }

    // db = sum(dout, dim=0)
    if (bias && bias->grad) {
        float *dout = output->grad->data;
        float *db = bias->grad->data;
        for (int i = 0; i < M; i++) {
            for (int j = 0; j < K; j++) {
                db[j] += dout[i * K + j];
            }
        }
    }
}

Linear *linear_create(int in_features, int out_features, int use_bias) {
    Linear *l = calloc(1, sizeof(Linear));
    l->in_features = in_features;
    l->out_features = out_features;

    // Kaiming initialization: scale = sqrt(2 / in_features)
    float scale = sqrtf(2.0f / (float)in_features);
    int w_shape[] = {out_features, in_features};
    l->weight = tensor_randn(w_shape, 2, DTYPE_FP32, 1, scale);

    if (use_bias) {
        int b_shape[] = {out_features};
        l->bias = tensor_zeros(b_shape, 1, DTYPE_FP32, 1);
    }
    return l;
}

void linear_free(Linear *l) {
    if (!l) return;
    tensor_free(l->weight);
    tensor_free(l->bias);
    free(l);
}

Tensor *linear_forward(Linear *l, Tensor *input, ComputeGraph *g) {
    // input: [batch*seq, in_features]
    // output: [batch*seq, out_features]
    int M = input->shape[0];
    int K = l->in_features;
    int N = l->out_features;

    int out_shape[] = {M, N};
    Tensor *output = tensor_zeros(out_shape, 2, DTYPE_FP32, 1);

    if (metal_is_initialized()) {
        // y = x @ W^T  =>  matmul(x, W, transpose_a=0, transpose_b=1)
        metal_matmul(input->metal_buf, l->weight->metal_buf,
                     output->metal_buf, M, K, N, 0, 1);
        metal_synchronize();

        // Add bias
        if (l->bias) {
            float *out = output->data;
            float *b = l->bias->data;
            for (int i = 0; i < M; i++) {
                for (int j = 0; j < N; j++) {
                    out[i * N + j] += b[j];
                }
            }
        }
    } else {
        // CPU fallback: y = x @ W^T + b
        float *x = input->data;
        float *w = l->weight->data;
        float *y = output->data;
        for (int i = 0; i < M; i++) {
            for (int j = 0; j < N; j++) {
                float sum = 0.0f;
                for (int k = 0; k < K; k++) {
                    sum += x[i * K + k] * w[j * K + k]; // W is [out, in], so W^T
                }
                y[i * N + j] = sum + (l->bias ? l->bias->data[j] : 0.0f);
            }
        }
    }

    // Register in graph
    if (g) {
        int n_inputs = l->bias ? 3 : 2;
        Tensor *inputs[] = {input, l->weight, l->bias};
        graph_add_node(g, output, inputs, n_inputs, linear_backward);
    }
    return output;
}

// ==================================================================
// Layer Norm
// ==================================================================

static void layer_norm_backward_fn(GraphNode *node) {
    Tensor *output = node->output;     // [batch, dim]
    Tensor *input  = node->inputs[0];  // [batch, dim]
    Tensor *gamma  = node->inputs[1];  // [dim]
    Tensor *beta   = node->inputs[2];  // [dim]

    int batch = input->shape[0];
    int dim   = input->shape[1];
    float eps = 1e-5f;

    if (metal_is_initialized()) {
        // Zero dgamma, dbeta before accumulation
        tensor_fill(gamma->grad, 0.0f);
        tensor_fill(beta->grad, 0.0f);
        metal_layer_norm_backward(input->metal_buf, output->grad->metal_buf,
                                  gamma->metal_buf, input->grad->metal_buf,
                                  gamma->grad->metal_buf, beta->grad->metal_buf,
                                  batch, dim, eps);
        metal_synchronize();
    } else {
        // CPU backward
        float *x = input->data;
        float *dy = output->grad->data;
        float *g = gamma->data;
        float *dx = input->grad ? input->grad->data : NULL;
        float *dg = gamma->grad->data;
        float *db = beta->grad->data;

        for (int b = 0; b < batch; b++) {
            int off = b * dim;
            // Compute mean, var
            float mean = 0, var = 0;
            for (int i = 0; i < dim; i++) mean += x[off + i];
            mean /= dim;
            for (int i = 0; i < dim; i++) {
                float d = x[off + i] - mean;
                var += d * d;
            }
            var /= dim;
            float inv_std = 1.0f / sqrtf(var + eps);

            float ds = 0, dm = 0;
            for (int i = 0; i < dim; i++) {
                float xhat = (x[off + i] - mean) * inv_std;
                dg[i] += dy[off + i] * xhat;
                db[i] += dy[off + i];
                ds += dy[off + i] * g[i] * xhat;
                dm += dy[off + i] * g[i];
            }
            if (dx) {
                for (int i = 0; i < dim; i++) {
                    float xhat = (x[off + i] - mean) * inv_std;
                    dx[off + i] += inv_std / (float)dim *
                        ((float)dim * dy[off + i] * g[i] - dm - xhat * ds);
                }
            }
        }
    }
}

LayerNorm *layer_norm_create(int dim, float eps) {
    LayerNorm *ln = calloc(1, sizeof(LayerNorm));
    ln->dim = dim;
    ln->eps = eps;
    int shape[] = {dim};
    ln->gamma = tensor_create(shape, 1, DTYPE_FP32, 1);
    ln->beta  = tensor_zeros(shape, 1, DTYPE_FP32, 1);
    tensor_fill(ln->gamma, 1.0f);
    return ln;
}

void layer_norm_free(LayerNorm *ln) {
    if (!ln) return;
    tensor_free(ln->gamma);
    tensor_free(ln->beta);
    free(ln);
}

Tensor *layer_norm_forward(LayerNorm *ln, Tensor *input, ComputeGraph *g) {
    // input: [batch, dim] or [batch*seq, dim]
    int batch = input->size / ln->dim;
    int out_shape[] = {(int)batch, ln->dim};
    Tensor *output = tensor_zeros(out_shape, 2, DTYPE_FP32, 1);

    if (metal_is_initialized()) {
        metal_layer_norm(input->metal_buf, output->metal_buf,
                         ln->gamma->metal_buf, ln->beta->metal_buf,
                         batch, ln->dim, ln->eps);
        metal_synchronize();
    } else {
        float *x = input->data;
        float *y = output->data;
        float *gm = ln->gamma->data;
        float *bt = ln->beta->data;
        int dim = ln->dim;

        for (int b = 0; b < (int)batch; b++) {
            int off = b * dim;
            float mean = 0, var = 0;
            for (int i = 0; i < dim; i++) mean += x[off + i];
            mean /= dim;
            for (int i = 0; i < dim; i++) {
                float d = x[off + i] - mean;
                var += d * d;
            }
            var /= dim;
            float inv_std = 1.0f / sqrtf(var + ln->eps);
            for (int i = 0; i < dim; i++) {
                y[off + i] = gm[i] * (x[off + i] - mean) * inv_std + bt[i];
            }
        }
    }

    if (g) {
        Tensor *inputs[] = {input, ln->gamma, ln->beta};
        graph_add_node(g, output, inputs, 3, layer_norm_backward_fn);
    }
    return output;
}

// ==================================================================
// RMSNorm
// ==================================================================

static void rms_norm_backward_fn(GraphNode *node) {
    Tensor *output = node->output;     // [batch, dim]
    Tensor *input  = node->inputs[0];  // [batch, dim]
    Tensor *gamma  = node->inputs[1];  // [dim]

    int batch = input->shape[0];
    int dim   = input->shape[1];
    float eps = 1e-6f;

    if (metal_is_initialized()) {
        tensor_fill(gamma->grad, 0.0f);
        metal_rms_norm_backward(input->metal_buf, output->grad->metal_buf,
                                gamma->metal_buf, input->grad->metal_buf,
                                gamma->grad->metal_buf,
                                batch, dim, eps);
        metal_synchronize();
    } else {
        float *x = input->data;
        float *dy = output->grad->data;
        float *g = gamma->data;
        float *dx = input->grad ? input->grad->data : NULL;
        float *dg = gamma->grad->data;

        for (int b = 0; b < batch; b++) {
            int off = b * dim;
            // Compute sum of squares
            float ss = 0;
            for (int i = 0; i < dim; i++) ss += x[off + i] * x[off + i];
            float inv_rms = 1.0f / sqrtf(ss / dim + eps);
            float inv_rms3 = inv_rms * inv_rms * inv_rms;

            // sum(dy * gamma * x)
            float dsum = 0;
            for (int i = 0; i < dim; i++) {
                dsum += dy[off + i] * g[i] * x[off + i];
                dg[i] += dy[off + i] * x[off + i] * inv_rms;
            }
            if (dx) {
                for (int i = 0; i < dim; i++) {
                    dx[off + i] += g[i] * dy[off + i] * inv_rms
                                 - x[off + i] * inv_rms3 * dsum / (float)dim;
                }
            }
        }
    }
}

RMSNorm *rms_norm_create(int dim, float eps) {
    RMSNorm *rn = calloc(1, sizeof(RMSNorm));
    rn->dim = dim;
    rn->eps = eps;
    int shape[] = {dim};
    rn->gamma = tensor_create(shape, 1, DTYPE_FP32, 1);
    tensor_fill(rn->gamma, 1.0f);
    return rn;
}

void rms_norm_free(RMSNorm *rn) {
    if (!rn) return;
    tensor_free(rn->gamma);
    free(rn);
}

Tensor *rms_norm_forward(RMSNorm *rn, Tensor *input, ComputeGraph *g) {
    int batch = input->size / rn->dim;
    int out_shape[] = {(int)batch, rn->dim};
    Tensor *output = tensor_zeros(out_shape, 2, DTYPE_FP32, 1);

    if (metal_is_initialized()) {
        metal_rms_norm(input->metal_buf, output->metal_buf,
                       rn->gamma->metal_buf,
                       batch, rn->dim, rn->eps);
        metal_synchronize();
    } else {
        float *x = input->data;
        float *y = output->data;
        float *gm = rn->gamma->data;
        int dim = rn->dim;

        for (int b = 0; b < (int)batch; b++) {
            int off = b * dim;
            float ss = 0;
            for (int i = 0; i < dim; i++) ss += x[off + i] * x[off + i];
            float inv_rms = 1.0f / sqrtf(ss / dim + rn->eps);
            for (int i = 0; i < dim; i++) {
                y[off + i] = gm[i] * x[off + i] * inv_rms;
            }
        }
    }

    if (g) {
        Tensor *inputs[] = {input, rn->gamma};
        graph_add_node(g, output, inputs, 2, rms_norm_backward_fn);
    }
    return output;
}

// ==================================================================
// Embedding
// ==================================================================

static void embedding_backward_fn(GraphNode *node) {
    Tensor *output  = node->output;     // [seq, dim]
    Tensor *indices = node->inputs[0];  // [seq] (DTYPE_UINT32)
    Tensor *weight  = node->inputs[1];  // [vocab, dim]

    int seq_len = output->shape[0];
    int dim     = output->shape[1];

    if (metal_is_initialized() && indices->metal_buf) {
        metal_embedding_backward(output->grad->metal_buf, indices->metal_buf,
                                 weight->grad->metal_buf,
                                 weight->shape[0], dim, seq_len);
        metal_synchronize();
    } else {
        // CPU
        float *dy = output->grad->data;
        float *dw = weight->grad->data;
        uint32_t *idx = indices->data_u32;
        if (!idx) {
            // Fallback: indices stored as float
            float *fidx = indices->data;
            for (int s = 0; s < seq_len; s++) {
                int token = (int)fidx[s];
                for (int d = 0; d < dim; d++) {
                    dw[token * dim + d] += dy[s * dim + d];
                }
            }
        } else {
            for (int s = 0; s < seq_len; s++) {
                int token = (int)idx[s];
                for (int d = 0; d < dim; d++) {
                    dw[token * dim + d] += dy[s * dim + d];
                }
            }
        }
    }
}

Embedding *embedding_create(int vocab_size, int dim) {
    Embedding *e = calloc(1, sizeof(Embedding));
    e->vocab_size = vocab_size;
    e->dim = dim;
    float scale = 0.02f;
    int shape[] = {vocab_size, dim};
    e->weight = tensor_randn(shape, 2, DTYPE_FP32, 1, scale);
    return e;
}

void embedding_free(Embedding *e) {
    if (!e) return;
    tensor_free(e->weight);
    free(e);
}

Tensor *embedding_forward(Embedding *e, Tensor *indices, ComputeGraph *g) {
    // indices: [seq_len] (DTYPE_UINT32 or DTYPE_FP32)
    int seq_len = indices->shape[0];
    int dim = e->dim;
    int out_shape[] = {seq_len, dim};
    Tensor *output = tensor_zeros(out_shape, 2, DTYPE_FP32, 1);

    if (metal_is_initialized() && indices->metal_buf) {
        metal_embedding(e->weight->metal_buf, indices->metal_buf,
                        output->metal_buf, e->vocab_size, dim, seq_len);
        metal_synchronize();
    } else {
        // CPU lookup
        float *w = e->weight->data;
        float *o = output->data;
        for (int s = 0; s < seq_len; s++) {
            int token;
            if (indices->dtype == DTYPE_UINT32) {
                token = (int)indices->data_u32[s];
            } else {
                token = (int)indices->data[s];
            }
            memcpy(&o[s * dim], &w[token * dim], dim * sizeof(float));
        }
    }

    if (g) {
        Tensor *inputs[] = {indices, e->weight};
        graph_add_node(g, output, inputs, 2, embedding_backward_fn);
    }
    return output;
}

// ==================================================================
// GELU
// ==================================================================

static void gelu_backward_fn(GraphNode *node) {
    Tensor *output = node->output;
    Tensor *input  = node->inputs[0];
    int n = (int)input->size;

    if (metal_is_initialized()) {
        metal_gelu_backward(input->metal_buf, output->grad->metal_buf,
                            input->grad->metal_buf, n);
        metal_synchronize();
    } else {
        float *x  = input->data;
        float *dy = output->grad->data;
        float *dx = input->grad->data;
        float c = 0.7978845608f;
        for (int i = 0; i < n; i++) {
            float xi = x[i];
            float inner = c * (xi + 0.044715f * xi * xi * xi);
            float tanh_inner = tanhf(inner);
            float sech2 = 1.0f - tanh_inner * tanh_inner;
            float dgelu = 0.5f * (1.0f + tanh_inner) +
                          0.5f * xi * sech2 * c * (1.0f + 3.0f * 0.044715f * xi * xi);
            dx[i] += dy[i] * dgelu;
        }
    }
}

Tensor *gelu_forward(Tensor *input, ComputeGraph *g) {
    int n = (int)input->size;
    Tensor *output = tensor_zeros(input->shape, input->ndim, DTYPE_FP32, 1);

    if (metal_is_initialized()) {
        metal_gelu(input->metal_buf, output->metal_buf, n);
        metal_synchronize();
    } else {
        float *x = input->data;
        float *y = output->data;
        float c = 0.7978845608f;
        for (int i = 0; i < n; i++) {
            float xi = x[i];
            float inner = c * (xi + 0.044715f * xi * xi * xi);
            y[i] = 0.5f * xi * (1.0f + tanhf(inner));
        }
    }

    if (g) {
        Tensor *inputs[] = {input};
        graph_add_node(g, output, inputs, 1, gelu_backward_fn);
    }
    return output;
}

// ==================================================================
// SiLU: x * sigmoid(x)
// ==================================================================

static void silu_backward_fn(GraphNode *node) {
    Tensor *output = node->output;
    Tensor *input  = node->inputs[0];
    int n = (int)input->size;

    if (metal_is_initialized()) {
        metal_silu_backward(input->metal_buf, output->grad->metal_buf,
                            input->grad->metal_buf, n);
        metal_synchronize();
    } else {
        float *x  = input->data;
        float *dy = output->grad->data;
        float *dx = input->grad->data;
        for (int i = 0; i < n; i++) {
            float xi = x[i];
            float sig = 1.0f / (1.0f + expf(-xi));
            float dsilu = sig * (1.0f + xi * (1.0f - sig));
            dx[i] += dy[i] * dsilu;
        }
    }
}

Tensor *silu_forward(Tensor *input, ComputeGraph *g) {
    int n = (int)input->size;
    Tensor *output = tensor_zeros(input->shape, input->ndim, DTYPE_FP32, 1);

    if (metal_is_initialized()) {
        metal_silu(input->metal_buf, output->metal_buf, n);
        metal_synchronize();
    } else {
        float *x = input->data;
        float *y = output->data;
        for (int i = 0; i < n; i++) {
            float xi = x[i];
            float sig = 1.0f / (1.0f + expf(-xi));
            y[i] = xi * sig;
        }
    }

    if (g) {
        Tensor *inputs[] = {input};
        graph_add_node(g, output, inputs, 1, silu_backward_fn);
    }
    return output;
}

// ==================================================================
// Element-wise multiply: output = a * b
// ==================================================================

static void elementwise_mul_backward(GraphNode *node) {
    Tensor *output = node->output;
    Tensor *a      = node->inputs[0];
    Tensor *b      = node->inputs[1];
    int n = (int)a->size;

    // da += dy * b, db += dy * a
    if (metal_is_initialized()) {
        if (a->grad) {
            metal_mul(output->grad->metal_buf, b->metal_buf, a->grad->metal_buf, n);
            metal_synchronize();
        }
        if (b->grad) {
            metal_mul(output->grad->metal_buf, a->metal_buf, b->grad->metal_buf, n);
            metal_synchronize();
        }
    } else {
        float *dy = output->grad->data;
        if (a->grad) {
            for (int i = 0; i < n; i++)
                a->grad->data[i] += dy[i] * b->data[i];
        }
        if (b->grad) {
            for (int i = 0; i < n; i++)
                b->grad->data[i] += dy[i] * a->data[i];
        }
    }
}

Tensor *elementwise_mul(Tensor *a, Tensor *b, ComputeGraph *g) {
    int n = (int)a->size;
    Tensor *output = tensor_zeros(a->shape, a->ndim, DTYPE_FP32, 1);

    if (metal_is_initialized()) {
        metal_mul(a->metal_buf, b->metal_buf, output->metal_buf, n);
        metal_synchronize();
    } else {
        for (int i = 0; i < n; i++) {
            output->data[i] = a->data[i] * b->data[i];
        }
    }

    if (g) {
        Tensor *inputs[] = {a, b};
        graph_add_node(g, output, inputs, 2, elementwise_mul_backward);
    }
    return output;
}

// ==================================================================
// Residual add: output = x + residual
// ==================================================================

static void residual_add_backward(GraphNode *node) {
    Tensor *output   = node->output;
    Tensor *x        = node->inputs[0];
    Tensor *residual = node->inputs[1];

    // Both inputs receive the full gradient
    if (x->grad) tensor_add_inplace(x->grad, output->grad);
    if (residual->grad) tensor_add_inplace(residual->grad, output->grad);
}

Tensor *residual_add(Tensor *x, Tensor *residual, ComputeGraph *g) {
    int n = (int)x->size;
    Tensor *output = tensor_zeros(x->shape, x->ndim, DTYPE_FP32, 1);

    if (metal_is_initialized()) {
        metal_add(x->metal_buf, residual->metal_buf, output->metal_buf, n);
        metal_synchronize();
    } else {
        for (int i = 0; i < n; i++) {
            output->data[i] = x->data[i] + residual->data[i];
        }
    }

    if (g) {
        Tensor *inputs[] = {x, residual};
        graph_add_node(g, output, inputs, 2, residual_add_backward);
    }
    return output;
}

// ==================================================================
// ParamList
// ==================================================================

ParamList *param_list_create(int capacity) {
    ParamList *pl = calloc(1, sizeof(ParamList));
    pl->params = calloc(capacity, sizeof(Tensor *));
    pl->capacity = capacity;
    return pl;
}

void param_list_add(ParamList *pl, Tensor *param) {
    assert(pl->n_params < pl->capacity);
    pl->params[pl->n_params++] = param;
}

void param_list_free(ParamList *pl) {
    if (!pl) return;
    free(pl->params);
    free(pl);
}
