#include "lora.h"
#include "../core/metal_backend.h"
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <assert.h>

// ==================================================================
// LoRA implementation
// ==================================================================

// Backward for LoRA delta: delta = scaling * x @ A^T @ B^T
// Inputs: [x, lora_A, lora_B]
// We need: dx += scaling * (dout @ B @ A)
//          dA += scaling * (B^T @ dout^T @ x)^T = scaling * (x^T @ dout @ B)^T
//          dB += scaling * dout^T @ (x @ A^T)
static void lora_backward(GraphNode *node) {
    Tensor *output = node->output;      // [M, out_features] (the LoRA delta)
    Tensor *input  = node->inputs[0];   // [M, in_features]
    Tensor *lora_A = node->inputs[1];   // [rank, in_features]
    Tensor *lora_B = node->inputs[2];   // [out_features, rank]

    float *scaling_ptr = (float *)node->saved_data;
    float scaling = *scaling_ptr;

    int M = input->shape[0];
    int in_f = lora_A->shape[1];
    int rank = lora_A->shape[0];
    int out_f = lora_B->shape[0];

    float *dout = output->grad->data;

    // Step 1: Compute intermediate h = x @ A^T -> [M, rank]
    float *h = calloc((size_t)M * rank, sizeof(float));
    for (int m = 0; m < M; m++) {
        for (int r = 0; r < rank; r++) {
            float sum = 0.0f;
            for (int k = 0; k < in_f; k++) {
                sum += input->data[m * in_f + k] * lora_A->data[r * in_f + k];
            }
            h[m * rank + r] = sum;
        }
    }

    // dB[out_f, rank] += scaling * dout^T[out_f, M] @ h[M, rank]
    if (lora_B->grad) {
        float *dB = lora_B->grad->data;
        if (metal_is_initialized()) {
            // Create temp tensors for Metal
            int dout_shape[] = {M, out_f};
            int h_shape[] = {M, rank};
            int dB_shape[] = {out_f, rank};
            Tensor *dout_t = tensor_create(dout_shape, 2, DTYPE_FP32, 0);
            Tensor *h_t = tensor_create(h_shape, 2, DTYPE_FP32, 0);
            Tensor *dB_t = tensor_zeros(dB_shape, 2, DTYPE_FP32, 0);
            memcpy(dout_t->data, dout, (size_t)M * out_f * sizeof(float));
            memcpy(h_t->data, h, (size_t)M * rank * sizeof(float));
            metal_matmul(dout_t->metal_buf, h_t->metal_buf, dB_t->metal_buf,
                         out_f, M, rank, 1, 0);
            metal_synchronize();
            for (int i = 0; i < out_f * rank; i++) {
                dB[i] += scaling * dB_t->data[i];
            }
            tensor_free(dout_t);
            tensor_free(h_t);
            tensor_free(dB_t);
        } else {
            for (int o = 0; o < out_f; o++) {
                for (int r = 0; r < rank; r++) {
                    float sum = 0.0f;
                    for (int m = 0; m < M; m++) {
                        sum += dout[m * out_f + o] * h[m * rank + r];
                    }
                    dB[o * rank + r] += scaling * sum;
                }
            }
        }
    }

    // Compute dh[M, rank] = dout[M, out_f] @ B[out_f, rank]
    float *dh = calloc((size_t)M * rank, sizeof(float));
    for (int m = 0; m < M; m++) {
        for (int r = 0; r < rank; r++) {
            float sum = 0.0f;
            for (int o = 0; o < out_f; o++) {
                sum += dout[m * out_f + o] * lora_B->data[o * rank + r];
            }
            dh[m * rank + r] = sum;
        }
    }

    // dA[rank, in_f] += scaling * dh^T[rank, M] @ x[M, in_f]
    if (lora_A->grad) {
        float *dA = lora_A->grad->data;
        if (metal_is_initialized()) {
            int dh_shape[] = {M, rank};
            int x_shape[] = {M, in_f};
            int dA_shape[] = {rank, in_f};
            Tensor *dh_t = tensor_create(dh_shape, 2, DTYPE_FP32, 0);
            Tensor *x_t = tensor_create(x_shape, 2, DTYPE_FP32, 0);
            Tensor *dA_t = tensor_zeros(dA_shape, 2, DTYPE_FP32, 0);
            memcpy(dh_t->data, dh, (size_t)M * rank * sizeof(float));
            memcpy(x_t->data, input->data, (size_t)M * in_f * sizeof(float));
            metal_matmul(dh_t->metal_buf, x_t->metal_buf, dA_t->metal_buf,
                         rank, M, in_f, 1, 0);
            metal_synchronize();
            for (int i = 0; i < rank * in_f; i++) {
                dA[i] += scaling * dA_t->data[i];
            }
            tensor_free(dh_t);
            tensor_free(x_t);
            tensor_free(dA_t);
        } else {
            for (int r = 0; r < rank; r++) {
                for (int k = 0; k < in_f; k++) {
                    float sum = 0.0f;
                    for (int m = 0; m < M; m++) {
                        sum += dh[m * rank + r] * input->data[m * in_f + k];
                    }
                    dA[r * in_f + k] += scaling * sum;
                }
            }
        }
    }

    // dx[M, in_f] += scaling * dh[M, rank] @ A[rank, in_f]
    if (input->grad) {
        float *dx = input->grad->data;
        for (int m = 0; m < M; m++) {
            for (int k = 0; k < in_f; k++) {
                float sum = 0.0f;
                for (int r = 0; r < rank; r++) {
                    sum += dh[m * rank + r] * lora_A->data[r * in_f + k];
                }
                dx[m * in_f + k] += scaling * sum;
            }
        }
    }

    free(h);
    free(dh);
}

LoRALinear *lora_create(Linear *base, int rank, float alpha) {
    LoRALinear *l = calloc(1, sizeof(LoRALinear));
    l->base = base;
    l->in_features = base->in_features;
    l->out_features = base->out_features;
    l->rank = rank;
    l->alpha = alpha;
    l->scaling = alpha / (float)rank;
    l->enabled = 1;

    // Freeze base weights
    base->weight->requires_grad = 0;
    if (base->bias) base->bias->requires_grad = 0;

    // Initialize A with Kaiming, B with zeros (standard LoRA init)
    float a_scale = sqrtf(2.0f / (float)base->in_features);
    int a_shape[] = {rank, base->in_features};
    l->lora_A = tensor_randn(a_shape, 2, DTYPE_FP32, 1, a_scale);

    int b_shape[] = {base->out_features, rank};
    l->lora_B = tensor_zeros(b_shape, 2, DTYPE_FP32, 1);

    return l;
}

void lora_free(LoRALinear *l) {
    if (!l) return;
    // base is not owned by LoRA — caller frees it
    tensor_free(l->lora_A);
    tensor_free(l->lora_B);
    free(l);
}

Tensor *lora_forward(LoRALinear *l, Tensor *input, ComputeGraph *g) {
    // Base forward: y = x @ W^T + b
    Tensor *base_out = linear_forward(l->base, input, g);

    if (!l->enabled) return base_out;

    int M = input->shape[0];
    int out_f = l->out_features;

    // Compute LoRA delta: delta = scaling * x @ A^T @ B^T
    int delta_shape[] = {M, out_f};
    Tensor *delta = tensor_zeros(delta_shape, 2, DTYPE_FP32, 1);

    if (metal_is_initialized()) {
        metal_lora_forward(input->metal_buf, l->lora_A->metal_buf,
                           l->lora_B->metal_buf, delta->metal_buf,
                           M, l->in_features, out_f, l->rank);
        metal_synchronize();
        // Apply scaling
        tensor_scale(delta, l->scaling);
    } else {
        // CPU: step 1: h = x @ A^T -> [M, rank]
        float *h = calloc((size_t)M * l->rank, sizeof(float));
        float *x = input->data;
        float *A = l->lora_A->data;
        float *B = l->lora_B->data;

        for (int m = 0; m < M; m++) {
            for (int r = 0; r < l->rank; r++) {
                float sum = 0.0f;
                for (int k = 0; k < l->in_features; k++) {
                    sum += x[m * l->in_features + k] * A[r * l->in_features + k];
                }
                h[m * l->rank + r] = sum;
            }
        }

        // step 2: delta = h @ B^T -> [M, out_f]
        float *d = delta->data;
        for (int m = 0; m < M; m++) {
            for (int o = 0; o < out_f; o++) {
                float sum = 0.0f;
                for (int r = 0; r < l->rank; r++) {
                    sum += h[m * l->rank + r] * B[o * l->rank + r];
                }
                d[m * out_f + o] = l->scaling * sum;
            }
        }
        free(h);
    }

    // Add delta to base output
    Tensor *output = residual_add(base_out, delta, g);

    // Register LoRA backward for delta computation
    if (g) {
        float *scaling_ptr = malloc(sizeof(float));
        *scaling_ptr = l->scaling;
        Tensor *inputs[] = {input, l->lora_A, l->lora_B};
        GraphNode *node = graph_add_node(g, delta, inputs, 3, lora_backward);
        node->saved_data = scaling_ptr;
    }

    return output;
}

void lora_merge(LoRALinear *l) {
    // W = W + scaling * B @ A
    // B: [out_f, rank], A: [rank, in_f] -> result: [out_f, in_f]
    float *W = l->base->weight->data;
    float *A = l->lora_A->data;
    float *B = l->lora_B->data;
    int out_f = l->out_features;
    int in_f = l->in_features;
    int rank = l->rank;

    for (int o = 0; o < out_f; o++) {
        for (int i = 0; i < in_f; i++) {
            float sum = 0.0f;
            for (int r = 0; r < rank; r++) {
                sum += B[o * rank + r] * A[r * in_f + i];
            }
            W[o * in_f + i] += l->scaling * sum;
        }
    }
}

void lora_collect_params(LoRALinear *l, ParamList *pl) {
    param_list_add(pl, l->lora_A);
    param_list_add(pl, l->lora_B);
}

// ==================================================================
// QLoRA implementation
// ==================================================================

QLoRALinear *qlora_create(int in_features, int out_features,
                          const uint8_t *packed, const float *scales,
                          const float *zeros, int group_size,
                          int rank, float alpha) {
    QLoRALinear *l = calloc(1, sizeof(QLoRALinear));
    l->in_features = in_features;
    l->out_features = out_features;
    l->n_elements = in_features * out_features;
    l->group_size = group_size;
    l->n_groups = (l->n_elements + group_size - 1) / group_size;
    l->rank = rank;
    l->alpha = alpha;
    l->scaling = alpha / (float)rank;
    l->enabled = 1;

    // Copy quantized weights
    int n_bytes = (l->n_elements + 1) / 2;
    l->packed_weights = malloc(n_bytes);
    memcpy(l->packed_weights, packed, n_bytes);

    l->scales = malloc(l->n_groups * sizeof(float));
    memcpy(l->scales, scales, l->n_groups * sizeof(float));

    l->zeros = malloc(l->n_groups * sizeof(float));
    memcpy(l->zeros, zeros, l->n_groups * sizeof(float));

    // Create Metal buffers
    if (metal_is_initialized()) {
        l->metal_packed = metal_create_shared_buffer(n_bytes);
        memcpy(metal_buffer_contents(l->metal_packed), packed, n_bytes);

        l->metal_scales = metal_create_shared_buffer(l->n_groups * sizeof(float));
        memcpy(metal_buffer_contents(l->metal_scales), scales, l->n_groups * sizeof(float));

        l->metal_zeros = metal_create_shared_buffer(l->n_groups * sizeof(float));
        memcpy(metal_buffer_contents(l->metal_zeros), zeros, l->n_groups * sizeof(float));
    }

    // LoRA adapters (trainable)
    float a_scale = sqrtf(2.0f / (float)in_features);
    int a_shape[] = {rank, in_features};
    l->lora_A = tensor_randn(a_shape, 2, DTYPE_FP32, 1, a_scale);

    int b_shape[] = {out_features, rank};
    l->lora_B = tensor_zeros(b_shape, 2, DTYPE_FP32, 1);

    return l;
}

void qlora_free(QLoRALinear *l) {
    if (!l) return;
    free(l->packed_weights);
    free(l->scales);
    free(l->zeros);
    if (l->metal_packed) metal_free_buffer(l->metal_packed);
    if (l->metal_scales) metal_free_buffer(l->metal_scales);
    if (l->metal_zeros)  metal_free_buffer(l->metal_zeros);
    tensor_free(l->lora_A);
    tensor_free(l->lora_B);
    free(l);
}

// CPU dequantize helper
static void cpu_dequantize_4bit(const uint8_t *packed, const float *scales,
                                const float *zeros, float *output,
                                int n_elements, int group_size) {
    for (int i = 0; i < (n_elements + 1) / 2; i++) {
        uint8_t byte = packed[i];
        int idx = i * 2;
        int group = idx / group_size;
        float scale = scales[group];
        float zero = zeros[group];

        output[idx] = (float)(byte & 0x0F) * scale + zero;
        if (idx + 1 < n_elements) {
            output[idx + 1] = (float)((byte >> 4) & 0x0F) * scale + zero;
        }
    }
}

Tensor *qlora_forward(QLoRALinear *l, Tensor *input, ComputeGraph *g) {
    int M = input->shape[0];
    int in_f = l->in_features;
    int out_f = l->out_features;

    // Dequantize base weights -> [out_f, in_f]
    int w_shape[] = {out_f, in_f};
    Tensor *W_deq = tensor_create(w_shape, 2, DTYPE_FP32, 0);

    if (metal_is_initialized() && l->metal_packed) {
        metal_dequantize_4bit(l->metal_packed, l->metal_scales, l->metal_zeros,
                              W_deq->metal_buf, l->n_elements, l->group_size);
        metal_synchronize();
    } else {
        cpu_dequantize_4bit(l->packed_weights, l->scales, l->zeros,
                            W_deq->data, l->n_elements, l->group_size);
    }

    // Base forward: y = x @ W_deq^T
    int out_shape[] = {M, out_f};
    Tensor *base_out = tensor_zeros(out_shape, 2, DTYPE_FP32, 1);

    if (metal_is_initialized()) {
        metal_matmul(input->metal_buf, W_deq->metal_buf, base_out->metal_buf,
                     M, in_f, out_f, 0, 1);
        metal_synchronize();
    } else {
        float *x = input->data;
        float *w = W_deq->data;
        float *y = base_out->data;
        for (int m = 0; m < M; m++) {
            for (int o = 0; o < out_f; o++) {
                float sum = 0.0f;
                for (int k = 0; k < in_f; k++) {
                    sum += x[m * in_f + k] * w[o * in_f + k];
                }
                y[m * out_f + o] = sum;
            }
        }
    }
    tensor_free(W_deq);

    if (!l->enabled) return base_out;

    // LoRA delta: delta = scaling * x @ A^T @ B^T
    int delta_shape[] = {M, out_f};
    Tensor *delta = tensor_zeros(delta_shape, 2, DTYPE_FP32, 1);

    if (metal_is_initialized()) {
        metal_lora_forward(input->metal_buf, l->lora_A->metal_buf,
                           l->lora_B->metal_buf, delta->metal_buf,
                           M, in_f, out_f, l->rank);
        metal_synchronize();
        tensor_scale(delta, l->scaling);
    } else {
        float *x = input->data;
        float *A = l->lora_A->data;
        float *B = l->lora_B->data;
        float *h = calloc((size_t)M * l->rank, sizeof(float));

        for (int m = 0; m < M; m++) {
            for (int r = 0; r < l->rank; r++) {
                float sum = 0.0f;
                for (int k = 0; k < in_f; k++) {
                    sum += x[m * in_f + k] * A[r * in_f + k];
                }
                h[m * l->rank + r] = sum;
            }
        }

        float *d = delta->data;
        for (int m = 0; m < M; m++) {
            for (int o = 0; o < out_f; o++) {
                float sum = 0.0f;
                for (int r = 0; r < l->rank; r++) {
                    sum += h[m * l->rank + r] * B[o * l->rank + r];
                }
                d[m * out_f + o] = l->scaling * sum;
            }
        }
        free(h);
    }

    // Add delta to base output
    Tensor *output = residual_add(base_out, delta, g);

    // Register backward for LoRA delta
    if (g) {
        float *scaling_ptr = malloc(sizeof(float));
        *scaling_ptr = l->scaling;
        Tensor *inputs[] = {input, l->lora_A, l->lora_B};
        GraphNode *node = graph_add_node(g, delta, inputs, 3, lora_backward);
        node->saved_data = scaling_ptr;
    }

    return output;
}

void qlora_collect_params(QLoRALinear *l, ParamList *pl) {
    param_list_add(pl, l->lora_A);
    param_list_add(pl, l->lora_B);
}
