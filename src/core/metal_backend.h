#ifndef METAL_BACKEND_H
#define METAL_BACKEND_H

#include <stddef.h>
#include <stdint.h>

// Initialize Metal device, command queue, load shader library
int  metal_init(const char *shader_path);
void metal_cleanup(void);
int  metal_is_initialized(void);

// Buffer management (MTLResourceStorageModeShared)
void *metal_create_shared_buffer(size_t bytes);
void  metal_free_buffer(void *buf);
void *metal_buffer_contents(void *buf);

// Kernel dispatch
// matmul: C[M,N] = A[M,K] @ B[K,N] (with optional transposes)
// transpose_a: if 1, A is stored as [K,M] and read transposed
// transpose_b: if 1, B is stored as [N,K] and read transposed
void metal_matmul(void *buf_a, void *buf_b, void *buf_c,
                  int M, int K, int N, int transpose_a, int transpose_b);

// matmul batched: for attention, shape [batch, M, K] @ [batch, K, N]
void metal_matmul_batched(void *buf_a, void *buf_b, void *buf_c,
                          int batch, int M, int K, int N, int transpose_a, int transpose_b);

// Fused layer norm + residual add
void metal_layer_norm_residual(void *buf_x, void *buf_residual, void *buf_out,
                               void *buf_gamma, void *buf_beta,
                               int batch, int dim, float eps);

// Flash Attention: Q,K,V,O shape [B, H, N, D]
void metal_flash_attention(void *buf_q, void *buf_k, void *buf_v, void *buf_o,
                           int B, int H, int N, int D, float scale, int is_causal);

// Flash Attention backward
void metal_flash_attention_backward(void *buf_q, void *buf_k, void *buf_v,
                                    void *buf_o, void *buf_do,
                                    void *buf_dq, void *buf_dk, void *buf_dv,
                                    int B, int H, int N, int D, float scale, int is_causal);

// Layer norm: out = (x - mean) / sqrt(var + eps) * gamma + beta
void metal_layer_norm(void *buf_x, void *buf_out, void *buf_gamma, void *buf_beta,
                      int batch, int dim, float eps);

// Layer norm backward
void metal_layer_norm_backward(void *buf_x, void *buf_dy, void *buf_gamma,
                               void *buf_dx, void *buf_dgamma, void *buf_dbeta,
                               int batch, int dim, float eps);

// GELU activation
void metal_gelu(void *buf_in, void *buf_out, int n);
void metal_gelu_backward(void *buf_in, void *buf_dy, void *buf_dx, int n);

// RoPE: Apply rotary position embeddings
void metal_rope(void *buf_x, void *buf_out, int B, int H, int N, int D, int offset);
void metal_rope_backward(void *buf_dy, void *buf_dx, int B, int H, int N, int D, int offset);

// Embedding lookup
void metal_embedding(void *buf_table, void *buf_indices, void *buf_out,
                     int vocab_size, int dim, int seq_len);
void metal_embedding_backward(void *buf_dy, void *buf_indices, void *buf_dtable,
                              int vocab_size, int dim, int seq_len);

// Softmax + CrossEntropy loss (fused)
void metal_softmax_ce_loss(void *buf_logits, void *buf_targets, void *buf_loss,
                           void *buf_dlogits, int batch, int vocab_size);

// AdamW optimizer step
void metal_adamw_update(void *buf_param, void *buf_grad,
                        void *buf_m, void *buf_v,
                        int n, float lr, float beta1, float beta2,
                        float eps, float weight_decay, int step);

// Residual add
void metal_add(void *buf_a, void *buf_b, void *buf_out, int n);

// Element-wise multiply
void metal_mul(void *buf_a, void *buf_b, void *buf_out, int n);

// RMSNorm: out = gamma * x / sqrt(mean(x²) + eps)
void metal_rms_norm(void *buf_x, void *buf_out, void *buf_gamma,
                    int batch, int dim, float eps);
void metal_rms_norm_backward(void *buf_x, void *buf_dy, void *buf_gamma,
                             void *buf_dx, void *buf_dgamma,
                             int batch, int dim, float eps);

// SiLU activation: out = x * sigmoid(x)
void metal_silu(void *buf_in, void *buf_out, int n);
void metal_silu_backward(void *buf_in, void *buf_dy, void *buf_dx, int n);

// LoRA forward: delta = x @ A^T @ B^T
void metal_lora_forward(void *buf_x, void *buf_a, void *buf_b, void *buf_delta,
                        int M, int in_features, int out_features, int rank);

// 4-bit dequantization (QLoRA)
void metal_dequantize_4bit(void *buf_packed, void *buf_scales, void *buf_zeros,
                           void *buf_output, int n_elements, int group_size);

// Synchronize GPU
void metal_synchronize(void);

#endif // METAL_BACKEND_H
