#ifndef FAST_METAL_H
#define FAST_METAL_H

#include <stddef.h>
#include <stdint.h>

// Initialize Metal device and compile compute kernels
int fast_metal_init(void);
void fast_metal_shutdown(void);

// Buffer management — all buffers are shared (CPU + GPU accessible)
typedef struct MetalBuf MetalBuf;

MetalBuf *metal_buf_create(size_t size);                    // empty buffer
MetalBuf *metal_buf_from_data(const void *data, size_t sz); // copy data in
void      metal_buf_free(MetalBuf *b);
void     *metal_buf_ptr(MetalBuf *b);  // CPU-accessible pointer

// Enqueue Q8_0 matvec: y = W @ x (SIMD, 8 simdgroups per threadgroup)
void metal_enqueue_q8_matvec(MetalBuf *W_buf, MetalBuf *x_buf, MetalBuf *y_buf,
                             int rows, int nb);

// Enqueue F16 matvec: y = W @ x (half-precision weights, float accumulation)
void metal_enqueue_f16_matvec(MetalBuf *W_buf, MetalBuf *x_buf, MetalBuf *y_buf,
                              int rows, int cols);

// Enqueue F16 batched matvec: Y[B,rows] = W[rows,cols] × X[B,cols]
void metal_enqueue_f16_batch_matvec(MetalBuf *W_buf, MetalBuf *X_buf, MetalBuf *Y_buf,
                                     int rows, int cols, int B);

// Enqueue RMSNorm: out = gamma * x / sqrt(mean(x²) + eps)
// B = number of independent vectors (dispatch B threadgroups)
void metal_enqueue_rms_norm(MetalBuf *x_buf, MetalBuf *gamma_buf, MetalBuf *out_buf,
                            int dim, float eps);
void metal_enqueue_rms_norm_batched(MetalBuf *x_buf, MetalBuf *gamma_buf, MetalBuf *out_buf,
                                     int dim, float eps, int B);

// Enqueue per-head RMSNorm: in-place, one threadgroup per head
void metal_enqueue_per_head_rms_norm(MetalBuf *vec_buf, MetalBuf *gamma_buf,
                                      int n_heads, int head_dim, float eps);

// Enqueue RoPE: rotary position embedding for Q and K
void metal_enqueue_rope(MetalBuf *q_buf, MetalBuf *k_buf,
                        int Hq, int Hkv, int hd, int pos, float theta);

// Enqueue KV cache store: copy K/V into cache at position pos
void metal_enqueue_kv_cache_store(MetalBuf *k_buf, MetalBuf *v_buf,
                                   MetalBuf *k_cache, MetalBuf *v_cache,
                                   int Hkv, int max_seq, int hd, int pos);

// Enqueue attention: fused Q@K^T + softmax + scores@V
void metal_enqueue_attention(MetalBuf *Q_buf, MetalBuf *Kc_buf, MetalBuf *Vc_buf,
                             MetalBuf *attn_out_buf, int n_attend, int hd,
                             int max_seq, int Hq, int group_ratio);

// Enqueue SiLU(gate) * up — modifies gate in place
void metal_enqueue_silu_mul(MetalBuf *gate_buf, MetalBuf *up_buf, int n);

// Enqueue residual add: x += y
void metal_enqueue_residual_add(MetalBuf *x_buf, MetalBuf *y_buf, int n);

// Offset-aware dispatch for batched inference (per-item kernels)
void metal_enqueue_per_head_rms_norm_off(MetalBuf *vec_buf, size_t vec_off,
                                          MetalBuf *gamma_buf,
                                          int n_heads, int head_dim, float eps);
void metal_enqueue_rope_off(MetalBuf *q_buf, size_t q_off,
                             MetalBuf *k_buf, size_t k_off,
                             int Hq, int Hkv, int hd, int pos, float theta);
void metal_enqueue_kv_cache_store_off(MetalBuf *k_buf, size_t k_off,
                                       MetalBuf *v_buf, size_t v_off,
                                       MetalBuf *k_cache, MetalBuf *v_cache,
                                       int Hkv, int max_seq, int hd, int pos);
void metal_enqueue_attention_off(MetalBuf *Q_buf, size_t q_off,
                                  MetalBuf *Kc_buf, MetalBuf *Vc_buf,
                                  MetalBuf *attn_out_buf, size_t out_off,
                                  int n_attend, int hd, int max_seq, int Hq,
                                  int group_ratio);

// Commit all pending GPU commands and wait for completion
void metal_flush(void);

// ===================== Training dispatches =====================

// F16 tiled matmul: C[M,N] = A[M,K] @ W_half[N,K]^T
void metal_enqueue_f16_matmul(MetalBuf *W_buf, MetalBuf *A_buf, MetalBuf *C_buf,
                               int M, int N, int K);

// F16 matmul backward: C[M,K] = A[M,N] @ W_half[N,K] (no transpose)
void metal_enqueue_f16_matmul_nt(MetalBuf *W_buf, MetalBuf *A_buf, MetalBuf *C_buf,
                                  int M, int N, int K);

// Transpose: [seq_len, n_heads*head_dim] -> [n_heads, seq_len, head_dim]
void metal_enqueue_transpose_heads_fwd(MetalBuf *src, MetalBuf *dst,
                                        int seq_len, int n_heads, int head_dim);

// Transpose back: [n_heads, seq_len, head_dim] -> [seq_len, n_heads*head_dim]
void metal_enqueue_transpose_heads_rev(MetalBuf *src, MetalBuf *dst,
                                        int seq_len, int n_heads, int head_dim);

// Repeat KV heads: [n_kv, seq_len, head_dim] -> [n_q, seq_len, head_dim]
void metal_enqueue_repeat_kv(MetalBuf *src, MetalBuf *dst,
                              int n_kv, int seq_len, int head_dim, int group_ratio);

// Batch RoPE: apply RoPE to Q[Hq,N,D] and K[Hkv,N,D] in-place
void metal_enqueue_rope_train(MetalBuf *q_buf, MetalBuf *k_buf,
                               int Hq, int Hkv, int hd, int seq_len, float theta);

// SiLU*mul backward
void metal_enqueue_silu_mul_backward(MetalBuf *dout, MetalBuf *gate, MetalBuf *up,
                                      MetalBuf *dgate, MetalBuf *dup, int n);

// RMSNorm backward for training
void metal_enqueue_rms_norm_backward(MetalBuf *x, MetalBuf *dy, MetalBuf *gamma,
                                      MetalBuf *dx, int batch, int dim, float eps);

// Scale tensor in-place: x *= alpha
void metal_enqueue_scale(MetalBuf *x, float alpha, int n);

// Add scaled: y += alpha * x
void metal_enqueue_add_scaled(MetalBuf *y, MetalBuf *x, float alpha, int n);

// Softmax CE loss + gradient
void metal_enqueue_softmax_ce(MetalBuf *logits, MetalBuf *targets, MetalBuf *losses,
                               MetalBuf *dlogits, int batch, int vocab_size);

// Attention training forward: causal self-attention with probs saving
void metal_enqueue_attn_train_fwd(MetalBuf *Q, MetalBuf *K, MetalBuf *V,
                                   MetalBuf *out, MetalBuf *probs,
                                   int H, int N, int D, float scale);

// Attention backward: compute dQ and d_score
void metal_enqueue_attn_bwd_dq(MetalBuf *d_out, MetalBuf *probs, MetalBuf *V,
                                MetalBuf *K, MetalBuf *d_score, MetalBuf *dQ,
                                int H, int N, int D, float scale);

// Attention backward: compute dK and dV
void metal_enqueue_attn_bwd_dkv(MetalBuf *d_score, MetalBuf *Q, MetalBuf *probs,
                                 MetalBuf *d_out, MetalBuf *dK, MetalBuf *dV,
                                 int H, int N, int D);

// Repeat KV backward: reduce expanded gradients to KV head count
void metal_enqueue_repeat_kv_bwd(MetalBuf *d_expanded, MetalBuf *d_kv,
                                  int n_kv, int seq_len, int head_dim, int group_ratio);

// RoPE backward: inverse rotation
void metal_enqueue_rope_train_bwd(MetalBuf *q_buf, MetalBuf *k_buf,
                                   int Hq, int Hkv, int hd, int seq_len, float theta);

// Float matmul: C[M,N] = A[M,K] @ W[N,K]^T (both float, for LoRA)
void metal_enqueue_float_matmul(MetalBuf *A, MetalBuf *W, MetalBuf *C,
                                 int M, int N, int K);

// Float matmul A-transposed: C[M,N] = A[K,M]^T @ B[K,N] (for LoRA grad)
void metal_enqueue_float_matmul_tn(MetalBuf *A, MetalBuf *B, MetalBuf *C,
                                    int M, int N, int K);

// Float matmul no-transpose: C[M,K] = A[M,N] @ W[N,K] (for LoRA backward)
void metal_enqueue_float_matmul_nt(MetalBuf *A, MetalBuf *W, MetalBuf *C,
                                    int M, int N, int K);

// Buffer copy: dst = src
void metal_enqueue_copy(MetalBuf *src, MetalBuf *dst, int n);

// Element-wise clamp: x = clamp(x, -max_val, max_val)
void metal_enqueue_clamp(MetalBuf *x, float max_val, int n);

#endif // FAST_METAL_H
