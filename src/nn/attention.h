#ifndef ATTENTION_H
#define ATTENTION_H

#include "../core/tensor.h"
#include "../core/autograd.h"
#include "layers.h"

// Multi-Head Attention with Flash Attention and RoPE
typedef struct {
    Linear *q_proj;
    Linear *k_proj;
    Linear *v_proj;
    Linear *o_proj;
    int d_model;
    int n_heads;
    int head_dim;
    float scale;
} MultiHeadAttention;

MultiHeadAttention *mha_create(int d_model, int n_heads);
void                mha_free(MultiHeadAttention *mha);

// Forward: x [batch, seq, d_model] -> output [batch, seq, d_model]
// For simplicity, x is [batch*seq, d_model] (2D)
Tensor *mha_forward(MultiHeadAttention *mha, Tensor *x,
                    int batch, int seq_len, ComputeGraph *g);

// Collect parameters
void mha_collect_params(MultiHeadAttention *mha, ParamList *pl);

// Forward declare LoRA for optional integration
struct LoRALinear;

// ==================================================================
// Grouped Query Attention (GQA) — for Qwen3, LLaMA, etc.
// n_q_heads query heads, n_kv_heads key/value heads (n_q_heads / n_kv_heads = group ratio)
// ==================================================================
typedef struct {
    int d_model;
    int n_q_heads;
    int n_kv_heads;
    int head_dim;
    float scale;
    float rope_theta;
    Linear *q_proj;   // [d_model, n_q_heads * head_dim]
    Linear *k_proj;   // [d_model, n_kv_heads * head_dim]
    Linear *v_proj;   // [d_model, n_kv_heads * head_dim]
    Linear *o_proj;   // [n_q_heads * head_dim, d_model]
    RMSNorm *q_norm;  // per-head QK norm (dim=head_dim)
    RMSNorm *k_norm;  // per-head QK norm (dim=head_dim)
} GroupedQueryAttention;

GroupedQueryAttention *gqa_create(int d_model, int n_q_heads, int n_kv_heads,
                                  int head_dim, float rope_theta);
void                   gqa_free(GroupedQueryAttention *gqa);

Tensor *gqa_forward(GroupedQueryAttention *gqa, Tensor *x,
                    int batch, int seq_len, ComputeGraph *g);

// GQA forward with optional LoRA on Q and V projections (pass NULL to skip)
Tensor *gqa_forward_lora(GroupedQueryAttention *gqa, Tensor *x,
                         int batch, int seq_len, ComputeGraph *g,
                         struct LoRALinear *q_lora, struct LoRALinear *v_lora);

void gqa_collect_params(GroupedQueryAttention *gqa, ParamList *pl);

// ==================================================================
// KV Cache for fast incremental inference
// ==================================================================
typedef struct {
    int n_layers;
    int n_kv_heads;
    int head_dim;
    int max_seq_len;
    int cur_len;        // current number of cached positions
    float **k_cache;    // [n_layers][n_kv_heads * max_seq_len * head_dim]
    float **v_cache;    // [n_layers][n_kv_heads * max_seq_len * head_dim]
} KVCache;

KVCache *kv_cache_create(int n_layers, int n_kv_heads, int head_dim, int max_seq_len);
void     kv_cache_free(KVCache *cache);
void     kv_cache_reset(KVCache *cache);

// Incremental forward: process only new tokens, use cached K/V for attention
Tensor *gqa_forward_cached(GroupedQueryAttention *gqa, Tensor *x,
                           int batch, int n_new, int layer_idx,
                           KVCache *cache);

#endif // ATTENTION_H
