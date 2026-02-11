#ifndef QWEN3_H
#define QWEN3_H

#include "../core/tensor.h"
#include "../core/autograd.h"
#include "layers.h"
#include "attention.h"
#include "lora.h"
#include "../data/gguf.h"
#include "../data/safetensors.h"

// ==================================================================
// Qwen3 Block: RMSNorm + GQA + RMSNorm + SwiGLU FFN
// ==================================================================
typedef struct {
    RMSNorm *input_norm;
    GroupedQueryAttention *attn;
    RMSNorm *post_attn_norm;
    Linear *gate_proj;   // [d_model, intermediate_size]
    Linear *up_proj;     // [d_model, intermediate_size]
    Linear *down_proj;   // [intermediate_size, d_model]
    int d_model;
} Qwen3Block;

// ==================================================================
// Qwen3 Model
// ==================================================================
typedef struct {
    Embedding *token_emb;
    Qwen3Block **blocks;
    RMSNorm *final_norm;
    Linear *lm_head;
    int n_layers;
    int d_model;
    int vocab_size;
    int n_q_heads;
    int n_kv_heads;
    int head_dim;
    int intermediate_size;
    float rope_theta;

    // LoRA adapters (NULL when not using LoRA)
    LoRALinear **q_loras;   // [n_layers]
    LoRALinear **v_loras;   // [n_layers]
} Qwen3Model;

typedef struct {
    int n_layers;
    int d_model;
    int n_q_heads;
    int n_kv_heads;
    int head_dim;
    int intermediate_size;
    int vocab_size;
    float rope_theta;
    float rms_norm_eps;
} Qwen3Config;

// Default config for Qwen3-4B
Qwen3Config qwen3_4b_config(void);

// Create model with random weights
Qwen3Model *qwen3_create(Qwen3Config config);

// Load model from GGUF file
Qwen3Model *qwen3_load_gguf(const char *gguf_path);

// Load model from safetensors file(s)
// path can be a single .safetensors file or directory with shards
Qwen3Model *qwen3_load_safetensors(const char *path);

// Free model
void qwen3_free(Qwen3Model *model);

// Forward: tokens [batch * seq_len] -> logits [batch*seq_len, vocab_size]
Tensor *qwen3_forward(Qwen3Model *model, Tensor *tokens,
                      int batch, int seq_len, ComputeGraph *g);

// Incremental forward with KV cache (fast inference)
// tokens: [batch * n_new] new token IDs
// Returns logits for new positions only: [batch * n_new, vocab_size]
Tensor *qwen3_forward_cached(Qwen3Model *model, Tensor *tokens,
                             int batch, int n_new, KVCache *cache);

// Create KV cache for this model
KVCache *qwen3_create_kv_cache(Qwen3Model *model, int max_seq_len);

// Attach LoRA adapters to Q and V projections
void qwen3_attach_lora(Qwen3Model *model, int rank, float alpha);

// Collect trainable parameters (LoRA only if attached, else all)
ParamList *qwen3_collect_params(Qwen3Model *model);

// Count total parameters
size_t qwen3_param_count(Qwen3Model *model);

// Save/load LoRA adapters
void qwen3_save_lora(Qwen3Model *model, const char *path);
int  qwen3_load_lora(Qwen3Model *model, const char *path);

#endif // QWEN3_H
