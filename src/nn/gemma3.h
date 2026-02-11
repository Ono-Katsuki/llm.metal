#ifndef GEMMA3_H
#define GEMMA3_H

#include "../core/tensor.h"
#include "../core/autograd.h"
#include "layers.h"
#include "attention.h"
#include "lora.h"
#include "../data/safetensors.h"

// ==================================================================
// Gemma3 Block: RMSNorm + GQA + RMSNorm + GeGLU FFN
// ==================================================================

typedef struct {
    RMSNorm *input_norm;
    GroupedQueryAttention *attn;
    RMSNorm *post_attn_norm;   // post-attention norm (before residual add)
    RMSNorm *pre_ff_norm;      // pre-feedforward norm
    Linear *gate_proj;   // [d_model, intermediate_size]
    Linear *up_proj;     // [d_model, intermediate_size]
    Linear *down_proj;   // [intermediate_size, d_model]
    RMSNorm *post_ff_norm;     // post-feedforward norm (before residual add)
    int d_model;
    int is_sliding;      // 1 = local (sliding_window), 0 = global
} Gemma3Block;

// ==================================================================
// Gemma3 Model
// ==================================================================

typedef struct {
    Embedding *token_emb;
    Gemma3Block **blocks;
    RMSNorm *final_norm;
    // lm_head is tied with token_emb (weight sharing)
    int n_layers;
    int d_model;
    int vocab_size;
    int n_q_heads;
    int n_kv_heads;
    int head_dim;
    int intermediate_size;
    int sliding_window;
    float local_rope_theta;
    float global_rope_theta;

    // LoRA adapters (NULL when not using LoRA)
    LoRALinear **q_loras;
    LoRALinear **v_loras;
} Gemma3Model;

typedef struct {
    int n_layers;
    int d_model;
    int n_q_heads;
    int n_kv_heads;
    int head_dim;
    int intermediate_size;
    int vocab_size;
    int sliding_window;
    float local_rope_theta;
    float global_rope_theta;
    float rms_norm_eps;
} Gemma3Config;

// Default configs
Gemma3Config gemma3_1b_config(void);
Gemma3Config gemma3_4b_config(void);

// Create model with allocated (zero) weights
Gemma3Model *gemma3_create(Gemma3Config config);

// Load from safetensors (path = single file or directory)
Gemma3Model *gemma3_load_safetensors(const char *path);

// Free model
void gemma3_free(Gemma3Model *model);

// Count parameters
size_t gemma3_param_count(Gemma3Model *model);

// Attach LoRA adapters to Q and V projections
void gemma3_attach_lora(Gemma3Model *model, int rank, float alpha);

// Save/load LoRA adapters
void gemma3_save_lora(Gemma3Model *model, const char *path);
int  gemma3_load_lora(Gemma3Model *model, const char *path);

#endif // GEMMA3_H
