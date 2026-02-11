#ifndef TRANSFORMER_H
#define TRANSFORMER_H

#include "../core/tensor.h"
#include "../core/autograd.h"
#include "layers.h"
#include "attention.h"

// Transformer Block: Pre-LayerNorm decoder block
typedef struct {
    LayerNorm *ln1;
    MultiHeadAttention *attn;
    LayerNorm *ln2;
    Linear *ff_up;     // d_model -> 4*d_model
    Linear *ff_down;   // 4*d_model -> d_model
    int d_model;
    int checkpoint;    // gradient checkpointing
} TransformerBlock;

TransformerBlock *transformer_block_create(int d_model, int n_heads, int checkpoint);
void              transformer_block_free(TransformerBlock *blk);
Tensor           *transformer_block_forward(TransformerBlock *blk, Tensor *x,
                                            int batch, int seq_len, ComputeGraph *g);
void              transformer_block_collect_params(TransformerBlock *blk, ParamList *pl);

// GPT Model: full decoder-only transformer
typedef struct {
    Embedding *token_emb;
    TransformerBlock **blocks;
    LayerNorm *final_ln;
    Linear *lm_head;        // Output projection to vocab
    int n_layers;
    int d_model;
    int n_heads;
    int vocab_size;
    int max_seq_len;
} GPTModel;

typedef struct {
    int n_layers;
    int d_model;
    int n_heads;
    int vocab_size;
    int max_seq_len;
    int gradient_checkpointing;
} GPTConfig;

GPTModel *gpt_create(GPTConfig config);
void      gpt_free(GPTModel *model);

// Forward: tokens [batch, seq_len] -> logits [batch*seq_len, vocab_size]
Tensor *gpt_forward(GPTModel *model, Tensor *tokens, int batch, int seq_len, ComputeGraph *g);

// Collect all parameters
ParamList *gpt_collect_params(GPTModel *model);

// Count parameters
size_t gpt_param_count(GPTModel *model);

#endif // TRANSFORMER_H
