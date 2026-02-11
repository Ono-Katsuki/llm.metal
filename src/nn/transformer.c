#include "transformer.h"
#include "../core/metal_backend.h"
#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <math.h>

// ==================================================================
// Gradient Checkpointing support
// ==================================================================

typedef struct {
    TransformerBlock *blk;
    Tensor *saved_input;    // only tensor saved (instead of all intermediates)
    int batch;
    int seq_len;
} CheckpointData;

static void checkpoint_cleanup(void *data) {
    // saved_input is not owned by checkpoint — it's the block input
    free(data);
}

static void checkpoint_backward(GraphNode *node) {
    CheckpointData *cd = (CheckpointData *)node->saved_data;
    Tensor *output = node->output;

    // Re-run forward pass with a temporary graph to reconstruct activations
    ComputeGraph *tmp_g = graph_create();
    Tensor *recomputed = transformer_block_forward(cd->blk, cd->saved_input,
                                                    cd->batch, cd->seq_len, tmp_g);

    // Set the recomputed output's grad to our output's grad
    if (recomputed->grad && output->grad) {
        memcpy(recomputed->grad->data, output->grad->data, output->grad->bytes);
    }

    // Run backward on temp graph
    graph_backward(tmp_g, recomputed);

    // Propagate gradient to input
    if (cd->saved_input->grad && cd->saved_input->grad->data) {
        // grad was accumulated during backward on tmp_g
        // already written to saved_input->grad by the temp graph backward
    }

    graph_destroy(tmp_g);
}

// ==================================================================
// Transformer Block
// ==================================================================

TransformerBlock *transformer_block_create(int d_model, int n_heads, int checkpoint) {
    TransformerBlock *blk = calloc(1, sizeof(TransformerBlock));
    blk->d_model = d_model;
    blk->checkpoint = checkpoint;
    blk->ln1 = layer_norm_create(d_model, 1e-5f);
    blk->attn = mha_create(d_model, n_heads);
    blk->ln2 = layer_norm_create(d_model, 1e-5f);
    blk->ff_up = linear_create(d_model, 4 * d_model, 1);
    blk->ff_down = linear_create(4 * d_model, d_model, 1);
    return blk;
}

void transformer_block_free(TransformerBlock *blk) {
    if (!blk) return;
    layer_norm_free(blk->ln1);
    mha_free(blk->attn);
    layer_norm_free(blk->ln2);
    linear_free(blk->ff_up);
    linear_free(blk->ff_down);
    free(blk);
}

// Standard forward (saves all activations in graph)
static Tensor *block_forward_full(TransformerBlock *blk, Tensor *x,
                                  int batch, int seq_len, ComputeGraph *g) {
    // Pre-LayerNorm Transformer Block:
    // x = x + Attention(LayerNorm(x))
    // x = x + FFN(LayerNorm(x))

    // Attention sub-block
    Tensor *ln1_out = layer_norm_forward(blk->ln1, x, g);
    Tensor *attn_out = mha_forward(blk->attn, ln1_out, batch, seq_len, g);
    Tensor *x2 = residual_add(x, attn_out, g);

    // FFN sub-block
    Tensor *ln2_out = layer_norm_forward(blk->ln2, x2, g);
    Tensor *ff_up = linear_forward(blk->ff_up, ln2_out, g);
    Tensor *ff_act = gelu_forward(ff_up, g);
    Tensor *ff_down = linear_forward(blk->ff_down, ff_act, g);
    Tensor *x3 = residual_add(x2, ff_down, g);

    return x3;
}

Tensor *transformer_block_forward(TransformerBlock *blk, Tensor *x,
                                  int batch, int seq_len, ComputeGraph *g) {
    if (!blk->checkpoint || !g) {
        // No checkpointing: standard forward
        return block_forward_full(blk, x, batch, seq_len, g);
    }

    // Gradient checkpointing: run forward without graph (eval mode)
    // to avoid saving intermediate activations
    Tensor *output = block_forward_full(blk, x, batch, seq_len, NULL);

    // Make output require grad so backward can propagate
    if (!output->grad) {
        output->requires_grad = 1;
        output->grad = tensor_zeros(output->shape, output->ndim, DTYPE_FP32, 0);
    }

    // Register a single checkpoint node in the main graph
    CheckpointData *cd = calloc(1, sizeof(CheckpointData));
    cd->blk = blk;
    cd->saved_input = x;
    cd->batch = batch;
    cd->seq_len = seq_len;

    Tensor *inputs[] = {x};
    GraphNode *node = graph_add_node(g, output, inputs, 1, checkpoint_backward);
    node->saved_data = cd;
    node->cleanup_fn = checkpoint_cleanup;
    node->checkpoint = 1;

    return output;
}

void transformer_block_collect_params(TransformerBlock *blk, ParamList *pl) {
    param_list_add(pl, blk->ln1->gamma);
    param_list_add(pl, blk->ln1->beta);
    mha_collect_params(blk->attn, pl);
    param_list_add(pl, blk->ln2->gamma);
    param_list_add(pl, blk->ln2->beta);
    param_list_add(pl, blk->ff_up->weight);
    if (blk->ff_up->bias) param_list_add(pl, blk->ff_up->bias);
    param_list_add(pl, blk->ff_down->weight);
    if (blk->ff_down->bias) param_list_add(pl, blk->ff_down->bias);
}

// ==================================================================
// GPT Model
// ==================================================================

GPTModel *gpt_create(GPTConfig config) {
    GPTModel *model = calloc(1, sizeof(GPTModel));
    model->n_layers = config.n_layers;
    model->d_model = config.d_model;
    model->n_heads = config.n_heads;
    model->vocab_size = config.vocab_size;
    model->max_seq_len = config.max_seq_len;

    model->token_emb = embedding_create(config.vocab_size, config.d_model);

    model->blocks = calloc(config.n_layers, sizeof(TransformerBlock *));
    for (int i = 0; i < config.n_layers; i++) {
        model->blocks[i] = transformer_block_create(config.d_model, config.n_heads,
                                                     config.gradient_checkpointing);
    }

    model->final_ln = layer_norm_create(config.d_model, 1e-5f);
    model->lm_head = linear_create(config.d_model, config.vocab_size, 0);

    return model;
}

void gpt_free(GPTModel *model) {
    if (!model) return;
    embedding_free(model->token_emb);
    for (int i = 0; i < model->n_layers; i++) {
        transformer_block_free(model->blocks[i]);
    }
    free(model->blocks);
    layer_norm_free(model->final_ln);
    linear_free(model->lm_head);
    free(model);
}

Tensor *gpt_forward(GPTModel *model, Tensor *tokens, int batch, int seq_len, ComputeGraph *g) {
    // tokens: [batch * seq_len] (flat array of token IDs)

    // Token embeddings -> [batch*seq_len, d_model]
    Tensor *x = embedding_forward(model->token_emb, tokens, g);

    // Transformer blocks
    for (int i = 0; i < model->n_layers; i++) {
        x = transformer_block_forward(model->blocks[i], x, batch, seq_len, g);
    }

    // Final layer norm
    x = layer_norm_forward(model->final_ln, x, g);

    // LM head: [BN, d_model] -> [BN, vocab_size]
    Tensor *logits = linear_forward(model->lm_head, x, g);

    return logits;
}

ParamList *gpt_collect_params(GPTModel *model) {
    // Estimate total params
    int cap = 2 + model->n_layers * 12 + 4;
    ParamList *pl = param_list_create(cap);

    param_list_add(pl, model->token_emb->weight);

    for (int i = 0; i < model->n_layers; i++) {
        transformer_block_collect_params(model->blocks[i], pl);
    }

    param_list_add(pl, model->final_ln->gamma);
    param_list_add(pl, model->final_ln->beta);
    param_list_add(pl, model->lm_head->weight);

    return pl;
}

size_t gpt_param_count(GPTModel *model) {
    ParamList *pl = gpt_collect_params(model);
    size_t total = 0;
    for (int i = 0; i < pl->n_params; i++) {
        total += pl->params[i]->size;
    }
    param_list_free(pl);
    return total;
}
