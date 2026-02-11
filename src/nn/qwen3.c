#include "qwen3.h"
#include "../core/metal_backend.h"
#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <math.h>
#include <assert.h>

// ==================================================================
// Qwen3 Config
// ==================================================================

Qwen3Config qwen3_4b_config(void) {
    return (Qwen3Config){
        .n_layers = 36,
        .d_model = 2560,
        .n_q_heads = 32,
        .n_kv_heads = 8,
        .head_dim = 128,
        .intermediate_size = 9728,
        .vocab_size = 151936,
        .rope_theta = 1000000.0f,
        .rms_norm_eps = 1e-6f,
    };
}

// ==================================================================
// Qwen3 Block
// ==================================================================

static Qwen3Block *qwen3_block_create(Qwen3Config config) {
    Qwen3Block *blk = calloc(1, sizeof(Qwen3Block));
    blk->d_model = config.d_model;
    blk->input_norm = rms_norm_create(config.d_model, config.rms_norm_eps);
    blk->attn = gqa_create(config.d_model, config.n_q_heads, config.n_kv_heads,
                            config.head_dim, config.rope_theta);
    blk->post_attn_norm = rms_norm_create(config.d_model, config.rms_norm_eps);
    blk->gate_proj = linear_create(config.d_model, config.intermediate_size, 0);
    blk->up_proj = linear_create(config.d_model, config.intermediate_size, 0);
    blk->down_proj = linear_create(config.intermediate_size, config.d_model, 0);
    return blk;
}

static void qwen3_block_free(Qwen3Block *blk) {
    if (!blk) return;
    rms_norm_free(blk->input_norm);
    gqa_free(blk->attn);
    rms_norm_free(blk->post_attn_norm);
    linear_free(blk->gate_proj);
    linear_free(blk->up_proj);
    linear_free(blk->down_proj);
    free(blk);
}

static Tensor *qwen3_block_forward(Qwen3Block *blk, Tensor *x,
                                    int batch, int seq_len, ComputeGraph *g,
                                    LoRALinear *q_lora, LoRALinear *v_lora) {
    // Pre-RMSNorm + GQA Attention + Residual
    Tensor *ln1_out = rms_norm_forward(blk->input_norm, x, g);

    // GQA attention — uses LoRA for Q/V projections when provided
    Tensor *attn_out = gqa_forward_lora(blk->attn, ln1_out, batch, seq_len, g,
                                         q_lora, v_lora);
    Tensor *x2 = residual_add(x, attn_out, g);

    // Pre-RMSNorm + SwiGLU FFN + Residual
    Tensor *ln2_out = rms_norm_forward(blk->post_attn_norm, x2, g);

    // SwiGLU: down_proj(silu(gate_proj(x)) * up_proj(x))
    Tensor *gate = linear_forward(blk->gate_proj, ln2_out, g);
    Tensor *gate_act = silu_forward(gate, g);
    Tensor *up = linear_forward(blk->up_proj, ln2_out, g);
    Tensor *product = elementwise_mul(gate_act, up, g);
    Tensor *ff_out = linear_forward(blk->down_proj, product, g);
    Tensor *x3 = residual_add(x2, ff_out, g);

    return x3;
}

// Cached block forward: only processes n_new tokens, uses KV cache for attention
static Tensor *qwen3_block_forward_cached(Qwen3Block *blk, Tensor *x,
                                           int batch, int n_new, int layer_idx,
                                           KVCache *cache) {
    // Pre-RMSNorm + GQA with KV cache + Residual
    Tensor *ln1_out = rms_norm_forward(blk->input_norm, x, NULL);
    metal_synchronize();
    Tensor *attn_out = gqa_forward_cached(blk->attn, ln1_out, batch, n_new, layer_idx, cache);

    // Residual: x + attn_out (CPU, no graph)
    int res_shape[] = {batch * n_new, blk->d_model};
    Tensor *x2 = tensor_create(res_shape, 2, DTYPE_FP32, 0);
    for (int i = 0; i < batch * n_new * blk->d_model; i++)
        x2->data[i] = x->data[i] + attn_out->data[i];

    // Pre-RMSNorm + SwiGLU FFN + Residual
    Tensor *ln2_out = rms_norm_forward(blk->post_attn_norm, x2, NULL);
    metal_synchronize();
    Tensor *gate = linear_forward(blk->gate_proj, ln2_out, NULL);
    metal_synchronize();
    Tensor *gate_act = silu_forward(gate, NULL);
    metal_synchronize();
    Tensor *up = linear_forward(blk->up_proj, ln2_out, NULL);
    metal_synchronize();
    Tensor *product = elementwise_mul(gate_act, up, NULL);
    metal_synchronize();
    Tensor *ff_out = linear_forward(blk->down_proj, product, NULL);
    metal_synchronize();

    // Residual
    Tensor *x3 = tensor_create(res_shape, 2, DTYPE_FP32, 0);
    for (int i = 0; i < batch * n_new * blk->d_model; i++)
        x3->data[i] = x2->data[i] + ff_out->data[i];

    tensor_free(ln1_out);
    tensor_free(attn_out);
    tensor_free(x2);
    tensor_free(ln2_out);
    tensor_free(gate);
    tensor_free(gate_act);
    tensor_free(up);
    tensor_free(product);
    tensor_free(ff_out);

    return x3;
}

static void qwen3_block_collect_params(Qwen3Block *blk, ParamList *pl) {
    param_list_add(pl, blk->input_norm->gamma);
    gqa_collect_params(blk->attn, pl);
    param_list_add(pl, blk->post_attn_norm->gamma);
    param_list_add(pl, blk->gate_proj->weight);
    param_list_add(pl, blk->up_proj->weight);
    param_list_add(pl, blk->down_proj->weight);
}

// ==================================================================
// Qwen3 Model
// ==================================================================

Qwen3Model *qwen3_create(Qwen3Config config) {
    Qwen3Model *model = calloc(1, sizeof(Qwen3Model));
    model->n_layers = config.n_layers;
    model->d_model = config.d_model;
    model->vocab_size = config.vocab_size;
    model->n_q_heads = config.n_q_heads;
    model->n_kv_heads = config.n_kv_heads;
    model->head_dim = config.head_dim;
    model->intermediate_size = config.intermediate_size;
    model->rope_theta = config.rope_theta;

    model->token_emb = embedding_create(config.vocab_size, config.d_model);
    model->blocks = calloc(config.n_layers, sizeof(Qwen3Block *));
    for (int i = 0; i < config.n_layers; i++) {
        model->blocks[i] = qwen3_block_create(config);
    }
    model->final_norm = rms_norm_create(config.d_model, config.rms_norm_eps);
    model->lm_head = linear_create(config.d_model, config.vocab_size, 0);

    return model;
}

void qwen3_free(Qwen3Model *model) {
    if (!model) return;
    embedding_free(model->token_emb);
    for (int i = 0; i < model->n_layers; i++) {
        qwen3_block_free(model->blocks[i]);
    }
    free(model->blocks);
    rms_norm_free(model->final_norm);
    linear_free(model->lm_head);

    if (model->q_loras) {
        for (int i = 0; i < model->n_layers; i++) {
            lora_free(model->q_loras[i]);
        }
        free(model->q_loras);
    }
    if (model->v_loras) {
        for (int i = 0; i < model->n_layers; i++) {
            lora_free(model->v_loras[i]);
        }
        free(model->v_loras);
    }
    free(model);
}

// ==================================================================
// GGUF Weight Loading
// ==================================================================

// Helper: load GGUF tensor data into a Tensor's data buffer
static int load_weight(GGUFFile *f, const char *gguf_name, Tensor *dst) {
    size_t n_elements = 0;
    float *data = gguf_load_tensor_f32(f, gguf_name, &n_elements);
    if (!data) {
        fprintf(stderr, "[Qwen3] Warning: tensor '%s' not found in GGUF\n", gguf_name);
        return -1;
    }
    if (n_elements != dst->size) {
        fprintf(stderr, "[Qwen3] Size mismatch for '%s': GGUF=%zu, model=%zu\n",
                gguf_name, n_elements, dst->size);
        free(data);
        return -1;
    }
    memcpy(dst->data, data, n_elements * sizeof(float));
    free(data);
    return 0;
}

Qwen3Model *qwen3_load_gguf(const char *gguf_path) {
    GGUFFile *f = gguf_open(gguf_path);
    if (!f) return NULL;

    Qwen3Config config = qwen3_4b_config();
    Qwen3Model *model = qwen3_create(config);

    printf("[Qwen3] Loading weights from GGUF...\n");

    // Token embeddings
    load_weight(f, "token_embd.weight", model->token_emb->weight);

    // Transformer blocks
    char name_buf[256];
    for (int i = 0; i < config.n_layers; i++) {
        Qwen3Block *blk = model->blocks[i];

        snprintf(name_buf, sizeof(name_buf), "blk.%d.attn_norm.weight", i);
        load_weight(f, name_buf, blk->input_norm->gamma);

        snprintf(name_buf, sizeof(name_buf), "blk.%d.attn_q.weight", i);
        load_weight(f, name_buf, blk->attn->q_proj->weight);

        snprintf(name_buf, sizeof(name_buf), "blk.%d.attn_k.weight", i);
        load_weight(f, name_buf, blk->attn->k_proj->weight);

        snprintf(name_buf, sizeof(name_buf), "blk.%d.attn_v.weight", i);
        load_weight(f, name_buf, blk->attn->v_proj->weight);

        snprintf(name_buf, sizeof(name_buf), "blk.%d.attn_q_norm.weight", i);
        load_weight(f, name_buf, blk->attn->q_norm->gamma);

        snprintf(name_buf, sizeof(name_buf), "blk.%d.attn_k_norm.weight", i);
        load_weight(f, name_buf, blk->attn->k_norm->gamma);

        snprintf(name_buf, sizeof(name_buf), "blk.%d.attn_output.weight", i);
        load_weight(f, name_buf, blk->attn->o_proj->weight);

        snprintf(name_buf, sizeof(name_buf), "blk.%d.ffn_norm.weight", i);
        load_weight(f, name_buf, blk->post_attn_norm->gamma);

        snprintf(name_buf, sizeof(name_buf), "blk.%d.ffn_gate.weight", i);
        load_weight(f, name_buf, blk->gate_proj->weight);

        snprintf(name_buf, sizeof(name_buf), "blk.%d.ffn_up.weight", i);
        load_weight(f, name_buf, blk->up_proj->weight);

        snprintf(name_buf, sizeof(name_buf), "blk.%d.ffn_down.weight", i);
        load_weight(f, name_buf, blk->down_proj->weight);

        if ((i + 1) % 6 == 0 || i == config.n_layers - 1) {
            printf("[Qwen3]   Loaded blocks 0-%d\n", i);
        }
    }

    // Final norm
    load_weight(f, "output_norm.weight", model->final_norm->gamma);

    // LM head (output projection)
    if (load_weight(f, "output.weight", model->lm_head->weight) != 0) {
        // Some GGUF files tie embeddings and lm_head
        printf("[Qwen3] output.weight not found, tying with token_embd.weight\n");
        memcpy(model->lm_head->weight->data, model->token_emb->weight->data,
               model->lm_head->weight->size * sizeof(float));
    }

    printf("[Qwen3] Weight loading complete.\n");
    gguf_close(f);
    return model;
}

// ==================================================================
// Safetensors Weight Loading (HuggingFace format)
// ==================================================================

static int load_weight_st(SafetensorsFile *sf, const char *name, Tensor *dst) {
    size_t n_elements = 0;
    float *data = safetensors_load_f32(sf, name, &n_elements);
    if (!data) {
        fprintf(stderr, "[Qwen3] Warning: tensor '%s' not found in safetensors\n", name);
        return -1;
    }
    if (n_elements != dst->size) {
        fprintf(stderr, "[Qwen3] Size mismatch for '%s': file=%zu, model=%zu\n",
                name, n_elements, dst->size);
        free(data);
        return -1;
    }
    memcpy(dst->data, data, n_elements * sizeof(float));
    free(data);
    return 0;
}

Qwen3Model *qwen3_load_safetensors(const char *path) {
    SafetensorsFile *sf = safetensors_open(path);
    if (!sf) return NULL;

    Qwen3Config config = qwen3_4b_config();
    Qwen3Model *model = qwen3_create(config);

    printf("[Qwen3] Loading weights from safetensors...\n");

    // Token embeddings
    load_weight_st(sf, "model.embed_tokens.weight", model->token_emb->weight);

    // Transformer blocks
    char name[256];
    for (int i = 0; i < config.n_layers; i++) {
        Qwen3Block *blk = model->blocks[i];

        snprintf(name, sizeof(name), "model.layers.%d.input_layernorm.weight", i);
        load_weight_st(sf, name, blk->input_norm->gamma);

        snprintf(name, sizeof(name), "model.layers.%d.self_attn.q_proj.weight", i);
        load_weight_st(sf, name, blk->attn->q_proj->weight);

        snprintf(name, sizeof(name), "model.layers.%d.self_attn.k_proj.weight", i);
        load_weight_st(sf, name, blk->attn->k_proj->weight);

        snprintf(name, sizeof(name), "model.layers.%d.self_attn.v_proj.weight", i);
        load_weight_st(sf, name, blk->attn->v_proj->weight);

        snprintf(name, sizeof(name), "model.layers.%d.self_attn.q_norm.weight", i);
        load_weight_st(sf, name, blk->attn->q_norm->gamma);

        snprintf(name, sizeof(name), "model.layers.%d.self_attn.k_norm.weight", i);
        load_weight_st(sf, name, blk->attn->k_norm->gamma);

        snprintf(name, sizeof(name), "model.layers.%d.self_attn.o_proj.weight", i);
        load_weight_st(sf, name, blk->attn->o_proj->weight);

        snprintf(name, sizeof(name), "model.layers.%d.post_attention_layernorm.weight", i);
        load_weight_st(sf, name, blk->post_attn_norm->gamma);

        snprintf(name, sizeof(name), "model.layers.%d.mlp.gate_proj.weight", i);
        load_weight_st(sf, name, blk->gate_proj->weight);

        snprintf(name, sizeof(name), "model.layers.%d.mlp.up_proj.weight", i);
        load_weight_st(sf, name, blk->up_proj->weight);

        snprintf(name, sizeof(name), "model.layers.%d.mlp.down_proj.weight", i);
        load_weight_st(sf, name, blk->down_proj->weight);

        if ((i + 1) % 6 == 0 || i == config.n_layers - 1)
            printf("[Qwen3]   Loaded blocks 0-%d\n", i);
    }

    // Final norm
    load_weight_st(sf, "model.norm.weight", model->final_norm->gamma);

    // LM head (may be tied with token embeddings)
    if (load_weight_st(sf, "lm_head.weight", model->lm_head->weight) != 0) {
        printf("[Qwen3] lm_head.weight not found, tying with embed_tokens\n");
        memcpy(model->lm_head->weight->data, model->token_emb->weight->data,
               model->lm_head->weight->size * sizeof(float));
    }

    printf("[Qwen3] Weight loading complete.\n");
    safetensors_close(sf);
    return model;
}

// ==================================================================
// Forward
// ==================================================================

Tensor *qwen3_forward(Qwen3Model *model, Tensor *tokens,
                      int batch, int seq_len, ComputeGraph *g) {
    // Token embeddings -> [batch*seq_len, d_model]
    Tensor *x = embedding_forward(model->token_emb, tokens, g);

    // Transformer blocks
    for (int i = 0; i < model->n_layers; i++) {
        LoRALinear *q_lora = model->q_loras ? model->q_loras[i] : NULL;
        LoRALinear *v_lora = model->v_loras ? model->v_loras[i] : NULL;
        x = qwen3_block_forward(model->blocks[i], x, batch, seq_len, g,
                                 q_lora, v_lora);
    }

    // Final norm
    x = rms_norm_forward(model->final_norm, x, g);

    // LM head
    Tensor *logits = linear_forward(model->lm_head, x, g);

    return logits;
}

// ==================================================================
// Cached Forward (fast inference with KV cache)
// ==================================================================

KVCache *qwen3_create_kv_cache(Qwen3Model *model, int max_seq_len) {
    return kv_cache_create(model->n_layers, model->n_kv_heads,
                           model->head_dim, max_seq_len);
}

Tensor *qwen3_forward_cached(Qwen3Model *model, Tensor *tokens,
                             int batch, int n_new, KVCache *cache) {
    // Token embeddings for new tokens -> [batch*n_new, d_model]
    Tensor *x = embedding_forward(model->token_emb, tokens, NULL);
    metal_synchronize();

    // Transformer blocks with KV cache
    for (int i = 0; i < model->n_layers; i++) {
        Tensor *x_next = qwen3_block_forward_cached(model->blocks[i], x,
                                                      batch, n_new, i, cache);
        tensor_free(x);
        x = x_next;
    }

    // Final norm
    Tensor *normed = rms_norm_forward(model->final_norm, x, NULL);
    metal_synchronize();

    // LM head: only for new positions
    Tensor *logits = linear_forward(model->lm_head, normed, NULL);
    metal_synchronize();

    tensor_free(x);
    tensor_free(normed);

    return logits;
}

// ==================================================================
// LoRA
// ==================================================================

void qwen3_attach_lora(Qwen3Model *model, int rank, float alpha) {
    model->q_loras = calloc(model->n_layers, sizeof(LoRALinear *));
    model->v_loras = calloc(model->n_layers, sizeof(LoRALinear *));

    // Freeze all base weights
    model->token_emb->weight->requires_grad = 0;
    model->final_norm->gamma->requires_grad = 0;
    model->lm_head->weight->requires_grad = 0;

    for (int i = 0; i < model->n_layers; i++) {
        Qwen3Block *blk = model->blocks[i];

        // Freeze all block weights
        blk->input_norm->gamma->requires_grad = 0;
        blk->post_attn_norm->gamma->requires_grad = 0;
        blk->attn->q_proj->weight->requires_grad = 0;
        blk->attn->k_proj->weight->requires_grad = 0;
        blk->attn->v_proj->weight->requires_grad = 0;
        blk->attn->o_proj->weight->requires_grad = 0;
        blk->attn->q_norm->gamma->requires_grad = 0;
        blk->attn->k_norm->gamma->requires_grad = 0;
        blk->gate_proj->weight->requires_grad = 0;
        blk->up_proj->weight->requires_grad = 0;
        blk->down_proj->weight->requires_grad = 0;

        // Attach LoRA to Q and V projections
        model->q_loras[i] = lora_create(blk->attn->q_proj, rank, alpha);
        model->v_loras[i] = lora_create(blk->attn->v_proj, rank, alpha);
    }

    // Count LoRA params
    size_t lora_params = 0;
    for (int i = 0; i < model->n_layers; i++) {
        lora_params += model->q_loras[i]->lora_A->size;
        lora_params += model->q_loras[i]->lora_B->size;
        lora_params += model->v_loras[i]->lora_A->size;
        lora_params += model->v_loras[i]->lora_B->size;
    }
    printf("[Qwen3] LoRA attached: rank=%d, alpha=%.1f, trainable params: %zu (%.2f MB)\n",
           rank, alpha, lora_params, (float)lora_params * 4.0f / (1024 * 1024));
}

ParamList *qwen3_collect_params(Qwen3Model *model) {
    if (model->q_loras) {
        // LoRA mode: only collect LoRA parameters
        int cap = model->n_layers * 4;
        ParamList *pl = param_list_create(cap);
        for (int i = 0; i < model->n_layers; i++) {
            lora_collect_params(model->q_loras[i], pl);
            lora_collect_params(model->v_loras[i], pl);
        }
        return pl;
    }

    // Full mode: collect all parameters
    // Per block: input_norm(1) + attn(q,k,v,o,q_norm,k_norm=6) + post_attn_norm(1) + ffn(gate,up,down=3) = 11
    int cap = 1 + model->n_layers * 11 + 2;
    ParamList *pl = param_list_create(cap);

    param_list_add(pl, model->token_emb->weight);
    for (int i = 0; i < model->n_layers; i++) {
        qwen3_block_collect_params(model->blocks[i], pl);
    }
    param_list_add(pl, model->final_norm->gamma);
    param_list_add(pl, model->lm_head->weight);

    return pl;
}

size_t qwen3_param_count(Qwen3Model *model) {
    ParamList *pl = qwen3_collect_params(model);
    size_t total = 0;
    for (int i = 0; i < pl->n_params; i++) {
        total += pl->params[i]->size;
    }
    param_list_free(pl);
    return total;
}

// ==================================================================
// LoRA Save/Load
// ==================================================================

void qwen3_save_lora(Qwen3Model *model, const char *path) {
    if (!model->q_loras) {
        fprintf(stderr, "[Qwen3] No LoRA adapters to save\n");
        return;
    }

    FILE *fp = fopen(path, "wb");
    if (!fp) {
        fprintf(stderr, "[Qwen3] Cannot open %s for writing\n", path);
        return;
    }

    // Header: magic, n_layers, rank
    uint32_t magic = 0x4C4F5241; // "LORA"
    int32_t n_layers = model->n_layers;
    int32_t rank = model->q_loras[0]->rank;
    fwrite(&magic, 4, 1, fp);
    fwrite(&n_layers, 4, 1, fp);
    fwrite(&rank, 4, 1, fp);

    // Save all LoRA A and B matrices
    for (int i = 0; i < model->n_layers; i++) {
        fwrite(model->q_loras[i]->lora_A->data, sizeof(float),
               model->q_loras[i]->lora_A->size, fp);
        fwrite(model->q_loras[i]->lora_B->data, sizeof(float),
               model->q_loras[i]->lora_B->size, fp);
        fwrite(model->v_loras[i]->lora_A->data, sizeof(float),
               model->v_loras[i]->lora_A->size, fp);
        fwrite(model->v_loras[i]->lora_B->data, sizeof(float),
               model->v_loras[i]->lora_B->size, fp);
    }

    fclose(fp);
    printf("[Qwen3] LoRA adapters saved to %s\n", path);
}

int qwen3_load_lora(Qwen3Model *model, const char *path) {
    if (!model->q_loras) {
        fprintf(stderr, "[Qwen3] LoRA not attached, cannot load\n");
        return -1;
    }

    FILE *fp = fopen(path, "rb");
    if (!fp) {
        fprintf(stderr, "[Qwen3] Cannot open %s for reading\n", path);
        return -1;
    }

    uint32_t magic;
    int32_t n_layers, rank;
    fread(&magic, 4, 1, fp);
    fread(&n_layers, 4, 1, fp);
    fread(&rank, 4, 1, fp);

    if (magic != 0x4C4F5241) {
        fprintf(stderr, "[Qwen3] Invalid LoRA file magic\n");
        fclose(fp);
        return -1;
    }
    if (n_layers != model->n_layers || rank != model->q_loras[0]->rank) {
        fprintf(stderr, "[Qwen3] LoRA config mismatch: file(layers=%d,rank=%d) vs model(layers=%d,rank=%d)\n",
                n_layers, rank, model->n_layers, model->q_loras[0]->rank);
        fclose(fp);
        return -1;
    }

    for (int i = 0; i < model->n_layers; i++) {
        fread(model->q_loras[i]->lora_A->data, sizeof(float),
              model->q_loras[i]->lora_A->size, fp);
        fread(model->q_loras[i]->lora_B->data, sizeof(float),
              model->q_loras[i]->lora_B->size, fp);
        fread(model->v_loras[i]->lora_A->data, sizeof(float),
              model->v_loras[i]->lora_A->size, fp);
        fread(model->v_loras[i]->lora_B->data, sizeof(float),
              model->v_loras[i]->lora_B->size, fp);
    }

    fclose(fp);
    printf("[Qwen3] LoRA adapters loaded from %s\n", path);
    return 0;
}
