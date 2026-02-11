#include "gemma3.h"
#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <math.h>

// ==================================================================
// Gemma3 Configs
// ==================================================================

Gemma3Config gemma3_1b_config(void) {
    return (Gemma3Config){
        .n_layers = 26,
        .d_model = 1152,
        .n_q_heads = 4,
        .n_kv_heads = 1,
        .head_dim = 256,
        .intermediate_size = 6912,
        .vocab_size = 262144,
        .sliding_window = 512,
        .local_rope_theta = 10000.0f,
        .global_rope_theta = 1000000.0f,
        .rms_norm_eps = 1e-6f,
    };
}

Gemma3Config gemma3_4b_config(void) {
    return (Gemma3Config){
        .n_layers = 34,
        .d_model = 2560,
        .n_q_heads = 8,
        .n_kv_heads = 4,
        .head_dim = 256,
        .intermediate_size = 10240,
        .vocab_size = 262144,
        .sliding_window = 1024,
        .local_rope_theta = 10000.0f,
        .global_rope_theta = 1000000.0f,
        .rms_norm_eps = 1e-6f,
    };
}

// ==================================================================
// Gemma3 Block
// ==================================================================

// Gemma3 uses 5:1 hybrid attention pattern:
// layer 0 = global, layers 1-5 = local(1,2,3,4,5), layer 6 = global, ...
// i.e. every 6th layer (0,6,12,...) is global, the rest are local/sliding.
static int is_sliding_layer(int layer_idx) {
    return (layer_idx % 6) != 0;
}

static Gemma3Block *gemma3_block_create(Gemma3Config config, int layer_idx) {
    Gemma3Block *blk = calloc(1, sizeof(Gemma3Block));
    blk->d_model = config.d_model;
    blk->is_sliding = is_sliding_layer(layer_idx);

    float rope_theta = blk->is_sliding ? config.local_rope_theta : config.global_rope_theta;

    blk->input_norm = rms_norm_create(config.d_model, config.rms_norm_eps);
    blk->attn = gqa_create(config.d_model, config.n_q_heads, config.n_kv_heads,
                            config.head_dim, rope_theta);
    blk->post_attn_norm = rms_norm_create(config.d_model, config.rms_norm_eps);
    blk->pre_ff_norm = rms_norm_create(config.d_model, config.rms_norm_eps);
    blk->gate_proj = linear_create(config.d_model, config.intermediate_size, 0);
    blk->up_proj = linear_create(config.d_model, config.intermediate_size, 0);
    blk->down_proj = linear_create(config.intermediate_size, config.d_model, 0);
    blk->post_ff_norm = rms_norm_create(config.d_model, config.rms_norm_eps);
    return blk;
}

static void gemma3_block_free(Gemma3Block *blk) {
    if (!blk) return;
    rms_norm_free(blk->input_norm);
    gqa_free(blk->attn);
    rms_norm_free(blk->post_attn_norm);
    rms_norm_free(blk->pre_ff_norm);
    linear_free(blk->gate_proj);
    linear_free(blk->up_proj);
    linear_free(blk->down_proj);
    rms_norm_free(blk->post_ff_norm);
    free(blk);
}

// ==================================================================
// Gemma3 Model
// ==================================================================

Gemma3Model *gemma3_create(Gemma3Config config) {
    Gemma3Model *model = calloc(1, sizeof(Gemma3Model));
    model->n_layers = config.n_layers;
    model->d_model = config.d_model;
    model->vocab_size = config.vocab_size;
    model->n_q_heads = config.n_q_heads;
    model->n_kv_heads = config.n_kv_heads;
    model->head_dim = config.head_dim;
    model->intermediate_size = config.intermediate_size;
    model->sliding_window = config.sliding_window;
    model->local_rope_theta = config.local_rope_theta;
    model->global_rope_theta = config.global_rope_theta;

    model->token_emb = embedding_create(config.vocab_size, config.d_model);
    model->blocks = calloc(config.n_layers, sizeof(Gemma3Block *));
    for (int i = 0; i < config.n_layers; i++) {
        model->blocks[i] = gemma3_block_create(config, i);
    }
    model->final_norm = rms_norm_create(config.d_model, config.rms_norm_eps);

    return model;
}

void gemma3_free(Gemma3Model *model) {
    if (!model) return;
    embedding_free(model->token_emb);
    for (int i = 0; i < model->n_layers; i++) {
        gemma3_block_free(model->blocks[i]);
    }
    free(model->blocks);
    rms_norm_free(model->final_norm);
    // No lm_head to free — it's tied with token_emb

    if (model->q_loras) {
        for (int i = 0; i < model->n_layers; i++)
            lora_free(model->q_loras[i]);
        free(model->q_loras);
    }
    if (model->v_loras) {
        for (int i = 0; i < model->n_layers; i++)
            lora_free(model->v_loras[i]);
        free(model->v_loras);
    }
    free(model);
}

size_t gemma3_param_count(Gemma3Model *model) {
    size_t total = 0;
    // token_emb
    total += model->token_emb->weight->size;
    // blocks
    for (int i = 0; i < model->n_layers; i++) {
        Gemma3Block *blk = model->blocks[i];
        total += blk->input_norm->gamma->size;
        total += blk->attn->q_proj->weight->size;
        total += blk->attn->k_proj->weight->size;
        total += blk->attn->v_proj->weight->size;
        total += blk->attn->o_proj->weight->size;
        total += blk->attn->q_norm->gamma->size;
        total += blk->attn->k_norm->gamma->size;
        total += blk->post_attn_norm->gamma->size;
        total += blk->pre_ff_norm->gamma->size;
        total += blk->gate_proj->weight->size;
        total += blk->up_proj->weight->size;
        total += blk->down_proj->weight->size;
        total += blk->post_ff_norm->gamma->size;
    }
    total += model->final_norm->gamma->size;
    // lm_head is tied → not counted separately
    return total;
}

// ==================================================================
// Safetensors Weight Loading (HuggingFace format)
// ==================================================================

static int load_weight_st(SafetensorsFile *sf, const char *name, Tensor *dst) {
    size_t n_elements = 0;
    float *data = safetensors_load_f32(sf, name, &n_elements);
    if (!data) {
        fprintf(stderr, "[Gemma3] Warning: tensor '%s' not found in safetensors\n", name);
        return -1;
    }
    if (n_elements != dst->size) {
        fprintf(stderr, "[Gemma3] Size mismatch for '%s': file=%zu, model=%zu\n",
                name, n_elements, dst->size);
        free(data);
        return -1;
    }
    memcpy(dst->data, data, n_elements * sizeof(float));
    free(data);
    return 0;
}

Gemma3Model *gemma3_load_safetensors(const char *path) {
    SafetensorsFile *sf = safetensors_open(path);
    if (!sf) return NULL;

    // TODO: Auto-detect config from safetensors metadata / config.json
    // For now, try to detect 1B vs 4B by checking tensor sizes
    Gemma3Config config = gemma3_4b_config();

    // Check if this is a 1B model by looking at embed_tokens shape
    STTensor *emb = safetensors_find(sf, "model.embed_tokens.weight");
    if (emb && emb->ndim == 2 && emb->shape[1] == 1152) {
        config = gemma3_1b_config();
        printf("[Gemma3] Detected 1B config (d_model=1152)\n");
    } else {
        printf("[Gemma3] Using 4B config (d_model=2560)\n");
    }

    Gemma3Model *model = gemma3_create(config);

    printf("[Gemma3] Loading weights from safetensors...\n");

    // Token embeddings
    load_weight_st(sf, "model.embed_tokens.weight", model->token_emb->weight);

    // Transformer blocks
    char name[256];
    for (int i = 0; i < config.n_layers; i++) {
        Gemma3Block *blk = model->blocks[i];

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

        snprintf(name, sizeof(name), "model.layers.%d.pre_feedforward_layernorm.weight", i);
        load_weight_st(sf, name, blk->pre_ff_norm->gamma);

        snprintf(name, sizeof(name), "model.layers.%d.mlp.gate_proj.weight", i);
        load_weight_st(sf, name, blk->gate_proj->weight);

        snprintf(name, sizeof(name), "model.layers.%d.mlp.up_proj.weight", i);
        load_weight_st(sf, name, blk->up_proj->weight);

        snprintf(name, sizeof(name), "model.layers.%d.mlp.down_proj.weight", i);
        load_weight_st(sf, name, blk->down_proj->weight);

        snprintf(name, sizeof(name), "model.layers.%d.post_feedforward_layernorm.weight", i);
        load_weight_st(sf, name, blk->post_ff_norm->gamma);

        if ((i + 1) % 6 == 0 || i == config.n_layers - 1)
            printf("[Gemma3]   Loaded blocks 0-%d\n", i);
    }

    // Final norm
    load_weight_st(sf, "model.norm.weight", model->final_norm->gamma);

    // Gemma3 RMSNorm uses (1 + weight) * norm(x), so add 1.0 to all norm weights
    // to convert to standard RMSNorm convention: weight * norm(x)
    printf("[Gemma3] Adjusting norm weights (+1.0 for Gemma3 convention)...\n");
    for (int i = 0; i < (int)model->final_norm->gamma->size; i++)
        model->final_norm->gamma->data[i] += 1.0f;
    for (int i = 0; i < config.n_layers; i++) {
        Gemma3Block *blk = model->blocks[i];
        for (int j = 0; j < (int)blk->input_norm->gamma->size; j++)
            blk->input_norm->gamma->data[j] += 1.0f;
        for (int j = 0; j < (int)blk->post_attn_norm->gamma->size; j++)
            blk->post_attn_norm->gamma->data[j] += 1.0f;
        for (int j = 0; j < (int)blk->pre_ff_norm->gamma->size; j++)
            blk->pre_ff_norm->gamma->data[j] += 1.0f;
        for (int j = 0; j < (int)blk->post_ff_norm->gamma->size; j++)
            blk->post_ff_norm->gamma->data[j] += 1.0f;
        for (int j = 0; j < (int)blk->attn->q_norm->gamma->size; j++)
            blk->attn->q_norm->gamma->data[j] += 1.0f;
        for (int j = 0; j < (int)blk->attn->k_norm->gamma->size; j++)
            blk->attn->k_norm->gamma->data[j] += 1.0f;
    }

    // lm_head is tied with embed_tokens — no separate weight to load
    printf("[Gemma3] lm_head tied with embed_tokens (weight sharing)\n");

    printf("[Gemma3] Weight loading complete.\n");
    safetensors_close(sf);
    return model;
}

// ==================================================================
// LoRA utilities
// ==================================================================

void gemma3_attach_lora(Gemma3Model *model, int rank, float alpha) {
    model->q_loras = calloc(model->n_layers, sizeof(LoRALinear *));
    model->v_loras = calloc(model->n_layers, sizeof(LoRALinear *));

    // Freeze all base weights
    model->token_emb->weight->requires_grad = 0;
    model->final_norm->gamma->requires_grad = 0;
    // lm_head is tied with token_emb, no separate weight

    for (int i = 0; i < model->n_layers; i++) {
        Gemma3Block *blk = model->blocks[i];

        blk->input_norm->gamma->requires_grad = 0;
        blk->post_attn_norm->gamma->requires_grad = 0;
        blk->pre_ff_norm->gamma->requires_grad = 0;
        blk->post_ff_norm->gamma->requires_grad = 0;
        blk->attn->q_proj->weight->requires_grad = 0;
        blk->attn->k_proj->weight->requires_grad = 0;
        blk->attn->v_proj->weight->requires_grad = 0;
        blk->attn->o_proj->weight->requires_grad = 0;
        blk->attn->q_norm->gamma->requires_grad = 0;
        blk->attn->k_norm->gamma->requires_grad = 0;
        blk->gate_proj->weight->requires_grad = 0;
        blk->up_proj->weight->requires_grad = 0;
        blk->down_proj->weight->requires_grad = 0;

        model->q_loras[i] = lora_create(blk->attn->q_proj, rank, alpha);
        model->v_loras[i] = lora_create(blk->attn->v_proj, rank, alpha);
    }

    size_t lora_params = 0;
    for (int i = 0; i < model->n_layers; i++) {
        lora_params += model->q_loras[i]->lora_A->size;
        lora_params += model->q_loras[i]->lora_B->size;
        lora_params += model->v_loras[i]->lora_A->size;
        lora_params += model->v_loras[i]->lora_B->size;
    }
    printf("[Gemma3] LoRA attached: rank=%d, alpha=%.1f, trainable params: %zu (%.2f MB)\n",
           rank, alpha, lora_params, (float)lora_params * 4.0f / (1024 * 1024));
}

void gemma3_save_lora(Gemma3Model *model, const char *path) {
    if (!model->q_loras) {
        fprintf(stderr, "[Gemma3] No LoRA adapters to save\n");
        return;
    }

    FILE *fp = fopen(path, "wb");
    if (!fp) {
        fprintf(stderr, "[Gemma3] Cannot open %s for writing\n", path);
        return;
    }

    uint32_t magic = 0x4C4F5241; // "LORA"
    int32_t n_layers = model->n_layers;
    int32_t rank = model->q_loras[0]->rank;
    fwrite(&magic, 4, 1, fp);
    fwrite(&n_layers, 4, 1, fp);
    fwrite(&rank, 4, 1, fp);

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
    printf("[Gemma3] LoRA adapters saved to %s\n", path);
}

int gemma3_load_lora(Gemma3Model *model, const char *path) {
    if (!model->q_loras) {
        fprintf(stderr, "[Gemma3] LoRA not attached, cannot load\n");
        return -1;
    }

    FILE *fp = fopen(path, "rb");
    if (!fp) {
        fprintf(stderr, "[Gemma3] Cannot open %s for reading\n", path);
        return -1;
    }

    uint32_t magic;
    int32_t n_layers, rank;
    fread(&magic, 4, 1, fp);
    fread(&n_layers, 4, 1, fp);
    fread(&rank, 4, 1, fp);

    if (magic != 0x4C4F5241) {
        fprintf(stderr, "[Gemma3] Invalid LoRA file magic\n");
        fclose(fp);
        return -1;
    }
    if (n_layers != model->n_layers || rank != model->q_loras[0]->rank) {
        fprintf(stderr, "[Gemma3] LoRA config mismatch: file(layers=%d,rank=%d) vs model(layers=%d,rank=%d)\n",
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
    printf("[Gemma3] LoRA adapters loaded from %s\n", path);
    return 0;
}
