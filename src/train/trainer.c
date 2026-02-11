#include "trainer.h"
#include "../core/metal_backend.h"
#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <math.h>
#include <time.h>
#include <sys/stat.h>

Trainer *trainer_create(TrainConfig config, const Tokenizer *tok) {
    Trainer *t = calloc(1, sizeof(Trainer));

    // Create model
    t->model = gpt_create(config.model_config);
    t->params = gpt_collect_params(t->model);
    t->opt_config = config.opt_config;
    t->optimizer = adamw_create(t->params, config.opt_config);

    // Create data loaders
    if (config.train_data) {
        t->train_loader = dataloader_create(config.train_data, tok,
                                            config.batch_size, config.seq_len);
    }
    if (config.val_data) {
        t->val_loader = dataloader_create(config.val_data, tok,
                                          config.batch_size, config.seq_len);
    }

    t->batch_size = config.batch_size;
    t->seq_len = config.seq_len;
    t->grad_accum_steps = config.grad_accum_steps > 0 ? config.grad_accum_steps : 1;
    t->max_steps = config.max_steps;
    t->log_interval = config.log_interval > 0 ? config.log_interval : 10;
    t->eval_interval = config.eval_interval > 0 ? config.eval_interval : 100;
    t->save_interval = config.save_interval > 0 ? config.save_interval : 1000;
    if (config.checkpoint_dir) {
        strncpy(t->checkpoint_dir, config.checkpoint_dir, 255);
    } else {
        strncpy(t->checkpoint_dir, "checkpoints", 255);
    }

    t->graph = graph_create();

    printf("[Trainer] Model: %zu parameters (%.2f M)\n",
           gpt_param_count(t->model),
           (float)gpt_param_count(t->model) / 1e6f);
    printf("[Trainer] Batch size: %d, Seq len: %d, Grad accum: %d\n",
           t->batch_size, t->seq_len, t->grad_accum_steps);
    printf("[Trainer] Effective batch size: %d\n",
           t->batch_size * t->grad_accum_steps);

    return t;
}

void trainer_free(Trainer *t) {
    if (!t) return;
    gpt_free(t->model);
    param_list_free(t->params);
    adamw_free(t->optimizer);
    if (t->train_loader) dataloader_free(t->train_loader);
    if (t->val_loader) dataloader_free(t->val_loader);
    graph_destroy(t->graph);
    free(t);
}

float trainer_step(Trainer *t) {
    float total_loss = 0.0f;

    // Gradient accumulation loop
    for (int acc = 0; acc < t->grad_accum_steps; acc++) {
        // Get batch
        int BN = t->batch_size * t->seq_len;
        int tok_shape[] = {BN};

        // Create uint32 tensors for token indices
        Tensor *input  = tensor_create(tok_shape, 1, DTYPE_UINT32, 0);
        Tensor *target = tensor_create(tok_shape, 1, DTYPE_UINT32, 0);

        if (t->train_loader) {
            // Dataloader writes to float tensors, so load into float then convert
            int float_shape[] = {BN};
            Tensor *input_f  = tensor_create(float_shape, 1, DTYPE_FP32, 0);
            Tensor *target_f = tensor_create(float_shape, 1, DTYPE_FP32, 0);
            int ret = dataloader_next_batch(t->train_loader, input_f, target_f);
            if (ret < 0) {
                dataloader_reset(t->train_loader);
                dataloader_shuffle(t->train_loader);
                dataloader_next_batch(t->train_loader, input_f, target_f);
            }
            // Convert float -> uint32
            for (int i = 0; i < BN; i++) {
                input->data_u32[i]  = (uint32_t)input_f->data[i];
                target->data_u32[i] = (uint32_t)target_f->data[i];
            }
            tensor_free(input_f);
            tensor_free(target_f);
        } else {
            // Dummy data for testing
            for (int i = 0; i < BN; i++) {
                input->data_u32[i]  = (uint32_t)(rand() % t->model->vocab_size);
                target->data_u32[i] = (uint32_t)(rand() % t->model->vocab_size);
            }
        }

        // Reset graph (frees previous step's intermediates)
        graph_reset(t->graph);

        // Forward pass
        Tensor *logits = gpt_forward(t->model, input, t->batch_size, t->seq_len, t->graph);

        // Compute loss: softmax cross-entropy
        int loss_shape[] = {BN};
        Tensor *losses = tensor_zeros(loss_shape, 1, DTYPE_FP32, 0);
        int dlogits_shape[] = {BN, t->model->vocab_size};
        Tensor *dlogits = tensor_zeros(dlogits_shape, 2, DTYPE_FP32, 0);

        if (metal_is_initialized() && logits->metal_buf) {
            // Target is already uint32 — pass directly to Metal kernel
            metal_softmax_ce_loss(logits->metal_buf, target->metal_buf,
                                  losses->metal_buf, dlogits->metal_buf,
                                  BN, t->model->vocab_size);
            metal_synchronize();
        } else {
            // CPU softmax + cross-entropy
            int V = t->model->vocab_size;
            for (int b = 0; b < BN; b++) {
                float *log_row = &logits->data[b * V];
                int tgt_id = (int)target->data_u32[b];

                // Max for numerical stability
                float max_val = log_row[0];
                for (int v = 1; v < V; v++)
                    if (log_row[v] > max_val) max_val = log_row[v];

                // Sum exp
                float sum_exp = 0;
                for (int v = 0; v < V; v++) sum_exp += expf(log_row[v] - max_val);
                float log_sum = logf(sum_exp) + max_val;

                losses->data[b] = log_sum - log_row[tgt_id];

                // Gradient: softmax - one_hot
                for (int v = 0; v < V; v++) {
                    float prob = expf(log_row[v] - log_sum);
                    dlogits->data[b * V + v] = (prob - (v == tgt_id ? 1.0f : 0.0f)) / (float)BN;
                }
            }
        }

        // Mean loss
        float batch_loss = 0;
        for (int i = 0; i < BN; i++) batch_loss += losses->data[i];
        batch_loss /= (float)BN;
        total_loss += batch_loss;

        // Set logits gradient for backward pass
        // logits->grad is allocated by tensor_create(requires_grad=1)
        if (logits->grad) {
            float scale = 1.0f / (float)t->grad_accum_steps;
            for (size_t i = 0; i < dlogits->size; i++) {
                logits->grad->data[i] = dlogits->data[i] * scale;
            }
        }
        // graph_backward no longer auto-seeds — we set grad above
        graph_backward(t->graph, logits);

        tensor_free(input);
        tensor_free(target);
        tensor_free(losses);
        tensor_free(dlogits);
    }

    total_loss /= (float)t->grad_accum_steps;

    // Gradient clipping
    grad_clip_norm(t->params, t->opt_config.max_grad_norm);

    // Optimizer step
    adamw_step(t->optimizer, t->params);

    // Zero gradients
    adamw_zero_grad(t->params);

    t->step++;
    t->running_loss = 0.95f * t->running_loss + 0.05f * total_loss;

    return total_loss;
}

float trainer_evaluate(Trainer *t) {
    if (!t->val_loader) return -1.0f;
    dataloader_reset(t->val_loader);

    float total_loss = 0;
    int n_batches = 0;
    int max_eval_batches = 20;

    int BN = t->batch_size * t->seq_len;
    int tok_shape[] = {BN};
    Tensor *input_f  = tensor_create(tok_shape, 1, DTYPE_FP32, 0);
    Tensor *target_f = tensor_create(tok_shape, 1, DTYPE_FP32, 0);

    while (n_batches < max_eval_batches) {
        if (dataloader_next_batch(t->val_loader, input_f, target_f) < 0) break;

        // Convert float indices to uint32
        Tensor *input_u = tensor_create(tok_shape, 1, DTYPE_UINT32, 0);
        Tensor *target_u = tensor_create(tok_shape, 1, DTYPE_UINT32, 0);
        for (int i = 0; i < BN; i++) {
            input_u->data_u32[i]  = (uint32_t)input_f->data[i];
            target_u->data_u32[i] = (uint32_t)target_f->data[i];
        }

        graph_reset(t->graph);
        Tensor *logits = gpt_forward(t->model, input_u, t->batch_size, t->seq_len, t->graph);

        // Compute loss (CPU)
        int V = t->model->vocab_size;
        float batch_loss = 0;
        for (int b = 0; b < BN; b++) {
            float *row = &logits->data[b * V];
            int tgt = (int)target_u->data_u32[b];
            float max_val = row[0];
            for (int v = 1; v < V; v++) if (row[v] > max_val) max_val = row[v];
            float sum_exp = 0;
            for (int v = 0; v < V; v++) sum_exp += expf(row[v] - max_val);
            batch_loss += logf(sum_exp) + max_val - row[tgt];
        }
        total_loss += batch_loss / (float)BN;
        n_batches++;

        tensor_free(input_u);
        tensor_free(target_u);
    }

    tensor_free(input_f);
    tensor_free(target_f);

    return n_batches > 0 ? total_loss / (float)n_batches : -1.0f;
}

void trainer_train(Trainer *t) {
    printf("\n=== Training Start ===\n");
    printf("Max steps: %d\n\n", t->max_steps);

    struct timespec start, now;
    clock_gettime(CLOCK_MONOTONIC, &start);

    for (int s = 0; s < t->max_steps; s++) {
        float loss = trainer_step(t);

        if (s % t->log_interval == 0) {
            clock_gettime(CLOCK_MONOTONIC, &now);
            double elapsed = (double)(now.tv_sec - start.tv_sec) +
                             (double)(now.tv_nsec - start.tv_nsec) / 1e9;
            float lr = adamw_get_lr(t->optimizer);
            printf("step %5d | loss %.4f | lr %.2e | %.1f tok/s\n",
                   s, loss, lr,
                   (double)(s + 1) * t->batch_size * t->seq_len / elapsed);
        }

        if (t->val_loader && s > 0 && s % t->eval_interval == 0) {
            float val_loss = trainer_evaluate(t);
            printf("  [eval] val_loss = %.4f\n", val_loss);
        }

        if (s > 0 && s % t->save_interval == 0) {
            char path[512];
            snprintf(path, sizeof(path), "%s/step_%06d.bin", t->checkpoint_dir, s);
            trainer_save_checkpoint(t, path);
        }
    }

    clock_gettime(CLOCK_MONOTONIC, &now);
    double total = (double)(now.tv_sec - start.tv_sec) +
                   (double)(now.tv_nsec - start.tv_nsec) / 1e9;
    printf("\n=== Training Complete ===\n");
    printf("Total time: %.1f seconds\n", total);
    printf("Final loss: %.4f\n", t->running_loss);
}

void trainer_save_checkpoint(Trainer *t, const char *path) {
    // Ensure directory exists
    char dir[512];
    strncpy(dir, path, sizeof(dir) - 1);
    char *last_slash = strrchr(dir, '/');
    if (last_slash) {
        *last_slash = '\0';
        mkdir(dir, 0755);
    }

    FILE *f = fopen(path, "wb");
    if (!f) {
        fprintf(stderr, "[Trainer] Cannot save checkpoint: %s\n", path);
        return;
    }

    // Header
    int header[4] = {0x4C4C4D31, t->step, t->params->n_params, 0}; // "LLM1"
    fwrite(header, sizeof(int), 4, f);

    // Save parameters
    for (int i = 0; i < t->params->n_params; i++) {
        Tensor *p = t->params->params[i];
        fwrite(&p->ndim, sizeof(int), 1, f);
        fwrite(p->shape, sizeof(int), p->ndim, f);
        fwrite(p->data, sizeof(float), p->size, f);
    }

    // Save optimizer state
    for (int i = 0; i < t->params->n_params; i++) {
        fwrite(t->optimizer->m[i]->data, sizeof(float), t->optimizer->m[i]->size, f);
        fwrite(t->optimizer->v[i]->data, sizeof(float), t->optimizer->v[i]->size, f);
    }

    fclose(f);
    printf("[Trainer] Checkpoint saved: %s\n", path);
}

int trainer_load_checkpoint(Trainer *t, const char *path) {
    FILE *f = fopen(path, "rb");
    if (!f) return -1;

    int header[4];
    fread(header, sizeof(int), 4, f);
    if (header[0] != 0x4C4C4D31) {
        fprintf(stderr, "[Trainer] Invalid checkpoint format\n");
        fclose(f);
        return -1;
    }

    t->step = header[1];
    t->optimizer->step = t->step;
    int n_params = header[2];
    if (n_params != t->params->n_params) {
        fprintf(stderr, "[Trainer] Parameter count mismatch: %d vs %d\n",
                n_params, t->params->n_params);
        fclose(f);
        return -1;
    }

    // Load parameters
    for (int i = 0; i < n_params; i++) {
        int ndim;
        int shape[TENSOR_MAX_DIMS];
        fread(&ndim, sizeof(int), 1, f);
        fread(shape, sizeof(int), ndim, f);
        fread(t->params->params[i]->data, sizeof(float), t->params->params[i]->size, f);
    }

    // Load optimizer state
    for (int i = 0; i < n_params; i++) {
        fread(t->optimizer->m[i]->data, sizeof(float), t->optimizer->m[i]->size, f);
        fread(t->optimizer->v[i]->data, sizeof(float), t->optimizer->v[i]->size, f);
    }

    fclose(f);
    printf("[Trainer] Checkpoint loaded: %s (step %d)\n", path, t->step);
    return 0;
}
