#ifndef TRAINER_H
#define TRAINER_H

#include "../nn/transformer.h"
#include "../nn/optimizer.h"
#include "../data/dataloader.h"

typedef struct {
    // Model
    GPTModel  *model;
    ParamList *params;

    // Optimizer
    AdamW     *optimizer;
    AdamWConfig opt_config;

    // Data
    DataLoader *train_loader;
    DataLoader *val_loader;

    // Training config
    int    batch_size;
    int    seq_len;
    int    grad_accum_steps;
    int    max_steps;
    int    log_interval;
    int    eval_interval;
    int    save_interval;
    char   checkpoint_dir[256];

    // State
    int    step;
    float  running_loss;
    ComputeGraph *graph;
} Trainer;

typedef struct {
    GPTConfig    model_config;
    AdamWConfig  opt_config;
    int          batch_size;
    int          seq_len;
    int          grad_accum_steps;
    int          max_steps;
    int          log_interval;
    int          eval_interval;
    int          save_interval;
    const char  *checkpoint_dir;
    const char  *train_data;
    const char  *val_data;
} TrainConfig;

Trainer *trainer_create(TrainConfig config, const Tokenizer *tok);
void     trainer_free(Trainer *t);

// Run training loop
void     trainer_train(Trainer *t);

// Single training step (returns loss)
float    trainer_step(Trainer *t);

// Evaluation
float    trainer_evaluate(Trainer *t);

// Checkpoint save/load
void     trainer_save_checkpoint(Trainer *t, const char *path);
int      trainer_load_checkpoint(Trainer *t, const char *path);

#endif // TRAINER_H
