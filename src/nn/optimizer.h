#ifndef OPTIMIZER_H
#define OPTIMIZER_H

#include "../core/tensor.h"
#include "layers.h"

typedef struct {
    float lr;
    float beta1;
    float beta2;
    float eps;
    float weight_decay;
    float max_grad_norm;

    // LR schedule
    int warmup_steps;
    int total_steps;
    float min_lr;
    float max_lr;

    // Dynamic loss scaling (for mixed precision)
    float loss_scale;
    int loss_scale_window;
    int loss_scale_counter;
    int loss_scale_growth_interval;
} AdamWConfig;

typedef struct {
    Tensor **m;         // first moment for each param
    Tensor **v;         // second moment for each param
    int n_params;
    int step;
    AdamWConfig config;
} AdamW;

// Create optimizer for a parameter list
AdamW *adamw_create(ParamList *params, AdamWConfig config);
void   adamw_free(AdamW *opt);

// Perform one optimization step
void   adamw_step(AdamW *opt, ParamList *params);

// Zero gradients for all parameters
void   adamw_zero_grad(ParamList *params);

// Get current learning rate (with schedule)
float  adamw_get_lr(AdamW *opt);

// Loss scaling for mixed precision
float  adamw_get_loss_scale(AdamW *opt);
void   adamw_update_loss_scale(AdamW *opt, int overflow_detected);

// Gradient clipping (max norm)
float  grad_clip_norm(ParamList *params, float max_norm);

// Default config
AdamWConfig adamw_default_config(void);

#endif // OPTIMIZER_H
