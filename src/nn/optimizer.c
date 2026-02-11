#include "optimizer.h"
#include "../core/metal_backend.h"
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <stdio.h>

AdamWConfig adamw_default_config(void) {
    return (AdamWConfig){
        .lr = 3e-4f,
        .beta1 = 0.9f,
        .beta2 = 0.999f,
        .eps = 1e-8f,
        .weight_decay = 0.01f,
        .max_grad_norm = 1.0f,
        .warmup_steps = 100,
        .total_steps = 10000,
        .min_lr = 1e-5f,
        .max_lr = 3e-4f,
        .loss_scale = 1.0f,
        .loss_scale_window = 0,
        .loss_scale_counter = 0,
        .loss_scale_growth_interval = 2000,
    };
}

AdamW *adamw_create(ParamList *params, AdamWConfig config) {
    AdamW *opt = calloc(1, sizeof(AdamW));
    opt->n_params = params->n_params;
    opt->config = config;
    opt->step = 0;

    opt->m = calloc(params->n_params, sizeof(Tensor *));
    opt->v = calloc(params->n_params, sizeof(Tensor *));

    for (int i = 0; i < params->n_params; i++) {
        Tensor *p = params->params[i];
        opt->m[i] = tensor_zeros(p->shape, p->ndim, DTYPE_FP32, 0);
        opt->v[i] = tensor_zeros(p->shape, p->ndim, DTYPE_FP32, 0);
    }
    return opt;
}

void adamw_free(AdamW *opt) {
    if (!opt) return;
    for (int i = 0; i < opt->n_params; i++) {
        tensor_free(opt->m[i]);
        tensor_free(opt->v[i]);
    }
    free(opt->m);
    free(opt->v);
    free(opt);
}

float adamw_get_lr(AdamW *opt) {
    int step = opt->step;
    AdamWConfig *c = &opt->config;

    if (step < c->warmup_steps) {
        // Linear warmup
        return c->max_lr * ((float)step / (float)c->warmup_steps);
    }
    // Cosine annealing
    int denom = c->total_steps - c->warmup_steps;
    if (denom <= 0) return c->max_lr;
    float progress = (float)(step - c->warmup_steps) / (float)denom;
    if (progress > 1.0f) progress = 1.0f;
    return c->min_lr + 0.5f * (c->max_lr - c->min_lr) * (1.0f + cosf((float)M_PI * progress));
}

float adamw_get_loss_scale(AdamW *opt) {
    return opt->config.loss_scale;
}

void adamw_update_loss_scale(AdamW *opt, int overflow_detected) {
    AdamWConfig *c = &opt->config;
    if (overflow_detected) {
        c->loss_scale *= 0.5f;
        c->loss_scale_counter = 0;
        if (c->loss_scale < 1.0f) c->loss_scale = 1.0f;
    } else {
        c->loss_scale_counter++;
        if (c->loss_scale_counter >= c->loss_scale_growth_interval) {
            c->loss_scale *= 2.0f;
            c->loss_scale_counter = 0;
        }
    }
}

float grad_clip_norm(ParamList *params, float max_norm) {
    float total_norm = 0.0f;
    for (int i = 0; i < params->n_params; i++) {
        Tensor *p = params->params[i];
        if (!p->grad) continue;
        float *g = p->grad->data;
        for (size_t j = 0; j < p->size; j++) {
            total_norm += g[j] * g[j];
        }
    }
    total_norm = sqrtf(total_norm);

    if (total_norm > max_norm) {
        float clip_coef = max_norm / (total_norm + 1e-6f);
        for (int i = 0; i < params->n_params; i++) {
            Tensor *p = params->params[i];
            if (!p->grad) continue;
            tensor_scale(p->grad, clip_coef);
        }
    }
    return total_norm;
}

void adamw_step(AdamW *opt, ParamList *params) {
    opt->step++;
    float lr = adamw_get_lr(opt);
    AdamWConfig *c = &opt->config;

    for (int i = 0; i < params->n_params; i++) {
        Tensor *p = params->params[i];
        if (!p->grad) continue;

        if (metal_is_initialized() && p->metal_buf && opt->m[i]->metal_buf) {
            metal_adamw_update(p->metal_buf, p->grad->metal_buf,
                               opt->m[i]->metal_buf, opt->v[i]->metal_buf,
                               (int)p->size, lr, c->beta1, c->beta2,
                               c->eps, c->weight_decay, opt->step);
        } else {
            // CPU fallback
            float *param = p->data;
            float *grad  = p->grad->data;
            float *m     = opt->m[i]->data;
            float *v     = opt->v[i]->data;

            float bc1 = 1.0f - powf(c->beta1, (float)opt->step);
            float bc2 = 1.0f - powf(c->beta2, (float)opt->step);

            for (size_t j = 0; j < p->size; j++) {
                float g = grad[j];
                m[j] = c->beta1 * m[j] + (1.0f - c->beta1) * g;
                v[j] = c->beta2 * v[j] + (1.0f - c->beta2) * g * g;

                float m_hat = m[j] / bc1;
                float v_hat = v[j] / bc2;

                param[j] = param[j] * (1.0f - lr * c->weight_decay)
                         - lr * m_hat / (sqrtf(v_hat) + c->eps);
            }
        }
    }

    if (metal_is_initialized()) metal_synchronize();
}

void adamw_zero_grad(ParamList *params) {
    for (int i = 0; i < params->n_params; i++) {
        if (params->params[i]->grad) {
            tensor_fill(params->params[i]->grad, 0.0f);
        }
    }
}
