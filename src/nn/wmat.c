#include "wmat.h"
#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <math.h>
#include <time.h>

// ==================================================================
// Q8_0 Quantization — same format as GGML
// ==================================================================

#define QK8 32

typedef struct {
    uint16_t d;
    int8_t   qs[QK8];
} __attribute__((packed)) block_q8;

static void quantize_row_q8(const float *src, block_q8 *dst, int n_blocks) {
    for (int b = 0; b < n_blocks; b++) {
        const float *sp = src + b * QK8;
        block_q8 *blk = &dst[b];
        float amax = 0.0f;
        for (int j = 0; j < QK8; j++) {
            float a = fabsf(sp[j]);
            if (a > amax) amax = a;
        }
        float scale = amax / 127.0f;
        __fp16 h = (__fp16)scale;
        memcpy(&blk->d, &h, 2);
        if (scale != 0.0f) {
            float inv = 1.0f / scale;
            for (int j = 0; j < QK8; j++) {
                int v = (int)roundf(sp[j] * inv);
                blk->qs[j] = (int8_t)(v < -128 ? -128 : (v > 127 ? 127 : v));
            }
        } else {
            memset(blk->qs, 0, QK8);
        }
    }
}

// ==================================================================
// Weight creation
// ==================================================================

WMat *wmat_create_q8(const float *src, int rows, int cols) {
    WMat *w = calloc(1, sizeof(WMat));
    w->rows = rows;
    w->cols = cols;
    w->nb = cols / QK8;
    size_t data_bytes = (size_t)rows * w->nb * sizeof(block_q8);
    block_q8 *data = malloc(data_bytes);

    for (int i = 0; i < rows; i++)
        quantize_row_q8(src + (size_t)i * cols, &data[(size_t)i * w->nb], w->nb);

    w->mbuf = metal_buf_from_data(data, data_bytes);
    free(data);
    return w;
}

WMat *wmat_create_f16(const float *src, int rows, int cols) {
    WMat *w = calloc(1, sizeof(WMat));
    w->rows = rows;
    w->cols = cols;
    w->nb = 0;
    size_t n = (size_t)rows * cols;
    __fp16 *data = malloc(n * sizeof(__fp16));
    for (size_t i = 0; i < n; i++)
        data[i] = (__fp16)src[i];
    w->mbuf = metal_buf_from_data(data, n * sizeof(__fp16));
    free(data);
    return w;
}

void wmat_free(WMat *w) {
    if (!w) return;
    metal_buf_free(w->mbuf);
    free(w);
}

// ==================================================================
// Bulk weight conversion + upload (model-independent)
// ==================================================================

LayerW *layers_convert_upload(const LayerWeightRef *refs, int n_layers, int use_fp16) {
    WMat *(*create_fn)(const float *, int, int) = use_fp16 ? wmat_create_f16 : wmat_create_q8;

    LayerW *layers = calloc(n_layers, sizeof(LayerW));
    for (int i = 0; i < n_layers; i++) {
        const LayerWeightRef *ref = &refs[i];
        LayerW *lw = &layers[i];

        lw->q_proj = create_fn(ref->q_proj.data, ref->q_proj.rows, ref->q_proj.cols);
        lw->k_proj = create_fn(ref->k_proj.data, ref->k_proj.rows, ref->k_proj.cols);
        lw->v_proj = create_fn(ref->v_proj.data, ref->v_proj.rows, ref->v_proj.cols);
        lw->o_proj = create_fn(ref->o_proj.data, ref->o_proj.rows, ref->o_proj.cols);
        lw->gate_proj = create_fn(ref->gate_proj.data, ref->gate_proj.rows, ref->gate_proj.cols);
        lw->up_proj = create_fn(ref->up_proj.data, ref->up_proj.rows, ref->up_proj.cols);
        lw->down_proj = create_fn(ref->down_proj.data, ref->down_proj.rows, ref->down_proj.cols);

        lw->input_norm_g = metal_buf_from_data(
            ref->input_norm_g, ref->d_model * sizeof(float));
        lw->post_attn_norm_g = metal_buf_from_data(
            ref->post_attn_norm_g, ref->d_model * sizeof(float));

        if (ref->pre_ff_norm_g) {
            lw->pre_ff_norm_g = metal_buf_from_data(
                ref->pre_ff_norm_g, ref->d_model * sizeof(float));
        }
        if (ref->post_ff_norm_g) {
            lw->post_ff_norm_g = metal_buf_from_data(
                ref->post_ff_norm_g, ref->d_model * sizeof(float));
        }
        if (ref->q_norm_g) {
            lw->q_norm_g = metal_buf_from_data(
                ref->q_norm_g, ref->head_dim * sizeof(float));
        }
        if (ref->k_norm_g) {
            lw->k_norm_g = metal_buf_from_data(
                ref->k_norm_g, ref->head_dim * sizeof(float));
        }

        if ((i + 1) % 6 == 0)
            printf("  Converted layers 0-%d\n", i);
    }
    return layers;
}

void layers_free(LayerW *layers, int n_layers) {
    if (!layers) return;
    for (int i = 0; i < n_layers; i++) {
        LayerW *lw = &layers[i];
        wmat_free(lw->q_proj); wmat_free(lw->k_proj);
        wmat_free(lw->v_proj); wmat_free(lw->o_proj);
        wmat_free(lw->gate_proj); wmat_free(lw->up_proj);
        wmat_free(lw->down_proj);
        metal_buf_free(lw->input_norm_g);
        metal_buf_free(lw->post_attn_norm_g);
        metal_buf_free(lw->pre_ff_norm_g);
        metal_buf_free(lw->post_ff_norm_g);
        metal_buf_free(lw->q_norm_g);
        metal_buf_free(lw->k_norm_g);
    }
    free(layers);
}

WMat *wmat_convert(const float *data, int rows, int cols, int use_fp16) {
    WMat *(*create_fn)(const float *, int, int) = use_fp16 ? wmat_create_f16 : wmat_create_q8;
    return create_fn(data, rows, cols);
}

MetalBuf *norm_upload(const float *gamma, int dim) {
    return metal_buf_from_data(gamma, dim * sizeof(float));
}
