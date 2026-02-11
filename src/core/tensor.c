#include "tensor.h"
#include "mem_pool.h"
#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <math.h>
#include <assert.h>

// Forward declaration - implemented in metal_backend
extern void *metal_create_shared_buffer(size_t bytes);
extern void  metal_free_buffer(void *buf);
extern void *metal_buffer_contents(void *buf);
extern int   metal_is_initialized(void);

size_t tensor_element_size(DType dtype) {
    switch (dtype) {
        case DTYPE_FP32:   return sizeof(float);
        case DTYPE_FP16:   return sizeof(_Float16);
        case DTYPE_UINT32: return sizeof(uint32_t);
    }
    return sizeof(float);
}

void tensor_compute_strides(Tensor *t) {
    if (t->ndim == 0) return;
    t->stride[t->ndim - 1] = 1;
    for (int i = t->ndim - 2; i >= 0; i--) {
        t->stride[i] = t->stride[i + 1] * t->shape[i + 1];
    }
}

int tensor_numel(const Tensor *t) {
    if (t->ndim == 0) return 0;
    int n = 1;
    for (int i = 0; i < t->ndim; i++) n *= t->shape[i];
    return n;
}

Tensor *tensor_create(const int *shape, int ndim, DType dtype, int requires_grad) {
    assert(ndim > 0 && ndim <= TENSOR_MAX_DIMS);
    Tensor *t = calloc(1, sizeof(Tensor));
    if (!t) return NULL;

    t->ndim = ndim;
    t->dtype = dtype;
    t->requires_grad = requires_grad;
    t->owns_data = 1;
    for (int i = 0; i < ndim; i++) t->shape[i] = shape[i];
    tensor_compute_strides(t);
    t->size = (size_t)tensor_numel(t);
    t->bytes = t->size * tensor_element_size(dtype);

    // Allocate via Metal shared buffer if Metal is up, else malloc
    if (metal_is_initialized()) {
        t->metal_buf = metal_create_shared_buffer(t->bytes);
        if (!t->metal_buf) { free(t); return NULL; }
        void *ptr = metal_buffer_contents(t->metal_buf);
        if (dtype == DTYPE_FP32)        t->data = (float *)ptr;
        else if (dtype == DTYPE_FP16)   t->data_f16 = (_Float16 *)ptr;
        else if (dtype == DTYPE_UINT32) t->data_u32 = (uint32_t *)ptr;
    } else {
        if (dtype == DTYPE_FP32) {
            t->data = (float *)calloc(t->size, sizeof(float));
        } else if (dtype == DTYPE_FP16) {
            t->data_f16 = (_Float16 *)calloc(t->size, sizeof(_Float16));
        } else if (dtype == DTYPE_UINT32) {
            t->data_u32 = (uint32_t *)calloc(t->size, sizeof(uint32_t));
        }
    }

    if (requires_grad) {
        int grad_shape[TENSOR_MAX_DIMS];
        memcpy(grad_shape, shape, ndim * sizeof(int));
        t->grad = tensor_create(grad_shape, ndim, DTYPE_FP32, 0);
    }
    return t;
}

Tensor *tensor_create_from_metal_buf(void *metal_buf, const int *shape, int ndim, DType dtype) {
    Tensor *t = calloc(1, sizeof(Tensor));
    if (!t) return NULL;
    t->ndim = ndim;
    t->dtype = dtype;
    t->owns_data = 0;
    t->metal_buf = metal_buf;
    for (int i = 0; i < ndim; i++) t->shape[i] = shape[i];
    tensor_compute_strides(t);
    t->size = (size_t)tensor_numel(t);
    t->bytes = t->size * tensor_element_size(dtype);
    void *ptr = metal_buffer_contents(metal_buf);
    if (dtype == DTYPE_FP32) t->data = (float *)ptr;
    else                     t->data_f16 = (_Float16 *)ptr;
    return t;
}

Tensor *tensor_zeros(const int *shape, int ndim, DType dtype, int requires_grad) {
    Tensor *t = tensor_create(shape, ndim, dtype, requires_grad);
    if (!t) return NULL;
    // Memory is already zeroed from calloc / Metal shared
    if (t->metal_buf) {
        memset(metal_buffer_contents(t->metal_buf), 0, t->bytes);
    }
    return t;
}

// Box-Muller for normal random
static float randn_val(void) {
    float u1 = ((float)rand() / RAND_MAX);
    float u2 = ((float)rand() / RAND_MAX);
    if (u1 < 1e-10f) u1 = 1e-10f;
    return sqrtf(-2.0f * logf(u1)) * cosf(2.0f * (float)M_PI * u2);
}

Tensor *tensor_randn(const int *shape, int ndim, DType dtype, int requires_grad, float scale) {
    Tensor *t = tensor_create(shape, ndim, dtype, requires_grad);
    if (!t) return NULL;
    for (size_t i = 0; i < t->size; i++) {
        float v = randn_val() * scale;
        if (dtype == DTYPE_FP32) t->data[i] = v;
        else                     t->data_f16[i] = (_Float16)v;
    }
    return t;
}

void tensor_free(Tensor *t) {
    if (!t) return;
    if (t->grad) { tensor_free(t->grad); t->grad = NULL; }
    if (t->owns_data) {
        if (t->metal_buf) {
            metal_free_buffer(t->metal_buf);
        } else {
            free(t->data);
            free(t->data_f16);
            free(t->data_u32);
        }
    }
    free(t);
}

Tensor *tensor_view(Tensor *src, const int *new_shape, int new_ndim) {
    Tensor *v = calloc(1, sizeof(Tensor));
    if (!v) return NULL;
    v->ndim = new_ndim;
    v->dtype = src->dtype;
    v->data = src->data;
    v->data_f16 = src->data_f16;
    v->data_u32 = src->data_u32;
    v->metal_buf = src->metal_buf;
    v->owns_data = 0;
    v->size = src->size;
    v->bytes = src->bytes;
    for (int i = 0; i < new_ndim; i++) v->shape[i] = new_shape[i];
    tensor_compute_strides(v);
    // Verify total elements match
    assert((size_t)tensor_numel(v) == src->size);
    return v;
}

Tensor *tensor_transpose(Tensor *src, int dim0, int dim1) {
    assert(dim0 < src->ndim && dim1 < src->ndim);
    Tensor *v = calloc(1, sizeof(Tensor));
    if (!v) return NULL;
    *v = *src;
    v->owns_data = 0;
    v->grad = NULL;
    // Swap shape and stride
    int tmp = v->shape[dim0]; v->shape[dim0] = v->shape[dim1]; v->shape[dim1] = tmp;
    tmp = v->stride[dim0]; v->stride[dim0] = v->stride[dim1]; v->stride[dim1] = tmp;
    return v;
}

Tensor *tensor_slice(Tensor *src, int dim, int start, int end) {
    assert(dim < src->ndim && start >= 0 && end <= src->shape[dim]);
    Tensor *v = calloc(1, sizeof(Tensor));
    if (!v) return NULL;
    *v = *src;
    v->owns_data = 0;
    v->grad = NULL;
    v->shape[dim] = end - start;
    // Offset data pointer
    size_t offset = (size_t)start * (size_t)v->stride[dim];
    if (v->dtype == DTYPE_FP32)        v->data += offset;
    else if (v->dtype == DTYPE_FP16)   v->data_f16 += offset;
    else if (v->dtype == DTYPE_UINT32) v->data_u32 += offset;
    v->size = (size_t)tensor_numel(v);
    v->bytes = v->size * tensor_element_size(v->dtype);
    return v;
}

float tensor_get(const Tensor *t, const int *indices) {
    size_t idx = 0;
    for (int i = 0; i < t->ndim; i++) idx += (size_t)indices[i] * (size_t)t->stride[i];
    if (t->dtype == DTYPE_FP32) return t->data[idx];
    return (float)t->data_f16[idx];
}

void tensor_set(Tensor *t, const int *indices, float val) {
    size_t idx = 0;
    for (int i = 0; i < t->ndim; i++) idx += (size_t)indices[i] * (size_t)t->stride[i];
    if (t->dtype == DTYPE_FP32) t->data[idx] = val;
    else                        t->data_f16[idx] = (_Float16)val;
}

void tensor_fill(Tensor *t, float val) {
    if (t->dtype == DTYPE_FP32) {
        for (size_t i = 0; i < t->size; i++) t->data[i] = val;
    } else if (t->dtype == DTYPE_FP16) {
        _Float16 v16 = (_Float16)val;
        for (size_t i = 0; i < t->size; i++) t->data_f16[i] = v16;
    } else if (t->dtype == DTYPE_UINT32) {
        uint32_t v32 = (uint32_t)val;
        for (size_t i = 0; i < t->size; i++) t->data_u32[i] = v32;
    }
}

void tensor_copy(Tensor *dst, const Tensor *src) {
    assert(dst->size == src->size);
    if (dst->dtype == src->dtype) {
        memcpy(dst->dtype == DTYPE_FP32 ? (void *)dst->data : (void *)dst->data_f16,
               src->dtype == DTYPE_FP32 ? (const void *)src->data : (const void *)src->data_f16,
               dst->bytes);
    } else {
        // Cross-dtype copy
        for (size_t i = 0; i < src->size; i++) {
            float v = src->dtype == DTYPE_FP32 ? src->data[i] : (float)src->data_f16[i];
            if (dst->dtype == DTYPE_FP32) dst->data[i] = v;
            else                          dst->data_f16[i] = (_Float16)v;
        }
    }
}

void tensor_scale(Tensor *t, float s) {
    if (t->dtype == DTYPE_FP32) {
        for (size_t i = 0; i < t->size; i++) t->data[i] *= s;
    } else {
        for (size_t i = 0; i < t->size; i++) t->data_f16[i] *= (_Float16)s;
    }
}

void tensor_add_inplace(Tensor *dst, const Tensor *src) {
    assert(dst->size == src->size);
    if (dst->dtype == DTYPE_FP32 && src->dtype == DTYPE_FP32) {
        for (size_t i = 0; i < dst->size; i++) dst->data[i] += src->data[i];
    } else {
        for (size_t i = 0; i < dst->size; i++) {
            float v = src->dtype == DTYPE_FP32 ? src->data[i] : (float)src->data_f16[i];
            if (dst->dtype == DTYPE_FP32) dst->data[i] += v;
            else                          dst->data_f16[i] += (_Float16)v;
        }
    }
}

void tensor_to_fp16(Tensor *dst, const Tensor *src) {
    assert(dst->dtype == DTYPE_FP16 && src->dtype == DTYPE_FP32);
    assert(dst->size == src->size);
    for (size_t i = 0; i < src->size; i++) {
        dst->data_f16[i] = (_Float16)src->data[i];
    }
}

void tensor_to_fp32(Tensor *dst, const Tensor *src) {
    assert(dst->dtype == DTYPE_FP32 && src->dtype == DTYPE_FP16);
    assert(dst->size == src->size);
    for (size_t i = 0; i < src->size; i++) {
        dst->data[i] = (float)src->data_f16[i];
    }
}

void tensor_print(const Tensor *t, const char *name) {
    printf("Tensor '%s': shape=[", name);
    for (int i = 0; i < t->ndim; i++) {
        printf("%d%s", t->shape[i], i < t->ndim - 1 ? "," : "");
    }
    const char *dtype_str = t->dtype == DTYPE_FP32 ? "fp32" :
                            t->dtype == DTYPE_FP16 ? "fp16" : "uint32";
    printf("], dtype=%s, size=%zu", dtype_str, t->size);
    if (t->requires_grad) printf(", requires_grad");
    printf("\n");
    // Print first few values
    int n = t->size < 8 ? (int)t->size : 8;
    printf("  data: [");
    for (int i = 0; i < n; i++) {
        if (t->dtype == DTYPE_UINT32) {
            printf("%u%s", t->data_u32[i], i < n - 1 ? ", " : "");
        } else {
            float v = t->dtype == DTYPE_FP32 ? t->data[i] : (float)t->data_f16[i];
            printf("%.4f%s", v, i < n - 1 ? ", " : "");
        }
    }
    if ((int)t->size > n) printf(", ...");
    printf("]\n");
}
