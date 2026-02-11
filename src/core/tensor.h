#ifndef TENSOR_H
#define TENSOR_H

#include <stddef.h>
#include <stdint.h>

#define TENSOR_MAX_DIMS 4

typedef enum {
    DTYPE_FP32,
    DTYPE_FP16,
    DTYPE_UINT32
} DType;

typedef struct Tensor {
    float *data;
    _Float16 *data_f16;
    uint32_t *data_u32;
    int ndim;
    int shape[TENSOR_MAX_DIMS];
    int stride[TENSOR_MAX_DIMS];
    size_t size;          // total number of elements
    size_t bytes;         // total bytes allocated
    DType dtype;
    void *metal_buf;      // id<MTLBuffer>
    struct Tensor *grad;
    int requires_grad;
    int owns_data;        // 1 if this tensor owns the buffer
} Tensor;

// Creation / destruction
Tensor *tensor_create(const int *shape, int ndim, DType dtype, int requires_grad);
Tensor *tensor_create_from_metal_buf(void *metal_buf, const int *shape, int ndim, DType dtype);
Tensor *tensor_zeros(const int *shape, int ndim, DType dtype, int requires_grad);
Tensor *tensor_randn(const int *shape, int ndim, DType dtype, int requires_grad, float scale);
void    tensor_free(Tensor *t);

// Views (zero-copy)
Tensor *tensor_view(Tensor *src, const int *new_shape, int new_ndim);
Tensor *tensor_transpose(Tensor *src, int dim0, int dim1);
Tensor *tensor_slice(Tensor *src, int dim, int start, int end);

// Element access
float   tensor_get(const Tensor *t, const int *indices);
void    tensor_set(Tensor *t, const int *indices, float val);

// Basic ops (CPU fallback)
void    tensor_fill(Tensor *t, float val);
void    tensor_copy(Tensor *dst, const Tensor *src);
void    tensor_scale(Tensor *t, float s);
void    tensor_add_inplace(Tensor *dst, const Tensor *src);

// Utility
size_t  tensor_element_size(DType dtype);
void    tensor_compute_strides(Tensor *t);
void    tensor_print(const Tensor *t, const char *name);
int     tensor_numel(const Tensor *t);

// FP16 <-> FP32 conversion
void    tensor_to_fp16(Tensor *dst_f16, const Tensor *src_f32);
void    tensor_to_fp32(Tensor *dst_f32, const Tensor *src_f16);

#endif // TENSOR_H
