#ifndef GGUF_H
#define GGUF_H

#include <stdint.h>
#include <stddef.h>

// GGML tensor types we support
#define GGML_TYPE_F32  0
#define GGML_TYPE_F16  1
#define GGML_TYPE_Q8_0 8

// GGUF metadata value types
#define GGUF_TYPE_UINT8   0
#define GGUF_TYPE_INT8    1
#define GGUF_TYPE_UINT16  2
#define GGUF_TYPE_INT16   3
#define GGUF_TYPE_UINT32  4
#define GGUF_TYPE_INT32   5
#define GGUF_TYPE_FLOAT32 6
#define GGUF_TYPE_BOOL    7
#define GGUF_TYPE_STRING  8
#define GGUF_TYPE_ARRAY   9
#define GGUF_TYPE_UINT64  10
#define GGUF_TYPE_INT64   11
#define GGUF_TYPE_FLOAT64 12

typedef struct {
    char     *name;
    uint32_t  type;       // GGML_TYPE_F32, F16, Q8_0
    uint64_t  offset;     // offset from data_offset
    int       ndim;
    uint64_t  shape[4];
    size_t    data_size;  // bytes of tensor data
} GGUFTensor;

typedef struct {
    int         n_tensors;
    GGUFTensor *tensors;
    void       *mmap_data;      // mmap'd file
    size_t      mmap_size;
    uint64_t    data_offset;    // tensor data starts here
} GGUFFile;

// Open a GGUF file (mmap'd for efficiency)
GGUFFile *gguf_open(const char *path);

// Close and free
void gguf_close(GGUFFile *f);

// Find a tensor by name. Returns NULL if not found.
GGUFTensor *gguf_find_tensor(GGUFFile *f, const char *name);

// Load a tensor as F32 array (handles F32, F16, Q8_0 dequantization).
// Caller must free the returned pointer.
// Sets *n_elements to the total number of float elements.
float *gguf_load_tensor_f32(GGUFFile *f, const char *name, size_t *n_elements);

#endif // GGUF_H
