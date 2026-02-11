#ifndef SAFETENSORS_H
#define SAFETENSORS_H

#include <stdint.h>
#include <stddef.h>

#define ST_DTYPE_F32  0
#define ST_DTYPE_F16  1
#define ST_DTYPE_BF16 2

typedef struct {
    char     *name;
    int       dtype;
    int       ndim;
    uint64_t  shape[4];
    uint64_t  data_start;
    uint64_t  data_end;
    int       shard_idx;
} STTensor;

typedef struct {
    void   *mmap_data;
    size_t  mmap_size;
    size_t  data_offset;  // 8 + header_size
} STShard;

typedef struct {
    int       n_tensors;
    STTensor *tensors;
    int       n_shards;
    STShard  *shards;
} SafetensorsFile;

// Open safetensors file(s).
// path can be a single .safetensors file or a directory containing shards.
SafetensorsFile *safetensors_open(const char *path);

// Close and free
void safetensors_close(SafetensorsFile *sf);

// Find tensor by name. Returns NULL if not found.
STTensor *safetensors_find(SafetensorsFile *sf, const char *name);

// Load tensor as F32 array (handles F32, F16, BF16 conversion).
// Caller must free the returned pointer.
float *safetensors_load_f32(SafetensorsFile *sf, const char *name, size_t *n_elements);

#endif // SAFETENSORS_H
