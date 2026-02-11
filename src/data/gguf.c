#include "gguf.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <unistd.h>
#include <math.h>

// GGUF magic and version
#define GGUF_MAGIC 0x46554747  // "GGUF" in little-endian

// Helper: read from buffer at offset, advance offset
static uint32_t read_u32(const uint8_t *buf, size_t *off) {
    uint32_t v;
    memcpy(&v, buf + *off, 4);
    *off += 4;
    return v;
}

static uint64_t read_u64(const uint8_t *buf, size_t *off) {
    uint64_t v;
    memcpy(&v, buf + *off, 8);
    *off += 8;
    return v;
}

static char *read_string(const uint8_t *buf, size_t *off) {
    uint64_t len = read_u64(buf, off);
    char *s = malloc(len + 1);
    memcpy(s, buf + *off, len);
    s[len] = '\0';
    *off += len;
    return s;
}

// Skip a metadata value based on its type
static void skip_metadata_value(const uint8_t *buf, size_t *off, uint32_t type) {
    switch (type) {
        case GGUF_TYPE_UINT8:
        case GGUF_TYPE_INT8:
        case GGUF_TYPE_BOOL:
            *off += 1;
            break;
        case GGUF_TYPE_UINT16:
        case GGUF_TYPE_INT16:
            *off += 2;
            break;
        case GGUF_TYPE_UINT32:
        case GGUF_TYPE_INT32:
        case GGUF_TYPE_FLOAT32:
            *off += 4;
            break;
        case GGUF_TYPE_UINT64:
        case GGUF_TYPE_INT64:
        case GGUF_TYPE_FLOAT64:
            *off += 8;
            break;
        case GGUF_TYPE_STRING: {
            uint64_t len = read_u64(buf, off);
            *off += len;
            break;
        }
        case GGUF_TYPE_ARRAY: {
            uint32_t arr_type = read_u32(buf, off);
            uint64_t arr_len = read_u64(buf, off);
            for (uint64_t i = 0; i < arr_len; i++) {
                skip_metadata_value(buf, off, arr_type);
            }
            break;
        }
        default:
            fprintf(stderr, "[GGUF] Unknown metadata type: %u\n", type);
            break;
    }
}

// Compute tensor data size in bytes
static size_t tensor_data_size(uint32_t type, uint64_t n_elements) {
    switch (type) {
        case GGML_TYPE_F32:
            return n_elements * 4;
        case GGML_TYPE_F16:
            return n_elements * 2;
        case GGML_TYPE_Q8_0: {
            // Q8_0: blocks of 32 elements, each block = 2 bytes (f16 scale) + 32 bytes (int8)
            uint64_t n_blocks = (n_elements + 31) / 32;
            return n_blocks * 34; // 2 + 32
        }
        default:
            fprintf(stderr, "[GGUF] Unsupported tensor type: %u\n", type);
            return 0;
    }
}

GGUFFile *gguf_open(const char *path) {
    int fd = open(path, O_RDONLY);
    if (fd < 0) {
        fprintf(stderr, "[GGUF] Cannot open file: %s\n", path);
        return NULL;
    }

    struct stat st;
    fstat(fd, &st);
    size_t file_size = st.st_size;

    void *data = mmap(NULL, file_size, PROT_READ, MAP_PRIVATE, fd, 0);
    close(fd);
    if (data == MAP_FAILED) {
        fprintf(stderr, "[GGUF] mmap failed for: %s\n", path);
        return NULL;
    }

    const uint8_t *buf = (const uint8_t *)data;
    size_t off = 0;

    // Read header
    uint32_t magic = read_u32(buf, &off);
    if (magic != GGUF_MAGIC) {
        fprintf(stderr, "[GGUF] Invalid magic: 0x%08X (expected 0x%08X)\n", magic, GGUF_MAGIC);
        munmap(data, file_size);
        return NULL;
    }

    uint32_t version = read_u32(buf, &off);
    if (version < 2 || version > 3) {
        fprintf(stderr, "[GGUF] Unsupported version: %u\n", version);
        munmap(data, file_size);
        return NULL;
    }

    uint64_t n_tensors = read_u64(buf, &off);
    uint64_t n_kv = read_u64(buf, &off);

    printf("[GGUF] Version %u, %llu tensors, %llu metadata KV pairs\n",
           version, n_tensors, n_kv);

    // Skip metadata KV pairs
    for (uint64_t i = 0; i < n_kv; i++) {
        char *key = read_string(buf, &off);
        uint32_t val_type = read_u32(buf, &off);
        skip_metadata_value(buf, &off, val_type);
        free(key);
    }

    // Parse tensor info
    GGUFTensor *tensors = calloc(n_tensors, sizeof(GGUFTensor));
    for (uint64_t i = 0; i < n_tensors; i++) {
        tensors[i].name = read_string(buf, &off);
        uint32_t ndim = read_u32(buf, &off);
        tensors[i].ndim = ndim;
        uint64_t n_elements = 1;
        for (uint32_t d = 0; d < ndim; d++) {
            tensors[i].shape[d] = read_u64(buf, &off);
            n_elements *= tensors[i].shape[d];
        }
        tensors[i].type = read_u32(buf, &off);
        tensors[i].offset = read_u64(buf, &off);
        tensors[i].data_size = tensor_data_size(tensors[i].type, n_elements);
    }

    // Data section starts at alignment boundary (32 bytes for GGUF v2/v3)
    size_t alignment = 32;
    uint64_t data_offset = (off + alignment - 1) & ~(alignment - 1);

    GGUFFile *f = calloc(1, sizeof(GGUFFile));
    f->n_tensors = (int)n_tensors;
    f->tensors = tensors;
    f->mmap_data = data;
    f->mmap_size = file_size;
    f->data_offset = data_offset;

    printf("[GGUF] Data offset: %llu, file size: %zu\n", data_offset, file_size);
    return f;
}

void gguf_close(GGUFFile *f) {
    if (!f) return;
    for (int i = 0; i < f->n_tensors; i++) {
        free(f->tensors[i].name);
    }
    free(f->tensors);
    if (f->mmap_data) {
        munmap(f->mmap_data, f->mmap_size);
    }
    free(f);
}

GGUFTensor *gguf_find_tensor(GGUFFile *f, const char *name) {
    for (int i = 0; i < f->n_tensors; i++) {
        if (strcmp(f->tensors[i].name, name) == 0) {
            return &f->tensors[i];
        }
    }
    return NULL;
}

// Dequantize F16 to F32
static void dequantize_f16(const uint8_t *src, float *dst, size_t n) {
    const uint16_t *f16 = (const uint16_t *)src;
    for (size_t i = 0; i < n; i++) {
        // F16 -> F32 conversion
        uint16_t h = f16[i];
        uint32_t sign = (h >> 15) & 1;
        uint32_t exp = (h >> 10) & 0x1F;
        uint32_t mant = h & 0x3FF;

        float val;
        if (exp == 0) {
            // Subnormal
            val = ldexpf((float)mant, -24);
        } else if (exp == 31) {
            // Inf/NaN
            val = (mant == 0) ? INFINITY : NAN;
        } else {
            val = ldexpf((float)(mant + 1024), (int)exp - 25);
        }
        dst[i] = sign ? -val : val;
    }
}

// Dequantize Q8_0 to F32
// Block format: f16 scale (2 bytes) + 32 x int8 (32 bytes) = 34 bytes per block
static void dequantize_q8_0(const uint8_t *src, float *dst, size_t n_elements) {
    size_t n_blocks = (n_elements + 31) / 32;
    for (size_t b = 0; b < n_blocks; b++) {
        const uint8_t *block = src + b * 34;
        // Read f16 scale
        uint16_t scale_f16;
        memcpy(&scale_f16, block, 2);
        float scale;
        // Quick f16->f32
        uint32_t sign = (scale_f16 >> 15) & 1;
        uint32_t exp = (scale_f16 >> 10) & 0x1F;
        uint32_t mant = scale_f16 & 0x3FF;
        if (exp == 0) {
            scale = ldexpf((float)mant, -24);
        } else if (exp == 31) {
            scale = (mant == 0) ? INFINITY : NAN;
        } else {
            scale = ldexpf((float)(mant + 1024), (int)exp - 25);
        }
        if (sign) scale = -scale;

        const int8_t *quants = (const int8_t *)(block + 2);
        size_t remaining = n_elements - b * 32;
        size_t count = remaining < 32 ? remaining : 32;
        for (size_t i = 0; i < count; i++) {
            dst[b * 32 + i] = (float)quants[i] * scale;
        }
    }
}

float *gguf_load_tensor_f32(GGUFFile *f, const char *name, size_t *n_elements) {
    GGUFTensor *t = gguf_find_tensor(f, name);
    if (!t) {
        fprintf(stderr, "[GGUF] Tensor not found: %s\n", name);
        *n_elements = 0;
        return NULL;
    }

    size_t total = 1;
    for (int d = 0; d < t->ndim; d++) {
        total *= t->shape[d];
    }
    *n_elements = total;

    const uint8_t *tensor_data = (const uint8_t *)f->mmap_data + f->data_offset + t->offset;

    float *out = malloc(total * sizeof(float));
    if (!out) {
        fprintf(stderr, "[GGUF] Failed to allocate %zu bytes for tensor %s\n",
                total * sizeof(float), name);
        return NULL;
    }

    switch (t->type) {
        case GGML_TYPE_F32:
            memcpy(out, tensor_data, total * sizeof(float));
            break;
        case GGML_TYPE_F16:
            dequantize_f16(tensor_data, out, total);
            break;
        case GGML_TYPE_Q8_0:
            dequantize_q8_0(tensor_data, out, total);
            break;
        default:
            fprintf(stderr, "[GGUF] Unsupported tensor type %u for %s\n", t->type, name);
            free(out);
            *n_elements = 0;
            return NULL;
    }

    return out;
}
