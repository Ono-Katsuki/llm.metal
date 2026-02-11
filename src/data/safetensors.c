#include "safetensors.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <unistd.h>
#include <dirent.h>
#include <math.h>

// ===================================================================
// Minimal JSON scanner for safetensors headers
// ===================================================================

static const char *skip_ws(const char *p) {
    while (*p == ' ' || *p == '\n' || *p == '\r' || *p == '\t') p++;
    return p;
}

static const char *parse_str(const char *p, char *buf, int max) {
    if (*p != '"') { buf[0] = 0; return p; }
    p++;
    int i = 0;
    while (*p && *p != '"') {
        if (*p == '\\' && *(p + 1)) p++;
        if (i < max - 1) buf[i++] = *p;
        p++;
    }
    buf[i] = 0;
    if (*p == '"') p++;
    return p;
}

static const char *skip_val(const char *p) {
    p = skip_ws(p);
    if (*p == '"') {
        p++;
        while (*p && *p != '"') { if (*p == '\\') p++; p++; }
        if (*p) p++;
    } else if (*p == '{') {
        int d = 1; p++;
        while (d > 0 && *p) {
            if (*p == '"') { p++; while (*p && *p != '"') { if (*p == '\\') p++; p++; } if (*p) p++; }
            else { if (*p == '{') d++; else if (*p == '}') d--; p++; }
        }
    } else if (*p == '[') {
        int d = 1; p++;
        while (d > 0 && *p) {
            if (*p == '"') { p++; while (*p && *p != '"') { if (*p == '\\') p++; p++; } if (*p) p++; }
            else { if (*p == '[') d++; else if (*p == ']') d--; p++; }
        }
    } else {
        while (*p && *p != ',' && *p != '}' && *p != ']') p++;
    }
    return p;
}

static const char *parse_int_array(const char *p, uint64_t *out, int max, int *n) {
    *n = 0;
    p = skip_ws(p);
    if (*p != '[') return p;
    p++;
    while (*n < max) {
        p = skip_ws(p);
        if (*p == ']') break;
        out[(*n)++] = strtoull(p, (char **)&p, 10);
        p = skip_ws(p);
        if (*p == ',') p++;
    }
    if (*p == ']') p++;
    return p;
}

static int dtype_from_str(const char *s) {
    if (strcmp(s, "F32") == 0)  return ST_DTYPE_F32;
    if (strcmp(s, "F16") == 0)  return ST_DTYPE_F16;
    if (strcmp(s, "BF16") == 0) return ST_DTYPE_BF16;
    return -1;
}

// ===================================================================
// Parse one shard's header JSON into tensor entries
// ===================================================================

static int parse_header(const char *json, STTensor **out, int *n_out, int shard_idx) {
    // Count tensors by counting "data_offsets" occurrences
    int cap = 0;
    const char *s = json;
    while ((s = strstr(s, "\"data_offsets\"")) != NULL) { cap++; s++; }
    if (cap == 0) { *out = NULL; *n_out = 0; return 0; }

    STTensor *tensors = calloc(cap, sizeof(STTensor));
    int idx = 0;

    const char *p = skip_ws(json);
    if (*p != '{') { free(tensors); return -1; }
    p++;

    char key[512], field[64];

    while (idx < cap) {
        p = skip_ws(p);
        if (*p == '}' || *p == 0) break;
        if (*p == ',') { p++; continue; }

        // Key
        p = parse_str(p, key, sizeof(key));
        p = skip_ws(p);
        if (*p == ':') p++;
        p = skip_ws(p);

        // Skip __metadata__
        if (strcmp(key, "__metadata__") == 0) { p = skip_val(p); continue; }

        // Tensor entry object
        if (*p != '{') { p = skip_val(p); continue; }
        p++;

        STTensor *t = &tensors[idx];
        t->name = strdup(key);
        t->shard_idx = shard_idx;
        t->dtype = -1;

        while (*p && *p != '}') {
            p = skip_ws(p);
            if (*p == ',') { p++; continue; }
            if (*p == '}') break;

            p = parse_str(p, field, sizeof(field));
            p = skip_ws(p);
            if (*p == ':') p++;
            p = skip_ws(p);

            if (strcmp(field, "dtype") == 0) {
                p = parse_str(p, field, sizeof(field));
                t->dtype = dtype_from_str(field);
            } else if (strcmp(field, "shape") == 0) {
                int n;
                p = parse_int_array(p, t->shape, 4, &n);
                t->ndim = n;
            } else if (strcmp(field, "data_offsets") == 0) {
                uint64_t off[2] = {0, 0};
                int n;
                p = parse_int_array(p, off, 2, &n);
                t->data_start = off[0];
                t->data_end = off[1];
            } else {
                p = skip_val(p);
            }
        }
        if (*p == '}') p++;
        idx++;
    }

    *out = tensors;
    *n_out = idx;
    return 0;
}

// ===================================================================
// Open a single shard file (mmap)
// ===================================================================

static int open_shard(const char *path, STShard *shard) {
    int fd = open(path, O_RDONLY);
    if (fd < 0) return -1;

    struct stat st;
    fstat(fd, &st);
    shard->mmap_size = st.st_size;
    shard->mmap_data = mmap(NULL, shard->mmap_size, PROT_READ, MAP_PRIVATE, fd, 0);
    close(fd);
    if (shard->mmap_data == MAP_FAILED) { shard->mmap_data = NULL; return -1; }

    uint64_t header_size;
    memcpy(&header_size, shard->mmap_data, 8);
    shard->data_offset = 8 + header_size;
    return 0;
}

// ===================================================================
// Public API
// ===================================================================

SafetensorsFile *safetensors_open(const char *path) {
    struct stat st;
    if (stat(path, &st) != 0) {
        fprintf(stderr, "[Safetensors] Cannot access: %s\n", path);
        return NULL;
    }

    SafetensorsFile *sf = calloc(1, sizeof(SafetensorsFile));

    if (S_ISDIR(st.st_mode)) {
        // Directory: collect all *.safetensors shard files
        DIR *dir = opendir(path);
        if (!dir) { free(sf); return NULL; }

        char **shard_paths = NULL;
        int n_shards = 0, shard_cap = 0;

        struct dirent *ent;
        while ((ent = readdir(dir)) != NULL) {
            size_t len = strlen(ent->d_name);
            if (len > 12 && strcmp(ent->d_name + len - 12, ".safetensors") == 0) {
                if (n_shards >= shard_cap) {
                    shard_cap = shard_cap ? shard_cap * 2 : 8;
                    shard_paths = realloc(shard_paths, shard_cap * sizeof(char *));
                }
                char full[1024];
                snprintf(full, sizeof(full), "%s/%s", path, ent->d_name);
                shard_paths[n_shards++] = strdup(full);
            }
        }
        closedir(dir);

        if (n_shards == 0) {
            fprintf(stderr, "[Safetensors] No .safetensors files in: %s\n", path);
            free(sf); free(shard_paths);
            return NULL;
        }

        // Sort alphabetically
        for (int i = 0; i < n_shards - 1; i++)
            for (int j = i + 1; j < n_shards; j++)
                if (strcmp(shard_paths[i], shard_paths[j]) > 0) {
                    char *tmp = shard_paths[i];
                    shard_paths[i] = shard_paths[j];
                    shard_paths[j] = tmp;
                }

        sf->n_shards = n_shards;
        sf->shards = calloc(n_shards, sizeof(STShard));

        int total = 0;
        STTensor **per_shard = calloc(n_shards, sizeof(STTensor *));
        int *per_shard_n = calloc(n_shards, sizeof(int));

        for (int i = 0; i < n_shards; i++) {
            if (open_shard(shard_paths[i], &sf->shards[i]) != 0) {
                fprintf(stderr, "[Safetensors] Failed to open: %s\n", shard_paths[i]);
                continue;
            }
            uint64_t hdr_size = sf->shards[i].data_offset - 8;
            char *hdr = malloc(hdr_size + 1);
            memcpy(hdr, (char *)sf->shards[i].mmap_data + 8, hdr_size);
            hdr[hdr_size] = '\0';

            parse_header(hdr, &per_shard[i], &per_shard_n[i], i);
            total += per_shard_n[i];
            free(hdr);

            printf("[Safetensors] Shard %d: %s (%d tensors)\n",
                   i, shard_paths[i], per_shard_n[i]);
        }

        // Merge tensor lists
        sf->n_tensors = total;
        sf->tensors = calloc(total, sizeof(STTensor));
        int off = 0;
        for (int i = 0; i < n_shards; i++) {
            if (per_shard[i]) {
                memcpy(&sf->tensors[off], per_shard[i], per_shard_n[i] * sizeof(STTensor));
                off += per_shard_n[i];
                free(per_shard[i]);
            }
        }
        free(per_shard);
        free(per_shard_n);
        for (int i = 0; i < n_shards; i++) free(shard_paths[i]);
        free(shard_paths);

    } else {
        // Single file
        sf->n_shards = 1;
        sf->shards = calloc(1, sizeof(STShard));
        if (open_shard(path, &sf->shards[0]) != 0) {
            fprintf(stderr, "[Safetensors] Failed to open: %s\n", path);
            free(sf->shards); free(sf);
            return NULL;
        }
        uint64_t hdr_size = sf->shards[0].data_offset - 8;
        char *hdr = malloc(hdr_size + 1);
        memcpy(hdr, (char *)sf->shards[0].mmap_data + 8, hdr_size);
        hdr[hdr_size] = '\0';

        parse_header(hdr, &sf->tensors, &sf->n_tensors, 0);
        free(hdr);
        printf("[Safetensors] %s (%d tensors)\n", path, sf->n_tensors);
    }

    printf("[Safetensors] Total: %d tensors, %d shard(s)\n", sf->n_tensors, sf->n_shards);
    return sf;
}

void safetensors_close(SafetensorsFile *sf) {
    if (!sf) return;
    for (int i = 0; i < sf->n_tensors; i++)
        free(sf->tensors[i].name);
    free(sf->tensors);
    for (int i = 0; i < sf->n_shards; i++) {
        if (sf->shards[i].mmap_data)
            munmap(sf->shards[i].mmap_data, sf->shards[i].mmap_size);
    }
    free(sf->shards);
    free(sf);
}

STTensor *safetensors_find(SafetensorsFile *sf, const char *name) {
    for (int i = 0; i < sf->n_tensors; i++) {
        if (strcmp(sf->tensors[i].name, name) == 0)
            return &sf->tensors[i];
    }
    return NULL;
}

// ===================================================================
// Dtype conversions
// ===================================================================

static void convert_f16(const uint8_t *src, float *dst, size_t n) {
    const uint16_t *f16 = (const uint16_t *)src;
    for (size_t i = 0; i < n; i++) {
        uint16_t h = f16[i];
        uint32_t sign = (h >> 15) & 1;
        uint32_t exp = (h >> 10) & 0x1F;
        uint32_t mant = h & 0x3FF;
        float val;
        if (exp == 0) val = ldexpf((float)mant, -24);
        else if (exp == 31) val = (mant == 0) ? INFINITY : NAN;
        else val = ldexpf((float)(mant + 1024), (int)exp - 25);
        dst[i] = sign ? -val : val;
    }
}

static void convert_bf16(const uint8_t *src, float *dst, size_t n) {
    const uint16_t *bf = (const uint16_t *)src;
    for (size_t i = 0; i < n; i++) {
        uint32_t bits = (uint32_t)bf[i] << 16;
        memcpy(&dst[i], &bits, 4);
    }
}

float *safetensors_load_f32(SafetensorsFile *sf, const char *name, size_t *n_elements) {
    STTensor *t = safetensors_find(sf, name);
    if (!t) {
        fprintf(stderr, "[Safetensors] Tensor not found: %s\n", name);
        *n_elements = 0;
        return NULL;
    }

    size_t total = 1;
    for (int d = 0; d < t->ndim; d++) total *= t->shape[d];
    *n_elements = total;

    STShard *shard = &sf->shards[t->shard_idx];
    const uint8_t *data = (const uint8_t *)shard->mmap_data + shard->data_offset + t->data_start;

    float *out = malloc(total * sizeof(float));
    if (!out) {
        fprintf(stderr, "[Safetensors] Alloc failed for %s (%zu elements)\n", name, total);
        *n_elements = 0;
        return NULL;
    }

    switch (t->dtype) {
        case ST_DTYPE_F32:
            memcpy(out, data, total * sizeof(float));
            break;
        case ST_DTYPE_F16:
            convert_f16(data, out, total);
            break;
        case ST_DTYPE_BF16:
            convert_bf16(data, out, total);
            break;
        default:
            fprintf(stderr, "[Safetensors] Unsupported dtype %d for %s\n", t->dtype, name);
            free(out);
            *n_elements = 0;
            return NULL;
    }
    return out;
}
