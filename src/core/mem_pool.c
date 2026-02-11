#include "mem_pool.h"
#include <stdlib.h>
#include <string.h>
#include <stdio.h>

extern void *metal_create_shared_buffer(size_t bytes);
extern void  metal_free_buffer(void *buf);
extern void *metal_buffer_contents(void *buf);
extern int   metal_is_initialized(void);

MemPool *mem_pool_create(size_t default_slab_size) {
    MemPool *pool = calloc(1, sizeof(MemPool));
    if (!pool) return NULL;
    pool->default_slab_size = default_slab_size;
    return pool;
}

static int mem_pool_add_slab(MemPool *pool, size_t min_size) {
    if (pool->n_slabs >= MEM_POOL_MAX_SLABS) return -1;
    size_t size = min_size > pool->default_slab_size ? min_size : pool->default_slab_size;
    MemSlab *slab = &pool->slabs[pool->n_slabs];
    if (metal_is_initialized()) {
        slab->metal_buf = metal_create_shared_buffer(size);
        if (!slab->metal_buf) return -1;
        slab->buffer = metal_buffer_contents(slab->metal_buf);
    } else {
        slab->buffer = malloc(size);
        if (!slab->buffer) return -1;
        slab->metal_buf = NULL;
    }
    slab->capacity = size;
    slab->used = 0;
    pool->n_slabs++;
    pool->total_allocated += size;
    return pool->n_slabs - 1;
}

void *mem_pool_alloc(MemPool *pool, size_t bytes, void **out_metal_buf) {
    // Align to 256 bytes for Metal
    bytes = (bytes + 255) & ~(size_t)255;

    // Try existing slabs
    for (int i = 0; i < pool->n_slabs; i++) {
        MemSlab *s = &pool->slabs[i];
        if (s->used + bytes <= s->capacity) {
            void *ptr = (char *)s->buffer + s->used;
            if (out_metal_buf) *out_metal_buf = s->metal_buf;
            s->used += bytes;
            size_t total_used = 0;
            for (int j = 0; j < pool->n_slabs; j++) total_used += pool->slabs[j].used;
            if (total_used > pool->peak_usage) pool->peak_usage = total_used;
            return ptr;
        }
    }

    // Need new slab
    int idx = mem_pool_add_slab(pool, bytes);
    if (idx < 0) return NULL;
    MemSlab *s = &pool->slabs[idx];
    void *ptr = s->buffer;
    if (out_metal_buf) *out_metal_buf = s->metal_buf;
    s->used = bytes;
    return ptr;
}

void mem_pool_reset(MemPool *pool) {
    for (int i = 0; i < pool->n_slabs; i++) {
        pool->slabs[i].used = 0;
    }
}

void mem_pool_destroy(MemPool *pool) {
    if (!pool) return;
    for (int i = 0; i < pool->n_slabs; i++) {
        if (pool->slabs[i].metal_buf) {
            metal_free_buffer(pool->slabs[i].metal_buf);
        } else {
            free(pool->slabs[i].buffer);
        }
    }
    free(pool);
}

void mem_pool_print_stats(const MemPool *pool) {
    printf("MemPool: %d slabs, %.2f MB allocated, %.2f MB peak\n",
           pool->n_slabs,
           (double)pool->total_allocated / (1024.0 * 1024.0),
           (double)pool->peak_usage / (1024.0 * 1024.0));
}
