#ifndef MEM_POOL_H
#define MEM_POOL_H

#include <stddef.h>

// Simple arena / slab allocator for temporary tensors during forward/backward
// Reduces Metal buffer creation overhead

#define MEM_POOL_MAX_SLABS 256

typedef struct {
    void *buffer;       // Metal shared buffer or malloc
    void *metal_buf;    // id<MTLBuffer> if Metal
    size_t capacity;
    size_t used;
} MemSlab;

typedef struct {
    MemSlab slabs[MEM_POOL_MAX_SLABS];
    int n_slabs;
    size_t default_slab_size;
    size_t total_allocated;
    size_t peak_usage;
} MemPool;

// Initialize pool with a default slab size
MemPool *mem_pool_create(size_t default_slab_size);
void     mem_pool_destroy(MemPool *pool);

// Allocate from pool (returns pointer to shared memory)
void    *mem_pool_alloc(MemPool *pool, size_t bytes, void **out_metal_buf);

// Reset pool (mark all memory as reusable without freeing)
void     mem_pool_reset(MemPool *pool);

// Stats
void     mem_pool_print_stats(const MemPool *pool);

#endif // MEM_POOL_H
