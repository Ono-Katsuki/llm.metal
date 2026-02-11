#ifndef DATALOADER_H
#define DATALOADER_H

#include <stddef.h>
#include "../core/tensor.h"
#include "tokenizer.h"

typedef struct {
    // Memory-mapped data
    char  *mmap_data;
    size_t mmap_size;
    int    fd;

    // Pre-tokenized data
    int   *token_ids;
    size_t n_tokens;

    // Batching config
    int    batch_size;
    int    seq_len;
    size_t current_pos;

    // Tokenizer reference
    const Tokenizer *tokenizer;
} DataLoader;

// Create data loader from a text file (mmap + tokenize)
DataLoader *dataloader_create(const char *filepath, const Tokenizer *tok,
                              int batch_size, int seq_len);

// Create data loader from pre-tokenized binary file
DataLoader *dataloader_create_from_tokens(const char *filepath,
                                          int batch_size, int seq_len);

void dataloader_free(DataLoader *dl);

// Get next batch: returns input tokens [batch_size * seq_len] and
// target tokens [batch_size * seq_len] (shifted by 1)
// Returns 0 on success, -1 on epoch end (wraps around)
int dataloader_next_batch(DataLoader *dl, Tensor *input, Tensor *target);

// Reset to beginning
void dataloader_reset(DataLoader *dl);

// Shuffle (randomize starting position)
void dataloader_shuffle(DataLoader *dl);

// Get total number of batches per epoch
size_t dataloader_n_batches(const DataLoader *dl);

#endif // DATALOADER_H
