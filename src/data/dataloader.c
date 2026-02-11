#include "dataloader.h"
#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <fcntl.h>
#include <unistd.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <time.h>

DataLoader *dataloader_create(const char *filepath, const Tokenizer *tok,
                              int batch_size, int seq_len) {
    DataLoader *dl = calloc(1, sizeof(DataLoader));
    dl->batch_size = batch_size;
    dl->seq_len = seq_len;
    dl->tokenizer = tok;

    // Memory-map the file
    dl->fd = open(filepath, O_RDONLY);
    if (dl->fd < 0) {
        fprintf(stderr, "[DataLoader] Cannot open: %s\n", filepath);
        free(dl);
        return NULL;
    }

    struct stat st;
    fstat(dl->fd, &st);
    dl->mmap_size = (size_t)st.st_size;
    dl->mmap_data = mmap(NULL, dl->mmap_size, PROT_READ, MAP_PRIVATE, dl->fd, 0);
    if (dl->mmap_data == MAP_FAILED) {
        fprintf(stderr, "[DataLoader] mmap failed for: %s\n", filepath);
        close(dl->fd);
        free(dl);
        return NULL;
    }

    // Tokenize the entire file
    int n_tokens = 0;
    dl->token_ids = tokenizer_encode(tok, dl->mmap_data, &n_tokens);
    dl->n_tokens = (size_t)n_tokens;
    dl->current_pos = 0;

    printf("[DataLoader] Loaded %s: %zu bytes, %zu tokens\n",
           filepath, dl->mmap_size, dl->n_tokens);
    printf("[DataLoader] Batch size: %d, Seq len: %d, Batches/epoch: %zu\n",
           batch_size, seq_len, dataloader_n_batches(dl));

    return dl;
}

DataLoader *dataloader_create_from_tokens(const char *filepath,
                                          int batch_size, int seq_len) {
    DataLoader *dl = calloc(1, sizeof(DataLoader));
    dl->batch_size = batch_size;
    dl->seq_len = seq_len;

    // Memory-map the token file
    dl->fd = open(filepath, O_RDONLY);
    if (dl->fd < 0) {
        fprintf(stderr, "[DataLoader] Cannot open: %s\n", filepath);
        free(dl);
        return NULL;
    }

    struct stat st;
    fstat(dl->fd, &st);
    dl->mmap_size = (size_t)st.st_size;
    dl->mmap_data = mmap(NULL, dl->mmap_size, PROT_READ, MAP_PRIVATE, dl->fd, 0);
    if (dl->mmap_data == MAP_FAILED) {
        close(dl->fd);
        free(dl);
        return NULL;
    }

    // Treat file as array of int32
    dl->n_tokens = dl->mmap_size / sizeof(int);
    dl->token_ids = (int *)dl->mmap_data; // zero-copy
    dl->current_pos = 0;

    printf("[DataLoader] Loaded %s: %zu tokens (pre-tokenized)\n",
           filepath, dl->n_tokens);
    return dl;
}

void dataloader_free(DataLoader *dl) {
    if (!dl) return;
    if (dl->mmap_data && dl->mmap_data != MAP_FAILED) {
        // If tokens were from tokenize (not mmap), free them
        if ((void *)dl->token_ids != (void *)dl->mmap_data) {
            free(dl->token_ids);
        }
        munmap(dl->mmap_data, dl->mmap_size);
    }
    if (dl->fd >= 0) close(dl->fd);
    free(dl);
}

int dataloader_next_batch(DataLoader *dl, Tensor *input, Tensor *target) {
    size_t batch_tokens = (size_t)dl->batch_size * (size_t)dl->seq_len;

    // Check if we have enough tokens
    if (dl->current_pos + batch_tokens + 1 > dl->n_tokens) {
        dl->current_pos = 0;
        return -1; // epoch end
    }

    // Fill input and target tensors
    // input[b*seq + s] = token_ids[current_pos + b*(seq_len) + s]
    // target[b*seq + s] = token_ids[current_pos + b*(seq_len) + s + 1]
    float *inp = input->data;
    float *tgt = target->data;

    for (int b = 0; b < dl->batch_size; b++) {
        size_t base = dl->current_pos + (size_t)b * (size_t)dl->seq_len;
        for (int s = 0; s < dl->seq_len; s++) {
            inp[b * dl->seq_len + s] = (float)dl->token_ids[base + s];
            tgt[b * dl->seq_len + s] = (float)dl->token_ids[base + s + 1];
        }
    }

    dl->current_pos += batch_tokens;
    return 0;
}

void dataloader_reset(DataLoader *dl) {
    dl->current_pos = 0;
}

void dataloader_shuffle(DataLoader *dl) {
    // Random starting offset within the data
    size_t max_offset = dl->n_tokens / 2;
    if (max_offset > 0) {
        dl->current_pos = (size_t)(rand() % (int)max_offset);
    }
}

size_t dataloader_n_batches(const DataLoader *dl) {
    size_t batch_tokens = (size_t)dl->batch_size * (size_t)dl->seq_len;
    if (batch_tokens == 0) return 0;
    return (dl->n_tokens - 1) / batch_tokens;
}
