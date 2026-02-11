#ifndef TOKENIZER_H
#define TOKENIZER_H

#include <stddef.h>

#define MAX_VOCAB_SIZE  65536
#define MAX_TOKEN_LEN   128
#define MAX_MERGES      65536

typedef struct {
    char tokens[MAX_VOCAB_SIZE][MAX_TOKEN_LEN];
    int vocab_size;

    // BPE merge rules
    struct {
        int left;
        int right;
        int result;
    } merges[MAX_MERGES];
    int n_merges;
} Tokenizer;

// Load tokenizer from vocab.json and merges.txt
Tokenizer *tokenizer_create(const char *vocab_path, const char *merges_path);

// Create a simple byte-level tokenizer (256 tokens, no merges)
Tokenizer *tokenizer_create_byte_level(void);

void tokenizer_free(Tokenizer *tok);

// Encode text to token IDs
int *tokenizer_encode(const Tokenizer *tok, const char *text, int *out_len);

// Decode token IDs to text
char *tokenizer_decode(const Tokenizer *tok, const int *tokens, int n_tokens);

#endif // TOKENIZER_H
