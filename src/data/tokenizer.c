#include "tokenizer.h"
#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <ctype.h>

Tokenizer *tokenizer_create_byte_level(void) {
    Tokenizer *tok = calloc(1, sizeof(Tokenizer));
    tok->vocab_size = 256;
    tok->n_merges = 0;
    for (int i = 0; i < 256; i++) {
        tok->tokens[i][0] = (char)i;
        tok->tokens[i][1] = '\0';
    }
    return tok;
}

Tokenizer *tokenizer_create(const char *vocab_path, const char *merges_path) {
    Tokenizer *tok = calloc(1, sizeof(Tokenizer));

    // Load vocab: simple text format, one token per line
    FILE *vf = fopen(vocab_path, "r");
    if (!vf) {
        fprintf(stderr, "[Tokenizer] Cannot open vocab: %s\n", vocab_path);
        // Fallback to byte-level
        free(tok);
        return tokenizer_create_byte_level();
    }

    char line[1024];
    tok->vocab_size = 0;
    while (fgets(line, sizeof(line), vf) && tok->vocab_size < MAX_VOCAB_SIZE) {
        // Remove trailing newline
        int len = (int)strlen(line);
        while (len > 0 && (line[len-1] == '\n' || line[len-1] == '\r')) line[--len] = '\0';
        strncpy(tok->tokens[tok->vocab_size], line, MAX_TOKEN_LEN - 1);
        tok->vocab_size++;
    }
    fclose(vf);

    // Load merges
    if (merges_path) {
        FILE *mf = fopen(merges_path, "r");
        if (mf) {
            tok->n_merges = 0;
            while (fgets(line, sizeof(line), mf) && tok->n_merges < MAX_MERGES) {
                int left, right, result;
                if (sscanf(line, "%d %d %d", &left, &right, &result) == 3) {
                    tok->merges[tok->n_merges].left = left;
                    tok->merges[tok->n_merges].right = right;
                    tok->merges[tok->n_merges].result = result;
                    tok->n_merges++;
                }
            }
            fclose(mf);
        }
    }

    printf("[Tokenizer] Loaded %d tokens, %d merges\n", tok->vocab_size, tok->n_merges);
    return tok;
}

void tokenizer_free(Tokenizer *tok) {
    free(tok);
}

// Simple byte-level encoding (when no BPE merges)
static int *encode_byte_level(const char *text, int *out_len) {
    int len = (int)strlen(text);
    int *tokens = malloc(len * sizeof(int));
    for (int i = 0; i < len; i++) {
        tokens[i] = (unsigned char)text[i];
    }
    *out_len = len;
    return tokens;
}

int *tokenizer_encode(const Tokenizer *tok, const char *text, int *out_len) {
    if (tok->n_merges == 0) {
        return encode_byte_level(text, out_len);
    }

    // Start with byte-level tokens
    int len = (int)strlen(text);
    int *ids = malloc(len * sizeof(int));
    int n = len;
    for (int i = 0; i < len; i++) {
        ids[i] = (unsigned char)text[i];
    }

    // Apply BPE merges greedily
    for (int m = 0; m < tok->n_merges; m++) {
        int left = tok->merges[m].left;
        int right = tok->merges[m].right;
        int result = tok->merges[m].result;

        int new_n = 0;
        int *new_ids = malloc(n * sizeof(int));
        for (int i = 0; i < n; i++) {
            if (i + 1 < n && ids[i] == left && ids[i + 1] == right) {
                new_ids[new_n++] = result;
                i++; // skip next
            } else {
                new_ids[new_n++] = ids[i];
            }
        }
        free(ids);
        ids = new_ids;
        n = new_n;
    }

    *out_len = n;
    return ids;
}

char *tokenizer_decode(const Tokenizer *tok, const int *tokens, int n_tokens) {
    // Estimate output size
    size_t total = 0;
    for (int i = 0; i < n_tokens; i++) {
        if (tokens[i] >= 0 && tokens[i] < tok->vocab_size) {
            total += strlen(tok->tokens[tokens[i]]);
        }
    }

    char *out = calloc(total + 1, 1);
    size_t pos = 0;
    for (int i = 0; i < n_tokens; i++) {
        if (tokens[i] >= 0 && tokens[i] < tok->vocab_size) {
            const char *t = tok->tokens[tokens[i]];
            size_t tlen = strlen(t);
            memcpy(out + pos, t, tlen);
            pos += tlen;
        }
    }
    out[pos] = '\0';
    return out;
}
