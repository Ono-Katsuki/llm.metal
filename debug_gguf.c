#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#include "src/core/tensor.h"
#include "src/core/metal_backend.h"
#include "src/data/gguf.h"

int main(void) {
    metal_init("shaders/kernels.metal");

    GGUFFile *f = gguf_open("models/Qwen3-4B-Q8_0.gguf");
    if (!f) { printf("Failed to open GGUF\n"); return 1; }

    // 1. Check output_norm.weight (F32)
    {
        size_t n = 0;
        float *data = gguf_load_tensor_f32(f, "output_norm.weight", &n);
        printf("output_norm.weight: n=%zu\n", n);
        printf("  First 20: ");
        for (int i = 0; i < 20 && i < (int)n; i++) printf("%.6f ", data[i]);
        printf("\n");
        free(data);
    }

    // 2. Check token_embd.weight (Q8_0) - first 20 values
    {
        size_t n = 0;
        float *data = gguf_load_tensor_f32(f, "token_embd.weight", &n);
        printf("\ntoken_embd.weight: n=%zu\n", n);
        printf("  First 20: ");
        for (int i = 0; i < 20 && i < (int)n; i++) printf("%.8f ", data[i]);
        printf("\n");

        // Token 151644 embedding
        int ne0 = 2560;
        int token = 151644;
        int start = token * ne0;
        printf("  Token %d embedding (first 10): ", token);
        for (int i = 0; i < 10; i++) printf("%.8f ", data[start + i]);
        printf("\n");
        free(data);
    }

    // 3. Check a Q8_0 weight matrix (attn_q)
    {
        size_t n = 0;
        float *data = gguf_load_tensor_f32(f, "blk.0.attn_q.weight", &n);
        printf("\nblk.0.attn_q.weight: n=%zu\n", n);
        printf("  First 10: ");
        for (int i = 0; i < 10 && i < (int)n; i++) printf("%.8f ", data[i]);
        printf("\n");
        free(data);
    }

    gguf_close(f);
    metal_cleanup();
    return 0;
}
