#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#include "src/core/tensor.h"
#include "src/core/metal_backend.h"
#include "src/core/autograd.h"
#include "src/nn/layers.h"
#include "src/nn/attention.h"
#include "src/nn/qwen3.h"

static void print_tensor_stats(const char *name, Tensor *t) {
    float sum = 0, abssum = 0, mx = -1e30f, mn = 1e30f;
    for (int i = 0; i < (int)t->size; i++) {
        sum += t->data[i];
        abssum += fabsf(t->data[i]);
        if (t->data[i] > mx) mx = t->data[i];
        if (t->data[i] < mn) mn = t->data[i];
    }
    printf("  %s: size=%zu, mean=%.6f, abssum=%.2f, min=%.6f, max=%.6f\n",
           name, t->size, sum / t->size, abssum, mn, mx);
    printf("    first 5: ");
    for (int i = 0; i < 5 && i < (int)t->size; i++) printf("%.6f ", t->data[i]);
    printf("\n");
}

int main(void) {
    metal_init("shaders/kernels.metal");

    printf("Loading model...\n");
    Qwen3Model *model = qwen3_load_gguf("models/Qwen3-4B-Q8_0.gguf");
    if (!model) { printf("Failed\n"); return 1; }

    // Single token test: token 872 = "user"
    int tok_shape[] = {1};
    Tensor *tokens = tensor_create(tok_shape, 1, DTYPE_UINT32, 0);
    tokens->data_u32[0] = 872;  // "user" token

    printf("\n=== Single token forward (token=872) ===\n");

    // Step 1: Embedding
    Tensor *embed = embedding_forward(model->token_emb, tokens, NULL);
    metal_synchronize();
    print_tensor_stats("embedding", embed);

    // Step 2: First layer input norm
    Tensor *ln1 = rms_norm_forward(model->blocks[0]->input_norm, embed, NULL);
    metal_synchronize();
    print_tensor_stats("layer0_input_norm", ln1);

    // Step 3: Q projection
    Tensor *q = linear_forward(model->blocks[0]->attn->q_proj, ln1, NULL);
    metal_synchronize();
    print_tensor_stats("layer0_q_proj", q);

    // Step 4: Full forward through model
    Tensor *logits = qwen3_forward(model, tokens, 1, 1, NULL);
    metal_synchronize();
    print_tensor_stats("logits", logits);

    // Top 5 tokens
    printf("\n  Top 5 logits:\n");
    float *ldata = logits->data;
    for (int rank = 0; rank < 5; rank++) {
        int best = 0;
        float best_val = -1e30f;
        for (int v = 0; v < model->vocab_size; v++) {
            if (ldata[v] > best_val) {
                best_val = ldata[v];
                best = v;
            }
        }
        printf("    [%d] token=%d, logit=%.4f\n", rank, best, best_val);
        ldata[best] = -1e30f;  // mask for next iteration
    }

    tensor_free(tokens);
    tensor_free(embed);
    tensor_free(ln1);
    tensor_free(q);
    tensor_free(logits);
    qwen3_free(model);
    metal_cleanup();
    return 0;
}
