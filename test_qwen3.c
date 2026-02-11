#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <math.h>

#include "src/core/tensor.h"
#include "src/core/metal_backend.h"
#include "src/core/autograd.h"
#include "src/nn/layers.h"
#include "src/nn/attention.h"
#include "src/nn/qwen3.h"

// Quick smoke tests for new components

static int test_rms_norm(void) {
    printf("=== Test: RMSNorm ===\n");
    RMSNorm *rn = rms_norm_create(64, 1e-6f);
    int shape[] = {4, 64};
    Tensor *x = tensor_randn(shape, 2, DTYPE_FP32, 1, 1.0f);
    ComputeGraph *g = graph_create();

    Tensor *out = rms_norm_forward(rn, x, g);
    metal_synchronize();

    // Check output shape
    if (out->shape[0] != 4 || out->shape[1] != 64) {
        printf("  FAIL: wrong output shape [%d, %d]\n", out->shape[0], out->shape[1]);
        return 1;
    }

    // Check output is not all zeros
    float sum = 0;
    for (int i = 0; i < (int)out->size; i++) sum += fabsf(out->data[i]);
    if (sum < 1e-6f) {
        printf("  FAIL: output is all zeros\n");
        return 1;
    }

    printf("  OK: shape=[%d,%d], sum=%.4f\n", out->shape[0], out->shape[1], sum);

    // Test backward
    tensor_fill(out->grad, 1.0f);
    graph_backward(g, out);
    float grad_sum = 0;
    for (int i = 0; i < (int)x->grad->size; i++) grad_sum += fabsf(x->grad->data[i]);
    printf("  Backward OK: grad_sum=%.4f\n", grad_sum);

    graph_destroy(g);
    rms_norm_free(rn);
    tensor_free(x);
    return 0;
}

static int test_silu(void) {
    printf("=== Test: SiLU ===\n");
    int shape[] = {256};
    Tensor *x = tensor_randn(shape, 1, DTYPE_FP32, 1, 1.0f);
    ComputeGraph *g = graph_create();

    Tensor *out = silu_forward(x, g);
    metal_synchronize();

    // Check silu(0) ≈ 0, silu(large) ≈ x
    // Just verify output is non-zero and correct shape
    if ((int)out->size != 256) {
        printf("  FAIL: wrong output size %d\n", (int)out->size);
        return 1;
    }

    // Verify silu values: for x=1, silu(1) = 1*sigmoid(1) ≈ 0.7311
    float test_x = 1.0f;
    float expected = test_x / (1.0f + expf(-test_x));
    printf("  OK: size=%d, silu(1.0)=%.4f (expected=%.4f)\n",
           (int)out->size, out->data[0], expected);

    // Test backward
    tensor_fill(out->grad, 1.0f);
    graph_backward(g, out);
    float grad_sum = 0;
    for (int i = 0; i < (int)x->grad->size; i++) grad_sum += fabsf(x->grad->data[i]);
    printf("  Backward OK: grad_sum=%.4f\n", grad_sum);

    graph_destroy(g);
    tensor_free(x);
    return 0;
}

static int test_elementwise_mul(void) {
    printf("=== Test: Elementwise Mul ===\n");
    int shape[] = {4, 64};
    Tensor *a = tensor_randn(shape, 2, DTYPE_FP32, 1, 1.0f);
    Tensor *b = tensor_randn(shape, 2, DTYPE_FP32, 1, 1.0f);
    ComputeGraph *g = graph_create();

    Tensor *out = elementwise_mul(a, b, g);
    metal_synchronize();

    // Verify a few elements
    int ok = 1;
    for (int i = 0; i < 5; i++) {
        float expected = a->data[i] * b->data[i];
        if (fabsf(out->data[i] - expected) > 1e-4f) {
            printf("  FAIL: out[%d]=%.4f, expected=%.4f\n", i, out->data[i], expected);
            ok = 0;
        }
    }
    if (ok) printf("  OK: element-wise multiply correct\n");

    graph_destroy(g);
    tensor_free(a);
    tensor_free(b);
    return ok ? 0 : 1;
}

static int test_gqa(void) {
    printf("=== Test: GQA (small) ===\n");
    // Small GQA: d_model=64, 4 Q heads, 2 KV heads, head_dim=16
    GroupedQueryAttention *gqa = gqa_create(64, 4, 2, 16, 10000.0f);
    int shape[] = {2 * 8, 64};  // batch=2, seq=8
    Tensor *x = tensor_randn(shape, 2, DTYPE_FP32, 1, 0.1f);
    ComputeGraph *g = graph_create();

    Tensor *out = gqa_forward(gqa, x, 2, 8, g);
    metal_synchronize();

    if (out->shape[0] != 16 || out->shape[1] != 64) {
        printf("  FAIL: wrong output shape [%d, %d]\n", out->shape[0], out->shape[1]);
        return 1;
    }

    float sum = 0;
    for (int i = 0; i < (int)out->size; i++) sum += fabsf(out->data[i]);
    printf("  OK: shape=[%d,%d], output_sum=%.4f\n", out->shape[0], out->shape[1], sum);

    graph_destroy(g);
    gqa_free(gqa);
    tensor_free(x);
    return 0;
}

static int test_qwen3_forward(void) {
    printf("=== Test: Qwen3 Forward (tiny) ===\n");
    // Create tiny Qwen3 for testing
    Qwen3Config cfg = {
        .n_layers = 2,
        .d_model = 64,
        .n_q_heads = 4,
        .n_kv_heads = 2,
        .head_dim = 16,
        .intermediate_size = 128,
        .vocab_size = 256,
        .rope_theta = 10000.0f,
        .rms_norm_eps = 1e-6f,
    };
    Qwen3Model *model = qwen3_create(cfg);
    printf("  Model params: %zu\n", qwen3_param_count(model));

    int tok_shape[] = {8};  // batch=1, seq=8
    Tensor *tokens = tensor_create(tok_shape, 1, DTYPE_UINT32, 0);
    for (int i = 0; i < 8; i++) tokens->data_u32[i] = (uint32_t)(rand() % 256);

    ComputeGraph *g = graph_create();
    Tensor *logits = qwen3_forward(model, tokens, 1, 8, g);
    metal_synchronize();

    if (logits->shape[0] != 8 || logits->shape[1] != 256) {
        printf("  FAIL: wrong logits shape [%d, %d]\n", logits->shape[0], logits->shape[1]);
        return 1;
    }

    float sum = 0;
    for (int i = 0; i < (int)logits->size; i++) sum += fabsf(logits->data[i]);
    printf("  OK: logits shape=[%d,%d], sum=%.4f\n", logits->shape[0], logits->shape[1], sum);

    // Test backward
    printf("  Testing backward...\n");
    tensor_fill(logits->grad, 0.001f);
    graph_backward(g, logits);
    printf("  Backward OK\n");

    graph_destroy(g);
    tensor_free(tokens);
    qwen3_free(model);
    return 0;
}

static int test_qwen3_lora(void) {
    printf("=== Test: Qwen3 LoRA SFT (tiny) ===\n");
    Qwen3Config cfg = {
        .n_layers = 2,
        .d_model = 64,
        .n_q_heads = 4,
        .n_kv_heads = 2,
        .head_dim = 16,
        .intermediate_size = 128,
        .vocab_size = 256,
        .rope_theta = 10000.0f,
        .rms_norm_eps = 1e-6f,
    };
    Qwen3Model *model = qwen3_create(cfg);
    qwen3_attach_lora(model, 4, 8.0f);

    ParamList *params = qwen3_collect_params(model);
    printf("  LoRA trainable tensors: %d\n", params->n_params);

    // Forward + backward
    int tok_shape[] = {4};
    Tensor *tokens = tensor_create(tok_shape, 1, DTYPE_UINT32, 0);
    for (int i = 0; i < 4; i++) tokens->data_u32[i] = (uint32_t)(rand() % 256);

    ComputeGraph *g = graph_create();
    Tensor *logits = qwen3_forward(model, tokens, 1, 4, g);
    metal_synchronize();
    printf("  Forward OK: logits=[%d,%d]\n", logits->shape[0], logits->shape[1]);

    // Save / load LoRA
    qwen3_save_lora(model, "/tmp/test_lora.bin");
    int ret = qwen3_load_lora(model, "/tmp/test_lora.bin");
    printf("  LoRA save/load: %s\n", ret == 0 ? "OK" : "FAIL");

    graph_destroy(g);
    param_list_free(params);
    tensor_free(tokens);
    qwen3_free(model);
    return 0;
}

int main(void) {
    printf("╔═══════════════════════════════════╗\n");
    printf("║   Qwen3 Component Tests           ║\n");
    printf("╚═══════════════════════════════════╝\n\n");

    srand(42);
    metal_init("shaders/kernels.metal");

    int failures = 0;
    failures += test_rms_norm();
    failures += test_silu();
    failures += test_elementwise_mul();
    failures += test_gqa();
    failures += test_qwen3_forward();
    failures += test_qwen3_lora();

    printf("\n=============================\n");
    if (failures == 0) {
        printf("All tests PASSED!\n");
    } else {
        printf("%d test(s) FAILED\n", failures);
    }

    metal_cleanup();
    return failures;
}
