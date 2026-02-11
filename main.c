#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <math.h>
#include <sys/stat.h>

#include "src/core/tensor.h"
#include "src/core/metal_backend.h"
#include "src/core/autograd.h"
#include "src/nn/transformer.h"
#include "src/nn/optimizer.h"
#include "src/nn/qwen3.h"
#include "src/nn/fast_inference.h"
#include "src/nn/fast_sft.h"
#include "src/data/tokenizer.h"
#include "src/data/dataloader.h"
#include "src/train/trainer.h"

// ==================================================================
// Configuration
// ==================================================================

typedef struct {
    // Model
    int n_layers;
    int d_model;
    int n_heads;
    int vocab_size;
    int max_seq_len;

    // Training
    int batch_size;
    int seq_len;
    int grad_accum_steps;
    int max_steps;
    float lr;
    float weight_decay;
    int warmup_steps;

    // Data
    char train_data[256];
    char val_data[256];
    char vocab_path[256];
    char merges_path[256];

    // Checkpointing
    char checkpoint_dir[256];
    char resume_from[256];
    int save_interval;
    int log_interval;
    int eval_interval;

    // Flags
    int gradient_checkpointing;
    int test_mode;

    // Qwen3 SFT mode
    char mode[32];           // "train", "sft", "generate"
    char gguf_path[512];     // path to GGUF model file
    char lora_adapter[512];  // path to LoRA adapter file
    int lora_rank;
    float lora_alpha;
    char prompt[1024];       // prompt for generate mode
    char token_file[512];    // pre-tokenized prompt file (uint32 binary)
    int max_gen_len;         // max generation length
    int use_fp16;            // 1 = F16 weights, 0 = Q8_0 quantized
    int gen_batch;           // batch size for parallel generation
} Config;

static Config default_config(void) {
    Config c = {0};
    // Small model for testing
    c.n_layers = 4;
    c.d_model = 128;
    c.n_heads = 4;
    c.vocab_size = 256;     // byte-level
    c.max_seq_len = 256;

    c.batch_size = 2;
    c.seq_len = 64;
    c.grad_accum_steps = 1;
    c.max_steps = 100;
    c.lr = 3e-4f;
    c.weight_decay = 0.01f;
    c.warmup_steps = 10;

    strncpy(c.checkpoint_dir, "checkpoints", 255);
    c.save_interval = 500;
    c.log_interval = 1;
    c.eval_interval = 50;

    c.gradient_checkpointing = 0;
    c.test_mode = 0;

    strncpy(c.mode, "train", 31);
    c.lora_rank = 16;
    c.lora_alpha = 32.0f;
    c.max_gen_len = 128;
    return c;
}

// ==================================================================
// Preset configurations
// ==================================================================

static Config config_125M(void) {
    Config c = default_config();
    c.n_layers = 12;
    c.d_model = 768;
    c.n_heads = 12;
    c.vocab_size = 32000;
    c.max_seq_len = 1024;
    c.batch_size = 4;
    c.seq_len = 512;
    c.grad_accum_steps = 4;
    c.max_steps = 10000;
    c.warmup_steps = 100;
    return c;
}

static Config config_350M(void) {
    Config c = default_config();
    c.n_layers = 24;
    c.d_model = 1024;
    c.n_heads = 16;
    c.vocab_size = 32000;
    c.max_seq_len = 1024;
    c.batch_size = 2;
    c.seq_len = 512;
    c.grad_accum_steps = 8;
    c.max_steps = 10000;
    c.warmup_steps = 200;
    c.gradient_checkpointing = 1;
    return c;
}

// ==================================================================
// CLI argument parsing
// ==================================================================

static void print_usage(const char *prog) {
    printf("Usage: %s [options]\n\n", prog);
    printf("Modes:\n");
    printf("  --mode train        GPT pre-training (default)\n");
    printf("  --mode sft          Qwen3 LoRA SFT with GGUF weights\n");
    printf("  --mode generate     Qwen3 text generation\n\n");
    printf("General options:\n");
    printf("  --preset <name>     Model preset: tiny, 125M, 350M\n");
    printf("  --layers <n>        Number of transformer layers\n");
    printf("  --dim <n>           Model dimension\n");
    printf("  --heads <n>         Number of attention heads\n");
    printf("  --vocab <n>         Vocabulary size\n");
    printf("  --batch <n>         Batch size\n");
    printf("  --seq <n>           Sequence length\n");
    printf("  --steps <n>         Max training steps\n");
    printf("  --lr <f>            Learning rate\n");
    printf("  --train <path>      Training data file\n");
    printf("  --val <path>        Validation data file\n");
    printf("  --checkpoint <dir>  Checkpoint directory\n");
    printf("  --resume <path>     Resume from checkpoint\n");
    printf("  --test              Run in test mode (random data)\n\n");
    printf("Qwen3 SFT options:\n");
    printf("  --model <path>      Model file/dir (.gguf or .safetensors)\n");
    printf("  --gguf <path>       GGUF model file (alias for --model)\n");
    printf("  --lora-rank <n>     LoRA rank (default: 16)\n");
    printf("  --lora-alpha <f>    LoRA alpha (default: 32)\n");
    printf("  --lora-adapter <p>  LoRA adapter file (load/save)\n\n");
    printf("Generate options:\n");
    printf("  --prompt <text>     Prompt text\n");
    printf("  --tokens <file>     Pre-tokenized prompt (uint32 binary)\n");
    printf("  --max-gen-len <n>   Max generation length (default: 128)\n");
    printf("  --fp16              Use F16 weights instead of Q8_0 quantization\n");
    printf("  --gen-batch <n>     Batch size for parallel generation (F16 only)\n");
    printf("  --help              Show this message\n");
}

static Config parse_args(int argc, char **argv) {
    Config c = default_config();

    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "--help") == 0) {
            print_usage(argv[0]);
            exit(0);
        } else if (strcmp(argv[i], "--mode") == 0 && i + 1 < argc) {
            strncpy(c.mode, argv[++i], 31);
        } else if ((strcmp(argv[i], "--gguf") == 0 || strcmp(argv[i], "--model") == 0) && i + 1 < argc) {
            strncpy(c.gguf_path, argv[++i], 511);
        } else if (strcmp(argv[i], "--lora-rank") == 0 && i + 1 < argc) {
            c.lora_rank = atoi(argv[++i]);
        } else if (strcmp(argv[i], "--lora-alpha") == 0 && i + 1 < argc) {
            c.lora_alpha = (float)atof(argv[++i]);
        } else if (strcmp(argv[i], "--lora-adapter") == 0 && i + 1 < argc) {
            strncpy(c.lora_adapter, argv[++i], 511);
        } else if (strcmp(argv[i], "--prompt") == 0 && i + 1 < argc) {
            strncpy(c.prompt, argv[++i], 1023);
        } else if (strcmp(argv[i], "--tokens") == 0 && i + 1 < argc) {
            strncpy(c.token_file, argv[++i], 511);
        } else if (strcmp(argv[i], "--max-gen-len") == 0 && i + 1 < argc) {
            c.max_gen_len = atoi(argv[++i]);
        } else if (strcmp(argv[i], "--preset") == 0 && i + 1 < argc) {
            i++;
            if (strcmp(argv[i], "125M") == 0) c = config_125M();
            else if (strcmp(argv[i], "350M") == 0) c = config_350M();
            else printf("Unknown preset: %s, using default\n", argv[i]);
        } else if (strcmp(argv[i], "--layers") == 0 && i + 1 < argc) {
            c.n_layers = atoi(argv[++i]);
        } else if (strcmp(argv[i], "--dim") == 0 && i + 1 < argc) {
            c.d_model = atoi(argv[++i]);
        } else if (strcmp(argv[i], "--heads") == 0 && i + 1 < argc) {
            c.n_heads = atoi(argv[++i]);
        } else if (strcmp(argv[i], "--vocab") == 0 && i + 1 < argc) {
            c.vocab_size = atoi(argv[++i]);
        } else if (strcmp(argv[i], "--batch") == 0 && i + 1 < argc) {
            c.batch_size = atoi(argv[++i]);
        } else if (strcmp(argv[i], "--seq") == 0 && i + 1 < argc) {
            c.seq_len = atoi(argv[++i]);
        } else if (strcmp(argv[i], "--steps") == 0 && i + 1 < argc) {
            c.max_steps = atoi(argv[++i]);
        } else if (strcmp(argv[i], "--lr") == 0 && i + 1 < argc) {
            c.lr = (float)atof(argv[++i]);
        } else if (strcmp(argv[i], "--train") == 0 && i + 1 < argc) {
            strncpy(c.train_data, argv[++i], 255);
        } else if (strcmp(argv[i], "--val") == 0 && i + 1 < argc) {
            strncpy(c.val_data, argv[++i], 255);
        } else if (strcmp(argv[i], "--checkpoint") == 0 && i + 1 < argc) {
            strncpy(c.checkpoint_dir, argv[++i], 255);
        } else if (strcmp(argv[i], "--resume") == 0 && i + 1 < argc) {
            strncpy(c.resume_from, argv[++i], 255);
        } else if (strcmp(argv[i], "--fp16") == 0) {
            c.use_fp16 = 1;
        } else if (strcmp(argv[i], "--gen-batch") == 0 && i + 1 < argc) {
            c.gen_batch = atoi(argv[++i]);
        } else if (strcmp(argv[i], "--test") == 0) {
            c.test_mode = 1;
        }
    }
    return c;
}

// ==================================================================
// Auto-detect model format
// ==================================================================

static Qwen3Model *load_model_auto(const char *path) {
    struct stat _st;
    if (stat(path, &_st) == 0 && S_ISDIR(_st.st_mode)) {
        printf("[Model] Loading safetensors from directory: %s\n", path);
        return qwen3_load_safetensors(path);
    }
    size_t len = strlen(path);
    if (len > 12 && strcmp(path + len - 12, ".safetensors") == 0) {
        printf("[Model] Loading safetensors: %s\n", path);
        return qwen3_load_safetensors(path);
    }
    return qwen3_load_gguf(path);
}

// ==================================================================
// Qwen3 SFT Training
// ==================================================================

static void run_sft(Config *cfg) {
    if (!cfg->gguf_path[0]) {
        fprintf(stderr, "[SFT] Error: --model is required for SFT mode\n");
        return;
    }

    float lr = cfg->lr > 0 ? cfg->lr : 1e-5f;
    printf("=== Qwen3 Fast LoRA SFT ===\n");
    printf("  Model: %s\n", cfg->gguf_path);
    printf("  LoRA rank: %d, alpha: %.1f\n", cfg->lora_rank, cfg->lora_alpha);
    printf("  Seq: %d, Steps: %d\n", cfg->seq_len, cfg->max_steps);
    printf("  LR: %.2e\n\n", lr);

    // Load model (auto-detect format)
    Qwen3Model *model = load_model_auto(cfg->gguf_path);
    if (!model) {
        fprintf(stderr, "[SFT] Failed to load model\n");
        return;
    }
    printf("[SFT] Total model params: %zu (%.2f B)\n",
           qwen3_param_count(model), (float)qwen3_param_count(model) / 1e9f);

    // Load existing LoRA adapter if specified
    if (cfg->lora_adapter[0]) {
        qwen3_attach_lora(model, cfg->lora_rank, cfg->lora_alpha);
        qwen3_load_lora(model, cfg->lora_adapter);
    }

    // Create fast SFT state (converts weights to F16, allocates GPU buffers)
    SFTState *sft = sft_state_create(model, cfg->seq_len,
                                      cfg->lora_rank, cfg->lora_alpha);
    if (!sft) {
        fprintf(stderr, "[SFT] Failed to create SFT state\n");
        qwen3_free(model);
        return;
    }

    // Setup data loader (pre-tokenized binary, batch=1 for fast SFT)
    DataLoader *train_loader = NULL;
    if (cfg->train_data[0]) {
        train_loader = dataloader_create_from_tokens(cfg->train_data, 1, cfg->seq_len);
        if (!train_loader) {
            fprintf(stderr, "[SFT] Warning: could not load train data: %s\n", cfg->train_data);
        }
    }

    // Allocate token buffers
    int BN = cfg->seq_len;
    uint32_t *input_tokens  = calloc(BN, sizeof(uint32_t));
    uint32_t *target_tokens = calloc(BN, sizeof(uint32_t));

    struct timespec start, now;
    clock_gettime(CLOCK_MONOTONIC, &start);

    printf("\n=== Fast SFT Training Start ===\n");
    for (int step = 0; step < cfg->max_steps; step++) {
        if (train_loader) {
            // Load batch via dataloader (float interface → cast to uint32)
            int tok_shape[] = {BN};
            Tensor *input_f  = tensor_create(tok_shape, 1, DTYPE_FP32, 0);
            Tensor *target_f = tensor_create(tok_shape, 1, DTYPE_FP32, 0);
            int ret = dataloader_next_batch(train_loader, input_f, target_f);
            if (ret < 0) {
                dataloader_reset(train_loader);
                dataloader_shuffle(train_loader);
                dataloader_next_batch(train_loader, input_f, target_f);
            }
            for (int i = 0; i < BN; i++) {
                input_tokens[i]  = (uint32_t)input_f->data[i];
                target_tokens[i] = (uint32_t)target_f->data[i];
            }
            tensor_free(input_f);
            tensor_free(target_f);
        } else {
            // Dummy data for testing: fixed repeating tokens
            srand(42);
            for (int i = 0; i < BN; i++) {
                input_tokens[i]  = (uint32_t)(rand() % 1000);  // small vocab subset
                target_tokens[i] = (i + 1 < BN) ? input_tokens[i + 1] : input_tokens[0];
            }
        }

        // Single GPU call: forward + loss + backward + flush
        float loss = sft_train_step(sft, input_tokens, target_tokens, lr);

        if (step % cfg->log_interval == 0) {
            clock_gettime(CLOCK_MONOTONIC, &now);
            double elapsed = (double)(now.tv_sec - start.tv_sec) +
                             (double)(now.tv_nsec - start.tv_nsec) / 1e9;
            printf("step %5d | loss %.4f | lr %.2e | %.1f tok/s\n",
                   step, loss, lr, (double)(step + 1) * BN / elapsed);
        }
    }

    // Sync LoRA weights back to model for saving
    sft_sync_lora_to_model(sft, model);

    // Save LoRA adapter
    char lora_path[512];
    if (cfg->lora_adapter[0]) {
        strncpy(lora_path, cfg->lora_adapter, 511);
    } else {
        snprintf(lora_path, sizeof(lora_path), "%s/lora_adapter.bin", cfg->checkpoint_dir);
    }
    qwen3_save_lora(model, lora_path);

    clock_gettime(CLOCK_MONOTONIC, &now);
    double total_time = (double)(now.tv_sec - start.tv_sec) +
                        (double)(now.tv_nsec - start.tv_nsec) / 1e9;
    printf("\n=== Fast SFT Complete ===\n");
    printf("Total time: %.1f seconds\n", total_time);
    printf("LoRA adapter saved: %s\n", lora_path);

    // Cleanup
    free(input_tokens);
    free(target_tokens);
    sft_state_free(sft);
    if (train_loader) dataloader_free(train_loader);
    qwen3_free(model);
}

// ==================================================================
// Qwen3 Text Generation (greedy)
// ==================================================================

static void run_generate(Config *cfg) {
    if (!cfg->gguf_path[0]) {
        fprintf(stderr, "[Generate] Error: --model is required\n");
        return;
    }

    printf("=== Qwen3 Text Generation ===\n");

    Qwen3Model *model = load_model_auto(cfg->gguf_path);
    if (!model) {
        fprintf(stderr, "[Generate] Failed to load model\n");
        return;
    }

    // Load LoRA adapter if specified
    if (cfg->lora_adapter[0]) {
        qwen3_attach_lora(model, cfg->lora_rank, cfg->lora_alpha);
        qwen3_load_lora(model, cfg->lora_adapter);
        // Merge LoRA into base weights for faster inference
        for (int i = 0; i < model->n_layers; i++) {
            lora_merge(model->q_loras[i]);
            lora_merge(model->v_loras[i]);
        }
        printf("[Generate] LoRA merged into base weights\n");
    }

    // Load prompt tokens
    int prompt_len = 0;
    int max_len = 0;
    uint32_t *tokens = NULL;

    if (cfg->token_file[0]) {
        // Load pre-tokenized prompt from binary file
        FILE *tf = fopen(cfg->token_file, "rb");
        if (!tf) {
            fprintf(stderr, "[Generate] Cannot open token file: %s\n", cfg->token_file);
            qwen3_free(model);
            return;
        }
        fseek(tf, 0, SEEK_END);
        long fsize = ftell(tf);
        fseek(tf, 0, SEEK_SET);
        prompt_len = (int)(fsize / sizeof(uint32_t));
        max_len = prompt_len + cfg->max_gen_len;
        if (max_len > 2048) max_len = 2048;
        tokens = calloc(max_len, sizeof(uint32_t));
        fread(tokens, sizeof(uint32_t), prompt_len, tf);
        fclose(tf);
        printf("Loaded %d tokens from %s\n", prompt_len, cfg->token_file);
        printf("Token IDs: ");
        for (int i = 0; i < prompt_len; i++) printf("%u ", tokens[i]);
        printf("\n");
    } else {
        // Byte-level fallback for quick testing
        const char *prompt = cfg->prompt[0] ? cfg->prompt : "Hello";
        prompt_len = (int)strlen(prompt);
        max_len = prompt_len + cfg->max_gen_len;
        if (max_len > 2048) max_len = 2048;
        tokens = calloc(max_len, sizeof(uint32_t));
        for (int i = 0; i < prompt_len; i++) {
            tokens[i] = (uint32_t)(unsigned char)prompt[i];
        }
        printf("Prompt (byte-level): \"%s\"\n", prompt);
    }
    printf("Generating up to %d tokens...\n", cfg->max_gen_len);

    // Fast inference with pre-allocated buffers + KV cache + Accelerate BLAS
    uint32_t *gen_tokens = calloc(cfg->max_gen_len, sizeof(uint32_t));
    int n_generated;
    if (cfg->gen_batch > 1) {
        printf("[Generate] Batched generation: batch=%d, F16\n", cfg->gen_batch);
        free(gen_tokens);
        gen_tokens = calloc((size_t)cfg->gen_batch * cfg->max_gen_len, sizeof(uint32_t));
        n_generated = qwen3_generate_fast_batch(model, tokens, prompt_len,
                                                 gen_tokens, cfg->max_gen_len, max_len,
                                                 cfg->gen_batch);
    } else {
        if (cfg->use_fp16)
            printf("[Generate] Using F16 precision (no quantization)\n");
        n_generated = qwen3_generate_fast(model, tokens, prompt_len,
                                          gen_tokens, cfg->max_gen_len, max_len,
                                          cfg->use_fp16);
    }

    // Copy generated tokens to the full sequence
    int cur_len = prompt_len;
    for (int i = 0; i < n_generated; i++) {
        tokens[cur_len++] = gen_tokens[i];
    }
    free(gen_tokens);
    printf("\n[Generate] Generated %d tokens.\n", n_generated);
    printf("[Generate] Token IDs: ");
    for (int i = prompt_len; i < cur_len; i++) printf("%u ", tokens[i]);
    printf("\n");

    // Save to binary file for Python decoding
    const char *out_path = "/tmp/generated_tokens.bin";
    FILE *out_f = fopen(out_path, "wb");
    if (out_f) {
        fwrite(&tokens[prompt_len], sizeof(uint32_t), n_generated, out_f);
        fclose(out_f);
        printf("[Generate] Saved to %s\n", out_path);
        printf("[Generate] Decode: python3 tools/decode_tokens.py %s\n", out_path);
    }

    free(tokens);
    qwen3_free(model);
}

// ==================================================================
// GPT Pre-training (original mode)
// ==================================================================

static void run_train(Config *cfg) {
    // Print config
    printf("Model config:\n");
    printf("  Layers:     %d\n", cfg->n_layers);
    printf("  Dimension:  %d\n", cfg->d_model);
    printf("  Heads:      %d\n", cfg->n_heads);
    printf("  Vocab size: %d\n", cfg->vocab_size);
    printf("  Max seq:    %d\n\n", cfg->max_seq_len);

    printf("Training config:\n");
    printf("  Batch size: %d\n", cfg->batch_size);
    printf("  Seq length: %d\n", cfg->seq_len);
    printf("  Grad accum: %d\n", cfg->grad_accum_steps);
    printf("  Max steps:  %d\n", cfg->max_steps);
    printf("  LR:         %.2e\n\n", cfg->lr);

    // Create tokenizer
    Tokenizer *tok;
    if (cfg->vocab_path[0]) {
        tok = tokenizer_create(cfg->vocab_path,
                               cfg->merges_path[0] ? cfg->merges_path : NULL);
    } else {
        tok = tokenizer_create_byte_level();
        printf("[Tokenizer] Using byte-level (256 tokens)\n");
    }

    // Setup training config
    GPTConfig model_cfg = {
        .n_layers = cfg->n_layers,
        .d_model = cfg->d_model,
        .n_heads = cfg->n_heads,
        .vocab_size = cfg->vocab_size,
        .max_seq_len = cfg->max_seq_len,
        .gradient_checkpointing = cfg->gradient_checkpointing,
    };

    AdamWConfig opt_cfg = adamw_default_config();
    opt_cfg.lr = cfg->lr;
    opt_cfg.max_lr = cfg->lr;
    opt_cfg.weight_decay = cfg->weight_decay;
    opt_cfg.warmup_steps = cfg->warmup_steps;
    opt_cfg.total_steps = cfg->max_steps;

    TrainConfig train_cfg = {
        .model_config = model_cfg,
        .opt_config = opt_cfg,
        .batch_size = cfg->batch_size,
        .seq_len = cfg->seq_len,
        .grad_accum_steps = cfg->grad_accum_steps,
        .max_steps = cfg->max_steps,
        .log_interval = cfg->log_interval,
        .eval_interval = cfg->eval_interval,
        .save_interval = cfg->save_interval,
        .checkpoint_dir = cfg->checkpoint_dir,
        .train_data = cfg->train_data[0] ? cfg->train_data : NULL,
        .val_data = cfg->val_data[0] ? cfg->val_data : NULL,
    };

    // Create trainer
    Trainer *trainer = trainer_create(train_cfg, tok);

    // Resume from checkpoint if specified
    if (cfg->resume_from[0]) {
        if (trainer_load_checkpoint(trainer, cfg->resume_from) != 0) {
            fprintf(stderr, "Failed to load checkpoint: %s\n", cfg->resume_from);
        }
    }

    // Run training
    trainer_train(trainer);

    // Cleanup
    trainer_free(trainer);
    tokenizer_free(tok);
}

// ==================================================================
// Main
// ==================================================================

int main(int argc, char **argv) {
    Config cfg = parse_args(argc, argv);

    printf("╔════════════════════════════════════════╗\n");
    printf("║    LLM Training System (Metal GPU)     ║\n");
    printf("╚════════════════════════════════════════╝\n\n");

    srand((unsigned int)time(NULL));

    // Initialize Metal backend
    if (metal_init("shaders/kernels.metal") != 0) {
        printf("[Warning] Metal init failed, falling back to CPU\n");
    }

    // Dispatch based on mode
    if (strcmp(cfg.mode, "sft") == 0) {
        run_sft(&cfg);
    } else if (strcmp(cfg.mode, "generate") == 0) {
        run_generate(&cfg);
    } else {
        run_train(&cfg);
    }

    metal_cleanup();
    printf("\nDone.\n");
    return 0;
}
