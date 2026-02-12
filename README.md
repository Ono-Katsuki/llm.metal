# llm.c — LLM Training & Inference on Apple Silicon

Pure C/Metal implementation of LLM training and inference, optimized for Apple Silicon (M4 Pro).
No PyTorch, no Python runtime — just C, Objective-C, and Metal shaders.

## Supported Models

| Model | Params | Precision | Speed (M4 Pro) |
|-------|--------|-----------|----------------|
| Gemma3 1B | 1B | F16 | ~94 tok/s (generate) |
| Qwen3 4B | 4B | F16 | ~27 tok/s (generate) |
| Qwen3 4B | 4B | Q8_0 | ~21 tok/s (generate) |

### SFT Training (LoRA rank=16, seq=64, M4 Pro)

| Model | Params | Speed |
|-------|--------|-------|
| Gemma3 1B | 1B | ~151 tok/s |
| Qwen3 4B | 4B | ~12 tok/s |

## Features

- **Inference**: Autoregressive generation with KV cache, F16 or Q8_0 matvec
- **SFT Training**: LoRA fine-tuning (rank 16, Q/V projections) with AdamW — **Qwen3 / Gemma3 両対応**
- **GRPO Training**: Group Relative Policy Optimization — LoRA / Full-param Online / Full-param Accurate の 3 モード
- **GPU Pipeline**: Single Metal command buffer — all ops (RMSNorm, RoPE, GQA, SwiGLU/GeGLU, matvec) in one GPU submission
- **Weight Loading**: HuggingFace SafeTensors (multi-shard), GGUF
- **Model Auto-detection**: Reads `config.json` to select Qwen3 or Gemma3 architecture

## Requirements

- macOS with Apple Silicon (M1/M2/M3/M4)
- Xcode Command Line Tools (`xcode-select --install`)
- Python 3 + `transformers` (for tokenization only)

## Build

```bash
make
```

## Quick Start

### 1. Download a Model

```bash
# Gemma3 1B (1.9 GB)
huggingface-cli download google/gemma-3-1b-pt --local-dir ~/models/gemma-3-1b-pt

# Qwen3 4B (8 GB)
huggingface-cli download Qwen/Qwen3-4B
```

### 2. Tokenize a Prompt

```bash
python3 tools/tokenize_prompt.py "The capital of France is" \
  --model google/gemma-3-1b-pt \
  --output /tmp/prompt.bin
```

### 3. Generate

```bash
# Gemma3 1B (F16)
./llm_train --mode generate --model ~/models/gemma-3-1b-pt \
  --fp16 --tokens /tmp/prompt.bin --max-gen-len 50

# Qwen3 4B (F16, faster)
./llm_train --mode generate --model ~/.cache/huggingface/hub/models--Qwen--Qwen3-4B/snapshots/*/ \
  --fp16 --tokens /tmp/prompt.bin --max-gen-len 50

# Qwen3 4B (Q8_0, less memory)
./llm_train --mode generate --model ~/.cache/huggingface/hub/models--Qwen--Qwen3-4B/snapshots/*/ \
  --tokens /tmp/prompt.bin --max-gen-len 50
```

### 4. Decode Output

```bash
python3 tools/decode_tokens.py /tmp/generated_tokens.bin
```

### 5. LoRA SFT (Qwen3 / Gemma3)

モデルタイプは `config.json` から自動判定されます。

```bash
# Prepare training data
python3 tools/tokenize_sft.py --input data/sft.jsonl --output data/sft_tokens.bin

# Train (Qwen3)
./llm_train --mode sft --model path/to/qwen3 \
  --train data/sft_tokens.bin --steps 100 --lr 1e-4 \
  --lora-rank 16 --lora-adapter sft/adapter.bin

# Train (Gemma3)
./llm_train --mode sft --model ~/models/gemma-3-1b-pt \
  --train data/sft_tokens.bin --steps 100 --lr 1e-4 \
  --lora-rank 16 --lora-adapter sft/gemma3_adapter.bin
```

### 6. GRPO Training

GRPO (Group Relative Policy Optimization) で強化学習ファインチューニング。
3つのモードから選択可能：

| モード | フラグ | 説明 | メモリ増分 |
|--------|--------|------|-----------|
| LoRA | (デフォルト) | LoRA アダプター上で AdamW 更新 | 低 |
| Full-param Online | `--full-param` | completion ごとに即座に SGD 更新（高速） | モデル重み F16 分 |
| Full-param Accurate | `--accurate` | F16 勾配蓄積 → 全 completion 後に 1 回 SGD（論文通り） | モデル重み F16 × 2 |

```bash
# Prepare GRPO prompts
python3 tools/tokenize_grpo.py --input data/prompts.jsonl --output data/grpo_prompts.bin

# LoRA GRPO
./llm_train --mode grpo --model ~/models/gemma-3-1b-pt --fp16 \
  --train data/grpo_prompts.bin --steps 100 --lr 1e-5 \
  --group-size 4 --temperature 0.7

# Full-param Online (fast, per-completion SGD)
./llm_train --mode grpo --model ~/models/gemma-3-1b-pt --fp16 \
  --train data/grpo_prompts.bin --steps 100 --lr 1e-5 \
  --full-param --group-size 4

# Full-param Accurate (paper-correct, grad accumulation)
./llm_train --mode grpo --model ~/models/gemma-3-1b-pt --fp16 \
  --train data/grpo_prompts.bin --steps 100 --lr 1e-5 \
  --accurate --group-size 4
```

**メモリ目安 (M4 Pro 24GB):**
- Gemma3-1B Accurate: ~1.9 GB (weights) + ~1.9 GB (grad accum) + KV cache 等
- Qwen3-4B Accurate: ~7.7 GB (weights) + ~7.7 GB (grad accum) + KV cache 等

## Project Structure

```
src/
  core/
    tensor.c/h          Tensor operations
    autograd.c/h        Automatic differentiation (training)
    metal_backend.m/h   Metal GPU backend
    mem_pool.c/h        Memory pool
  nn/
    qwen3.c/h           Qwen3 model definition + SafeTensors loader
    gemma3.c/h          Gemma3 model definition + SafeTensors loader
    fast_inference.c/h  GPU inference engine (F16/Q8_0 matvec, KV cache)
    fast_sft.c/h        GPU SFT training pipeline (Qwen3 + Gemma3)
    fast_metal.m/h      Metal kernel dispatch functions
    wmat.c/h            Weight matrix (Q8_0/F16) + model-independent conversion
    layers.c/h          Linear, Embedding, RMSNorm, etc.
    attention.c/h       Grouped Query Attention
    lora.c/h            LoRA adapters
    transformer.c/h     GPT-2 style transformer (pre-training)
    optimizer.c/h       AdamW optimizer
  data/
    safetensors.c/h     HuggingFace SafeTensors loader (multi-shard, BF16)
    gguf.c/h            GGUF format loader
    tokenizer.c/h       Byte-level tokenizer
    dataloader.c/h      Training data loader
  train/
    trainer.c/h         Training loop
shaders/
  q8_kernels.metal      GPU kernels (Q8_0/F16 matvec, RMSNorm, RoPE, attention, SiLU, GELU, etc.)
  kernels.metal         Legacy kernels
tools/
  tokenize_prompt.py    Tokenize text to binary (uses HuggingFace tokenizers)
  decode_tokens.py      Decode binary token IDs to text
  tokenize_sft.py       Prepare SFT training data
  preprocess_sft.py     SFT data preprocessing
  server.py             Inference server
  inspect_gguf.py       GGUF file inspector
main.c                  CLI entry point
Makefile
```

## CLI Reference

```
./llm_train [options]

Modes:
  --mode generate       Text generation (default)
  --mode sft            LoRA SFT training
  --mode train          GPT pre-training
  --mode serve          Persistent inference server

Model:
  --model <path>        Model directory (SafeTensors) or .gguf file
  --fp16                Use F16 weights (faster, more memory)
                        Default: Q8_0 quantization (slower, ~half memory)

Generation:
  --tokens <file>       Pre-tokenized prompt (uint32 binary)
  --prompt <text>       Text prompt (byte-level tokenizer, limited)
  --max-gen-len <n>     Max tokens to generate (default: 128)
  --gen-batch <n>       Batch size for parallel generation (F16 only)

SFT Training:
  --train <file>        Training data (tokenized binary)
  --steps <n>           Training steps
  --lr <f>              Learning rate (default: 1e-4)
  --lora-rank <n>       LoRA rank (default: 16)
  --lora-alpha <f>      LoRA alpha (default: 32)
  --lora-adapter <p>    LoRA adapter path (load/save)

Pre-training:
  --preset <name>       Model preset: tiny, 125M, 350M
  --layers/--dim/--heads/--vocab/--batch/--seq  Model dimensions
  --checkpoint <dir>    Checkpoint directory
  --resume <path>       Resume from checkpoint
```

## Architecture Notes

### Gemma3
- GeGLU activation (GELU, not SiLU)
- 4 RMSNorm layers per block (input, post-attn, pre-FF, post-FF)
- RMSNorm convention: `(1 + weight) * norm(x)` — weights adjusted at load time
- 5:1 hybrid attention: layer % 6 == 0 is global (RoPE theta=1M), rest are local (theta=10K)
- Embedding scaling: `x * sqrt(d_model)`
- Tied embeddings: lm_head shares embed_tokens weight

### Qwen3
- SwiGLU activation (SiLU)
- 2 RMSNorm layers per block (input, post-attn)
- QK norm (per-head RMSNorm on Q and K)
- Tied embeddings

### SFT Training Pipeline
- Forward + backward in single Metal command buffer (no CPU-GPU sync per op)
- LoRA on Q/V projections, AdamW optimizer on CPU
- Gemma3: GeGLU backward (`gelu_mul_backward`), 4-norm structure, per-layer RoPE theta, embedding scaling
- Qwen3: SwiGLU backward (`silu_mul_backward`), 2-norm structure
- Global gradient norm clipping (max_norm=1.0)

### Metal Kernels
- All inference ops run in a single Metal command buffer per token
- Q8_0 matvec: GGML-compatible block quantization (32 elements per block)
- F16 matvec: Half-precision matrix-vector multiply
- Fused kernels: `silu_mul`, `gelu_mul`, `silu_mul_backward`, `gelu_mul_backward`, `residual_add`, `rope`, `rms_norm`
