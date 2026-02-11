#!/usr/bin/env python3
"""
Qwen3 SFT Data Preprocessor

Converts JSONL instruction/response data to tokenized binary format
compatible with the C DataLoader (int32 token array).

Usage:
    python tools/preprocess_sft.py \
        --input sft/data/sample.jsonl \
        --output sft/data/train.bin \
        --model Qwen/Qwen3-4B \
        --max-seq-len 256

Input JSONL format:
    {"instruction": "...", "response": "..."}

Output: flat int32 binary file of token IDs
"""

import argparse
import json
import struct
import sys


def main():
    parser = argparse.ArgumentParser(description="Preprocess SFT data for Qwen3")
    parser.add_argument("--input", required=True, help="Input JSONL file")
    parser.add_argument("--output", required=True, help="Output binary file")
    parser.add_argument("--model", default="Qwen/Qwen3-4B", help="HuggingFace model name for tokenizer")
    parser.add_argument("--max-seq-len", type=int, default=256, help="Max sequence length")
    parser.add_argument("--val-split", type=float, default=0.1, help="Validation split ratio")
    parser.add_argument("--val-output", default=None, help="Validation output binary file")
    args = parser.parse_args()

    try:
        from transformers import AutoTokenizer
    except ImportError:
        print("Error: transformers package required. Install with: pip install transformers")
        sys.exit(1)

    print(f"Loading tokenizer: {args.model}")
    tok = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)

    # Read JSONL
    samples = []
    with open(args.input, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            samples.append(json.loads(line))

    print(f"Loaded {len(samples)} samples from {args.input}")

    # Tokenize using chat template
    all_tokens = []
    for sample in samples:
        instruction = sample.get("instruction", "")
        response = sample.get("response", "")

        # Build chat messages
        messages = [
            {"role": "user", "content": instruction},
            {"role": "assistant", "content": response},
        ]

        # Apply chat template
        try:
            text = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
            tokens = tok.encode(text, add_special_tokens=False)
        except Exception:
            # Fallback: simple concatenation
            text = f"<|im_start|>user\n{instruction}<|im_end|>\n<|im_start|>assistant\n{response}<|im_end|>"
            tokens = tok.encode(text, add_special_tokens=False)

        # Truncate to max length
        if len(tokens) > args.max_seq_len:
            tokens = tokens[:args.max_seq_len]

        all_tokens.extend(tokens)

    print(f"Total tokens: {len(all_tokens)}")

    # Split into train/val
    if args.val_output and args.val_split > 0:
        split_idx = int(len(all_tokens) * (1.0 - args.val_split))
        train_tokens = all_tokens[:split_idx]
        val_tokens = all_tokens[split_idx:]
    else:
        train_tokens = all_tokens
        val_tokens = []

    # Write binary (int32)
    def write_tokens(tokens, path):
        with open(path, "wb") as f:
            for t in tokens:
                f.write(struct.pack("<i", t))
        print(f"Written {len(tokens)} tokens to {path} ({len(tokens) * 4} bytes)")

    write_tokens(train_tokens, args.output)
    if val_tokens and args.val_output:
        write_tokens(val_tokens, args.val_output)

    print("Done!")


if __name__ == "__main__":
    main()
