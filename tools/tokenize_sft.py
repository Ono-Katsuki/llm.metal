#!/usr/bin/env python3
"""Tokenize SFT JSONL data using Qwen3 tokenizer → raw int32 binary.

Usage:
    python3 tools/tokenize_sft.py [--input sft/data/sample.jsonl] [--output sft/data]

The dataloader expects flat int32 token sequences where input[i]=tok[i],
target[i]=tok[i+1] (next-token prediction).
"""
import argparse
import json
import struct
import os

from transformers import AutoTokenizer


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="sft/data/sample.jsonl")
    parser.add_argument("--output", default="sft/data")
    parser.add_argument("--val-ratio", type=float, default=0.1,
                        help="Fraction of examples for validation")
    parser.add_argument("--repeat", type=int, default=8,
                        help="Repeat dataset N times for more training data")
    args = parser.parse_args()

    print("Loading Qwen3-4B tokenizer...")
    tok = AutoTokenizer.from_pretrained("Qwen/Qwen3-4B")
    print(f"  vocab_size={tok.vocab_size}, eos={tok.eos_token_id}")

    # Read JSONL
    examples = []
    with open(args.input) as f:
        for line in f:
            line = line.strip()
            if line:
                examples.append(json.loads(line))
    print(f"  Loaded {len(examples)} examples")

    # Tokenize each example using chat template
    all_token_seqs = []
    for ex in examples:
        messages = [
            {"role": "user", "content": ex["instruction"]},
            {"role": "assistant", "content": ex["response"]},
        ]
        text = tok.apply_chat_template(messages, tokenize=False,
                                       add_generation_prompt=False)
        token_ids = tok.encode(text, add_special_tokens=False)
        # Append EOS
        if token_ids[-1] != tok.eos_token_id:
            token_ids.append(tok.eos_token_id)
        all_token_seqs.append(token_ids)

    # Print stats
    lengths = [len(s) for s in all_token_seqs]
    total_tokens = sum(lengths)
    print(f"  Total tokens: {total_tokens}")
    print(f"  Avg tokens/example: {total_tokens / len(lengths):.0f}")
    print(f"  Min/Max: {min(lengths)}/{max(lengths)}")

    # Split into train/val
    n_val = max(1, int(len(all_token_seqs) * args.val_ratio))
    val_seqs = all_token_seqs[:n_val]
    train_seqs = all_token_seqs[n_val:]
    print(f"  Train examples: {len(train_seqs)}, Val examples: {len(val_seqs)}")

    # Concatenate tokens into flat sequences
    # Repeat training data for more steps
    train_tokens = []
    for _ in range(args.repeat):
        for seq in train_seqs:
            train_tokens.extend(seq)

    val_tokens = []
    for seq in val_seqs:
        val_tokens.extend(seq)

    print(f"  Train tokens (after {args.repeat}x repeat): {len(train_tokens)}")
    print(f"  Val tokens: {len(val_tokens)}")

    # Write binary files
    os.makedirs(args.output, exist_ok=True)

    train_path = os.path.join(args.output, "train.bin")
    with open(train_path, "wb") as f:
        for t in train_tokens:
            f.write(struct.pack("<i", t))
    print(f"  Wrote {train_path}: {len(train_tokens)} tokens "
          f"({len(train_tokens) * 4} bytes)")

    val_path = os.path.join(args.output, "valid.bin")
    with open(val_path, "wb") as f:
        for t in val_tokens:
            f.write(struct.pack("<i", t))
    print(f"  Wrote {val_path}: {len(val_tokens)} tokens "
          f"({len(val_tokens) * 4} bytes)")

    # Print first few tokens for verification
    print("\n  First 20 train tokens:", train_tokens[:20])
    print("  Decoded:", tok.decode(train_tokens[:20]))


if __name__ == "__main__":
    main()
