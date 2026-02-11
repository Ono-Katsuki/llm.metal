#!/usr/bin/env python3
"""Tokenize a prompt and save as binary token IDs."""
import argparse
import struct
from transformers import AutoTokenizer

def main():
    parser = argparse.ArgumentParser(description="Tokenize a prompt to binary token IDs")
    parser.add_argument("prompt", help="Text prompt to tokenize")
    parser.add_argument("--model", default="Qwen/Qwen3-4B", help="HuggingFace model name (default: Qwen/Qwen3-4B)")
    parser.add_argument("--output", default="/tmp/prompt.bin", help="Output binary file (default: /tmp/prompt.bin)")
    args = parser.parse_args()

    tok = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)

    messages = [{"role": "user", "content": args.prompt}]
    try:
        text = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    except Exception:
        text = args.prompt

    ids = tok.encode(text)
    print(f"Prompt: {args.prompt}")
    print(f"Template: {text}")
    print(f"Token IDs ({len(ids)}): {ids}")

    with open(args.output, "wb") as f:
        for tid in ids:
            f.write(struct.pack("<I", tid))

    print(f"Saved {len(ids)} tokens to {args.output}")

if __name__ == "__main__":
    main()
