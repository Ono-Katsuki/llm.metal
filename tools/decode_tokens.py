#!/usr/bin/env python3
"""Decode binary token IDs to text."""
import argparse
import struct
from transformers import AutoTokenizer

def main():
    parser = argparse.ArgumentParser(description="Decode binary token IDs to text")
    parser.add_argument("input", help="Binary token file to decode")
    parser.add_argument("--model", default="Qwen/Qwen3-4B", help="HuggingFace model name (default: Qwen/Qwen3-4B)")
    args = parser.parse_args()

    tok = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)

    with open(args.input, "rb") as f:
        data = f.read()

    ids = [struct.unpack("<I", data[i:i+4])[0] for i in range(0, len(data), 4)]
    print(f"Token IDs ({len(ids)}): {ids}")
    print(f"\nDecoded:\n{tok.decode(ids)}")

if __name__ == "__main__":
    main()
