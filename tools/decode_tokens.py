#!/usr/bin/env python3
"""Decode binary token IDs using Qwen3 tokenizer."""
import struct
import sys
from transformers import AutoTokenizer

path = sys.argv[1] if len(sys.argv) > 1 else "/tmp/generated_tokens.bin"
tok = AutoTokenizer.from_pretrained("Qwen/Qwen3-4B", trust_remote_code=True)

with open(path, "rb") as f:
    data = f.read()

ids = [struct.unpack("<I", data[i:i+4])[0] for i in range(0, len(data), 4)]
print(f"Token IDs ({len(ids)}): {ids}")
print(f"\nDecoded:\n{tok.decode(ids)}")
