#!/usr/bin/env python3
"""Create simple test binary data for SFT training.
Writes raw int32 tokens representing a simple repeating pattern.
"""
import struct
import os

# Use common Chinese/English token IDs (mid-range to avoid special tokens)
# Pattern: repeating sequence of 512 tokens, total 4096 tokens
pattern = []
# Simple arithmetic pattern: 5000, 5001, 5002, ...
for i in range(512):
    pattern.append(5000 + (i % 200))

# Repeat pattern to get enough tokens
tokens = pattern * 8  # 4096 tokens

os.makedirs("sft/data", exist_ok=True)
with open("sft/data/train.bin", "wb") as f:
    for t in tokens:
        f.write(struct.pack("<i", t))

# Smaller validation set
val_tokens = pattern * 2  # 1024 tokens
with open("sft/data/valid.bin", "wb") as f:
    for t in val_tokens:
        f.write(struct.pack("<i", t))

print(f"Created train.bin: {len(tokens)} tokens ({len(tokens)*4} bytes)")
print(f"Created valid.bin: {len(val_tokens)} tokens ({len(val_tokens)*4} bytes)")
