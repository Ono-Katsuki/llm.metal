#!/usr/bin/env python3
"""Verify GGUF Q8_0 dequantization against reference implementation."""
import struct
import numpy as np
import sys

def read_u32(f): return struct.unpack("<I", f.read(4))[0]
def read_u64(f): return struct.unpack("<Q", f.read(8))[0]
def read_string(f):
    length = read_u64(f)
    return f.read(length).decode("utf-8", errors="replace")

def skip_value(f, vtype):
    if vtype in (0, 1, 7): f.read(1)
    elif vtype in (2, 3): f.read(2)
    elif vtype in (4, 5, 6): f.read(4)
    elif vtype in (10, 11, 12): f.read(8)
    elif vtype == 8: read_string(f)
    elif vtype == 9:
        arr_type = read_u32(f)
        arr_len = read_u64(f)
        for _ in range(arr_len):
            skip_value(f, arr_type)

def dequantize_q8_0(data, n_elements):
    """Reference Q8_0 dequantization."""
    n_blocks = (n_elements + 31) // 32
    result = np.zeros(n_elements, dtype=np.float32)
    for b in range(n_blocks):
        block = data[b*34:(b+1)*34]
        # f16 scale
        scale_f16 = struct.unpack("<e", block[:2])[0]  # numpy half-precision
        quants = np.frombuffer(block[2:34], dtype=np.int8)
        remaining = min(32, n_elements - b * 32)
        result[b*32:b*32+remaining] = quants[:remaining].astype(np.float32) * scale_f16
    return result

def main():
    path = sys.argv[1] if len(sys.argv) > 1 else "models/Qwen3-4B-Q8_0.gguf"
    tensor_name = sys.argv[2] if len(sys.argv) > 2 else "output_norm.weight"

    with open(path, "rb") as f:
        magic = read_u32(f)
        version = read_u32(f)
        n_tensors = read_u64(f)
        n_kv = read_u64(f)

        # Skip metadata
        for _ in range(n_kv):
            read_string(f)
            vtype = read_u32(f)
            skip_value(f, vtype)

        # Read tensor info
        tensors = {}
        for _ in range(n_tensors):
            name = read_string(f)
            ndim = read_u32(f)
            shape = [read_u64(f) for _ in range(ndim)]
            ttype = read_u32(f)
            offset = read_u64(f)
            n_elements = 1
            for s in shape:
                n_elements *= s
            tensors[name] = {"shape": shape, "type": ttype, "offset": offset, "n_elements": n_elements}

        # Alignment
        alignment = 32
        data_offset = (f.tell() + alignment - 1) & ~(alignment - 1)

        if tensor_name not in tensors:
            print(f"Tensor '{tensor_name}' not found. Available:")
            for name in list(tensors.keys())[:20]:
                print(f"  {name}: shape={tensors[name]['shape']}, type={tensors[name]['type']}")
            return

        t = tensors[tensor_name]
        print(f"Tensor: {tensor_name}")
        print(f"  Shape: {t['shape']}, Type: {t['type']}, Offset: {t['offset']}")
        print(f"  Elements: {t['n_elements']}")

        # Read raw data
        f.seek(data_offset + t["offset"])
        if t["type"] == 0:  # F32
            raw = f.read(t["n_elements"] * 4)
            values = np.frombuffer(raw, dtype=np.float32)
        elif t["type"] == 8:  # Q8_0
            n_blocks = (t["n_elements"] + 31) // 32
            raw = f.read(n_blocks * 34)
            values = dequantize_q8_0(raw, t["n_elements"])
        else:
            print(f"  Unsupported type: {t['type']}")
            return

        print(f"  First 20 values: {values[:20]}")
        print(f"  Mean: {values.mean():.6f}, Std: {values.std():.6f}")
        print(f"  Min: {values.min():.6f}, Max: {values.max():.6f}")

        # Check for token 151644 embedding
        if tensor_name == "token_embd.weight":
            ne0 = t["shape"][0]  # 2560
            token_id = 151644
            start = token_id * ne0
            end = start + ne0
            embed = values[start:end]
            print(f"\n  Token {token_id} embedding (first 10): {embed[:10]}")
            print(f"  Token {token_id} embed mean: {embed.mean():.6f}, std: {embed.std():.6f}")

if __name__ == "__main__":
    main()
