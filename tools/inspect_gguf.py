#!/usr/bin/env python3
"""Inspect a GGUF file's tensor names and shapes."""
import struct
import sys

GGUF_MAGIC = 0x46554747

TYPE_SIZES = {
    0: ("F32", 4), 1: ("F16", 2), 2: ("Q4_0", 0), 3: ("Q4_1", 0),
    6: ("Q5_0", 0), 7: ("Q5_1", 0), 8: ("Q8_0", 0), 9: ("Q8_1", 0),
}

META_TYPES = {
    0: "uint8", 1: "int8", 2: "uint16", 3: "int16", 4: "uint32", 5: "int32",
    6: "float32", 7: "bool", 8: "string", 9: "array", 10: "uint64", 11: "int64",
    12: "float64",
}

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

def main():
    path = sys.argv[1]
    with open(path, "rb") as f:
        magic = read_u32(f)
        assert magic == GGUF_MAGIC, f"Bad magic: {magic:#010x}"
        version = read_u32(f)
        n_tensors = read_u64(f)
        n_kv = read_u64(f)
        print(f"Version: {version}, Tensors: {n_tensors}, KV pairs: {n_kv}\n")

        # Read metadata
        print("=== Metadata ===")
        for _ in range(n_kv):
            key = read_string(f)
            vtype = read_u32(f)
            pos_before = f.tell()
            if vtype in (4, 5):  # uint32/int32
                val = read_u32(f)
                print(f"  {key} ({META_TYPES.get(vtype,'?')}) = {val}")
            elif vtype == 6:  # float32
                val = struct.unpack("<f", f.read(4))[0]
                print(f"  {key} ({META_TYPES.get(vtype,'?')}) = {val}")
            elif vtype == 8:  # string
                val = read_string(f)
                print(f"  {key} (string) = {val[:100]}")
            else:
                skip_value(f, vtype)
                size = f.tell() - pos_before
                print(f"  {key} ({META_TYPES.get(vtype,'?')}) [skipped {size} bytes]")

        # Read tensor info
        print(f"\n=== Tensors ({n_tensors}) ===")
        for i in range(n_tensors):
            name = read_string(f)
            ndim = read_u32(f)
            shape = [read_u64(f) for _ in range(ndim)]
            ttype = read_u32(f)
            offset = read_u64(f)
            type_name = TYPE_SIZES.get(ttype, (f"type_{ttype}", 0))[0]
            print(f"  [{i:3d}] {name:50s} {type_name:6s} shape={shape} offset={offset}")

if __name__ == "__main__":
    main()
