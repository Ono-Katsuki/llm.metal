#!/usr/bin/env python3
"""Tokenize a prompt using Qwen3 tokenizer and save as binary token IDs."""
import sys
import struct
from transformers import AutoTokenizer

def main():
    prompt = sys.argv[1] if len(sys.argv) > 1 else "Hello, how are you?"
    output = sys.argv[2] if len(sys.argv) > 2 else "/tmp/prompt_tokens.bin"

    tok = AutoTokenizer.from_pretrained("Qwen/Qwen3-4B", trust_remote_code=True)

    # Apply chat template
    messages = [{"role": "user", "content": prompt}]
    try:
        text = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    except Exception:
        text = f"<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n"

    ids = tok.encode(text)
    print(f"Prompt: {prompt}")
    print(f"Template: {text}")
    print(f"Token IDs ({len(ids)}): {ids}")

    # Write as uint32 binary
    with open(output, "wb") as f:
        for tid in ids:
            f.write(struct.pack("<I", tid))

    print(f"Saved {len(ids)} tokens to {output}")

if __name__ == "__main__":
    main()
