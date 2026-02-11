#!/usr/bin/env python3
"""FastAPI HTTP server for Qwen3 inference via C subprocess pipe."""

import argparse
import struct
import subprocess
import sys
import signal
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import uvicorn

# Globals set at startup
proc = None
tokenizer = None


def send_request(prompt_tokens: list[int], max_gen_len: int) -> list[int]:
    """Send a binary request to the C process and read the response."""
    prompt_len = len(prompt_tokens)
    # Header: [uint32 prompt_len] [uint32 max_gen_len]
    header = struct.pack("<II", prompt_len, max_gen_len)
    # Tokens: [prompt_len × uint32]
    tokens_data = struct.pack(f"<{prompt_len}I", *prompt_tokens)
    proc.stdin.write(header + tokens_data)
    proc.stdin.flush()

    # Read response: [uint32 n_generated]
    resp_header = proc.stdout.read(4)
    if len(resp_header) < 4:
        raise RuntimeError("C process closed unexpectedly")
    n_gen = struct.unpack("<I", resp_header)[0]

    # Read [n_generated × uint32]
    if n_gen > 0:
        resp_data = proc.stdout.read(n_gen * 4)
        if len(resp_data) < n_gen * 4:
            raise RuntimeError("C process returned incomplete data")
        output_tokens = list(struct.unpack(f"<{n_gen}I", resp_data))
    else:
        output_tokens = []

    return output_tokens


def shutdown_proc():
    """Send shutdown signal (prompt_len=0) and wait for C process to exit."""
    global proc
    if proc and proc.poll() is None:
        try:
            proc.stdin.write(struct.pack("<II", 0, 0))
            proc.stdin.flush()
            proc.wait(timeout=10)
        except Exception:
            proc.kill()
        proc = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    yield
    shutdown_proc()


app = FastAPI(title="Qwen3 Inference Server", lifespan=lifespan)


class GenerateRequest(BaseModel):
    prompt: str
    max_gen_len: int = 256
    system: str = ""


class GenerateResponse(BaseModel):
    text: str
    prompt_tokens: int
    generated_tokens: int
    token_ids: list[int]


@app.get("/health")
def health():
    alive = proc is not None and proc.poll() is None
    return {"status": "ok" if alive else "error"}


@app.post("/generate", response_model=GenerateResponse)
def generate(req: GenerateRequest):
    if proc is None or proc.poll() is not None:
        raise HTTPException(status_code=503, detail="C process not running")

    # Build chat-template messages
    messages = []
    if req.system:
        messages.append({"role": "system", "content": req.system})
    messages.append({"role": "user", "content": req.prompt})

    text = tokenizer.apply_chat_template(messages, tokenize=False,
                                         add_generation_prompt=True)
    prompt_tokens = tokenizer.encode(text)

    output_ids = send_request(prompt_tokens, req.max_gen_len)

    # Strip EOS tokens
    eos_ids = {tokenizer.eos_token_id, 151645, 151643}
    cleaned = []
    for tid in output_ids:
        if tid in eos_ids:
            break
        cleaned.append(tid)

    decoded = tokenizer.decode(cleaned, skip_special_tokens=True)

    return GenerateResponse(
        text=decoded,
        prompt_tokens=len(prompt_tokens),
        generated_tokens=len(cleaned),
        token_ids=cleaned,
    )


def main():
    global proc, tokenizer

    parser = argparse.ArgumentParser(description="Qwen3 HTTP Inference Server")
    parser.add_argument("--model", required=True, help="Model path for C binary")
    parser.add_argument("--tokenizer", default=None,
                        help="HuggingFace tokenizer name/path (default: Qwen/Qwen3-4B)")
    parser.add_argument("--binary", default="./llm_train", help="Path to C binary")
    parser.add_argument("--max-gen-len", type=int, default=256)
    parser.add_argument("--lora-adapter", default=None)
    parser.add_argument("--fp16", action="store_true")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()

    # Load tokenizer
    tokenizer_name = args.tokenizer or "Qwen/Qwen3-4B"
    print(f"[Server] Loading tokenizer: {tokenizer_name}")
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name, trust_remote_code=True)
    print(f"[Server] Tokenizer loaded (vocab_size={tokenizer.vocab_size})")

    # Build C subprocess command
    cmd = [args.binary, "--mode", "serve", "--model", args.model,
           "--max-gen-len", str(args.max_gen_len)]
    if args.fp16:
        cmd.append("--fp16")
    if args.lora_adapter:
        cmd.extend(["--lora-adapter", args.lora_adapter])

    print(f"[Server] Starting C process: {' '.join(cmd)}")
    proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                            stderr=sys.stderr)

    # Wait for "Ready" message from C process (it prints to stderr)
    # We just proceed — C process logs to stderr, we'll know if it fails
    print(f"[Server] C process started (pid={proc.pid})")

    def sigint_handler(sig, frame):
        print("\n[Server] Shutting down...")
        shutdown_proc()
        sys.exit(0)

    signal.signal(signal.SIGINT, sigint_handler)

    print(f"[Server] Starting HTTP server on {args.host}:{args.port}")
    uvicorn.run(app, host=args.host, port=args.port, log_level="info")


if __name__ == "__main__":
    main()
