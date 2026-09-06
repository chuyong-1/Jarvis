"""Stage 1: terminal prompt-in/response-out loop against a local GGUF model.

Run:
    venv\\Scripts\\python.exe src\\chat.py

If you hit a CUDA out-of-memory error, lower --n-gpu-layers (try 20, then 10)
until it loads. Layers left on GPU still get accelerated; the rest run on CPU.
"""
import argparse
import time
from pathlib import Path

from llama_cpp import Llama

DEFAULT_MODEL = Path(__file__).parent.parent / "models" / "qwen2.5-7b-instruct-q4_k_m-00001-of-00002.gguf"

SYSTEM_PROMPT = "You are Jarvis, a helpful local voice assistant. Keep answers concise."


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--n-gpu-layers", type=int, default=-1, help="-1 = offload all layers to GPU")
    # 8192 measured at 5217/6141 MiB VRAM and 34.4 tok/s -- full speed with ~900MB
    # left for the Windows desktop. 16384 fits (5893 MiB) but leaves only ~250MB,
    # and 32768 silently spills into shared system RAM and drops to ~30 tok/s.
    parser.add_argument("--n-ctx", type=int, default=8192)
    args = parser.parse_args()

    if not args.model.exists():
        raise SystemExit(f"Model not found at {args.model}. Run scripts/download_model.py first.")

    print(f"Loading {args.model.name} (n_gpu_layers={args.n_gpu_layers}, n_ctx={args.n_ctx})...")
    llm = Llama(
        model_path=str(args.model),
        n_gpu_layers=args.n_gpu_layers,
        n_ctx=args.n_ctx,
        chat_format="chatml",
        verbose=False,
    )
    print("Loaded. Type your message, or 'exit' to quit.\n")

    messages = [{"role": "system", "content": SYSTEM_PROMPT}]

    while True:
        user_input = input("You: ").strip()
        if user_input.lower() in ("exit", "quit"):
            break
        if not user_input:
            continue

        messages.append({"role": "user", "content": user_input})

        print("Jarvis: ", end="", flush=True)
        start = time.time()
        token_count = 0
        reply_chunks = []

        for chunk in llm.create_chat_completion(messages=messages, stream=True, max_tokens=512):
            delta = chunk["choices"][0]["delta"]
            text = delta.get("content")
            if text:
                print(text, end="", flush=True)
                reply_chunks.append(text)
                token_count += 1

        elapsed = time.time() - start
        print()
        if elapsed > 0:
            print(f"  [{token_count} chunks in {elapsed:.1f}s]\n")

        messages.append({"role": "assistant", "content": "".join(reply_chunks)})


if __name__ == "__main__":
    main()
