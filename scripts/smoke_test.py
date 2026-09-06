"""Stage 1 verification test: one fixed prompt, non-interactive.

Run:
    venv\\Scripts\\python.exe scripts\\smoke_test.py

Passes if:
  - model loads without error
  - llama.cpp reports GPU layers offloaded > 0 (confirms CUDA is actually used)
  - a coherent reply comes back within a reasonable time
  - decode speed is reported in tokens/sec
"""
import sys
import time
from pathlib import Path

from llama_cpp import Llama

MODEL_PATH = Path(__file__).resolve().parent.parent / "models" / "qwen2.5-7b-instruct-q4_k_m-00001-of-00002.gguf"
PROMPT = "In one sentence, what is the capital of France?"


def main():
    if not MODEL_PATH.exists():
        print(f"FAIL: model not found at {MODEL_PATH}")
        sys.exit(1)

    print(f"Loading {MODEL_PATH.name} with GPU offload...")
    load_start = time.time()
    llm = Llama(
        model_path=str(MODEL_PATH),
        n_gpu_layers=-1,
        n_ctx=2048,
        chat_format="chatml",
        verbose=True,  # prints llama.cpp's own GPU offload summary to stderr
    )
    print(f"Loaded in {time.time() - load_start:.1f}s\n")

    print(f"Prompt: {PROMPT}")
    start = time.time()
    result = llm.create_chat_completion(
        messages=[{"role": "user", "content": PROMPT}],
        max_tokens=128,
    )
    elapsed = time.time() - start

    reply = result["choices"][0]["message"]["content"]
    usage = result["usage"]
    tok_per_sec = usage["completion_tokens"] / elapsed if elapsed > 0 else 0

    print(f"\nReply: {reply}")
    print(f"\n{usage['completion_tokens']} tokens in {elapsed:.2f}s = {tok_per_sec:.1f} tok/s")

    if "paris" not in reply.lower():
        print("\nWARN: expected 'Paris' in the reply — model loaded but output looks wrong.")
        sys.exit(1)

    print("\nPASS")


if __name__ == "__main__":
    main()
