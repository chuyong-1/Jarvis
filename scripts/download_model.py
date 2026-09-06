"""Download the Stage 1 GGUF model from Hugging Face (no API key needed, public repo).

Q4_K_M for this model ships as two shards; llama.cpp loads them automatically
as long as both files live in the same directory and you point at part 1.
"""
from pathlib import Path

from huggingface_hub import hf_hub_download

REPO_ID = "Qwen/Qwen2.5-7B-Instruct-GGUF"
FILENAMES = [
    "qwen2.5-7b-instruct-q4_k_m-00001-of-00002.gguf",
    "qwen2.5-7b-instruct-q4_k_m-00002-of-00002.gguf",
]
MODELS_DIR = Path(__file__).resolve().parent.parent / "models"

if __name__ == "__main__":
    for filename in FILENAMES:
        path = hf_hub_download(repo_id=REPO_ID, filename=filename, local_dir=str(MODELS_DIR))
        print(f"Downloaded to: {path}")
