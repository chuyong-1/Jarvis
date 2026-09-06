"""Stage 2.1 verification: record from the mic and transcribe with faster-whisper.

Run:
    venv\\Scripts\\python.exe scripts\\test_stt.py
    venv\\Scripts\\python.exe scripts\\test_stt.py --duration 8 --model base.en
    venv\\Scripts\\python.exe scripts\\test_stt.py --device cpu   # force CPU

MEASURED on this machine (i7-12650H, CPU int8, 5.31s of clean synthetic speech,
RTF = transcribe_time / audio_seconds, lower is better):
    base.en   74M   beam 1  RTF 0.25   <- default: 3.3x faster, same text
    base.en   74M   beam 5  RTF 0.28
    small.en  244M  beam 1  RTF 0.81
    small.en  244M  beam 5  RTF 0.90
    medium.en 769M          not tested -- too slow on CPU to be usable here

Model size dominates; beam size barely matters. CAVEAT: those numbers come from
clean TTS-generated audio (no accent, no disfluency, no room noise). Real mic
input is harder and should favour small.en more than this suggests -- re-run the
comparison with real recordings before locking base.en in permanently.

NOTE: default device is "auto", which TRIES cuda then falls back to cpu, so this
test can measure both. The integrated assistant should pin STT to CPU and leave
all VRAM to the LLM -- see the VRAM tradeoff discussion.
"""
import argparse
import sys
import time
from pathlib import Path

import numpy as np
import sounddevice as sd

SAMPLE_RATE = 16000
# Keep whisper weights on D: -- the C: drive is nearly full.
DOWNLOAD_ROOT = Path(__file__).resolve().parent.parent / "models" / "whisper"


def load_model(requested_device: str, model_size: str):
    """Return (model, device_label). Tries CUDA first unless CPU is forced."""
    from faster_whisper import WhisperModel

    attempts = []
    if requested_device in ("auto", "cuda"):
        attempts.append(("cuda", "int8_float16"))
    if requested_device in ("auto", "cpu"):
        attempts.append(("cpu", "int8"))

    last_error = None
    for device, compute_type in attempts:
        try:
            print(f"Trying device={device} compute_type={compute_type} ...")
            model = WhisperModel(
                model_size,
                device=device,
                compute_type=compute_type,
                download_root=str(DOWNLOAD_ROOT),
            )
            # CTranslate2 loads its CUDA libs LAZILY: constructing the model on
            # "cuda" succeeds even when cublas/cudnn DLLs are missing, and it only
            # explodes on the first real encode. Force that here so the fallback
            # actually catches it -- otherwise it crashes mid-conversation later.
            warmup_segments, _ = model.transcribe(np.zeros(SAMPLE_RATE, dtype=np.float32))
            list(warmup_segments)  # generator is lazy; must consume it to run the encoder
            print(f"  -> loaded on {device.upper()} ({compute_type})")
            return model, f"{device} ({compute_type})"
        except Exception as exc:  # noqa: BLE001 - we want the reason printed, not swallowed
            last_error = exc
            print(f"  -> FAILED on {device}: {type(exc).__name__}: {exc}")
            if requested_device == "cuda":
                break

    raise SystemExit(f"Could not load model on any device. Last error: {last_error}")


def record(duration: float) -> np.ndarray:
    print(f"\nRecording {duration:.0f}s of audio -- SPEAK NOW ...")
    audio = sd.rec(int(duration * SAMPLE_RATE), samplerate=SAMPLE_RATE, channels=1, dtype="float32")
    sd.wait()
    return audio.flatten()


def load_wav(path: Path) -> np.ndarray:
    """Read a wav as 16kHz mono float32 (resampling if needed)."""
    import soundfile as sf

    data, rate = sf.read(str(path), dtype="float32", always_2d=True)
    audio = data.mean(axis=1)  # downmix to mono
    if rate != SAMPLE_RATE:
        # linear resample is fine for a smoke test
        n_out = int(round(len(audio) * SAMPLE_RATE / rate))
        audio = np.interp(
            np.linspace(0, len(audio) - 1, n_out, dtype=np.float64),
            np.arange(len(audio), dtype=np.float64),
            audio.astype(np.float64),
        ).astype(np.float32)
        print(f"Resampled {rate} Hz -> {SAMPLE_RATE} Hz")
    return audio


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--duration", type=float, default=5.0)
    parser.add_argument("--model", default="base.en")
    parser.add_argument("--device", choices=["auto", "cuda", "cpu"], default="auto")
    parser.add_argument("--file", type=Path, default=None,
                        help="transcribe this wav instead of recording (skips the mic entirely)")
    parser.add_argument("--beam-size", type=int, default=1,
                        help="1 = greedy (default); 5 costs ~12%% more time for no measured gain here")
    args = parser.parse_args()

    DOWNLOAD_ROOT.mkdir(parents=True, exist_ok=True)

    if args.file is None:
        try:
            default_in = sd.query_devices(kind="input")
            print(f"Input device: {default_in['name']}")
        except Exception as exc:  # noqa: BLE001
            print(f"FAIL: no usable microphone: {type(exc).__name__}: {exc}")
            sys.exit(1)

    print(f"Loading faster-whisper '{args.model}' (first run downloads weights to {DOWNLOAD_ROOT})")
    model, device_label = load_model(args.device, args.model)

    if args.file is not None:
        if not args.file.exists():
            print(f"FAIL: no such file: {args.file}")
            sys.exit(1)
        print(f"\nReading {args.file}")
        audio = load_wav(args.file)
        args.duration = len(audio) / SAMPLE_RATE
        print(f"Audio duration: {args.duration:.2f}s")
    else:
        audio = record(args.duration)

    # Distinguishes "mic captured silence" from "transcription failed".
    rms = float(np.sqrt(np.mean(audio**2)))
    peak = float(np.max(np.abs(audio)))
    print(f"Captured: RMS={rms:.5f} peak={peak:.5f}")
    if peak < 0.001:
        print("WARN: recording is essentially silent -- mic muted, wrong input device, or nobody spoke.")

    print("Transcribing ...")
    start = time.time()
    segments, info = model.transcribe(audio, beam_size=args.beam_size)
    text = " ".join(seg.text.strip() for seg in segments).strip()
    elapsed = time.time() - start

    rtf = elapsed / args.duration if args.duration else 0.0

    print("\n--- RESULT ---")
    print(f"Device:        {device_label}")
    print(f"Language:      {info.language} (confidence {info.language_probability:.2f})")
    print(f"Transcription: {text!r}")
    print(f"Transcribe:    {elapsed:.2f}s for {args.duration:.0f}s audio -> RTF {rtf:.2f}")

    if not text:
        print("\nFAIL: empty transcription")
        sys.exit(1)

    print("\nPASS")


if __name__ == "__main__":
    main()
