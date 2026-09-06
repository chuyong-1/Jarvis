"""Stage 2.3 verification: text-to-speech with Piper (standalone binary, CPU only).

Piper is deliberately NOT a pip dependency here: it ships as a self-contained
binary + onnx voice, runs fine on CPU, and stays out of the CUDA/torch
environment so it never competes with the LLM for VRAM.

Run:
    venv\\Scripts\\python.exe scripts\\test_tts.py
    venv\\Scripts\\python.exe scripts\\test_tts.py --text "Deleting three files. Confirm?" --play

Expected layout (see --help output / project docs for download steps):
    D:\\project-jarvis\\piper\\piper.exe
    D:\\project-jarvis\\models\\piper\\en_US-lessac-medium.onnx
    D:\\project-jarvis\\models\\piper\\en_US-lessac-medium.onnx.json
"""
import argparse
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
PIPER_BIN = ROOT / "piper" / "piper.exe"
VOICE = ROOT / "models" / "piper" / "en_US-lessac-medium.onnx"
OUT_WAV = ROOT / "models" / "piper" / "_tts_test_out.wav"

DOWNLOAD_HELP = f"""
Piper is not set up yet. Two downloads, no installer:

1) Binary -- https://github.com/rhasspy/piper/releases/latest
   Get: piper_windows_amd64.zip
   Extract the ENTIRE contents (piper.exe, its .dll files, and the
   espeak-ng-data\\ folder -- piper.exe will not run without its siblings) to:
       {PIPER_BIN.parent}

2) Voice -- https://huggingface.co/rhasspy/piper-voices
   Get BOTH files from en/en_US/lessac/medium/ :
       en_US-lessac-medium.onnx        (~63 MB)
       en_US-lessac-medium.onnx.json   (small config -- required)
   Put both in:
       {VOICE.parent}

Then re-run this script.
"""


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--text", default="All set. Stage two is working.")
    parser.add_argument("--play", action="store_true", help="play the generated audio back")
    parser.add_argument("--piper", type=Path, default=PIPER_BIN)
    parser.add_argument("--voice", type=Path, default=VOICE)
    args = parser.parse_args()

    missing = []
    if not args.piper.exists():
        missing.append(f"  piper binary: {args.piper}")
    if not args.voice.exists():
        missing.append(f"  voice model:  {args.voice}")
    if not args.voice.with_suffix(".onnx.json").exists():
        missing.append(f"  voice config: {args.voice.with_suffix('.onnx.json')}")

    if missing:
        print("FAIL: missing required files:")
        print("\n".join(missing))
        print(DOWNLOAD_HELP)
        sys.exit(1)

    OUT_WAV.parent.mkdir(parents=True, exist_ok=True)
    if OUT_WAV.exists():
        OUT_WAV.unlink()

    cmd = [str(args.piper), "--model", str(args.voice), "--output_file", str(OUT_WAV)]
    print(f"Text: {args.text!r}")
    print(f"Running: {' '.join(cmd)}")

    start = time.time()
    proc = subprocess.run(
        cmd,
        input=args.text.encode("utf-8"),
        capture_output=True,
        timeout=60,
    )
    elapsed = time.time() - start

    if proc.returncode != 0:
        print(f"\nFAIL: piper exited {proc.returncode}")
        print(proc.stderr.decode("utf-8", errors="replace"))
        sys.exit(1)

    if not OUT_WAV.exists():
        print(f"\nFAIL: piper reported success but produced no file at {OUT_WAV}")
        sys.exit(1)

    import soundfile as sf

    data, samplerate = sf.read(str(OUT_WAV))
    audio_seconds = len(data) / samplerate
    rtf = elapsed / audio_seconds if audio_seconds else 0.0

    print("\n--- RESULT ---")
    print(f"Output:     {OUT_WAV} ({OUT_WAV.stat().st_size / 1024:.0f} KB)")
    print(f"Audio:      {audio_seconds:.2f}s @ {samplerate} Hz")
    print(f"Generation: {elapsed:.2f}s -> RTF {rtf:.3f} (lower is better; <1.0 is faster than realtime)")

    if args.play:
        import sounddevice as sd

        print("Playing ...")
        sd.play(data, samplerate)
        sd.wait()

    print("\nPASS")


if __name__ == "__main__":
    main()
