"""Stage 2.2 verification: continuous wake-word detection with openWakeWord.

Uses the BUNDLED pretrained models (no custom training yet).

Run:
    venv\\Scripts\\python.exe scripts\\test_wakeword.py
    venv\\Scripts\\python.exe scripts\\test_wakeword.py --threshold 0.3
    venv\\Scripts\\python.exe scripts\\test_wakeword.py --duration 20   # self-terminating

Ctrl+C to stop; prints total detections and elapsed time.
"""
import argparse
import sys
import time


import numpy as np
import sounddevice as sd

SAMPLE_RATE = 16000
CHUNK = 1280  # 80ms at 16kHz -- openWakeWord's expected frame size


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--duration", type=float, default=None,
                        help="stop automatically after N seconds (default: run until Ctrl+C)")
    parser.add_argument("--models", nargs="*", default=None,
                        help="wake words to load, e.g. --models hey_jarvis. "
                             "Default loads ALL pretrained models (noisier: models you "
                             "never say can still false-fire)")
    parser.add_argument("--warmup", type=float, default=1.5,
                        help="ignore detections for the first N seconds while the mic AGC settles")
    args = parser.parse_args()

    import openwakeword
    from openwakeword.model import Model

    # Download to openWakeWord's own resources dir rather than a custom path:
    # Model() resolves the shared melspectrogram/embedding models relative to the
    # package, so a custom target_directory just desynchronises the two. The venv
    # is already on D:, so these land on D: anyway (~10MB total).
    print("Ensuring pretrained models are present ...")
    openwakeword.utils.download_models()

    try:
        if args.models:
            oww = Model(wakeword_models=args.models, inference_framework="onnx")
        else:
            oww = Model(inference_framework="onnx")
    except Exception as exc:  # noqa: BLE001
        print(f"FAIL: could not build the openWakeWord model: {type(exc).__name__}: {exc}")
        sys.exit(1)

    names = list(oww.models.keys())
    print(f"\nLoaded {len(names)} pretrained wake-word model(s):")
    for name in names:
        print(f"  - {name}")
    print(f"\nThreshold: {args.threshold}")
    print("Say one of the above wake words. Ctrl+C to stop.\n")

    try:
        default_in = sd.query_devices(kind="input")
        print(f"Input device: {default_in['name']}\n")
    except Exception as exc:  # noqa: BLE001
        print(f"FAIL: no usable microphone: {type(exc).__name__}: {exc}")
        sys.exit(1)

    detections = 0
    # NOT keyed off oww.models -- predict() returns MORE keys than models.keys()
    # (the single "timer" model emits 1_minute_timer, 5_minute_timer, ...), so
    # pre-seeding from models.keys() KeyErrors mid-stream. An exception raised
    # inside the InputStream context also deadlocks PortAudio's MME backend on
    # teardown, which looks like a hang rather than a crash. Fill in lazily.
    peak_scores: dict[str, float] = {}
    start = time.time()

    try:
        with sd.InputStream(samplerate=SAMPLE_RATE, channels=1, dtype="int16",
                            blocksize=CHUNK) as stream:
            while True:
                if args.duration is not None and (time.time() - start) >= args.duration:
                    print(f"\n(reached --duration {args.duration:.0f}s, stopping)")
                    break

                frame, overflowed = stream.read(CHUNK)
                if overflowed:
                    print("WARN: input overflow (dropped audio)")

                settling = (time.time() - start) < args.warmup

                scores = oww.predict(frame.flatten())
                for name, score in scores.items():
                    score = float(score)
                    if settling:
                        # The Intel mic array's AGC ramps hard for the first second or
                        # two, and that transient reliably false-fires wake words.
                        continue
                    peak_scores[name] = max(peak_scores.get(name, 0.0), score)
                    if score >= args.threshold:
                        detections += 1
                        elapsed = time.time() - start
                        print(f"[{elapsed:7.2f}s] DETECTED {name!r}  score={score:.3f}")
    except KeyboardInterrupt:
        print("\n(Ctrl+C)")
    except Exception as exc:  # noqa: BLE001 - never die inside the stream context
        print(f"\nERROR in capture loop: {type(exc).__name__}: {exc}")

    elapsed = time.time() - start
    print("\n--- RESULT ---")
    print(f"Listened:   {elapsed:.1f}s")
    print(f"Detections: {detections}")
    print("Peak score per model (useful for picking a threshold):")
    for name, peak in sorted(peak_scores.items(), key=lambda kv: -kv[1]):
        print(f"  {peak:.3f}  {name}")

    if detections == 0:
        print("\nNo detections. Try:")
        print("  - saying one of the wake words listed above, clearly, ~1 ft from the mic")
        print(f"  - lowering the threshold: --threshold 0.3 (current: {args.threshold})")
        print("  - checking the peak scores above: if all are ~0.000 the mic may be muted")


if __name__ == "__main__":
    main()
