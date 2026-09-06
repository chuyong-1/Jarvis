"""Voice output: backend selection, synthesis, and what actually gets spoken.

Piper is an optional manual download, so before this the assistant was silent on
any machine where it had not been fetched -- which is every fresh one. Windows
ships SAPI, so there is now a second local backend and "it cannot talk" is a
real failure rather than the default state.

    venv\\Scripts\\python.exe scripts\\test_voice_out.py
    venv\\Scripts\\python.exe scripts\\test_voice_out.py --play   (audible)
"""
import argparse
import sys
import time
import wave
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from agent import _safe_emit_len  # noqa: E402
from speech import SPEAK_WAV, SpeechIO, _speakable  # noqa: E402

FAILURES = []


def check(label, ok, detail=""):
    print(f"  [{'ok ' if ok else 'BAD'}] {label}" + (f"   {detail}" if detail else ""))
    if not ok:
        FAILURES.append(label)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--play", action="store_true", help="actually play the audio")
    args = ap.parse_args()

    print("--- backend ---")
    io = SpeechIO()
    name = io.tts_name()
    check("a local voice backend was found", name != "OFFLINE", name)
    check("tts_available agrees with the chosen backend",
          SpeechIO.tts_available() == (name != "OFFLINE"))
    # The SAPI probe spawns PowerShell (~1s), so it must happen once per
    # instance, not once per spoken reply.
    fresh = SpeechIO()
    check("nothing is probed until it is needed", fresh._tts_backend is None)
    fresh.tts_name()
    check("the probe result is cached after the first call",
          fresh._tts_backend is not None, fresh._tts_backend)

    print("\n--- what gets spoken ---")
    spoken = _speakable(r"Done. See **D:\jarvis-sandbox\notes\ideas.md** for the `details`.")
    check("markdown is stripped", "*" not in spoken and "`" not in spoken, spoken)
    check("a long path is reduced to its filename",
          "ideas.md" in spoken and "jarvis-sandbox" not in spoken, spoken)
    check("an empty reply is not spoken", _speakable("   ") == "")
    long_reply = ". ".join(f"Sentence number {i}" for i in range(200))
    check("a runaway reply is cut short", len(_speakable(long_reply)) <= 620,
          f"{len(_speakable(long_reply))} chars")

    print("\n--- tool calls never reach the transcript or the voice ---")
    for text, want in [("Hello there", 11), ("<tool_call>{}", 0), ("<tool_c", 0),
                       ("Let me look. <tool_call>{}", 13), ("Let me look. <too", 13),
                       ("a < b", 5)]:
        got = _safe_emit_len(text)
        check(f"emit {text[:got]!r} from {text!r}", got == want, f"got {got}, want {want}")

    print("\n--- synthesis ---")
    if name == "OFFLINE":
        print("  (skipped: no backend on this machine)")
    else:
        start = time.time()
        ok = io.speak("Systems nominal." if args.play else "Test.") if args.play else \
            (io._synth_piper if name == "PIPER" else io._synth_sapi)("Test.", SPEAK_WAV)
        elapsed = time.time() - start
        check("synthesis produced audio", bool(ok) and SPEAK_WAV.exists())
        if SPEAK_WAV.exists():
            with wave.open(str(SPEAK_WAV), "rb") as wf:
                seconds = wf.getnframes() / wf.getframerate()
            check("the wav has real audio in it", seconds > 0.1, f"{seconds:.2f}s")
            check("faster than realtime", elapsed < max(seconds * 4, 3.0),
                  f"{elapsed:.2f}s to make {seconds:.2f}s")

    print(f"\n{'PASS' if not FAILURES else 'FAIL'}: {len(FAILURES)} failing check(s)")
    for name_ in FAILURES:
        print(f"   - {name_}")
    sys.exit(1 if FAILURES else 0)


if __name__ == "__main__":
    main()
