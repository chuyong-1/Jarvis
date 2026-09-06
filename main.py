"""Jarvis entry point.

    venv\\Scripts\\python.exe main.py            # full app
    venv\\Scripts\\python.exe main.py --ui-only  # UI without loading the LLM
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

from app import JarvisApp  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ui-only", action="store_true",
                    help="launch the window without loading models (UI development)")
    ap.add_argument("--check", action="store_true",
                    help="verify native libraries load, write the result to jarvis_check.log, exit")
    args = ap.parse_args()

    if args.check:
        # A packaged windowed app has no console, so report to a file. This is
        # the check that matters for a frozen build: llama_cpp only loads if the
        # CUDA runtime DLLs were found (see packaging/runtime_hook_cuda.py).
        lines = [f"frozen={getattr(sys, 'frozen', False)}", f"exe={sys.executable}"]
        for name in ("llama_cpp", "faster_whisper", "openwakeword", "chromadb",
                     "networkx", "sounddevice"):
            try:
                mod = __import__(name)
                lines.append(f"OK   {name} {getattr(mod, '__version__', '')}")
            except Exception as exc:  # noqa: BLE001
                lines.append(f"FAIL {name}: {type(exc).__name__}: {exc}")
        Path("jarvis_check.log").write_text("\n".join(lines), encoding="utf-8")
        print("\n".join(lines))
        return

    if args.ui_only:
        app = JarvisApp(pipeline_factory=None)
    else:
        from pipeline import JarvisPipeline
        # The UI owns the app allowlist, because that is where it is curated.
        # The pipeline shares the same object, so ticking a box in the APPS tab
        # takes effect on the very next turn.
        app = JarvisApp(pipeline_factory=lambda ui: JarvisPipeline(ui=ui, apps=ui.apps))
    app.run()


if __name__ == "__main__":
    main()
