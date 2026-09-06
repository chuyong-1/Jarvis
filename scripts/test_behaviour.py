"""Does the model reach for a tool exactly when it must — and not otherwise?

This is the suite that catches system-prompt regressions, which are invisible to
every other test here: the tool layer stays green while the assistant quietly
starts making things up. Each case below is a failure that was actually observed
on this model:

  * "How is the machine holding up?" answered with invented vitals ("32GB out of
    64GB" on a 16GB machine) because a colloquial question read as chit-chat.
  * "Are you running out of memory?" answered with fabricated JSON that imitated
    tool output without any tool having run.
  * "Delete the file X" answered "Deleting X." with no delete_file call at all.

Run:
    venv\\Scripts\\python.exe scripts\\test_behaviour.py
    venv\\Scripts\\python.exe scripts\\test_behaviour.py --verbose

Loads the model, so it takes a couple of minutes.
"""
import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

MODEL = ROOT / "models" / "qwen2.5-7b-instruct-q4_k_m-00001-of-00002.gguf"

# (question, requirement)
#   "tool_name" -> that tool must be called
#   None        -> no tool is the right answer
#   "ANY"       -> either is fine; the ANSWER is what is checked
CASES = [
    ("How is the machine holding up?", "system_status", None),
    ("Are you running out of memory?", "system_status", None),
    ("What is your system status?", "system_status", None),
    ("Delete the file notes/disposable.txt", "delete_file", None),
    ("Open my meeting notes.", "find_files", None),
    ("Which carrier contract is up for renewal?", "search_files", "contoso"),
    # The date is injected into the system prompt, so answering with no tool is
    # correct AND faster. What matters is that the answer is right.
    ("What is today's date?", "ANY", "2026"),
    ("Hello, how are you?", None, None),
    ("What is the capital of France?", None, "paris"),
]

FAILURES = []


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--verbose", action="store_true")
    ap.add_argument("--runs", type=int, default=1,
                    help="repeat every case N times and report the pass RATE")
    args = ap.parse_args()

    from llama_cpp import Llama

    from agent import ToolCallingAgent
    from audit import AuditLog
    from config import ALLOWED_ROOTS, AUDIT_DB
    from index_store import FileIndex
    from tools import ToolExecutor

    if not MODEL.exists():
        raise SystemExit(f"Model not found at {MODEL}. Run scripts/download_model.py first.")

    llm = Llama(model_path=str(MODEL), n_gpu_layers=-1, n_ctx=8192, verbose=False)
    audit = AuditLog(AUDIT_DB)
    index = FileIndex(ALLOWED_ROOTS)
    disposable = ALLOWED_ROOTS[0] / "notes" / "disposable.txt"

    passes = {q: 0 for q, _t, _x in CASES}
    for run in range(args.runs):
        if args.runs > 1:
            print(f"\n=== run {run + 1} of {args.runs} ===")
        for question, want_tool, want_text in CASES:
            disposable.write_text("delete me", encoding="utf-8")
            # Declining every confirmation keeps the suite side-effect free: what
            # is under test is which tool the model CHOSE, not what the tool then
            # does with it.
            ex = ToolExecutor(ALLOWED_ROOTS, audit, index=index, confirm_fn=lambda *a: False)
            agent = ToolCallingAgent(llm, ex)
            reply = agent.chat(question)
            called = [c["name"] for c in agent.last_tool_calls]

            if want_tool == "ANY":
                ok = True
            elif want_tool:
                ok = want_tool in called
            else:
                ok = not called
            if want_text:
                ok = ok and want_text.lower() in reply.lower()

            passes[question] += bool(ok)
            print(f"  [{'ok ' if ok else 'BAD'}] {question!r}")
            print(f"        called={called or 'none'}  want={want_tool or 'none'}")
            if not ok or args.verbose:
                print(f"        reply={reply[:140].strip()!r}")

    disposable.unlink(missing_ok=True)
    audit.close()

    # Reported as a RATE. These cases are not deterministic even at temperature
    # 0: the system prompt carries the current time, so the prompt itself
    # differs minute to minute, and the index changes as files change. Printing
    # a single run as pass/fail hid that -- the same suite gave 9/9 and 8/9 on
    # consecutive runs, which looks like a regression and is not.
    total = len(CASES) * args.runs
    won = sum(passes.values())
    flaky = [q for q, n in passes.items() if 0 < n < args.runs]
    FAILURES.extend(q for q, n in passes.items() if n == 0)

    print(f"\n{'PASS' if not FAILURES else 'FAIL'}: {won}/{total} behaved correctly"
          + (f"  over {args.runs} runs" if args.runs > 1 else ""))
    for q in FAILURES:
        print(f"   FAILED EVERY RUN: {q}")
    for q in flaky:
        print(f"   flaky ({passes[q]}/{args.runs}): {q}")
    sys.exit(1 if FAILURES else 0)


if __name__ == "__main__":
    main()
