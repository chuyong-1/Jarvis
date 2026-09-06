"""Conversation memory across restarts, and the staleness rule that bounds it.

The knowledge graph remembers facts; this remembers the thread. Closing the
window used to reset the assistant mid-task, so "open that one" after a restart
referred to nothing.

The risk being tested is not "does it save" -- it is "does it know when to
FORGET". Restoring yesterday's turns into today's prompt is worse than restoring
nothing, because the model then answers a question the user no longer remembers
asking.

    venv\\Scripts\\python.exe scripts\\test_memory.py
"""
import json
import sys
import tempfile
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from conversation import MAX_STORED_MESSAGES, ConversationStore  # noqa: E402

FAILURES = []


def check(label, ok, detail=""):
    print(f"  [{'ok ' if ok else 'BAD'}] {label}" + (f"   {detail}" if detail else ""))
    if not ok:
        FAILURES.append(label)


def age_to(path: Path, hours: float) -> None:
    data = json.loads(path.read_text(encoding="utf-8"))
    data["saved_at"] = time.time() - hours * 3600
    path.write_text(json.dumps(data), encoding="utf-8")


def main():
    tmp = Path(tempfile.mkdtemp(prefix="jarvis_mem_"))
    path = tmp / "conversation.json"
    turn = [{"role": "user", "content": "Who is drafting the Contoso renewal?"},
            {"role": "assistant", "content": "Dana Whitfield is.", "tool_calls": []}]

    print("--- round trip ---")
    store = ConversationStore(path)
    check("nothing to restore on a first run", store.load() == [])
    check("age note is empty with nothing saved", store.age_note() == "")
    store.save(turn)
    check("saving creates the file", path.exists())

    fresh = ConversationStore(path)
    restored = fresh.load()
    check("a new process restores the exchange", len(restored) == 2, str(len(restored)))
    check("content survives intact", restored[0]["content"] == turn[0]["content"])
    check("the age is described", "ago" in fresh.age_note(), fresh.age_note())

    print("\n--- staleness ---")
    age_to(path, 3)
    check("3 hours old is still the same conversation",
          len(ConversationStore(path).load()) == 2)
    age_to(path, 40)
    check("40 hours old is NOT restored", ConversationStore(path).load() == [])
    check("a custom window is respected",
          len(ConversationStore(path, max_age_hours=72).load()) == 2)

    print("\n--- robustness ---")
    path.write_text("{ this is not json", encoding="utf-8")
    check("corrupt file restores nothing instead of raising",
          ConversationStore(path).load() == [])
    path.write_text(json.dumps({"saved_at": time.time(),
                                "messages": [{"role": "user", "content": "ok"},
                                             "junk", {"bad": 1},
                                             {"role": "system", "content": "x"}]}),
                    encoding="utf-8")
    kept = ConversationStore(path).load()
    check("malformed entries are dropped, valid ones kept", kept == [{"role": "user",
                                                                     "content": "ok"}],
          str(kept))
    check("missing file is not an error", ConversationStore(tmp / "nope.json").load() == [])

    print("\n--- bounded growth ---")
    store = ConversationStore(path)
    store.save([{"role": "user", "content": f"message {i}"} for i in range(500)])
    stored = json.loads(path.read_text(encoding="utf-8"))["messages"]
    check("history on disk is capped", len(stored) == MAX_STORED_MESSAGES,
          f"{len(stored)} kept")
    check("it keeps the NEWEST messages", stored[-1]["content"] == "message 499",
          stored[-1]["content"])

    store.clear()
    check("clear removes it", not path.exists() and store.age_note() == "")

    print(f"\n{'PASS' if not FAILURES else 'FAIL'}: {len(FAILURES)} failing check(s)")
    for name in FAILURES:
        print(f"   - {name}")
    sys.exit(1 if FAILURES else 0)


if __name__ == "__main__":
    main()
