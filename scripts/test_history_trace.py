"""History must record what a turn DID, not only what it said.

Storing just the prose reply meant every past turn looked like the assistant
asserting "Excel is opening" with no call attached. Replayed against a history
of those, the model stopped calling tools entirely and invented the outcomes
instead -- while cold, the same words called launch_app every time. These check
the evidence survives, and that trimming never orphans a tool result.
"""
import json
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from conversation import ConversationStore  # noqa: E402
from pipeline import tail_turns  # noqa: E402

fails = 0


def check(label, ok):
    global fails
    if not ok:
        fails += 1
    print(f"  [{'ok ' if ok else 'FAIL'}] {label}")


def turn(user, tool=None, reply="done"):
    out = [{"role": "user", "content": user}]
    if tool:
        out.append({"role": "assistant", "content": f"<tool_call>{tool}</tool_call>",
                    "tool_calls": []})
        out.append({"role": "tool", "content": f"{tool} ok"})
    out.append({"role": "assistant", "content": reply, "tool_calls": []})
    return out


history = turn("open excel", "launch_app") + turn("hi", None, "Hello!") + \
    turn("open notepad", "launch_app")

print("tail_turns cuts only on a user message")
for n in range(1, 5):
    cut = tail_turns(history, n)
    check(f"{n} turn(s): starts on a user message",
          not cut or cut[0]["role"] == "user")
    users = sum(1 for m in cut if m["role"] == "user")
    check(f"{n} turn(s): keeps exactly min({n}, 3) turns", users == min(n, 3))
    check(f"{n} turn(s): no tool row without a call before it",
          all(cut[i - 1]["role"] == "assistant"
              for i, m in enumerate(cut) if m["role"] == "tool" and i > 0))

check("zero turns is empty", tail_turns(history, 0) == [])
check("more turns than exist returns everything", tail_turns(history, 99) == history)
check("empty history stays empty", tail_turns([], 3) == [])

print("\nthe evidence survives a save/restore cycle")
tmp = Path(tempfile.mkdtemp(prefix="jarvis_hist_")) / "conversation.json"
store = ConversationStore(tmp)
store.save(history)
back = store.load()
check("tool rows are restored, not filtered away",
      any(m["role"] == "tool" for m in back))
check("the assistant's tool call is restored",
      any("<tool_call>" in m["content"] for m in back if m["role"] == "assistant"))
check("round trip preserves every message", len(back) == len(history))

print("\na history cut mid-turn is repaired on load")
orphan = [{"role": "tool", "content": "launch_app ok"},
          {"role": "assistant", "content": "Excel is open.", "tool_calls": []}] + \
    turn("hi", None, "Hello!")
tmp2 = Path(tempfile.mkdtemp(prefix="jarvis_hist2_")) / "conversation.json"
store2 = ConversationStore(tmp2)
store2.save(orphan)
back2 = store2.load()
check("restored history begins on a user message",
      bool(back2) and back2[0]["role"] == "user")
check("the orphaned tool result was dropped",
      not any(m["content"] == "launch_app ok" for m in back2))

print("\nstale history is still dropped entirely")
raw = json.loads(tmp.read_text(encoding="utf-8"))
raw["saved_at"] = 0.0
tmp.write_text(json.dumps(raw), encoding="utf-8")
check("no saved_at means no restore", ConversationStore(tmp).load() == [])

print(f"\n{'PASS' if not fails else 'FAIL'}: {fails} failing")
sys.exit(1 if fails else 0)
