"""Live-model check: does Qwen pick the right LOCAL-ACTION tool and use it well?

The interesting failure here is not a crash. It is the model reaching for
write_file to change one line of a note, which silently destroys everything it
did not bother to repeat. This asserts it edits surgically instead.

Run:
    venv\\Scripts\\python.exe scripts\\test_local_actions.py
"""
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from agent import ToolCallingAgent  # noqa: E402
from audit import AuditLog  # noqa: E402
from config import ALLOWED_ROOTS  # noqa: E402
from index_store import FileIndex  # noqa: E402
from knowledge import KnowledgeGraph  # noqa: E402
from tools import ToolExecutor  # noqa: E402

ROOT = ALLOWED_ROOTS[0]
NOTE = ROOT / "notes" / "shopping.txt"
ORIGINAL = ("Shopping list\n"
            "- whole milk\n"
            "- sourdough bread\n"
            "- coffee beans\n"
            "- olive oil\n")

results = []


def check(name, cond, detail=""):
    results.append((name, bool(cond)))
    print(f"  [{'ok ' if cond else 'BAD'}] {name}" + (f"  -> {detail}" if not cond else ""))


def main():
    from llama_cpp import Llama

    print("Loading model ...")
    llm = Llama(model_path="D:/project-jarvis/models/qwen2.5-7b-instruct-q4_k_m-00001-of-00002.gguf",
                n_gpu_layers=-1, n_ctx=8192, verbose=False)
    audit = AuditLog(Path(tempfile.mkdtemp(prefix="jloc_")) / "audit.db")
    kg = KnowledgeGraph(Path(tempfile.mkdtemp(prefix="jloc2_")) / "kg.json")
    prompts = []

    def confirm(tool, args, preview):
        prompts.append((tool, preview))
        return True

    ex = ToolExecutor(ALLOWED_ROOTS, audit, index=FileIndex(ALLOWED_ROOTS),
                      confirm_fn=confirm, kg=kg)
    agent = ToolCallingAgent(llm, ex)

    # ---- 1. change one item, keep the rest -----------------------------
    NOTE.write_text(ORIGINAL, encoding="utf-8")
    print("\n>>> swap one item")
    reply = agent.chat(f"In {NOTE}, change 'whole milk' to 'oat milk'. Leave everything else alone.")
    print(f"    tools: {[c['name'] for c in agent.last_tool_calls]}")
    print(f"    reply: {reply[:120].strip()!r}")
    body = NOTE.read_text(encoding="utf-8")
    print(f"    file now:\n{body}")

    check("used edit_file rather than rewriting the file",
          any(c["name"] == "edit_file" for c in agent.last_tool_calls),
          str([c["name"] for c in agent.last_tool_calls]))
    check("the item was changed", "oat milk" in body and "whole milk" not in body)
    check("NOTHING ELSE was lost",
          all(x in body for x in ("sourdough bread", "coffee beans", "olive oil")), repr(body))

    # ---- 2. add an item -------------------------------------------------
    print("\n>>> add an item")
    reply = agent.chat(f"Add 'greek yoghurt' to the list in {NOTE}.")
    print(f"    tools: {[c['name'] for c in agent.last_tool_calls]}")
    print(f"    reply: {reply[:200].strip()!r}")
    body = NOTE.read_text(encoding="utf-8")
    print(f"    file now:\n{body}")
    check("added the new item", "greek yoghurt" in body.lower(), repr(body))
    check("still kept the original items",
          all(x in body for x in ("sourdough bread", "coffee beans", "olive oil")), repr(body))

    # ---- 3. make a folder -----------------------------------------------
    print("\n>>> create a folder")
    target = ROOT / "receipts"
    reply = agent.chat(f"Create a folder called receipts inside {ROOT}.")
    print(f"    tools: {[c['name'] for c in agent.last_tool_calls]}")
    print(f"    reply: {reply[:160].strip()!r}")
    check("folder created", target.is_dir(), str(target))

    # ---- 4. every one of those asked permission -------------------------
    check("every destructive action prompted", len(prompts) >= 3,
          f"{len(prompts)} prompts: {[p[0] for p in prompts]}")

    NOTE.unlink(missing_ok=True)
    target.rmdir() if target.is_dir() else None
    audit.close()

    bad = [n for n, ok in results if not ok]
    print(f"\n{'FAIL' if bad else 'PASS'}: {len(bad)} failing of {len(results)}")
    for n in bad:
        print(f"    failed: {n}")
    sys.exit(1 if bad else 0)


if __name__ == "__main__":
    main()
