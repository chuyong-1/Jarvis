"""Verification for the local-action tools: edit / append / mkdir / open / launch.

Run:
    venv\\Scripts\\python.exe scripts\\test_edit_tools.py
"""
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from audit import AuditLog  # noqa: E402
from config import ALLOWED_ROOTS  # noqa: E402
from tools import ToolExecutor  # noqa: E402

ROOT = ALLOWED_ROOTS[0]
results = []


def check(name, cond, detail=""):
    results.append((name, bool(cond)))
    print(f"  [{'ok ' if cond else 'BAD'}] {name}" + (f"  -> {detail}" if not cond else ""))


class Confirmer:
    def __init__(self, answer=True):
        self.answer, self.prompts = answer, []

    def __call__(self, tool, args, preview):
        self.prompts.append(preview)
        return self.answer


def main():
    audit = AuditLog(Path(tempfile.mkdtemp(prefix="jedit_")) / "audit.db")
    yes = Confirmer(True)
    no = Confirmer(False)
    ex = ToolExecutor(ALLOWED_ROOTS, audit, index=None, confirm_fn=yes)
    ex_no = ToolExecutor(ALLOWED_ROOTS, audit, index=None, confirm_fn=no)

    note = ROOT / "notes" / "todo.txt"
    note.write_text("TODO\n- buy milk\n- call Dana about the renewal\n- book flights\n",
                    encoding="utf-8")

    # ---- edit_file -----------------------------------------------------
    print("\n--- edit_file ---")
    r = ex.execute("edit_file", {"path": str(note),
                                 "old_string": "- buy milk",
                                 "new_string": "- buy oat milk"})
    body = note.read_text(encoding="utf-8")
    check("edit replaces the snippet", r.ok and "- buy oat milk" in body, r.output[:70])
    check("edit leaves the rest untouched",
          "call Dana about the renewal" in body and "book flights" in body)
    check("confirmation preview contains a real diff",
          any("@@" in p and "-- buy milk" in p.replace("- - buy milk", "-- buy milk")
              or "+- buy oat milk" in p for p in yes.prompts), yes.prompts[-1][:120])

    r = ex.execute("edit_file", {"path": str(note), "old_string": "nonexistent text",
                                 "new_string": "x"})
    check("missing old_string is refused", not r.ok and "not found" in r.output, r.output[:70])

    # ambiguity must be refused, not silently applied to the first match
    dup = ROOT / "notes" / "dup.txt"
    dup.write_text("alpha\nalpha\n", encoding="utf-8")
    r = ex.execute("edit_file", {"path": str(dup), "old_string": "alpha", "new_string": "beta"})
    check("ambiguous old_string is refused", not r.ok and "2 times" in r.output, r.output[:70])
    check("ambiguous edit changed nothing", dup.read_text(encoding="utf-8") == "alpha\nalpha\n")
    dup.unlink()

    before = note.read_text(encoding="utf-8")
    r = ex_no.execute("edit_file", {"path": str(note), "old_string": "TODO",
                                    "new_string": "WRECKED"})
    check("declined edit changes nothing",
          (not r.ok) and note.read_text(encoding="utf-8") == before)

    # ---- append_to_file ------------------------------------------------
    print("\n--- append_to_file ---")
    r = ex.execute("append_to_file", {"path": str(note), "content": "- renew passport"})
    body = note.read_text(encoding="utf-8")
    check("append adds the line", r.ok and body.rstrip().endswith("- renew passport"), r.output[:70])
    check("append preserves earlier content", "- buy oat milk" in body and "book flights" in body)

    glued = ROOT / "notes" / "noeol.txt"
    glued.write_text("first line", encoding="utf-8")   # no trailing newline
    ex.execute("append_to_file", {"path": str(glued), "content": "second line"})
    check("append does not glue onto an unterminated last line",
          glued.read_text(encoding="utf-8") == "first line\nsecond line\n",
          repr(glued.read_text(encoding="utf-8")))
    glued.unlink()

    # ---- create_directory ----------------------------------------------
    print("\n--- create_directory ---")
    newdir = ROOT / "projects" / "2026" / "q1"
    r = ex.execute("create_directory", {"path": str(newdir)})
    check("creates nested folders", r.ok and newdir.is_dir(), r.output[:70])
    r = ex.execute("create_directory", {"path": str(newdir)})
    check("existing folder is reported, not silently ok", not r.ok, r.output[:70])
    r = ex.execute("create_directory", {"path": r"C:\Windows\jarvis_should_not_exist"})
    check("cannot create outside the roots",
          (not r.ok) and not Path(r"C:\Windows\jarvis_should_not_exist").exists(), r.output[:70])

    # ---- open_file / launch_app security -------------------------------
    print("\n--- open_file / launch_app guards ---")
    evil = ROOT / "notes" / "payload.bat"
    evil.write_text("@echo should never run\n", encoding="utf-8")
    r = ex.execute("open_file", {"path": str(evil)})
    check("open_file refuses executable types (would RUN them)",
          (not r.ok) and "executable" in r.output, r.output[:80])
    evil.unlink()

    r = ex.execute("open_file", {"path": r"C:\Windows\notepad.exe"})
    check("open_file outside roots denied", not r.ok, r.output[:80])

    # The allowlist is no longer hardcoded here -- it is the user's, curated in
    # the APPS tab and covered in detail by scripts/test_apps.py. What this
    # suite still checks is that an executor with NO registry refuses to launch
    # anything at all, rather than falling back to some default.
    r = ex.execute("launch_app", {"app": "definitely-not-an-app"})
    check("unknown app refused when no allowlist is attached", not r.ok, r.output[:90])

    r = ex_no.execute("launch_app", {"app": "notepad"})
    check("declined launch does not start anything", not r.ok, r.output[:70])
    check("launch prompted for confirmation", any("launch notepad" in p for p in no.prompts),
          str(no.prompts[-1])[:70])

    # cleanup
    note.unlink(missing_ok=True)
    import shutil
    shutil.rmtree(ROOT / "projects", ignore_errors=True)
    audit.close()

    bad = [n for n, ok in results if not ok]
    print(f"\n{'FAIL' if bad else 'PASS'}: {len(bad)} failing of {len(results)}")
    for n in bad:
        print(f"    failed: {n}")
    sys.exit(1 if bad else 0)


if __name__ == "__main__":
    main()
