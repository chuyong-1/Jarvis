"""Locating and opening files: find_files, not-found hints, open_file, machine tools.

The gap this covers: the existing suites verified that open_file REFUSES the
things it should (executables, paths outside the roots) but never that a
successful open works, and nothing exercised finding a file by name at all.
Live testing showed why that mattered -- asked to "open my meeting notes" the
model invented "notes\\my meeting notes.txt" and gave up when it did not exist.

    venv\\Scripts\\python.exe scripts\\test_find_open.py

No model is loaded here; this is the tool layer only.
"""
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

import tools as tools_mod  # noqa: E402
from audit import AuditLog  # noqa: E402
from tools import ToolExecutor, _not_found  # noqa: E402

FAILURES = []


def check(label, ok, detail=""):
    print(f"  [{'ok ' if ok else 'BAD'}] {label}" + (f"   {detail}" if detail else ""))
    if not ok:
        FAILURES.append(label)


def main():
    tmp = Path(tempfile.mkdtemp(prefix="jarvis_find_"))
    root = tmp / "sandbox"
    (root / "notes").mkdir(parents=True)
    (root / "reports").mkdir()
    (root / "notes" / "meeting_notes.txt").write_text("team sync", encoding="utf-8")
    (root / "reports" / "q1_budget.txt").write_text("numbers", encoding="utf-8")
    (root / "reports" / "annual_review.txt").write_text("review", encoding="utf-8")

    audit = AuditLog(tmp / "audit.db")
    ex = ToolExecutor([root], audit, confirm_fn=lambda *a: True)

    print("--- find_files ---")
    # find_files LEADS with the single best path and tells the model to use it,
    # rather than returning a bare list. A flat list made the model stop and ask
    # the user which of several candidates it should open.
    r = ex.execute("find_files", {"name": "meeting notes"})
    first = r.output.splitlines()[0]
    check("loose words find the underscored file", "meeting_notes.txt" in r.output, first)
    check("the best match is named first, as an absolute path",
          first.startswith("Best match: ") and str(root) in first, first)
    check("and the model is told to act on it", "Use this path" in r.output)

    r = ex.execute("find_files", {"name": "budget"})
    check("single word matches", "q1_budget.txt" in r.output)

    r = ex.execute("find_files", {"name": "q1_budget.txt"})
    check("best match ranks first", r.output.splitlines()[0].endswith("q1_budget.txt"),
          r.output.splitlines()[0])

    r = ex.execute("find_files", {"name": "nothinglikethis"})
    check("a miss still lists what exists, rather than dead-ending",
          r.ok and "meeting_notes.txt" in r.output)

    r = ex.execute("find_files", {"name": ""})
    check("empty query is rejected", not r.ok)

    print("\n--- not-found hints ---")
    hint = _not_found(root / "notes" / "my meeting notes.txt")
    check("names the real neighbour", "meeting_notes.txt" in hint)
    hint_root = _not_found(root / "meeting_notes.txt")
    check("a folder holding only folders is NOT reported as empty",
          "empty" not in hint_root and "notes" in hint_root, hint_root.splitlines()[1])
    hint_deep = _not_found(root / "nope" / "deeper" / "file.txt")
    check("walks up to the nearest real folder", "notes" in hint_deep)

    print("\n--- open_file ---")
    opened = []
    real_startfile = getattr(tools_mod.os, "startfile", None)
    # Patched so the suite never launches a real application.
    tools_mod.os.startfile = opened.append
    try:
        target = root / "notes" / "meeting_notes.txt"
        r = ex.execute("open_file", {"path": str(target)})
        check("a successful open reports ok", r.ok, r.output)
        # Compared RESOLVED: containment expands 8.3 short names, so the path
        # handed to the OS is the long form and will not match the input string.
        check("it really handed the file to the OS",
              [Path(p).resolve() for p in opened] == [target.resolve()], str(opened))

        r = ex.execute("open_file", {"path": str(root / "notes" / "ghost.txt")})
        check("a missing file fails with a usable hint",
              not r.ok and "meeting_notes.txt" in r.output)
        check("nothing was launched for the missing file", len(opened) == 1)
    finally:
        if real_startfile is not None:
            tools_mod.os.startfile = real_startfile

    print("\n--- machine awareness ---")
    r = ex.execute("get_datetime", {})
    check("get_datetime answers", r.ok and any(c.isdigit() for c in r.output), r.output)
    r = ex.execute("system_status", {})
    check("system_status reports memory and uptime",
          r.ok and "Memory" in r.output and "uptime" in r.output.lower())
    check("system_status degrades instead of raising", r.ok)

    audit.close()
    print(f"\n{'PASS' if not FAILURES else 'FAIL'}: {len(FAILURES)} failing check(s)")
    for name in FAILURES:
        print(f"   - {name}")
    sys.exit(1 if FAILURES else 0)


if __name__ == "__main__":
    main()
