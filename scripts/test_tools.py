"""Stage 3 verification: tool dispatch, permission tiers, and audit trail.

Asserts that:
  - read-only tools auto-execute (confirmation handler is never called)
  - destructive tools ALWAYS prompt, and do nothing when declined
  - every path argument is contained, including move destinations
  - run_command has no shell, honours the denylist, and times out
  - the audit log records all of it, denials included

Run:
    venv\\Scripts\\python.exe scripts\\test_tools.py
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


def check(name, condition, detail=""):
    results.append((name, bool(condition), detail))
    print(f"  [{'ok ' if condition else 'BAD'}] {name}" + (f"  -> {detail}" if not condition else ""))


class Confirmer:
    """Records prompts and answers with a scripted sequence of yes/no."""
    def __init__(self, answers):
        self.answers = list(answers)
        self.prompts = []

    def __call__(self, tool, args, preview):
        self.prompts.append((tool, preview))
        return self.answers.pop(0) if self.answers else False


def main():
    tmpdb = Path(tempfile.mkdtemp(prefix="jtools_")) / "audit.db"
    audit = AuditLog(tmpdb)

    # ---- read-only tools never prompt ----------------------------------
    never = Confirmer([])
    ex = ToolExecutor(ALLOWED_ROOTS, audit, index=None, confirm_fn=never)

    r = ex.execute("list_directory", {"path": str(ROOT / "reports")})
    check("list_directory works", r.ok and "q1_budget.xlsx" in r.output, r.output[:80])
    check("read-only did NOT prompt", not never.prompts)

    r = ex.execute("read_file", {"path": str(ROOT / "notes" / "meeting_notes.txt")})
    check("read_file works", r.ok and "Helsinki" in r.output, r.output[:80])

    r = ex.execute("read_file", {"path": str(ROOT / "reports" / "q1_budget.xlsx")})
    check("read_file extracts xlsx", r.ok and "Budget" in r.output, r.output[:80])

    # ---- containment ---------------------------------------------------
    r = ex.execute("read_file", {"path": r"C:\Windows\System32\drivers\etc\hosts"})
    check("read outside root denied", (not r.ok) and "Denied" in r.output, r.output[:80])

    r = ex.execute("list_directory", {"path": str(ROOT / ".." / "project-jarvis")})
    check("'..' escape denied", (not r.ok) and "Denied" in r.output, r.output[:80])

    # ---- destructive: declined -----------------------------------------
    decline = Confirmer([False])
    ex_d = ToolExecutor(ALLOWED_ROOTS, audit, index=None, confirm_fn=decline)
    victim = ROOT / "notes" / "meeting_notes.txt"
    before = victim.read_text(encoding="utf-8")
    r = ex_d.execute("delete_file", {"path": str(victim)})
    check("declined delete refuses", not r.ok, r.output[:60])
    check("declined delete prompted", len(decline.prompts) == 1, str(decline.prompts))
    check("declined delete did NOT touch file", victim.exists() and victim.read_text(encoding="utf-8") == before)

    # ---- destructive: approved -----------------------------------------
    approve = Confirmer([True, True, True, True, True])
    ex_a = ToolExecutor(ALLOWED_ROOTS, audit, index=None, confirm_fn=approve)

    scratch = ROOT / "notes" / "scratch.txt"
    r = ex_a.execute("write_file", {"path": str(scratch), "content": "hello from jarvis"})
    check("write_file works", r.ok and scratch.exists(), r.output[:60])

    moved = ROOT / "notes" / "scratch_moved.txt"
    r = ex_a.execute("move_file", {"source": str(scratch), "destination": str(moved)})
    check("move_file works", r.ok and moved.exists() and not scratch.exists(), r.output[:60])

    r = ex_a.execute("delete_file", {"path": str(moved)})
    check("delete_file works", r.ok and not moved.exists(), r.output[:60])

    # move OUT of the root must be refused even though the source is legal
    outside = Path(tempfile.gettempdir()) / "jarvis_exfil.txt"
    scratch.write_text("x", encoding="utf-8")
    r = ex_a.execute("move_file", {"source": str(scratch), "destination": str(outside)})
    check("move to outside root denied", (not r.ok) and not outside.exists(), r.output[:80])
    scratch.unlink(missing_ok=True)

    # ---- run_command ---------------------------------------------------
    ok_cmd = Confirmer([True]*8)
    ex_c = ToolExecutor(ALLOWED_ROOTS, audit, index=None, confirm_fn=ok_cmd)

    r = ex_c.execute("run_command", {"command": 'python -c "print(2+2)"', "cwd": str(ROOT)})
    check("run_command executes", r.ok and "4" in r.output, r.output[:80])

    # A pipe is NOT blocked (shell=False makes it inert) but it must also not
    # pipe anything -- it should arrive as a literal argument to python.
    r = ex_c.execute("run_command", {"command": 'python -c "import sys; print(len(sys.argv))" | more',
                                     "cwd": str(ROOT)})
    check("pipe is inert, not interpreted", r.ok and "more" not in r.output.lower(), r.output[:80])

    # Asserted on the OUTCOME, not on the wording. COMMAND_MODE moved from
    # "denylist" to "allowlist" when ALLOWED_ROOTS grew past the sandbox, so the
    # refusal reason changed from "denied executable" to "not in the allowlist".
    # The property under test -- a shell is unreachable -- is unchanged, and
    # under an allowlist it is strictly stronger.
    r = ex_c.execute("run_command", {"command": "powershell -Command Get-Process", "cwd": str(ROOT)})
    check("shell interpreter blocked", (not r.ok) and "refused" in r.output, r.output[:80])

    r = ex_c.execute("run_command", {"command": "rm -rf /", "cwd": str(ROOT)})
    check("rm blocked", (not r.ok) and "refused" in r.output, r.output[:80])

    r = ex_c.execute("run_command", {"command": 'python -c "import time; time.sleep(60)"', "cwd": str(ROOT)})
    check("timeout enforced", (not r.ok) and "timed out" in r.output, r.output[:80])

    # ---- audit trail ---------------------------------------------------
    rows = audit.recent(100)
    denied = [r for r in rows if r["outcome"] == "denied"]
    destructive_ok = [r for r in rows if r["tier"] == "destructive" and r["outcome"] == "ok"]
    check("audit recorded everything", len(rows) >= 14, f"{len(rows)} rows")
    check("audit recorded denials", len(denied) >= 4, f"{len(denied)} denials")
    check("every successful destructive call is marked confirmed",
          all(r["confirmed"] == 1 for r in destructive_ok), f"{len(destructive_ok)} rows")

    print("\n  --- audit tail ---")
    for row in rows[:8]:
        print(f"  {row['ts_utc']}  {row['tool']:<14} {row['tier']:<11} conf={row['confirmed']} "
              f"{row['outcome']:<7} {str(row['result'])[:44]}")

    audit.close()
    bad = [n for n, ok, _ in results if not ok]
    print(f"\n{'FAIL' if bad else 'PASS'}: {len(bad)} failing check(s) of {len(results)}")
    sys.exit(1 if bad else 0)


if __name__ == "__main__":
    main()
