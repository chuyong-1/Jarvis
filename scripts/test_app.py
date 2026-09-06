"""Stage 6 verification: the UI's thread-safety bridge, without a human.

The dangerous part of this app is not the widgets, it is that a WORKER thread
blocks on a decision made by the UI thread. If a dialog can ever be dismissed
without releasing the worker, the assistant hangs forever mid-turn. These tests
drive tkinter's event loop manually and assert the worker always gets released.

Run:
    venv\\Scripts\\python.exe scripts\\test_app.py
"""
import sys
import tempfile
import threading
import time
import tkinter as tk
from pathlib import Path
from tkinter import ttk

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from app import JarvisApp  # noqa: E402

results = []


def check(name, cond, detail=""):
    results.append((name, bool(cond)))
    print(f"  [{'ok ' if cond else 'BAD'}] {name}" + (f"  -> {detail}" if not cond else ""))


def pump(app, seconds=1.5):
    """Run the tkinter event loop manually for a while."""
    end = time.time() + seconds
    while time.time() < end:
        app.root.update()
        time.sleep(0.01)


def find_dialog(app):
    return [w for w in app.root.winfo_children() if isinstance(w, tk.Toplevel)]


def click(dialog, label):
    """Find a button by its text anywhere in the dialog and invoke it."""
    stack = [dialog]
    while stack:
        widget = stack.pop()
        stack.extend(widget.winfo_children())
        if isinstance(widget, ttk.Button) and widget.cget("text") == label:
            widget.invoke()
            return True
    return False


def run_confirm_case(app, label, closer):
    """Ask for confirmation from a worker thread; resolve it via `closer`."""
    box = {}

    def worker():
        box["result"] = app.confirm("delete_file", {"path": "x"}, "permanently delete x")
        box["done"] = True

    t = threading.Thread(target=worker, daemon=True)
    t.start()
    pump(app, 0.8)

    dialogs = find_dialog(app)
    check(f"{label}: dialog appeared", len(dialogs) == 1, f"{len(dialogs)} dialogs")
    check(f"{label}: worker is BLOCKED until answered", not box.get("done"), str(box))

    if dialogs:
        closer(dialogs[0])
    pump(app, 0.8)
    t.join(timeout=3)

    check(f"{label}: worker was released", box.get("done") is True, str(box))
    check(f"{label}: no dialog left behind", not find_dialog(app))
    return box.get("result")


def main():
    db = Path(tempfile.mkdtemp(prefix="japp_")) / "audit.db"
    # tray=False: pystray spawns an OS-level icon thread that outlives the
    # test and cannot be driven headlessly.
    app = JarvisApp(pipeline_factory=None, audit_db=db, tray=False)
    pump(app, 0.4)

    check("window built", app.root.winfo_exists() == 1)
    check("starts idle", app._state == "idle")

    # logging + state must be callable from a worker thread
    threading.Thread(target=lambda: (app.log("user", "from worker"),
                                     app.set_state("thinking")), daemon=True).start()
    pump(app, 0.6)
    text = app.transcript.get("1.0", "end")
    check("worker-thread log reached the UI", "from worker" in text)
    check("worker-thread state change applied", app._state == "thinking", app._state)

    print("\n  --- confirmation bridge ---")
    approved = run_confirm_case(app, "approve", lambda d: click(d, "Approve"))
    check("approve returns True", approved is True, str(approved))

    denied = run_confirm_case(app, "deny", lambda d: click(d, "Deny"))
    check("deny returns False", denied is False, str(denied))

    # THE failure mode that would hang the assistant forever: a user who closes
    # the dialog with the X instead of answering. protocol() with one argument
    # returns the registered handler's command name, which tk.call can invoke.
    closed = run_confirm_case(
        app, "window closed",
        lambda d: d.tk.call(d.protocol("WM_DELETE_WINDOW")))
    check("closing the window denies (never approves)", closed is False, str(closed))

    escaped = run_confirm_case(app, "escape key",
                               lambda d: d.event_generate("<Escape>", when="now"))
    check("Escape denies", escaped is False, str(escaped))

    # audit viewer
    app.audit.log_denied("delete_file", {"path": "x"}, "user declined confirmation")
    app.refresh_audit()
    pump(app, 0.3)
    rows = app.audit_table.get_children()
    check("audit viewer shows rows", len(rows) >= 1, f"{len(rows)} rows")

    app.on_close()
    bad = [n for n, ok in results if not ok]
    print(f"\n{'FAIL' if bad else 'PASS'}: {len(bad)} failing of {len(results)}")
    for n in bad:
        print(f"    failed: {n}")
    sys.exit(1 if bad else 0)


if __name__ == "__main__":
    main()
