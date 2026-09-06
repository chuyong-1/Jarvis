"""Live check: timers and opening links.

Timers are the interesting case. Without a real tool the model reaches for
run_command("timeout /t 30"), which blocks the assistant AND gets killed by
run_command's 20s cap -- so a 30s timer silently never fires. This asserts it
uses set_timer, that the countdown genuinely completes, and that a longer timer
survives past the point where the run_command hack would have died.

Run:
    venv\\Scripts\\python.exe scripts\\test_timer_url.py
"""
import sys
import tempfile
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from agent import ToolCallingAgent  # noqa: E402
from audit import AuditLog  # noqa: E402
from config import ALLOWED_ROOTS  # noqa: E402
from timers import TimerManager  # noqa: E402
from tools import ToolExecutor  # noqa: E402

results = []
fired = []


def check(name, cond, detail=""):
    results.append((name, bool(cond)))
    print(f"  [{'ok ' if cond else 'BAD'}] {name}" + (f"  -> {detail}" if not cond else ""))


def main():
    from llama_cpp import Llama

    print("Loading model ...")
    llm = Llama(model_path="D:/project-jarvis/models/qwen2.5-7b-instruct-q4_k_m-00001-of-00002.gguf",
                n_gpu_layers=-1, n_ctx=8192, verbose=False)
    audit = AuditLog(Path(tempfile.mkdtemp(prefix="jtu_")) / "audit.db")
    tm = TimerManager(on_fire=lambda t: fired.append((t.id, t.label, time.time())))

    opened = []
    def confirm(tool, args, preview):
        opened.append((tool, preview))
        print(f"    [CONFIRM] {tool}: {preview[:110]}")
        return False        # never actually open a browser in a test

    ex = ToolExecutor(ALLOWED_ROOTS, audit, index=None, confirm_fn=confirm, timers=tm)

    # ---- 1. short timer, end to end ------------------------------------
    print("\n>>> Set a timer for 5 seconds.")
    agent = ToolCallingAgent(llm, ex)
    reply = agent.chat("Set a timer for 5 seconds.")
    called = [c["name"] for c in agent.last_tool_calls]
    print(f"    tools: {called}   args: {[c['arguments'] for c in agent.last_tool_calls]}")
    print(f"    reply: {reply[:150].strip()!r}")

    check("used set_timer, not run_command", "set_timer" in called and "run_command" not in called,
          str(called))
    check("timer is counting down", len(tm.active()) == 1, str(tm.active()))

    # Measure from when the TIMER was created, not from now: the model turn
    # itself took a couple of seconds, during which the countdown was already
    # running, which made a correct timer look ~2s early.
    started = tm.active()[0].created if tm.active() else time.time()
    print("    waiting for it to fire ...")
    while not fired and time.time() - started < 20:
        time.sleep(0.25)
    check("the timer actually fired", bool(fired), "never fired")
    if fired:
        elapsed = fired[0][2] - started
        check("fired at roughly the right time (4.5-6.5s)", 4.5 <= elapsed <= 6.5, f"{elapsed:.1f}s")
    check("no longer listed as active", not tm.active(), str(tm.active()))

    # ---- 2. minutes get converted, and outlive the run_command cap ------
    print("\n>>> Set a timer for 2 minutes called pasta.")
    agent = ToolCallingAgent(llm, ex)
    reply = agent.chat("Set a timer for 2 minutes called pasta.")
    args = [c["arguments"] for c in agent.last_tool_calls]
    print(f"    tools: {[c['name'] for c in agent.last_tool_calls]}   args: {args}")
    secs = args[0].get("seconds") if args else None
    check("converted 2 minutes to 120 seconds", secs == 120, str(secs))
    check("a 120s timer is allowed at all (run_command caps at 20s)",
          any(t.seconds == 120 for t in tm.active()), str([t.seconds for t in tm.active()]))
    check("label captured", any("pasta" in (t.label or "").lower() for t in tm.active()),
          str([t.label for t in tm.active()]))

    # ---- 3. list + cancel ----------------------------------------------
    agent = ToolCallingAgent(llm, ex)
    reply = agent.chat("What timers do I have running?")
    print(f"\n>>> list timers -> {reply[:160].strip()!r}")
    check("lists the running timer", "pasta" in reply.lower() or "2m" in reply.lower(),
          reply[:100])

    tm.cancel_all()

    # ---- 4. open a link -------------------------------------------------
    print("\n>>> Open amazon on google.")
    agent = ToolCallingAgent(llm, ex)
    reply = agent.chat("Open amazon on google.")
    called = [c["name"] for c in agent.last_tool_calls]
    args = [c["arguments"] for c in agent.last_tool_calls]
    print(f"    tools: {called}   args: {args}")
    print(f"    reply: {reply[:160].strip()!r}")
    check("chose open_url", "open_url" in called, str(called))
    url = (args[0].get("url", "") if args else "").lower()
    check("URL points at amazon or a google search for it",
          "amazon" in url, url or "(none)")
    check("opening a link asked permission", any(t == "open_url" for t, _ in opened),
          str([t for t, _ in opened]))
    check("the declined link did NOT open", True)   # confirm_fn returned False throughout

    # ---- 5. scheme guard -------------------------------------------------
    before = len(opened)
    r = ex.execute("open_url", {"url": "file:///C:/Windows/win.ini"})
    check("file:// scheme refused", not r.ok and "http" in r.output, r.output[:80])
    r = ex.execute("open_url", {"url": "javascript:alert(1)"})
    check("javascript: scheme refused", not r.ok, r.output[:80])
    r = ex.execute("open_url", {"url": "https:///nohost"})
    check("URL with no host refused", not r.ok, r.output[:80])
    check("bad schemes never reached the confirmation prompt", len(opened) == before,
          f"{len(opened) - before} prompts shown for URLs that were never permissible")

    audit.close()
    bad = [n for n, ok in results if not ok]
    print(f"\n{'FAIL' if bad else 'PASS'}: {len(bad)} failing of {len(results)}")
    for n in bad:
        print(f"    failed: {n}")
    sys.exit(1 if bad else 0)


if __name__ == "__main__":
    main()
