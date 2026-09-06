"""The app allowlist: discovery, curation, and the boundary it enforces.

launch_app is the one tool that starts arbitrary programs, so the allowlist is
its entire security story. It used to be seven hardcoded Windows accessories --
safe, and useless, since none of them is an app anyone works in. It is now
discovered from the machine and curated by the user, which is far more useful
and therefore has to be tested far more carefully.

The property that matters: **installed is not the same as allowed.** PowerShell
is on every Windows machine and will be in the catalog; it must be unreachable
until the user deliberately enables it.

    venv\\Scripts\\python.exe scripts\\test_apps.py

Discovery is exercised against the real machine; everything else runs against a
synthetic catalog so the checks do not depend on what happens to be installed.
"""
import json
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

import tools as tools_mod  # noqa: E402
from apps import AppEntry, AppRegistry, _is_risky, discover  # noqa: E402
from audit import AuditLog  # noqa: E402
from tools import ToolExecutor  # noqa: E402

FAILURES = []


def check(label, ok, detail=""):
    print(f"  [{'ok ' if ok else 'BAD'}] {label}" + (f"   {detail}" if detail else ""))
    if not ok:
        FAILURES.append(label)


def fake_registry(tmp: Path, exe: Path) -> AppRegistry:
    reg = AppRegistry(tmp / "apps.json")
    reg.catalog = {
        "test editor": AppEntry("Test Editor", target=str(exe), risky=False),
        "test browser": AppEntry("Test Browser", target=str(exe), risky=False),
        "powershell": AppEntry("PowerShell", target=r"C:\Windows\System32\powershell.exe",
                               risky=True),
        "tiled app": AppEntry("Tiled App", target="", launcher=str(exe.with_suffix(".lnk")),
                              risky=False),
    }
    return reg


def main():
    tmp = Path(tempfile.mkdtemp(prefix="jarvis_apps_"))
    root = tmp / "sandbox"
    root.mkdir()
    doc = root / "notes.txt"
    doc.write_text("hello", encoding="utf-8")
    evil = root / "payload.bat"
    evil.write_text("echo hi", encoding="utf-8")
    outside = tmp / "outside.txt"
    outside.write_text("secret", encoding="utf-8")
    exe = tmp / "editor.exe"
    exe.write_text("", encoding="utf-8")

    print("--- discovery (this machine) ---")
    found = discover()
    check("found installed applications", len(found) > 5, f"{len(found)} apps")
    check("names are populated", all(a.name for a in found))
    risky = [a for a in found if a.risky]
    check("shells and system tools are flagged risky", len(risky) >= 1,
          f"{len(risky)} flagged")
    shells = [a for a in found if a.target.lower().endswith("powershell.exe")]
    check("every shell found is flagged", all(a.risky for a in shells),
          f"{len(shells)} shell entries")
    check("a cmd.exe shortcut is caught by its TARGET, not its label",
          _is_risky("Node.js command prompt", r"C:\Windows\System32\cmd.exe"))
    check("an ordinary app is not flagged",
          not _is_risky("Visual Studio Code", r"D:\Microsoft VS Code\Code.exe"))

    print("\n--- curation ---")
    reg = fake_registry(tmp, exe)
    check("nothing is enabled by default", reg.allowed() == [])
    check("an unknown name cannot be enabled", not reg.enable("Photoshop"))
    check("a known name can be enabled", reg.enable("Test Editor"))
    check("enabling persists to disk", (tmp / "apps.json").exists())
    saved = json.loads((tmp / "apps.json").read_text(encoding="utf-8"))
    check("the file records the enabled name", "test editor" in saved["enabled"])
    check("a fresh registry reads it back",
          [a.name for a in AppRegistry(tmp / "apps.json").allowed()] == ["Test Editor"])
    check("disabling works", reg.enable("Test Editor", False) and reg.allowed() == [])
    reg.enable("Test Editor")

    print("\n--- installed is NOT allowed ---")
    check("an installed but unenabled app does not resolve",
          reg.resolve("Test Browser") is None)
    check("PowerShell is in the catalog but unreachable",
          "powershell" in reg.catalog and reg.resolve("powershell") is None)
    check("an enabled app resolves", reg.resolve("Test Editor") is not None)
    check("a partial name resolves", reg.resolve("editor") is not None)
    reg.enable("Test Browser")
    check("an ambiguous name resolves to nothing rather than guessing",
          reg.resolve("test") is None)
    reg.enable("Test Browser", False)

    print("\n--- launch_app / open_with ---")
    audit = AuditLog(tmp / "audit.db")
    launched = []
    real_popen = tools_mod.subprocess.Popen
    tools_mod.subprocess.Popen = lambda argv, **kw: launched.append(list(argv))
    try:
        ex = ToolExecutor([root], audit, confirm_fn=lambda *a: True, apps=reg)

        r = ex.execute("launch_app", {"app": "Test Editor"})
        check("an enabled app launches", r.ok, r.output)
        check("it launched the allowlisted executable", launched == [[str(exe)]],
              str(launched))

        launched.clear()
        r = ex.execute("launch_app", {"app": "PowerShell"})
        check("an unenabled shell is refused", not r.ok, r.output[:70])
        check("and nothing was started", launched == [])

        # The model supplies a NAME, never a path. If it tries a path, the
        # lookup must fail rather than treating it as an executable to run.
        r = ex.execute("launch_app", {"app": r"C:\Windows\System32\cmd.exe"})
        check("a raw executable path is not accepted as an app name", not r.ok)
        check("still nothing started", launched == [])

        r = ex.execute("open_with", {"app": "Test Editor", "path": str(doc)})
        check("open_with opens a contained file", r.ok, r.output)
        check("with the right argv", launched == [[str(exe), str(doc.resolve())]],
              str(launched))

        launched.clear()
        r = ex.execute("open_with", {"app": "Test Editor", "path": str(outside)})
        check("open_with refuses a file outside the roots", not r.ok, r.output[:60])
        check("nothing started for the denied path", launched == [])

        r = ex.execute("open_with", {"app": "Test Editor", "path": str(evil)})
        check("open_with refuses executable file types", not r.ok, r.output[:60])

        r = ex.execute("open_with", {"app": "Test Editor", "path": str(root / "ghost.txt")})
        check("a missing file gives a usable hint", not r.ok and "notes.txt" in r.output)

        # Must be ENABLED first, or this passes for the wrong reason: refused as
        # unauthorised rather than as unable to take an argument.
        reg.enable("Tiled App")
        r = ex.execute("open_with", {"app": "Tiled App", "path": str(doc)})
        check("a shortcut-only app cannot be handed a file",
              not r.ok and "cannot be given a file" in r.output, r.output[:70])
        r = ex.execute("launch_app", {"app": "Tiled App"})
        check("but a shortcut-only app still launches on its own", r.ok, r.output[:70])
        reg.enable("Tiled App", False)

        print("\n--- list_apps ---")
        r = ex.execute("list_apps", {})
        check("lists enabled apps only",
              r.ok and "Test Editor" in r.output and "PowerShell" not in r.output)
    finally:
        tools_mod.subprocess.Popen = real_popen

    print("\n--- confirmation tier ---")
    denied = []
    ex_deny = ToolExecutor([root], audit, apps=reg,
                           confirm_fn=lambda *a: denied.append(a) or False)
    tools_mod.subprocess.Popen = lambda argv, **kw: launched.append(list(argv))
    try:
        launched.clear()
        ex_deny.execute("launch_app", {"app": "Test Editor"})
        ex_deny.execute("open_with", {"app": "Test Editor", "path": str(doc)})
        check("both app tools asked permission", len(denied) == 2, f"{len(denied)} prompts")
        check("a declined launch starts nothing", launched == [])
    finally:
        tools_mod.subprocess.Popen = real_popen

    audit.close()
    print(f"\n{'PASS' if not FAILURES else 'FAIL'}: {len(FAILURES)} failing check(s)")
    for name in FAILURES:
        print(f"   - {name}")
    sys.exit(1 if FAILURES else 0)


if __name__ == "__main__":
    main()
