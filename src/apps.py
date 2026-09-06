"""Installed-app discovery and the user-curated launch allowlist.

WHY THIS EXISTS: launch_app used to carry seven hardcoded names (notepad, calc,
paint...). That is safe and nearly useless -- none of them are the apps anyone
actually works in. The fix is not to drop the allowlist, which is the only thing
standing between a tool-calling model and every executable on the machine. It is
to make the allowlist *the user's*, discovered automatically and enabled by hand.

So:
  * discover() finds what is installed, from the Start Menu and the registry.
  * Anything that is a shell, a system administration tool, or an uninstaller is
    marked `risky` and is NOT offered by default.
  * AppRegistry persists which names the user enabled, in apps.json.
  * Only an enabled name can be launched. The model never supplies a path -- it
    supplies a NAME, which is looked up here. A model that cannot name an
    executable path cannot invent one.

The risky flag is a default, not a prohibition: the user can enable PowerShell if
they truly want to. It is marked so that enabling it is a deliberate act rather
than an accident of discovery.
"""
import json
import os
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
APPS_FILE = ROOT / "apps.json"

CREATE_NO_WINDOW = 0x08000000 if sys.platform == "win32" else 0

# Reaching a shell would dissolve the entire run_command design, which
# deliberately never uses one: shell=False with an argv list, so ';' and '|'
# arrive as inert text. Launching cmd.exe hands back everything that guard
# removes, so shells are held back the hardest.
SHELL_EXES = {
    "cmd.exe", "powershell.exe", "pwsh.exe", "wt.exe", "bash.exe", "sh.exe",
    "wscript.exe", "cscript.exe", "powershell_ise.exe", "conhost.exe",
}
# System administration and recovery tools: launching these is not dangerous in
# the way a shell is, but none of them is what "open my editor" means.
SYSTEM_EXES = {
    "regedit.exe", "taskmgr.exe", "mmc.exe", "services.exe", "gpedit.msc",
    "diskmgmt.msc", "compmgmt.msc", "eventvwr.exe", "control.exe", "msconfig.exe",
    "cleanmgr.exe", "dfrgui.exe", "recdisc.exe", "systempropertiesprotection.exe",
    "sdclt.exe", "perfmon.exe", "resmon.exe", "odbcad32.exe", "iscsicpl.exe",
}
RISKY_NAME_HINTS = ("uninstall", "uninstal", "repair", "recovery drive",
                    "registry editor", "command prompt", "powershell",
                    "firewall", "administrative tools", "reset")

# Suggested on first run only, and only if actually installed. Deliberately dull:
# a text editor, a browser, the file manager, the office apps people name out
# loud. Nothing here can do anything a confirmation prompt would not already
# have caught, and every one of them can be unticked.
SUGGESTED_DEFAULTS = (
    "notepad", "file explorer", "visual studio code", "google chrome",
    "microsoft edge", "word", "excel", "powerpoint", "calculator", "paint",
)


@dataclass
class AppEntry:
    name: str
    target: str = ""          # the .exe, when it could be resolved
    launcher: str = ""        # the .lnk, always launchable even if target is unknown
    risky: bool = False
    source: str = "start-menu"

    def to_dict(self) -> dict:
        return {"name": self.name, "target": self.target, "launcher": self.launcher,
                "risky": self.risky, "source": self.source}

    @classmethod
    def from_dict(cls, d: dict) -> "AppEntry":
        return cls(name=d.get("name", ""), target=d.get("target", ""),
                   launcher=d.get("launcher", ""), risky=bool(d.get("risky")),
                   source=d.get("source", "start-menu"))

    @property
    def runnable(self) -> str:
        """What to hand the OS. Prefers the exe; a .lnk still opens correctly."""
        return self.target or self.launcher

    @property
    def can_open_files(self) -> bool:
        """A .lnk cannot take an argument; only a real exe can open a file."""
        return bool(self.target) and self.target.lower().endswith(".exe")


def _is_risky(name: str, target: str) -> bool:
    exe = Path(target).name.lower() if target else ""
    if exe in SHELL_EXES or exe in SYSTEM_EXES:
        return True
    low = name.lower()
    return any(hint in low for hint in RISKY_NAME_HINTS)


def _start_menu_apps() -> list[AppEntry]:
    """Every Start Menu shortcut, with its target resolved.

    Shortcut targets are resolved in ONE PowerShell call rather than per file:
    130 shortcuts would otherwise be 130 process spawns, which is several
    seconds of stall on a UI thread.
    """
    if sys.platform != "win32":
        return []
    script = r"""
$ErrorActionPreference = 'SilentlyContinue'
$sh = New-Object -ComObject WScript.Shell
$roots = @("$env:ProgramData\Microsoft\Windows\Start Menu\Programs",
           "$env:APPDATA\Microsoft\Windows\Start Menu\Programs")
$out = foreach ($r in $roots) {
  if (Test-Path $r) {
    Get-ChildItem -Path $r -Recurse -Filter *.lnk | ForEach-Object {
      $t = ''
      try { $t = $sh.CreateShortcut($_.FullName).TargetPath } catch {}
      [pscustomobject]@{ name = $_.BaseName; lnk = $_.FullName; target = $t }
    }
  }
}
$out | ConvertTo-Json -Compress
"""
    try:
        proc = subprocess.run(
            ["powershell", "-NoProfile", "-NonInteractive", "-Command", script],
            capture_output=True, timeout=90, creationflags=CREATE_NO_WINDOW)
        raw = proc.stdout.decode("utf-8", errors="replace").strip()
        data = json.loads(raw) if raw else []
    except Exception:  # noqa: BLE001 - discovery must never be fatal
        return []
    if isinstance(data, dict):          # a single shortcut does not come back as a list
        data = [data]

    apps = []
    for item in data:
        name = (item.get("name") or "").strip()
        target = (item.get("target") or "").strip()
        if not name:
            continue
        # Shortcuts to documents and folders are not apps.
        if target and not target.lower().endswith((".exe", ".msc", ".bat", ".cmd")):
            target = ""
        apps.append(AppEntry(name=name, target=target, launcher=item.get("lnk", ""),
                             risky=_is_risky(name, target), source="start-menu"))
    return apps


def _registry_apps() -> list[AppEntry]:
    """Canonical exe paths from App Paths -- how Windows itself resolves 'chrome'."""
    if sys.platform != "win32":
        return []
    import winreg

    found = []
    for hive in (winreg.HKEY_LOCAL_MACHINE, winreg.HKEY_CURRENT_USER):
        try:
            key = winreg.OpenKey(
                hive, r"SOFTWARE\Microsoft\Windows\CurrentVersion\App Paths")
        except OSError:
            continue
        with key:
            for i in range(winreg.QueryInfoKey(key)[0]):
                try:
                    sub = winreg.EnumKey(key, i)
                    with winreg.OpenKey(key, sub) as app_key:
                        target = winreg.QueryValueEx(app_key, "")[0].strip('"')
                except OSError:
                    continue
                if not sub.lower().endswith(".exe") or not target:
                    continue
                name = Path(sub).stem
                found.append(AppEntry(name=name, target=target,
                                      risky=_is_risky(name, target),
                                      source="registry"))
    return found


def discover() -> list[AppEntry]:
    """Everything installed, de-duplicated, sorted by display name."""
    seen: dict[str, AppEntry] = {}
    # Start Menu first: its names are the ones a person would say out loud
    # ("Visual Studio Code"), where the registry has "code".
    for app in _start_menu_apps() + _registry_apps():
        key = app.name.lower()
        existing = seen.get(key)
        if existing is None:
            seen[key] = app
        elif not existing.target and app.target:
            existing.target = app.target      # fill in a target we could not resolve

    # Second pass, by TARGET. The registry and the Start Menu describe the same
    # executable under different names, which would otherwise show the user
    # "chrome" and "Google Chrome" as two separate apps to tick. The friendlier
    # name wins; the terse one still resolves through resolve()'s substring match.
    by_target: dict[str, AppEntry] = {}
    for app in seen.values():
        if not app.target:
            by_target[f"::{app.name.lower()}"] = app      # nothing to merge on
            continue
        key = app.target.lower()
        best = by_target.get(key)
        if best is None or _friendlier(app, best):
            by_target[key] = app
    return sorted(by_target.values(), key=lambda a: a.name.lower())


def _friendlier(candidate: AppEntry, current: AppEntry) -> bool:
    """Prefer the Start Menu's human name over the registry's exe stem."""
    if candidate.source != current.source:
        return candidate.source == "start-menu"
    return len(candidate.name) > len(current.name)


class AppRegistry:
    """The catalog of installed apps plus the names the user has enabled."""

    def __init__(self, path: Path = APPS_FILE):
        self.path = Path(path)
        self.catalog: dict[str, AppEntry] = {}
        self.enabled: set[str] = set()
        self.load()

    # ---- persistence ----------------------------------------------------

    def load(self) -> None:
        if not self.path.exists():
            return
        try:
            data = json.loads(self.path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return
        self.catalog = {k.lower(): AppEntry.from_dict(v)
                        for k, v in (data.get("catalog") or {}).items()}
        self.enabled = {n.lower() for n in (data.get("enabled") or [])}

    def save(self) -> None:
        payload = {
            "version": 1,
            "enabled": sorted(self.enabled),
            "catalog": {k: v.to_dict() for k, v in self.catalog.items()},
        }
        tmp = self.path.with_suffix(".tmp")
        tmp.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        os.replace(tmp, self.path)

    # ---- discovery ------------------------------------------------------

    def rescan(self) -> int:
        """Refresh the catalog. Enabled names the user picked are preserved."""
        self.catalog = {a.name.lower(): a for a in discover()}
        # Drop enabled entries for apps that have since been uninstalled, so the
        # list cannot slowly fill with names that resolve to nothing.
        self.enabled &= set(self.catalog)
        self.save()
        return len(self.catalog)

    def ensure_scanned(self) -> None:
        first_run = not self.path.exists()
        if not self.catalog:
            self.rescan()
        if first_run:
            self.seed_defaults()

    def seed_defaults(self) -> list[str]:
        """Tick a few obvious, non-risky apps on the very first run.

        An empty allowlist is the correct DEFAULT but a poor first experience:
        "open this in my editor" fails until the user has found the APPS tab. So
        a handful of common, harmless applications are suggested -- and only
        suggested. They are ordinary entries the user can untick, nothing is
        added that is not already installed, and anything flagged risky is
        skipped no matter what it is called.
        """
        seeded = []
        for want in SUGGESTED_DEFAULTS:
            for key, entry in self.catalog.items():
                if entry.risky or key in self.enabled:
                    continue
                if key == want or key.startswith(want) or want in key:
                    self.enabled.add(key)
                    seeded.append(entry.name)
                    break
        if seeded:
            self.save()
        return seeded

    # ---- the allowlist --------------------------------------------------

    def enable(self, name: str, on: bool = True) -> bool:
        key = name.lower()
        if key not in self.catalog:
            return False
        self.enabled.add(key) if on else self.enabled.discard(key)
        self.save()
        return True

    def allowed(self) -> list[AppEntry]:
        return sorted((self.catalog[n] for n in self.enabled if n in self.catalog),
                      key=lambda a: a.name.lower())

    def resolve(self, name: str) -> AppEntry | None:
        """Look up an ENABLED app by name, tolerating how a model would say it.

        Exact match first, then a unique prefix or substring: asked for "chrome"
        the user's entry may be "Google Chrome". Ambiguity resolves to nothing --
        launching one of two possible apps because the name was close is worse
        than saying it was unclear.
        """
        key = (name or "").strip().lower()
        if not key:
            return None
        if key in self.enabled and key in self.catalog:
            return self.catalog[key]
        hits = [self.catalog[n] for n in self.enabled
                if n in self.catalog and (n.startswith(key) or key in n)]
        return hits[0] if len(hits) == 1 else None

    def suggestions(self, name: str, limit: int = 6) -> list[str]:
        """Names to show when a lookup fails -- enabled ones only."""
        import difflib
        names = [self.catalog[n].name for n in self.enabled if n in self.catalog]
        close = difflib.get_close_matches(name or "", names, n=limit, cutoff=0.3)
        return close or sorted(names)[:limit]
