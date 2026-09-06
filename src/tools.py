"""Tool schema + dispatcher with path containment, tiered permissions, audit.

Execution rules, enforced centrally in ToolExecutor.execute() so no individual
tool can forget them:

  1. Every path argument goes through resolve_within() BEFORE the tool runs.
  2. Read-only tools auto-execute. Destructive tools require an explicit
     confirmation callback returning True -- every single call, no caching,
     no "remember this choice".
  3. Every call is audited: intent first, outcome after. Denials included.

On run_command: it never touches a shell. The command is parsed with shlex and
passed as an argv list to subprocess with shell=False, so shell metacharacters
(; | & > $() `) are inert rather than interpreted.

HONEST LIMITATION -- the denylist below is a safety net, not a security
boundary. Denylists are inherently bypassable (rename the binary, use an
alias, base64 a payload). The real protections are: no shell, argv-only,
forced cwd inside a root, a hard timeout, and mandatory human confirmation on
every invocation. If you later want an actual boundary, switch COMMAND_MODE to
"allowlist" and enumerate the handful of executables you truly need.
"""
import difflib
import os
import re
import shlex
import shutil
import subprocess
import sys
import time
import urllib.parse
import webbrowser
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from config import COMMAND_TIMEOUT_SECONDS, DESTRUCTIVE_TOOLS, READ_ONLY_TOOLS  # noqa: E402
from safety import PathNotAllowed, resolve_within  # noqa: E402
from timers import format_duration  # noqa: E402
from walk import walk_files  # noqa: E402

MAX_READ_CHARS = 20000
MAX_OUTPUT_CHARS = 8000

START_TIME = time.time()

# Windows-only flag; harmless elsewhere because the helper is only spawned on
# Windows. Keeps nvidia-smi from flashing a console window over the UI.
_NO_WINDOW = 0x08000000 if sys.platform == "win32" else 0
_GPU_CACHE: dict = {"at": 0.0, "info": None}
_GPU_TTL = 4.0


def gpu_probe() -> dict | None:
    """{'name', 'used', 'total', 'util'} from nvidia-smi, or None.

    Cached: the HUD asks once a second and spawning a process per frame would
    cost more than everything else the UI does. None is an ordinary answer --
    no NVIDIA GPU, no driver, or no nvidia-smi on PATH are all normal states,
    not errors.
    """
    now = time.time()
    if now - _GPU_CACHE["at"] < _GPU_TTL:
        return _GPU_CACHE["info"]
    info = None
    try:
        proc = subprocess.run(
            ["nvidia-smi",
             "--query-gpu=memory.used,memory.total,utilization.gpu,name",
             "--format=csv,noheader,nounits"],
            capture_output=True, timeout=6, creationflags=_NO_WINDOW)
        lines = proc.stdout.decode(errors="replace").strip().splitlines()
        if proc.returncode == 0 and lines:
            used, total, util, name = [p.strip() for p in lines[0].split(",")][:4]
            info = {"name": name, "used": int(used), "total": int(total),
                    "util": int(util)}
    except Exception:  # noqa: BLE001
        info = None
    _GPU_CACHE.update(at=now, info=info)
    return info


def match_files(roots, query: str, limit: int = 8) -> list[Path]:
    """Files across `roots` whose NAME resembles `query`, best first.

    Shared by find_files and by the not-found hint, so a guessed path and an
    explicit search agree about which file was probably meant.
    """
    words = [w for w in re.split(r"[^a-z0-9]+", (query or "").lower()) if w]
    if not words:
        return []
    from extract import SUPPORTED_SUFFIXES

    joined = " ".join(words)
    scored = []
    for path in walk_files(roots):
        stem = path.name.lower()
        # Any requested word appearing in the name counts; scoring by how many
        # do puts "meeting notes" above a file merely called "notes". The ratio
        # breaks ties on overall similarity.
        hits = sum(1 for w in words if w in stem)
        if not hits:
            continue
        ratio = difflib.SequenceMatcher(None, joined, path.stem.lower()).ratio()
        # This is a DOCUMENT assistant, so a document outranks a source file that
        # matches equally well: "the budget" means q1_budget.xlsx, not the
        # budget.py that happens to sit in a checked-out repo in Downloads. Code
        # is still findable, just ranked below.
        bonus = 0.6 if path.suffix.lower() in SUPPORTED_SUFFIXES else 0.0
        scored.append((hits + ratio + bonus, path))
    scored.sort(key=lambda m: -m[0])
    return [p for _s, p in scored[:limit]]


def _not_found(path: Path, roots=None, limit: int = 6) -> str:
    """A 'no such file' message that names what IS there.

    A model asked to "open my meeting notes" will guess a filename -- observed:
    "notes\\my meeting notes.txt" for a file actually called meeting_notes.txt.
    A bare "does not exist" leaves it nothing to correct with, so it either
    gives up or guesses again. Listing the real neighbours, closest first, turns
    a dead end into a retry that succeeds on the next round.
    """
    # Best hint of all: the file the user probably meant, wherever it actually
    # lives. A guessed path is usually the right NAME in the wrong FOLDER --
    # "my meeting notes.md" in Documents when meeting_notes.txt is in the
    # sandbox -- and listing the wrong folder's contents does not help at all.
    if roots:
        near = match_files(roots, path.stem, limit=4)
        if near:
            # Phrased as an INSTRUCTION, not as information. Handed a list of
            # candidates the model would report them back and ask the user which
            # to use -- with the answer already in front of it. Naming the single
            # best path and telling it to retry turns the dead end into the next
            # tool call instead of into a question.
            others = "".join(f"\n  {p}" for p in near[1:])
            return (f"does not exist: {path}\n"
                    f'RETRY this same tool call now with path="{near[0]}"'
                    + (f"\nOther possibilities, if that is the wrong one:{others}"
                       if others else ""))

    # The guessed path may name a folder that does not exist either, so walk up
    # to the nearest real ancestor before listing anything.
    parent = path.parent
    while not parent.is_dir() and parent != parent.parent:
        parent = parent.parent
    try:
        entries = sorted(parent.iterdir(), key=lambda p: (p.is_file(), p.name.lower()))
    except OSError:
        return f"does not exist: {path}"
    if not entries:
        return f"does not exist: {path} ({parent} is empty)"

    # Directories are listed too, and marked. Listing only files reported a
    # root holding just subfolders as "empty", which is both wrong and a dead
    # end -- the model needs to see that notes\ exists in order to look in it.
    names = [p.name + ("" if p.is_file() else "\\") for p in entries]
    close = difflib.get_close_matches(path.name, names, n=limit, cutoff=0.3)
    shown = close or names[:limit]
    return (f"does not exist: {path}\n"
            f"{parent} actually contains: {', '.join(shown)}\n"
            f"Names ending in a backslash are folders — look inside them. Use an "
            f"exact name from this list, or call search_files to locate the file.")


def memory_status() -> tuple[int, float, float]:
    """(percent used, GB free, GB total) via GlobalMemoryStatusEx.

    ctypes rather than psutil: one more pip dependency, and ~10MB in the
    bundle, to read three numbers Windows already exposes.
    Returns zeros rather than raising when the call is unavailable.
    """
    try:
        import ctypes

        class _MemStatus(ctypes.Structure):
            _fields_ = [("dwLength", ctypes.c_ulong),
                        ("dwMemoryLoad", ctypes.c_ulong),
                        ("ullTotalPhys", ctypes.c_ulonglong),
                        ("ullAvailPhys", ctypes.c_ulonglong),
                        ("ullTotalPageFile", ctypes.c_ulonglong),
                        ("ullAvailPageFile", ctypes.c_ulonglong),
                        ("ullTotalVirtual", ctypes.c_ulonglong),
                        ("ullAvailVirtual", ctypes.c_ulonglong),
                        ("ullAvailExtendedVirtual", ctypes.c_ulonglong)]

        stat = _MemStatus()
        stat.dwLength = ctypes.sizeof(_MemStatus)
        ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(stat))
        gb = 1024 ** 3
        return int(stat.dwMemoryLoad), stat.ullAvailPhys / gb, stat.ullTotalPhys / gb
    except Exception:  # noqa: BLE001
        return 0, 0.0, 0.0


def gpu_status() -> str:
    """The same reading as gpu_probe(), as a sentence for the model to read."""
    info = gpu_probe()
    if info is None:
        return "no NVIDIA GPU detected"
    return (f"{info['name']} - {info['used']}/{info['total']} MiB VRAM in use, "
            f"{info['util']}% busy")

# Extensions Windows EXECUTES rather than opens. os.startfile() on any of these
# is arbitrary code execution with the user's privileges, so open_file refuses
# them outright -- containment says WHERE a file may be, not that running it is
# safe. A malicious .bat inside an allowed root is still a malicious .bat.
EXECUTABLE_SUFFIXES = {
    ".exe", ".com", ".bat", ".cmd", ".ps1", ".psm1", ".vbs", ".vbe", ".js", ".jse",
    ".wsf", ".wsh", ".msi", ".msp", ".scr", ".cpl", ".hta", ".jar", ".reg", ".lnk",
    ".pif", ".inf", ".sct", ".application", ".gadget",
}

# Apps reachable via launch_app are an ALLOWLIST, not free-form: letting the
# model start any executable it names would make every other guard here
# decorative. The list itself now lives in apps.py and is curated by the user
# (see apps.AppRegistry) rather than hardcoded to seven Windows accessories
# nobody actually works in.
#
# What the model supplies is a NAME, which is looked up in that allowlist. It
# never supplies an executable path, so it cannot invent one.

# Search hits are fed to the MODEL, not shown to a human, so a short preview is
# actively harmful: truncating to a couple of hundred characters hands back the
# top of a document and hides the part that answers the question, and the model
# then answers plausibly from the fragment. Chunks are ~1200 chars, so return
# the whole chunk -- the chunk is the retrieval unit.
SEARCH_SNIPPET_CHARS = 1400

# Upper bound on a single timer. Longer than this is a calendar entry, not a
# timer, and these do not survive an app restart.
MAX_TIMER_SECONDS = 12 * 60 * 60

# ALLOWLIST, now that ALLOWED_ROOTS covers real documents and not just a
# sandbox. The module docstring above is explicit that the denylist is a safety
# net rather than a boundary -- fine while the only thing reachable was throwaway
# test data, not fine when Documents and Desktop are in scope. An allowlist is a
# real boundary: anything not named here simply cannot run, renamed binary or not.
COMMAND_MODE = "allowlist"
COMMAND_ALLOWLIST = {"git", "python", "pip", "dir", "ls", "type", "cat", "findstr"}

# Denied PROGRAMS, matched on the parsed executable name -- not on raw substrings.
#
# Note what is deliberately NOT here: shell metacharacters (| ; & > $() `).
# With shell=False the command is passed as an argv list, so those characters
# arrive as inert literal arguments -- blocking them buys no safety and
# falsely rejects legitimate commands (a ';' inside a quoted Python one-liner,
# a '>' in a search string). The meaningful control is refusing the shell
# interpreters themselves, below, so a shell is never reachable to begin with.
DENIED_EXECUTABLES = {
    # shell / script interpreters -- these would reintroduce shell semantics
    "cmd", "powershell", "pwsh", "wscript", "cscript", "mshta", "rundll32",
    # destructive system tools
    "format", "diskpart", "vssadmin", "bcdedit", "cipher", "reg", "regedit",
    "shutdown", "taskkill", "sc", "net", "netsh", "schtasks", "wmic", "fsutil",
    # network fetchers (exfiltration / payload retrieval)
    "curl", "wget", "certutil", "bitsadmin", "ftp", "scp",
    # unix-y destructive tools reachable via git-bash on Windows
    "rm", "rmdir", "del", "mkfs", "dd",
}

# Dangerous argument patterns, checked per-token against the PARSED argv so
# quoted content is compared as its own token rather than as a raw substring.
DENIED_ARG_PATTERNS = ["-encodedcommand", "-enc", "invoke-expression", "iex", "-rf", "--no-preserve-root"]


class ToolError(Exception):
    pass


class ConfirmationDenied(Exception):
    pass


@dataclass
class ToolResult:
    ok: bool
    output: str
    meta: dict = field(default_factory=dict)


# --------------------------------------------------------------------------
# Schema handed to the LLM (Stage 4 will feed this to Qwen's native tools API)
# --------------------------------------------------------------------------

TOOL_SCHEMA = [
    {
        "type": "function",
        "function": {
            "name": "search_files",
            "description": "Semantic search over indexed file contents. Use this to find "
                           "documents by meaning, not exact filename.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "What to look for"},
                    "n_results": {"type": "integer", "description": "Max hits (default 5)"},
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "read_file",
            "description": "Read the text content of a file inside the allowed roots.",
            "parameters": {
                "type": "object",
                "properties": {"path": {"type": "string"}},
                "required": ["path"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "list_directory",
            "description": "List files and subdirectories of a directory.",
            "parameters": {
                "type": "object",
                "properties": {"path": {"type": "string"}},
                "required": ["path"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "remember_fact",
            "description": "Store a lasting fact or preference about the user in long-term "
                           "memory, as a subject/predicate/object triple. Use for things "
                           "worth recalling in future conversations (names, roles, "
                           "preferences, ongoing projects) -- not for passing details. "
                           "The triple must still make sense read back on its own, months "
                           "later, with no conversation around it.",
            "parameters": {
                "type": "object",
                "properties": {
                    "subject": {"type": "string",
                                "description": "Who or what the fact is about: "
                                               "'the user', 'Dana', 'the Helsinki project'"},
                    "predicate": {"type": "string",
                                  "description": "The relationship, as a short verb phrase: "
                                                 "'prefers', 'works on', 'is responsible for'"},
                    # The failure this wording exists to prevent: told "always
                    # export spreadsheets as CSV, not XLSX", the model stored
                    # (the user, are_exported_as, spreadsheets) -- a triple that
                    # dropped the only word that mattered.
                    "object": {"type": "string",
                               "description": "The specific VALUE being remembered. Keep the "
                                              "detail that makes the fact useful. For 'I want "
                                              "spreadsheets exported as CSV, not XLSX' the "
                                              "object is 'CSV for exported spreadsheets, not "
                                              "XLSX' -- never just 'spreadsheets'."},
                },
                "required": ["subject", "predicate", "object"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "recall_facts",
            "description": "Look up previously remembered facts about the user or their "
                           "projects from long-term memory.",
            "parameters": {
                "type": "object",
                "properties": {"query": {"type": "string"}},
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "find_files",
            "description": "Find files by NAME across every allowed directory. Give the "
                           "words you would expect in the filename ('meeting notes', "
                           "'budget'). Returns exact absolute paths. Use this before "
                           "opening, reading or editing a file whose path you do not "
                           "already know. Use search_files instead to search file CONTENT.",
            "parameters": {
                "type": "object",
                "properties": {
                    "name": {"type": "string",
                             "description": "Part of the filename, e.g. 'budget'"},
                },
                "required": ["name"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_datetime",
            "description": "The current local date and time on this machine. Use this for "
                           "anything time-dependent -- 'today', 'this week', how long until "
                           "a deadline, or how old a document is. Never guess the date.",
            "parameters": {"type": "object", "properties": {}},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "system_status",
            "description": "This computer's own vitals: memory in use, free disk space, "
                           "GPU memory, and how long the assistant has been running. Use it "
                           "when asked about the machine, resources, or whether it is healthy.",
            "parameters": {"type": "object", "properties": {}},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "write_file",
            "description": "Create or overwrite a file. DESTRUCTIVE - requires confirmation.",
            "parameters": {
                "type": "object",
                "properties": {"path": {"type": "string"}, "content": {"type": "string"}},
                "required": ["path", "content"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "set_timer",
            "description": "Start a countdown timer that notifies the user when it finishes. "
                           "Use this for any timer, alarm or reminder request. Do NOT use "
                           "run_command for timing — it blocks and is capped at 20 seconds.",
            "parameters": {
                "type": "object",
                "properties": {
                    "seconds": {"type": "number",
                                "description": "Duration in SECONDS (convert minutes/hours first)"},
                    "label": {"type": "string", "description": "Optional, e.g. 'tea', 'standup'"},
                },
                "required": ["seconds"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "list_timers",
            "description": "List timers that are still counting down, with time remaining.",
            "parameters": {"type": "object", "properties": {}},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "cancel_timer",
            "description": "Cancel a running timer by its id (see list_timers).",
            "parameters": {
                "type": "object",
                "properties": {"timer_id": {"type": "integer"}},
                "required": ["timer_id"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "open_url",
            "description": "Open a web address in the user's normal browser (e.g. "
                           "https://www.amazon.com, or a Google search URL). This hands the "
                           "link to the browser; it does not read the page. "
                           "DESTRUCTIVE - requires confirmation.",
            "parameters": {
                "type": "object",
                "properties": {
                    "url": {"type": "string", "description": "Full http:// or https:// URL"},
                },
                "required": ["url"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "edit_file",
            "description": "Replace an exact snippet inside an existing text file, leaving "
                           "the rest untouched. PREFER THIS over write_file when changing "
                           "part of a file — write_file replaces the whole thing and loses "
                           "everything you did not repeat. DESTRUCTIVE - requires confirmation.",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string"},
                    "old_string": {"type": "string",
                                   "description": "Exact text to replace. Must appear EXACTLY "
                                                  "once in the file; include surrounding "
                                                  "context to make it unique."},
                    "new_string": {"type": "string", "description": "Replacement text."},
                },
                "required": ["path", "old_string", "new_string"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "append_to_file",
            "description": "Add text to the end of a file without touching existing content. "
                           "Use for adding an item to a list or a note. Creates the file if "
                           "missing. DESTRUCTIVE - requires confirmation.",
            "parameters": {
                "type": "object",
                "properties": {"path": {"type": "string"}, "content": {"type": "string"}},
                "required": ["path", "content"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "create_directory",
            "description": "Create a new folder (and any missing parents). "
                           "DESTRUCTIVE - requires confirmation.",
            "parameters": {
                "type": "object",
                "properties": {"path": {"type": "string"}},
                "required": ["path"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "open_file",
            "description": "Open a file in whatever application Windows normally uses for it "
                           "(e.g. a .docx in Word, a .pdf in the PDF viewer). "
                           "DESTRUCTIVE - requires confirmation.",
            "parameters": {
                "type": "object",
                "properties": {"path": {"type": "string"}},
                "required": ["path"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "launch_app",
            "description": "Start one of the user's applications by name, optionally with a "
                           "file. Only applications the user has enabled can be launched; "
                           "call list_apps to see them. DESTRUCTIVE - requires confirmation.",
            "parameters": {
                "type": "object",
                "properties": {
                    "app": {"type": "string",
                            "description": "The app's name, e.g. 'Visual Studio Code', "
                                           "'Google Chrome', 'Excel'"},
                    "path": {"type": "string",
                             "description": "Optional file or folder to open with it, "
                                            "which must be inside the allowed roots."},
                },
                "required": ["app"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "list_apps",
            "description": "List the applications the user has authorised you to use. Call "
                           "this when you are unsure whether an app is available, or when "
                           "the user asks what you can open things with.",
            "parameters": {"type": "object", "properties": {}},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "open_with",
            "description": "Open a specific file in a specific application — 'open the budget "
                           "in Excel', 'open this in VS Code'. Use open_file instead when the "
                           "user does not care which app is used. DESTRUCTIVE - requires "
                           "confirmation.",
            "parameters": {
                "type": "object",
                "properties": {
                    "app": {"type": "string", "description": "An enabled app's name"},
                    "path": {"type": "string", "description": "Absolute path to the file"},
                },
                "required": ["app", "path"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "draw_image",
            "description": (
                "DRAW a picture and save it as a PNG — use this whenever the user asks you "
                "to draw, sketch, or make a picture of something. You compose the picture "
                "yourself out of shapes; never hand the drawing back to the user to do. Set "
                "open_in to an app name such as 'mspaint' to open the finished picture in "
                "it. Example — a rocket: "
                '[{"kind":"polygon","points":[[450,80],[390,240],[510,240]],"fill":"red"},'
                '{"kind":"rectangle","box":[390,240,510,520],"fill":"white","outline":"black","width":3},'
                '{"kind":"polygon","points":[[390,430],[320,545],[390,520]],"fill":"red"},'
                '{"kind":"polygon","points":[[510,430],[580,545],[510,520]],"fill":"red"},'
                '{"kind":"ellipse","box":[425,300,475,350],"fill":"skyblue","outline":"black","width":3},'
                '{"kind":"polygon","points":[[420,520],[450,650],[480,520]],"fill":"orange"}]. '
                "DESTRUCTIVE - requires confirmation."),
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string",
                             "description": "Absolute .png path inside an allowed root"},
                    "shapes": {
                        "type": "array",
                        "description": (
                            "The picture, drawn in order — later shapes cover earlier ones. "
                            "Each shape is an object with 'kind' plus its geometry: "
                            "rectangle and ellipse take box=[x0,y0,x1,y1]; polygon takes "
                            "points=[[x,y],...] with 3 or more; line takes points and width; "
                            "text takes at=[x,y] and text. All accept fill and outline, as a "
                            "colour name or #rrggbb. Use enough shapes for the subject to be "
                            "RECOGNISABLE — usually 6 to 15, including the details that "
                            "identify it: a face needs two eyes and a curved mouth, a house "
                            "needs a roof, a door and windows. Fill most of the canvas rather "
                            "than leaving a small drawing in one corner."),
                        "items": {"type": "object"},
                    },
                    "width": {"type": "integer", "description": "Canvas width, default 900"},
                    "height": {"type": "integer", "description": "Canvas height, default 700"},
                    "background": {"type": "string",
                                   "description": "Canvas colour, default white"},
                    "open_in": {"type": "string",
                                "description": "Optional enabled app to open the result in, "
                                               "e.g. 'mspaint'"},
                },
                "required": ["path", "shapes"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "move_file",
            "description": "Move or rename a file. DESTRUCTIVE - requires confirmation.",
            "parameters": {
                "type": "object",
                "properties": {"source": {"type": "string"}, "destination": {"type": "string"}},
                "required": ["source", "destination"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "delete_file",
            "description": "Delete a file. DESTRUCTIVE - requires confirmation.",
            "parameters": {
                "type": "object",
                "properties": {"path": {"type": "string"}},
                "required": ["path"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "run_command",
            "description": "Run a command inside an allowed root. No shell, argv only, "
                           "hard timeout. DESTRUCTIVE - requires confirmation.",
            "parameters": {
                "type": "object",
                "properties": {
                    "command": {"type": "string"},
                    "cwd": {"type": "string", "description": "Working dir inside an allowed root"},
                },
                "required": ["command"],
            },
        },
    },
]


class ToolExecutor:
    def __init__(self, allowed_roots, audit, index=None, confirm_fn=None, kg=None,
                 timers=None, apps=None):
        """confirm_fn(tool, args, preview) -> bool. Required for destructive tools."""
        self.roots = allowed_roots
        self.audit = audit
        self.index = index
        self.confirm_fn = confirm_fn
        self.kg = kg
        self.timers = timers
        # The user's curated launch allowlist. None disables app tools entirely,
        # which is the right default for a headless executor.
        self.apps = apps

    # ---- helpers -------------------------------------------------------

    def _tier(self, tool: str) -> str:
        if tool in READ_ONLY_TOOLS:
            return "read_only"
        if tool in DESTRUCTIVE_TOOLS:
            return "destructive"
        raise ToolError(f"unknown tool: {tool!r}")

    def _preview(self, tool: str, args: dict) -> str:
        """Human-readable description shown in the confirmation prompt.

        For edits this renders a real unified diff. Approving a sentence that
        merely CLAIMS what will change is not informed consent -- the point of
        the prompt is to see the actual before/after.
        """
        if tool == "write_file":
            n = len(args.get("content", ""))
            exists = Path(args["path"]).exists()
            verb = "OVERWRITE existing" if exists else "create new"
            return f"{verb} file {args['path']} ({n} chars)"
        if tool == "edit_file":
            diff = difflib.unified_diff(
                args.get("old_string", "").splitlines(),
                args.get("new_string", "").splitlines(),
                fromfile="before", tofile="after", lineterm="", n=1)
            body = "\n".join(list(diff)[:40]) or "(no textual change)"
            return f"edit {args['path']}\n\n{body}"
        if tool == "append_to_file":
            addition = args.get("content", "")
            shown = addition if len(addition) <= 400 else addition[:400] + "…"
            return f"append to {args['path']}:\n\n{shown}"
        if tool == "create_directory":
            return f"create folder {args['path']}"
        if tool == "open_file":
            return f"open {args['path']} in its default application"
        if tool == "open_url":
            # Shown in full and never elided: a URL is how data would leave
            # this machine, and a truncated one hides exactly that.
            return "open this link in your browser:\n\n" + str(args.get("url"))
        if tool == "launch_app":
            extra = f" with {args['path']}" if args.get("path") else ""
            return f"launch {args.get('app')}{extra}"
        if tool == "open_with":
            return (f"open {args.get('path')}\n\n"
                    f"in the application: {args.get('app')}")
        if tool == "draw_image":
            shapes = args.get("shapes") or []
            n = len(shapes) if isinstance(shapes, (list, tuple)) else 0
            where = f"\n\nand open it in: {args['open_in']}" if args.get("open_in") else ""
            return f"draw a picture ({n} shape(s)) and save it to {args.get('path')}{where}"
        if tool == "move_file":
            return f"move {args['source']}  ->  {args['destination']}"
        if tool == "delete_file":
            return f"permanently delete {args['path']}"
        if tool == "run_command":
            return f"run {args['command']!r} in {args.get('cwd') or self.roots[0]}"
        return f"{tool} {args}"

    # ---- dispatch ------------------------------------------------------

    def execute(self, tool: str, args: dict) -> ToolResult:
        try:
            tier = self._tier(tool)
        except ToolError as exc:
            self.audit.log_denied(tool, args, str(exc))
            return ToolResult(False, str(exc))

        # Containment is checked before confirmation: the user should never be
        # asked to approve something that was never permitted in the first place.
        try:
            resolved = self._resolve_args(tool, args)
        except PathNotAllowed as exc:
            self.audit.log_denied(tool, args, f"path denied: {exc}")
            return ToolResult(False, f"Denied: {exc}")

        confirmed = False
        if tier == "destructive":
            if self.confirm_fn is None:
                self.audit.log_denied(tool, args, "no confirmation handler available")
                return ToolResult(False, "Denied: destructive tool with no confirmation handler")
            if not self.confirm_fn(tool, args, self._preview(tool, args)):
                self.audit.log_denied(tool, args, "user declined confirmation")
                return ToolResult(False, "Denied: user declined")
            confirmed = True

        row_id, started = self.audit.begin(tool, args, tier, confirmed)
        try:
            result = getattr(self, f"_do_{tool}")(args, resolved)
            self.audit.complete(row_id, started, "ok" if result.ok else "error", result.output)
            return result
        except Exception as exc:  # noqa: BLE001
            self.audit.complete(row_id, started, "error", f"{type(exc).__name__}: {exc}")
            return ToolResult(False, f"{type(exc).__name__}: {exc}")

    def _resolve_args(self, tool: str, args: dict) -> dict:
        """Validate every path-shaped argument up front."""
        out = {}
        if tool in ("read_file", "list_directory", "write_file", "delete_file",
                    "edit_file", "append_to_file", "create_directory", "open_file",
                    "open_with", "draw_image"):
            out["path"] = resolve_within(args["path"], self.roots)
        elif tool == "launch_app":
            # 'path' is optional here, but if given it is still contained.
            if args.get("path"):
                out["path"] = resolve_within(args["path"], self.roots)
        elif tool == "open_url":
            # Validated HERE, before confirmation, for the same reason paths are:
            # the user must never be shown an approve/deny prompt for something
            # that was never permissible. A file:// or javascript: URL is
            # rejected outright rather than becoming a decision they could
            # mistakenly approve.
            raw = str(args.get("url", "")).strip()
            parsed = urllib.parse.urlparse(raw)
            if parsed.scheme.lower() not in ("http", "https"):
                raise PathNotAllowed(
                    f"only http and https links are allowed (got "
                    f"{parsed.scheme or 'no scheme'!r})")
            if not parsed.netloc:
                raise PathNotAllowed("link has no host")
            out["url"] = raw
        elif tool == "move_file":
            out["source"] = resolve_within(args["source"], self.roots)
            out["destination"] = resolve_within(args["destination"], self.roots)
        elif tool == "run_command":
            cwd = args.get("cwd") or self.roots[0]
            out["cwd"] = resolve_within(cwd, self.roots)
        return out

    # ---- read-only tools -----------------------------------------------

    def _do_search_files(self, args, resolved) -> ToolResult:
        if self.index is None:
            return ToolResult(False, "no index available")
        hits = self.index.search(args["query"], n_results=int(args.get("n_results", 5)))
        if not hits:
            return ToolResult(True, "No matches.")
        lines = []
        for h in hits:
            snippet = h["text"][:SEARCH_SNIPPET_CHARS].strip()
            if len(h["text"]) > SEARCH_SNIPPET_CHARS:
                snippet += "\n    ...[chunk truncated]"
            lines.append(f"--- {h['name']}  ({h['source']})\n{snippet}")
        return ToolResult(True, "\n\n".join(lines), {"hits": len(hits)})

    def _do_find_files(self, args, resolved) -> ToolResult:
        """Locate files by name across the roots.

        The gap this fills: search_files searches CONTENT, and list_directory
        needs the folder already. Asked to "open my meeting notes", a model with
        neither had to invent a path -- and it did, repeatedly. Ranking by
        similarity to the requested words puts the intended file first.
        """
        query = str(args.get("name", "")).strip()
        if not query:
            return ToolResult(False, "give some part of the filename to look for")

        matches = match_files(self.roots, query)
        if not matches:
            listing = ", ".join(sorted(p.name for p in walk_files(self.roots))[:12])
            return ToolResult(True, f"No filename matches {query!r}. "
                                    f"Files available: {listing or 'none'}")

        # LEAD with the best match and say to use it. A flat list of candidates
        # made the model stop and ask the user which file they meant -- fine when
        # one root held five files, useless across four real folders where
        # "meeting notes" also matches Rdbms notes.pdf and RE_Notes_QA.docx.
        # Results are already ranked; saying so is what makes them actionable.
        best, others = matches[0], matches[1:5]
        text = (f"Best match: {best}\n"
                f"Use this path unless the user clearly meant a different file.")
        if others:
            text += "\nOther matches:" + "".join(f"\n  {p}" for p in others)
        return ToolResult(True, text, {"count": len(matches), "best": str(best)})

    def _do_get_datetime(self, args, resolved) -> ToolResult:
        now = datetime.now()
        return ToolResult(True, now.strftime("%A, %d %B %Y, %H:%M:%S (local time)"),
                          {"iso": now.isoformat(timespec="seconds")})

    def _do_system_status(self, args, resolved) -> ToolResult:
        """Machine vitals, gathered without adding a psutil dependency.

        Memory comes from GlobalMemoryStatusEx via ctypes, disk from shutil, and
        the GPU from nvidia-smi if it is on PATH. Every part degrades to a note
        rather than an error: a status report that raises is worse than one that
        says a figure is unavailable.
        """
        lines = [f"Assistant uptime: {format_duration(time.time() - START_TIME)}"]

        load, free_gb, total_gb = memory_status()
        lines.append(f"Memory: {load}% used, {free_gb:.1f} GB free of {total_gb:.1f} GB"
                     if total_gb else "Memory: unavailable")

        # Reported per DRIVE, not per root. Documents, Desktop and Downloads all
        # live on C:, and listing "3.1 GB free" three times reads like three
        # separate findings instead of one.
        seen_drives = set()
        for root in self.roots:
            drive = Path(root).anchor or str(root)
            if drive in seen_drives:
                continue
            seen_drives.add(drive)
            try:
                usage = shutil.disk_usage(root)
                lines.append(f"Disk {drive} {usage.free / 1024 ** 3:.1f} GB free "
                             f"of {usage.total / 1024 ** 3:.0f} GB")
            except Exception:  # noqa: BLE001
                lines.append(f"Disk {drive} unavailable")

        lines.append(f"GPU: {gpu_status()}")
        return ToolResult(True, "\n".join(lines))

    def _do_remember_fact(self, args, resolved) -> ToolResult:
        if self.kg is None:
            return ToolResult(False, "no knowledge graph available")
        is_new = self.kg.add_fact(args["subject"], args["predicate"], args["object"])
        self.kg.save()
        verb = "Remembered" if is_new else "Already knew (confidence raised)"
        return ToolResult(True, f"{verb}: {args['subject']} {args['predicate']} {args['object']}")

    def _do_recall_facts(self, args, resolved) -> ToolResult:
        if self.kg is None:
            return ToolResult(False, "no knowledge graph available")
        facts = self.kg.query(args["query"])
        if not facts:
            return ToolResult(True, "Nothing remembered about that.")
        return ToolResult(True, "\n".join(f"- {f['text']} (confidence {f['confidence']:.2f})"
                                          for f in facts), {"facts": len(facts)})

    def _do_set_timer(self, args, resolved) -> ToolResult:
        if self.timers is None:
            return ToolResult(False, "timers are not available")
        try:
            seconds = float(args["seconds"])
        except (TypeError, ValueError):
            return ToolResult(False, f"invalid duration: {args.get('seconds')!r}")
        if seconds <= 0:
            return ToolResult(False, "duration must be positive")
        if seconds > MAX_TIMER_SECONDS:
            return ToolResult(False, f"too long — the maximum is "
                                     f"{format_duration(MAX_TIMER_SECONDS)}")
        label = str(args.get("label") or "").strip()
        timer = self.timers.set_timer(seconds, label)
        pretty = format_duration(seconds)
        return ToolResult(True, f"Timer #{timer.id} set for {pretty}"
                                + (f" ({label})" if label else "") + ".",
                          {"timer_id": timer.id})

    def _do_list_timers(self, args, resolved) -> ToolResult:
        if self.timers is None:
            return ToolResult(False, "timers are not available")
        active = self.timers.active()
        if not active:
            return ToolResult(True, "No timers are running.")
        lines = [f"#{t.id}  {format_duration(t.remaining)} left"
                 + (f"  ({t.label})" if t.label else "") for t in active]
        return ToolResult(True, "\n".join(lines), {"count": len(active)})

    def _do_cancel_timer(self, args, resolved) -> ToolResult:
        if self.timers is None:
            return ToolResult(False, "timers are not available")
        try:
            timer_id = int(args["timer_id"])
        except (TypeError, ValueError, KeyError):
            return ToolResult(False, f"invalid timer id: {args.get('timer_id')!r}")
        ok = self.timers.cancel(timer_id)
        return ToolResult(ok, f"Cancelled timer #{timer_id}." if ok
                          else f"No running timer with id {timer_id}.")

    def _do_open_url(self, args, resolved) -> ToolResult:
        # Scheme/host were already validated in _resolve_args, before the user
        # was ever asked to approve this.
        raw = resolved["url"]
        try:
            webbrowser.open(raw)
        except Exception as exc:  # noqa: BLE001
            return ToolResult(False, f"could not open browser: {exc}")
        return ToolResult(True, f"opened {raw} in the browser")

    def _do_read_file(self, args, resolved) -> ToolResult:
        path = resolved["path"]
        if not path.is_file():
            # Same reasoning as open_file: name the neighbours so a guessed
            # filename can be corrected on the next round instead of dead-ending.
            return ToolResult(False, _not_found(path, self.roots) if not path.exists()
                              else f"not a file: {path}")
        from extract import ExtractionError, extract_text
        try:
            text = extract_text(path)
        except ExtractionError as exc:
            return ToolResult(False, str(exc))
        truncated = len(text) > MAX_READ_CHARS
        return ToolResult(True, text[:MAX_READ_CHARS] + ("\n...[truncated]" if truncated else ""),
                          {"chars": len(text)})

    def _do_list_directory(self, args, resolved) -> ToolResult:
        path = resolved["path"]
        if not path.is_dir():
            return ToolResult(False, f"not a directory: {path}")
        entries = []
        for p in sorted(path.iterdir()):
            kind = "dir " if p.is_dir() else "file"
            size = "" if p.is_dir() else f"  {p.stat().st_size:,d} B"
            entries.append(f"  [{kind}] {p.name}{size}")
        return ToolResult(True, "\n".join(entries) or "  (empty)", {"count": len(entries)})

    # ---- destructive tools ---------------------------------------------

    def _do_write_file(self, args, resolved) -> ToolResult:
        path = resolved["path"]
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(args["content"], encoding="utf-8")
        return ToolResult(True, f"wrote {len(args['content'])} chars to {path}")

    def _do_edit_file(self, args, resolved) -> ToolResult:
        path = resolved["path"]
        if not path.is_file():
            return ToolResult(False, f"not a file: {path}")
        old, new = args["old_string"], args["new_string"]
        if not old:
            return ToolResult(False, "old_string must not be empty (use append_to_file to add)")

        text = path.read_text(encoding="utf-8", errors="replace")
        count = text.count(old)
        if count == 0:
            return ToolResult(False, "old_string not found in the file — read it first and "
                                     "copy the text exactly, including whitespace")
        if count > 1:
            # Editing an ambiguous match would silently change the wrong line.
            return ToolResult(False, f"old_string appears {count} times — include more "
                                     f"surrounding context so it matches exactly once")

        path.write_text(text.replace(old, new, 1), encoding="utf-8")
        return ToolResult(True, f"edited {path}: replaced {len(old)} chars with {len(new)}")

    def _do_append_to_file(self, args, resolved) -> ToolResult:
        path = resolved["path"]
        path.parent.mkdir(parents=True, exist_ok=True)
        addition = args["content"]
        existing = path.read_text(encoding="utf-8", errors="replace") if path.exists() else ""
        # Join with a newline unless the file already ends with one, so appended
        # items do not get glued onto the last line.
        sep = "" if (not existing or existing.endswith("\n")) else "\n"
        with path.open("a", encoding="utf-8") as fh:
            fh.write(sep + addition + ("" if addition.endswith("\n") else "\n"))
        return ToolResult(True, f"appended {len(addition)} chars to {path}")

    def _do_create_directory(self, args, resolved) -> ToolResult:
        path = resolved["path"]
        if path.exists():
            return ToolResult(False, f"already exists: {path}")
        path.mkdir(parents=True, exist_ok=True)
        return ToolResult(True, f"created directory {path}")

    def _do_open_file(self, args, resolved) -> ToolResult:
        path = resolved["path"]
        if not path.exists():
            return ToolResult(False, _not_found(path, self.roots))
        if path.suffix.lower() in EXECUTABLE_SUFFIXES:
            return ToolResult(False, f"refused: {path.suffix} is an executable type — "
                                     f"opening it would run it")
        try:
            os.startfile(str(path))  # noqa: S606 - Windows default handler, path is contained
        except OSError as exc:
            return ToolResult(False, f"could not open: {exc}")
        return ToolResult(True, f"opened {path} in its default application")

    def _do_list_apps(self, args, resolved) -> ToolResult:
        """Which applications the user has actually authorised."""
        if self.apps is None:
            return ToolResult(False, "the app registry is not available")
        allowed = self.apps.allowed()
        if not allowed:
            return ToolResult(True, "No applications have been enabled yet. The user "
                                    "enables them in the APPS tab of the window.")
        lines = [a.name + ("" if a.can_open_files else "  (can be launched, "
                                                       "but cannot be given a file)")
                 for a in allowed]
        return ToolResult(True, "\n".join(lines), {"count": len(lines)})

    def _resolve_app(self, name: str):
        """An enabled app, or a ToolResult explaining why not. Never a path."""
        if self.apps is None:
            return None, ToolResult(False, "the app registry is not available")
        entry = self.apps.resolve(name)
        if entry is None:
            options = ", ".join(self.apps.suggestions(name)) or "none yet"
            return None, ToolResult(
                False, f"{name!r} is not an app you have enabled. Enabled apps: {options}. "
                       f"The user can enable more in the APPS tab.")
        return entry, None

    def _do_launch_app(self, args, resolved) -> ToolResult:
        entry, problem = self._resolve_app(str(args.get("app", "")))
        if problem is not None:
            return problem
        target = resolved.get("path")
        argv = [entry.runnable]
        if target is not None:
            if not entry.can_open_files:
                return ToolResult(False, f"{entry.name} was installed as a shortcut, so it "
                                         f"cannot be handed a file. Launch it on its own, "
                                         f"or use open_file to let Windows choose the app.")
            if target.suffix.lower() in EXECUTABLE_SUFFIXES:
                return ToolResult(False, f"refused: {target.suffix} is an executable type")
            argv.append(str(target))
        try:
            # argv[0] comes from the user's enabled allowlist, never from the
            # model; argv[1] has already been through resolve_within().
            subprocess.Popen(argv, shell=False)  # noqa: S603
        except (OSError, FileNotFoundError) as exc:
            return ToolResult(False, f"could not launch {entry.name}: {exc}")
        return ToolResult(True, f"launched {entry.name}"
                                + (f" with {target.name}" if target else ""))

    def _do_open_with(self, args, resolved) -> ToolResult:
        """Open a specific file in a specific application."""
        entry, problem = self._resolve_app(str(args.get("app", "")))
        if problem is not None:
            return problem
        target = resolved["path"]
        if not target.exists():
            return ToolResult(False, _not_found(target, self.roots))
        if not entry.can_open_files:
            return ToolResult(False, f"{entry.name} cannot be given a file to open. "
                                     f"Use open_file to let Windows choose the app.")
        if target.suffix.lower() in EXECUTABLE_SUFFIXES:
            return ToolResult(False, f"refused: {target.suffix} is an executable type — "
                                     f"opening it would run it")
        try:
            subprocess.Popen([entry.target, str(target)], shell=False)  # noqa: S603
        except (OSError, FileNotFoundError) as exc:
            return ToolResult(False, f"could not open {target.name} in {entry.name}: {exc}")
        return ToolResult(True, f"opened {target.name} in {entry.name}")

    def _do_draw_image(self, args, resolved) -> ToolResult:
        """Compose a PNG from the model's shape list, optionally showing it.

        The picture is a normal file in an allowed root, so nothing new is
        reachable: opening it goes through the same app allowlist as open_with,
        and a bad shape list fails with a message the model can act on rather
        than a PIL traceback it cannot.
        """
        from drawing import DrawError, render

        path = resolved["path"]
        if path.suffix.lower() not in (".png", ".jpg", ".jpeg", ".bmp", ".gif"):
            return ToolResult(False, f"{path.suffix or 'that'} is not an image type — "
                                     f"save the drawing as .png")

        # The app is resolved BEFORE the file is written: being told the app is
        # not enabled after a picture has already appeared on disk is confusing,
        # and leaves a file the user did not ask for.
        entry = None
        if args.get("open_in"):
            entry, problem = self._resolve_app(str(args["open_in"]))
            if problem is not None:
                return problem
            if not entry.can_open_files:
                return ToolResult(False, f"{entry.name} cannot be handed a file to open, so "
                                         f"the picture cannot be shown in it.")

        try:
            img = render(args.get("shapes"), args.get("width"), args.get("height"),
                         args.get("background") or "white")
        except DrawError as exc:
            return ToolResult(False, f"could not draw that: {exc}")

        path.parent.mkdir(parents=True, exist_ok=True)
        if path.suffix.lower() in (".jpg", ".jpeg"):
            img = img.convert("RGB")
        img.save(str(path))
        note = f"drew {img.width}x{img.height} picture to {path}"

        if entry is not None:
            try:
                subprocess.Popen([entry.target, str(path)], shell=False)  # noqa: S603
            except (OSError, FileNotFoundError) as exc:
                # The drawing exists and that half succeeded; say so precisely
                # rather than reporting the whole call as a failure.
                return ToolResult(True, f"{note}, but could not open it in "
                                        f"{entry.name}: {exc}")
            note += f", and opened it in {entry.name}"
        return ToolResult(True, note)

    def _do_move_file(self, args, resolved) -> ToolResult:
        src, dst = resolved["source"], resolved["destination"]
        if not src.exists():
            return ToolResult(False, f"source does not exist: {src}")
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(src), str(dst))
        return ToolResult(True, f"moved {src} -> {dst}")

    def _do_delete_file(self, args, resolved) -> ToolResult:
        path = resolved["path"]
        if not path.exists():
            return ToolResult(False, f"does not exist: {path}")
        if path.is_dir():
            return ToolResult(False, "refusing to delete a directory (files only)")
        path.unlink()
        return ToolResult(True, f"deleted {path}")

    @staticmethod
    def _parse_argv(raw: str) -> list[str]:
        """Split a command line into argv.

        posix=False is required on Windows so backslashes in paths are not
        eaten as escapes -- but it also LEAVES the quotes attached to each
        token, which would hand `python -c` the literal string '"print(1)"'.
        So strip one layer of matching outer quotes afterwards.
        """
        argv = shlex.split(raw, posix=False)
        cleaned = []
        for tok in argv:
            if len(tok) >= 2 and tok[0] == tok[-1] and tok[0] in ("'", '"'):
                tok = tok[1:-1]
            cleaned.append(tok)
        return cleaned

    def _do_run_command(self, args, resolved) -> ToolResult:
        try:
            argv = self._parse_argv(args["command"])
        except ValueError as exc:
            return ToolResult(False, f"could not parse command: {exc}")
        if not argv:
            return ToolResult(False, "empty command")

        exe = Path(argv[0]).name.lower().removesuffix(".exe")

        if COMMAND_MODE == "allowlist":
            if exe not in COMMAND_ALLOWLIST:
                return ToolResult(False, f"refused: {exe!r} is not in the allowlist")
        else:
            if exe in DENIED_EXECUTABLES:
                return ToolResult(False, f"refused: {exe!r} is a denied executable")
            for tok in argv[1:]:
                low = tok.lower()
                for pattern in DENIED_ARG_PATTERNS:
                    # Compare per-token: a ';' inside a quoted script body is
                    # its own token and harmless, unlike a bare dangerous flag.
                    if low == pattern or low.startswith(pattern + "="):
                        return ToolResult(False, f"refused: argument matches denied pattern {pattern!r}")

        try:
            proc = subprocess.run(
                argv,
                cwd=str(resolved["cwd"]),
                shell=False,                      # never a shell
                capture_output=True,
                text=True,
                timeout=COMMAND_TIMEOUT_SECONDS,
                env={**os.environ, "PYTHONIOENCODING": "utf-8"},
            )
        except subprocess.TimeoutExpired:
            return ToolResult(False, f"timed out after {COMMAND_TIMEOUT_SECONDS}s")
        except FileNotFoundError:
            return ToolResult(False, f"executable not found: {argv[0]}")

        out = (proc.stdout or "") + (("\n[stderr]\n" + proc.stderr) if proc.stderr else "")
        if len(out) > MAX_OUTPUT_CHARS:
            out = out[:MAX_OUTPUT_CHARS] + "\n...[truncated]"
        return ToolResult(proc.returncode == 0, out.strip() or "(no output)",
                          {"returncode": proc.returncode})
