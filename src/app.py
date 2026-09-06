"""Stage 6: the desktop app shell — HUD-styled window plus system tray icon.

Shows listening state, a live transcript, modal confirmation prompts for
destructive tools, and an audit log viewer.

TOOLKIT CHOICE — tkinter, not PySide6/PyQt:
  * tkinter ships with CPython, so it adds nothing to download on a slow link
    and nothing to a PyInstaller bundle that is already ~2GB because of
    ggml-cuda.dll (703MB) and the CUDA runtime (~736MB). PySide6 would add
    roughly another 200MB plus LGPL packaging obligations for a distributed
    binary.
  * The HUD look is drawn on a Canvas, which tkinter does perfectly well.
    Widget COLOURS need the 'clam' ttk theme — the native Windows theme
    silently ignores most colour options (see theme.apply_ttk_theme).

THREADING — the part that actually matters:
tkinter may only be touched from the thread that created the root window, but
the voice loop and the LLM must not run there or the UI freezes for the whole
turn. So the pipeline runs on a worker thread and communicates through a queue
that the main thread drains via after(). Confirmation is the delicate case: the
worker must BLOCK until the user answers, so it posts a request carrying a
threading.Event and waits on it while the UI thread shows the dialog and sets
the result. Never call a tkinter method from the worker.
"""
import queue
import sys
import threading
import time
import tkinter as tk
from datetime import datetime
from pathlib import Path
from tkinter import ttk

sys.path.insert(0, str(Path(__file__).resolve().parent))

import theme  # noqa: E402
from apps import AppRegistry  # noqa: E402
from audit import AuditLog  # noqa: E402
from config import ALLOWED_ROOTS, AUDIT_DB  # noqa: E402
from reactor import Reactor  # noqa: E402
from tools import ToolExecutor, gpu_probe, memory_status  # noqa: E402

ROLE_STYLE = {
    "user":   ("YOU",    theme.CYAN),
    "jarvis": ("JARVIS", theme.GREEN),
    "tool":   ("EXEC",   theme.AMBER),
    "system": ("SYS",    theme.TEXT_DIM),
    "denied": ("DENIED", theme.RED),
}


def _human_size(n: int) -> str:
    for unit in ("B", "KB", "MB", "GB"):
        if n < 1024 or unit == "GB":
            return f"{n:.0f} {unit}" if unit == "B" else f"{n:.1f} {unit}"
        n /= 1024.0
    return f"{n:.1f} GB"


# Starting points, so the first thing a new user sees is what it can DO rather
# than an empty prompt.
QUICK_ACTIONS = [
    ("WHAT'S HERE?", "What files do I have, and what is in them?"),
    ("SUMMARISE", "Summarise my meeting notes and list the action items."),
    ("STATUS", "What is your system status?"),
    ("TIME", "What is the date and time?"),
]


class ConfirmRequest:
    """A destructive tool call awaiting the user's decision."""

    def __init__(self, tool, args, preview):
        self.tool, self.args, self.preview = tool, args, preview
        self.event = threading.Event()
        self.approved = False


class JarvisApp:
    def __init__(self, pipeline_factory=None, audit_db: Path = AUDIT_DB, tray=True):
        # Before Tk exists, or Windows stretches the finished window instead of
        # letting it draw at the display's real resolution.
        theme.setup_dpi()
        self.root = tk.Tk()
        self.scale = theme.scale_factor(self.root)
        self.root.title("J.A.R.V.I.S.")
        # Position explicitly, do not let the window manager place it. Size-only
        # geometry leaves placement to Windows, which cascades each new window
        # down and right from the last one -- and once it runs out of room it
        # wraps to NEGATIVE coordinates. A second launch measured at x=-42, off
        # the left edge and behind whatever was already open, which reads to the
        # user as "it didn't start". Centred, clamped on screen, every time.
        w, h = self.px(1120), self.px(700)
        sw, sh = self.root.winfo_screenwidth(), self.root.winfo_screenheight()
        x = max(0, (sw - w) // 2)
        y = max(0, (sh - h) // 3)      # slightly above centre reads better
        self.root.geometry(f"{w}x{h}+{x}+{y}")
        self.root.minsize(self.px(940), self.px(600))
        self.root.configure(bg=theme.BG)

        self.style = ttk.Style(self.root)
        theme.apply_ttk_theme(self.style, self.scale)

        self.ui_queue: queue.Queue = queue.Queue()
        self.stop_event = threading.Event()
        self.audit = AuditLog(audit_db)
        self.pipeline_factory = pipeline_factory
        self.worker: threading.Thread | None = None
        self._cached_pipeline = None      # per-instance, NOT a class attribute
        self._pipeline_lock = threading.Lock()
        self._state = "idle"
        self._tray_icon = None
        self._boot_time = time.time()
        self._stream_open = False
        self._stream_chars = 0
        self._stream_started = 0.0
        self._tok_per_s = 0.0
        self._tts_cache = "VOICE OUT  PROBING…"
        self._entry_history: list[str] = []
        self._entry_pos = 0

        # A second, model-free executor for actions the USER takes directly in
        # the file browser. It reuses the same containment and the same audit
        # log as the agent's, so a UI-initiated read is recorded exactly like a
        # model-initiated one -- but it exists before (and without) the LLM, so
        # browsing files never waits on 4.5GB of weights.
        #
        # The app allowlist is owned by the UI, because the APPS tab is where it
        # is curated. The pipeline shares this same object, so ticking a box
        # takes effect on the very next turn: the system prompt is rebuilt each
        # turn and reads the enabled list live.
        self.apps = AppRegistry()
        self.apps.ensure_scanned()
        self._ui_exec = ToolExecutor(ALLOWED_ROOTS, self.audit, confirm_fn=self.confirm,
                                     apps=self.apps)

        self._build_ui()
        self._bind_shortcuts()
        self.root.protocol("WM_DELETE_WINDOW", self.on_close)
        self.root.after(50, self._drain_queue)
        # Come to the front on launch. A new Tk window does not raise itself
        # above the browser or editor the user already has open, so without this
        # a successful start is invisible -- indistinguishable from a failed one.
        self.root.after(80, self._raise_to_front)
        self._probe_tts()
        if tray:
            self._start_tray()

    # ---- UI construction -------------------------------------------------

    def px(self, n: int) -> int:
        """Scale a pixel dimension for the display. Fonts scale themselves."""
        return int(n * self.scale)

    def _build_ui(self):
        # ---------- title strip
        title = tk.Canvas(self.root, height=self.px(42), bg=theme.BG,
                          highlightthickness=0)
        title.pack(fill="x", padx=14, pady=(10, 0))
        title.create_text(self.px(6), self.px(21), anchor="w",
                          text="J . A . R . V . I . S .",
                          fill=theme.CYAN, font=(theme.MONO, 15, "bold"))
        # Placed past the measured width of the wordmark rather than a guessed
        # x, which collided with the final "S." at this font size.
        wordmark_right = title.bbox(title.find_all()[-1])[2]
        title.create_text(wordmark_right + self.px(26), self.px(23), anchor="w",
                          text="LOCAL  ·  OFFLINE  ·  NO CLOUD",
                          fill=theme.TEXT_DIM, font=(theme.MONO, 8))
        self._title_canvas = title
        title.bind("<Configure>", self._redraw_title)

        # Status strip is packed BEFORE the body: pack honours call order, so an
        # expand=True body declared first swallows the leftover space and squeezes
        # a later side="bottom" strip down to nothing.
        status = tk.Canvas(self.root, height=self.px(26), bg=theme.BG,
                           highlightthickness=0)
        status.pack(fill="x", side="bottom")
        self._status_canvas = status
        status.bind("<Configure>", self._redraw_status)

        body = ttk.Frame(self.root)
        body.pack(fill="both", expand=True, padx=14, pady=10)

        # ---------- left column: reactor + state
        left = ttk.Frame(body)
        left.pack(side="left", fill="y", padx=(0, 14))

        self.reactor = Reactor(left, size=self.px(210))
        self.reactor.pack()

        self.state_label = ttk.Label(left, text="STANDBY", style="Head.TLabel",
                                     anchor="center")
        self.state_label.pack(fill="x", pady=(6, 0))
        self.substate_label = ttk.Label(left, text="press  ENGAGE  to listen",
                                        style="Dim.TLabel", anchor="center")
        self.substate_label.pack(fill="x")

        self.listen_btn = ttk.Button(left, text="ENGAGE", command=self.toggle_listening)
        self.listen_btn.pack(fill="x", pady=(14, 0))

        # telemetry block
        self.telemetry = tk.Text(left, height=6, width=30, wrap="none", bg=theme.PANEL,
                                 fg=theme.TEXT_DIM, font=(theme.MONO, 8),
                                 relief="flat", highlightthickness=1,
                                 highlightbackground=theme.PANEL_EDGE, padx=8, pady=6)
        self.telemetry.pack(fill="x", pady=(14, 0))
        self.telemetry.configure(state="disabled")

        # Shortcut legend. The left column had dead space below the telemetry,
        # and keyboard shortcuts are undiscoverable unless they are written down
        # somewhere the user is already looking.
        keys = tk.Text(left, height=5, width=30, wrap="none", bg=theme.BG,
                       fg=theme.TEXT_DIM, font=(theme.MONO, 8), relief="flat",
                       highlightthickness=0, padx=8, pady=6)
        keys.insert("1.0", "\n".join([
            "ENTER       send",
            "UP / DOWN   previous commands",
            "CTRL+ENTER  engage microphone",
            "ESC         stop speaking",
            "CTRL+L      clear transcript",
        ]))
        keys.configure(state="disabled")
        keys.pack(fill="x", pady=(10, 0))

        # ---------- right column: notebook
        right = ttk.Frame(body)
        right.pack(side="left", fill="both", expand=True)

        notebook = ttk.Notebook(right)
        notebook.pack(fill="both", expand=True)
        self._notebook = notebook

        # --- transcript tab
        transcript_tab = ttk.Frame(notebook, padding=(0, 8, 2, 0))
        notebook.add(transcript_tab, text="  TRANSCRIPT  ")

        # Packed FIRST with side="bottom": pack gives each widget its requested
        # space in call order, so an expand=True text area declared first would
        # swallow the frame and push the entry row off the window.
        entry_row = ttk.Frame(transcript_tab, padding=(0, 8, 0, 0))
        entry_row.pack(side="bottom", fill="x")
        ttk.Button(entry_row, text="SEND", style="Go.TButton",
                   command=self.submit_typed).pack(side="right", padx=(8, 0))
        self.entry = ttk.Entry(entry_row, font=(theme.MONO, 10))
        self.entry.pack(side="left", fill="x", expand=True, ipady=3)
        self.entry.bind("<Return>", lambda _e: self.submit_typed())

        chips = ttk.Frame(transcript_tab, padding=(0, 6, 0, 0))
        chips.pack(side="bottom", fill="x")
        for label, prompt in QUICK_ACTIONS:
            ttk.Button(chips, text=label, style="Chip.TButton",
                       command=lambda p=prompt: self.fill_and_send(p)
                       ).pack(side="left", padx=(0, 6))

        text_wrap = tk.Frame(transcript_tab, bg=theme.PANEL_EDGE, padx=1, pady=1)
        text_wrap.pack(fill="both", expand=True)
        self.transcript = tk.Text(text_wrap, wrap="word", state="disabled",
                                  font=(theme.MONO, 10), background=theme.PANEL,
                                  foreground=theme.TEXT, insertbackground=theme.CYAN,
                                  relief="flat", padx=12, pady=10, spacing1=2, spacing3=6)
        scroll = ttk.Scrollbar(text_wrap, command=self.transcript.yview,
                               style="Vertical.TScrollbar")
        self.transcript.configure(yscrollcommand=scroll.set)
        scroll.pack(side="right", fill="y")
        self.transcript.pack(side="left", fill="both", expand=True)

        for role, (_prefix, colour) in ROLE_STYLE.items():
            self.transcript.tag_configure(role, foreground=colour)
            self.transcript.tag_configure(f"{role}_tag", foreground=colour,
                                          font=(theme.MONO, 9, "bold"))
        self.transcript.tag_configure("ts", foreground="#2c4a57", font=(theme.MONO, 8))

        # --- audit tab
        audit_tab = ttk.Frame(notebook, padding=(0, 8, 0, 0))
        notebook.add(audit_tab, text="  AUDIT LOG  ")

        bar = ttk.Frame(audit_tab)
        bar.pack(side="bottom", fill="x", pady=(8, 0))
        ttk.Button(bar, text="REFRESH", command=self.refresh_audit).pack(side="right")
        self.audit_summary = ttk.Label(bar, text="", style="Dim.TLabel")
        self.audit_summary.pack(side="left")

        cols = ("time", "tool", "tier", "confirmed", "outcome", "result")
        self.audit_table = ttk.Treeview(audit_tab, columns=cols, show="headings")
        widths = {"time": 135, "tool": 108, "tier": 88, "confirmed": 78,
                  "outcome": 76, "result": 420}
        for col in cols:
            self.audit_table.heading(col, text=col.upper())
            self.audit_table.column(col, width=self.px(widths[col]), anchor="w")
        self.audit_table.tag_configure("denied", foreground=theme.RED)
        self.audit_table.tag_configure("destructive", foreground=theme.AMBER)
        self.audit_table.tag_configure("read_only", foreground=theme.TEXT_DIM)

        ascroll = ttk.Scrollbar(audit_tab, command=self.audit_table.yview,
                                style="Vertical.TScrollbar")
        self.audit_table.configure(yscrollcommand=ascroll.set)
        ascroll.pack(side="right", fill="y")
        self.audit_table.pack(side="left", fill="both", expand=True)

        self._build_files_tab(notebook)
        self._build_apps_tab(notebook)

        self.refresh_audit()
        self._refresh_telemetry()
        self._boot_sequence()

    # ---- file browser ----------------------------------------------------

    def _build_files_tab(self, notebook):
        """Everything inside the allowed roots, browsable without the model.

        Opening a file is the single most common thing asked of this assistant,
        so it should not require a round trip through a 7B model to do it by
        hand. Reads go through the same ToolExecutor the agent uses, so they are
        contained and audited identically.
        """
        tab = ttk.Frame(notebook, padding=(0, 8, 0, 0))
        notebook.add(tab, text="  FILES  ")

        bar = ttk.Frame(tab)
        bar.pack(side="bottom", fill="x", pady=(8, 0))
        ttk.Button(bar, text="RESCAN", command=self.refresh_files).pack(side="right")
        ttk.Button(bar, text="OPEN EXTERNALLY", style="Go.TButton",
                   command=self.open_selected_file).pack(side="right", padx=8)
        self.files_hint = ttk.Label(
            bar, text="select a file to preview  ·  double-click to open it here",
            style="Dim.TLabel")
        self.files_hint.pack(side="left")

        split = ttk.Frame(tab)
        split.pack(fill="both", expand=True)

        tree_wrap = tk.Frame(split, bg=theme.PANEL_EDGE, padx=1, pady=1)
        tree_wrap.pack(side="left", fill="y")
        self.files_tree = ttk.Treeview(tree_wrap, columns=("size",), show="tree headings",
                                       height=18, selectmode="browse")
        self.files_tree.heading("#0", text="PATH")
        self.files_tree.heading("size", text="SIZE")
        self.files_tree.column("#0", width=self.px(300), anchor="w")
        self.files_tree.column("size", width=self.px(80), anchor="e")
        self.files_tree.pack(side="left", fill="y")
        self.files_tree.bind("<<TreeviewSelect>>", lambda _e: self.preview_selected_file())
        self.files_tree.bind("<Double-1>", lambda _e: self.preview_selected_file())

        prev_wrap = tk.Frame(split, bg=theme.PANEL_EDGE, padx=1, pady=1)
        prev_wrap.pack(side="left", fill="both", expand=True, padx=(10, 0))
        self.file_preview = tk.Text(prev_wrap, wrap="word", state="disabled",
                                    font=(theme.MONO, 9), background=theme.PANEL,
                                    foreground=theme.TEXT, relief="flat",
                                    padx=12, pady=10)
        pscroll = ttk.Scrollbar(prev_wrap, command=self.file_preview.yview,
                                style="Vertical.TScrollbar")
        self.file_preview.configure(yscrollcommand=pscroll.set)
        pscroll.pack(side="right", fill="y")
        self.file_preview.pack(side="left", fill="both", expand=True)

        self.refresh_files()

    def refresh_files(self):
        for row in self.files_tree.get_children():
            self.files_tree.delete(row)
        for root in ALLOWED_ROOTS:
            node = self.files_tree.insert("", "end", text=str(root), open=True,
                                          values=("",), tags=("dir",))
            self._insert_children(node, Path(root))
        self.files_tree.tag_configure("dir", foreground=theme.CYAN_DIM)
        self.files_tree.tag_configure("file", foreground=theme.TEXT)

    def _insert_children(self, parent, directory: Path, depth: int = 0):
        # Depth-limited: an accidentally wide root should not freeze the UI
        # thread enumerating a disk.
        if depth > 3:
            return
        try:
            entries = sorted(directory.iterdir(),
                             key=lambda p: (p.is_file(), p.name.lower()))
        except OSError:
            return
        for entry in entries[:400]:
            if entry.is_dir():
                node = self.files_tree.insert(parent, "end", text=entry.name + "\\",
                                              open=depth < 1, values=("",), tags=("dir",))
                self._insert_children(node, entry, depth + 1)
            else:
                try:
                    size = entry.stat().st_size
                except OSError:
                    size = 0
                self.files_tree.insert(parent, "end", text=entry.name, tags=("file",),
                                       values=(_human_size(size),))

    def _selected_path(self) -> Path | None:
        sel = self.files_tree.selection()
        if not sel:
            return None
        parts, node = [], sel[0]
        while node:
            parts.append(self.files_tree.item(node, "text").rstrip("\\"))
            node = self.files_tree.parent(node)
        return Path(*reversed(parts))

    def preview_selected_file(self):
        path = self._selected_path()
        if path is None or path.is_dir():
            return
        result = self._ui_exec.execute("read_file", {"path": str(path)})
        self.file_preview.configure(state="normal")
        self.file_preview.delete("1.0", "end")
        self.file_preview.insert("1.0", result.output if result.ok
                                 else f"[could not read]\n\n{result.output}")
        self.file_preview.configure(state="disabled")
        self.files_hint.configure(text=str(path))

    def open_selected_file(self):
        """Hand the file to Windows. Confirmed like any other launch."""
        path = self._selected_path()
        if path is None:
            self.log("system", "Select a file first.")
            return
        threading.Thread(
            target=lambda: self._log_tool_result(
                "open_file", self._ui_exec.execute("open_file", {"path": str(path)})),
            daemon=True).start()

    def _log_tool_result(self, name, result):
        self.log("tool" if result.ok else "denied", f"{name}: {result.output[:200]}")
        self.post("audit")

    # ---- app allowlist ----------------------------------------------------

    def _build_apps_tab(self, notebook):
        """Which of the user's applications Jarvis may launch.

        The allowlist is the whole security story for launch_app, so it stays an
        allowlist -- but it is now the USER'S list rather than seven hardcoded
        Windows accessories. Discovery proposes; only ticking here authorises.
        """
        tab = ttk.Frame(notebook, padding=(0, 8, 2, 0))
        notebook.add(tab, text="  APPS  ")

        bar = ttk.Frame(tab)
        bar.pack(side="bottom", fill="x", pady=(8, 0))
        ttk.Button(bar, text="RESCAN", command=self.rescan_apps).pack(side="right")
        self.apps_summary = ttk.Label(bar, text="", style="Dim.TLabel")
        self.apps_summary.pack(side="left")

        top = ttk.Frame(tab)
        top.pack(side="top", fill="x", pady=(0, 8))
        ttk.Label(top, text="ENABLE THE APPS JARVIS MAY OPEN",
                  style="Head.TLabel").pack(side="left")
        self.apps_filter = ttk.Entry(top, font=(theme.MONO, 9), width=24)
        self.apps_filter.pack(side="right", ipady=2)
        self.apps_filter.bind("<KeyRelease>", lambda _e: self.refresh_apps())
        ttk.Label(top, text="filter  ", style="Dim.TLabel").pack(side="right")

        wrap = tk.Frame(tab, bg=theme.PANEL_EDGE, padx=1, pady=1)
        wrap.pack(fill="both", expand=True)
        cols = ("on", "name", "path")
        self.apps_tree = ttk.Treeview(wrap, columns=cols, show="headings",
                                      selectmode="browse")
        for col, title, width in (("on", "ON", 40), ("name", "APPLICATION", 250),
                                  ("path", "LOCATION", 460)):
            self.apps_tree.heading(col, text=title)
            self.apps_tree.column(col, width=self.px(width),
                                  anchor="center" if col == "on" else "w")
        self.apps_tree.tag_configure("on", foreground=theme.GREEN)
        self.apps_tree.tag_configure("off", foreground=theme.TEXT_DIM)
        # Risky entries stay visible and remain enable-able, but they are
        # coloured as a warning: enabling PowerShell should be a deliberate act,
        # not something that slips through with everything else.
        self.apps_tree.tag_configure("risky", foreground=theme.AMBER)
        self.apps_tree.bind("<Double-1>", lambda _e: self.toggle_app())
        self.apps_tree.bind("<Return>", lambda _e: self.toggle_app())
        self.apps_tree.bind("<space>", lambda _e: self.toggle_app())

        ascroll = ttk.Scrollbar(wrap, command=self.apps_tree.yview,
                                style="Vertical.TScrollbar")
        self.apps_tree.configure(yscrollcommand=ascroll.set)
        ascroll.pack(side="right", fill="y")
        self.apps_tree.pack(side="left", fill="both", expand=True)

        self.refresh_apps()

    def refresh_apps(self):
        for row in self.apps_tree.get_children():
            self.apps_tree.delete(row)
        needle = (self.apps_filter.get() or "").strip().lower()
        shown = 0
        for entry in sorted(self.apps.catalog.values(), key=lambda a: a.name.lower()):
            if needle and needle not in entry.name.lower():
                continue
            on = entry.name.lower() in self.apps.enabled
            tag = "on" if on else ("risky" if entry.risky else "off")
            note = "  ⚠ shell / system tool" if entry.risky else ""
            self.apps_tree.insert("", "end", tags=(tag,), values=(
                "✓" if on else "·", entry.name,
                (entry.target or entry.launcher or "")[:110] + note))
            shown += 1
        enabled = len(self.apps.allowed())
        self.apps_summary.configure(
            text=f"{enabled} enabled of {len(self.apps.catalog)} installed"
                 f"   ·   showing {shown}   ·   double-click or press SPACE to toggle")

    def toggle_app(self):
        sel = self.apps_tree.selection()
        if not sel:
            return
        name = self.apps_tree.item(sel[0], "values")[1]
        entry = self.apps.catalog.get(name.lower())
        if entry is None:
            return
        now_on = name.lower() not in self.apps.enabled
        self.apps.enable(name, now_on)
        if now_on and entry.risky:
            # Said out loud, once, at the moment it happens.
            self.log("denied", f"{entry.name} is a shell or system tool. Enabling it "
                               f"gives the assistant a way around the no-shell guard.")
        self.log("system", f"{'Enabled' if now_on else 'Disabled'}: {entry.name}")
        self.refresh_apps()

    def rescan_apps(self):
        """Re-read installed apps. Slow enough to belong off the UI thread."""
        self.apps_summary.configure(text="scanning…")

        def work():
            count = self.apps.rescan()
            self.post("apps")
            self.log("system", f"App scan complete: {count} installed.")

        threading.Thread(target=work, daemon=True).start()

    # ---- boot -------------------------------------------------------------

    def _boot_sequence(self):
        """Staggered start-up lines. Cosmetic, but it makes the app feel awake.

        Scheduled through after() rather than slept through, so the window is
        interactive from the first frame.
        """
        steps = [
            (0,    "system", "Cold start — initialising local subsystems."),
            (260,  "system", f"Containment online. {len(ALLOWED_ROOTS)} root(s) authorised."),
            (520,  "system", "Audit log attached. Every tool call is recorded."),
            (780,  "system", "Speech, index and knowledge graph standing by."),
            (1040, "jarvis", "All systems nominal. Type a command, or press ENGAGE to listen."),
        ]
        for delay, who, text in steps:
            self.root.after(delay, lambda w=who, t=text: self._log_now(w, t))

    def _bind_shortcuts(self):
        self.root.bind("<Control-l>", lambda _e: self.clear_transcript())
        self.root.bind("<Control-Return>", lambda _e: self.toggle_listening())
        self.root.bind("<Escape>", lambda _e: self.hush())
        self.entry.bind("<Up>", lambda _e: self._recall_history(-1))
        self.entry.bind("<Down>", lambda _e: self._recall_history(1))

    def clear_transcript(self):
        self.transcript.configure(state="normal")
        self.transcript.delete("1.0", "end")
        self.transcript.configure(state="disabled")
        self._stream_open = False
        self.log("system", "Transcript cleared.")

    def hush(self):
        """Escape: stop talking. Barge-in matters when a reply runs long."""
        if self._cached_pipeline is not None:
            self._cached_pipeline.speech.stop_speaking()
            self.log("system", "Muted.")

    def _recall_history(self, direction):
        """Up/Down through previously typed commands, like a shell."""
        if not self._entry_history:
            return
        self._entry_pos = max(0, min(len(self._entry_history),
                                     self._entry_pos + direction))
        self.entry.delete(0, "end")
        if self._entry_pos < len(self._entry_history):
            self.entry.insert(0, self._entry_history[self._entry_pos])

    def _redraw_title(self, _event=None):
        c = self._title_canvas
        c.delete("deco")
        w = c.winfo_width()
        y = self.px(38)
        c.create_line(0, y, w, y, fill=theme.PANEL_EDGE, tags="deco")
        for i in range(0, self.px(90), self.px(10)):   # angular hash marks, right edge
            c.create_line(w - self.px(120) + i, self.px(30),
                          w - self.px(108) + i, self.px(12),
                          fill=theme.blend(theme.CYAN, theme.BG, 0.72), tags="deco")

    def _redraw_status(self, _event=None):
        c = self._status_canvas
        c.delete("all")
        w = c.winfo_width()
        c.create_line(0, 0, w, 0, fill=theme.PANEL_EDGE)
        roots = ", ".join(str(r) for r in ALLOWED_ROOTS) or "NONE — all access denied"
        c.create_text(self.px(14), self.px(14), anchor="w", text=f"ROOTS  {roots}",
                      fill=theme.TEXT_DIM, font=(theme.MONO, 8))
        c.create_text(w - self.px(14), self.px(14), anchor="e", text=self._tts_status(),
                      fill=theme.TEXT_DIM, font=(theme.MONO, 8))

    def _tts_status(self):
        return self._tts_cache

    def _probe_tts(self):
        """Identify the voice backend off the UI thread.

        The SAPI probe spawns PowerShell and takes about a second. Doing that
        inline while building the window is a visible freeze before the first
        frame is ever drawn, for a caption.
        """
        def work():
            try:
                from speech import SpeechIO
                self._tts_cache = "VOICE OUT  " + SpeechIO().tts_name()
            except Exception:  # noqa: BLE001
                self._tts_cache = "VOICE OUT  UNKNOWN"
            self.post("status")

        threading.Thread(target=work, daemon=True).start()

    def _refresh_telemetry(self):
        """Live vitals, redrawn once a second on the UI thread.

        Everything here is cheap or cached: the audit count is a SELECT over a
        few hundred rows, memory is one ctypes call, and gpu_status() caches
        nvidia-smi for several seconds, so the 1Hz tick never spawns a process
        per frame.
        """
        rows = self.audit.recent(500)
        denied = sum(1 for r in rows if r["outcome"] == "denied")
        load, free_gb, _total_gb = memory_status()
        up = int(time.time() - self._boot_time)

        gpu = gpu_probe()
        vram = f"{gpu['used']}/{gpu['total']} MiB {gpu['util']:>3d}%" if gpu else "n/a"

        rate = f"{self._tok_per_s:.1f} tok/s" if self._tok_per_s else "idle"
        model_state = "ONLINE" if self._cached_pipeline is not None else "STANDBY"

        lines = [
            f"MODEL      Qwen2.5-7B {model_state}",
            f"GENERATE   {rate}",
            f"VRAM       {vram}",
            f"RAM        {load}% · {free_gb:.1f} GB free",
            f"UPTIME     {up // 3600:02d}:{up % 3600 // 60:02d}:{up % 60:02d}",
            f"TOOLS      {len(rows)} calls · {denied} denied",
        ]
        self.telemetry.configure(state="normal")
        self.telemetry.delete("1.0", "end")
        self.telemetry.insert("1.0", "\n".join(lines))
        self.telemetry.configure(state="disabled")
        self.root.after(1000, self._refresh_telemetry)

    # ---- thread-safe posting (callable FROM the worker) ------------------

    def post(self, kind, *payload):
        self.ui_queue.put((kind, payload))

    def log(self, who, text):
        self.post("log", who, text)

    def set_state(self, state):
        self.post("state", state)

    def set_level(self, level):
        self.post("level", level)

    def stream_token(self, piece):
        """Called from the worker for every token of a reply."""
        self.post("stream", piece)

    def end_stream(self):
        self.post("stream_end")

    def confirm(self, tool, args, preview) -> bool:
        """Called on the WORKER thread. Blocks until the user decides."""
        req = ConfirmRequest(tool, args, preview)
        self.post("confirm", req)
        req.event.wait()          # released by the UI thread
        return req.approved

    # ---- main-thread queue drain ----------------------------------------

    def _drain_queue(self):
        try:
            while True:
                kind, payload = self.ui_queue.get_nowait()
                if kind == "log":
                    self._log_now(*payload)
                elif kind == "state":
                    self._set_state_now(*payload)
                elif kind == "level":
                    self.reactor.set_level(payload[0])
                elif kind == "confirm":
                    self._show_confirm(payload[0])
                elif kind == "stream":
                    self._stream_now(payload[0])
                elif kind == "stream_end":
                    self._stream_end_now()
                elif kind == "status":
                    self._redraw_status()
                elif kind == "apps":
                    self.refresh_apps()
                elif kind == "engaged":
                    self._set_engaged(payload[0])
                elif kind == "audit":
                    self.refresh_audit()
        except queue.Empty:
            pass
        self.root.after(50, self._drain_queue)

    # ---- streamed reply --------------------------------------------------

    def _stream_now(self, piece):
        """Append reply text as it is generated, under one JARVIS heading."""
        self.transcript.configure(state="normal")
        if not self._stream_open:
            self._begin_line("jarvis")
            self._stream_open = True
            self._stream_chars = 0
            self._stream_started = time.time()
        self.transcript.insert("end", piece, "jarvis")
        self._stream_chars += len(piece)
        self.transcript.see("end")
        self.transcript.configure(state="disabled")

    def _stream_end_now(self):
        if not self._stream_open:
            return
        self.transcript.configure(state="normal")
        self.transcript.insert("end", "\n", "jarvis")
        self.transcript.see("end")
        self.transcript.configure(state="disabled")
        elapsed = time.time() - self._stream_started
        # ~4 chars/token is close enough for a HUD readout and costs no
        # tokeniser call on the UI thread. Replies shorter than a third of a
        # second are not measured at all: dividing a handful of characters by a
        # near-zero interval produced readings like "26841 tok/s".
        if elapsed > 0.3:
            self._tok_per_s = (self._stream_chars / 4.0) / elapsed
        self._stream_open = False

    def _begin_line(self, who):
        """Timestamp + role tag. Caller must already have set state='normal'."""
        prefix, _colour = ROLE_STYLE.get(who, (who.upper(), theme.TEXT))
        self.transcript.insert("end", f"{datetime.now():%H:%M:%S}  ", "ts")
        self.transcript.insert("end", f"{prefix:<7}", f"{who}_tag")

    def _log_now(self, who, text):
        self.transcript.configure(state="normal")
        # A timer firing mid-reply must not splice itself into the streamed
        # line: close that line first so both stay readable.
        if self._stream_open:
            self.transcript.insert("end", "\n", "jarvis")
            self._stream_open = False
        self._begin_line(who)
        self.transcript.insert("end", f"{text}\n", who)
        self.transcript.see("end")
        self.transcript.configure(state="disabled")

    def _set_state_now(self, state):
        self._state = state
        colour, label, _speed = theme.STATES.get(state, theme.STATES["idle"])
        self.reactor.set_state(state)
        self.state_label.configure(text=label, foreground=colour)

    # ---- confirmation modal ---------------------------------------------

    def _show_confirm(self, req: ConfirmRequest):
        win = tk.Toplevel(self.root)
        win.title("CONFIRM DESTRUCTIVE ACTION")
        win.transient(self.root)
        win.configure(bg=theme.BG)
        win.grab_set()
        win.resizable(False, False)

        width = self.px(620)
        head = tk.Canvas(win, height=self.px(52), bg=theme.BG,
                         highlightthickness=0, width=width)
        head.pack(fill="x")
        head.create_text(self.px(20), self.px(20), anchor="w",
                         text="⚠  DESTRUCTIVE ACTION REQUESTED",
                         fill=theme.RED, font=(theme.MONO, 12, "bold"))
        head.create_text(self.px(20), self.px(38), anchor="w",
                         text="This cannot be undone. Review the exact path before approving.",
                         fill=theme.TEXT_DIM, font=(theme.MONO, 8))
        head.create_line(0, self.px(51), width, self.px(51), fill="#5c1a17")

        frame = ttk.Frame(win, padding=18)
        frame.pack(fill="both", expand=True)

        box = tk.Text(frame, height=6, width=68, wrap="word", bg=theme.PANEL,
                      fg=theme.TEXT, font=(theme.MONO, 10), relief="flat",
                      padx=12, pady=10, highlightthickness=1,
                      highlightbackground="#5c1a17")
        box.insert("1.0", f"TOOL   {req.tool}\n\n{req.preview}")
        box.tag_add("tool", "1.0", "1.end")
        box.tag_configure("tool", foreground=theme.AMBER, font=(theme.MONO, 10, "bold"))
        box.configure(state="disabled")
        box.pack(fill="both", expand=True)

        def answer(approved):
            req.approved = approved
            req.event.set()               # unblocks the worker thread
            win.grab_release()
            win.destroy()
            self.log("system" if approved else "denied",
                     f"{'APPROVED' if approved else 'DENIED'}: {req.tool}")
            self.post("audit")

        buttons = ttk.Frame(frame)
        buttons.pack(fill="x", pady=(14, 0))
        ttk.Label(buttons, text="ESC to deny", style="Dim.TLabel").pack(side="left")
        ttk.Button(buttons, text="Deny", style="Danger.TButton",
                   command=lambda: answer(False)).pack(side="right")
        ttk.Button(buttons, text="Approve", command=lambda: answer(True)).pack(
            side="right", padx=8)

        # Closing the window is a DENIAL, never an approval, and must still
        # release the worker or the pipeline hangs forever.
        win.protocol("WM_DELETE_WINDOW", lambda: answer(False))
        win.bind("<Escape>", lambda _e: answer(False))
        # A modal that never takes keyboard focus silently ignores Escape, and
        # the user has to hunt for the mouse to refuse a destructive action.
        win.focus_force()
        win.update_idletasks()
        x = self.root.winfo_rootx() + (self.root.winfo_width() - win.winfo_width()) // 2
        y = self.root.winfo_rooty() + 110
        win.geometry(f"+{max(x, 0)}+{max(y, 0)}")

    # ---- audit viewer ----------------------------------------------------

    def refresh_audit(self):
        for row in self.audit_table.get_children():
            self.audit_table.delete(row)
        rows = self.audit.recent(300)
        for r in rows:
            tag = "denied" if r["outcome"] == "denied" else r["tier"]
            self.audit_table.insert("", "end", tags=(tag,), values=(
                r["ts_utc"].replace("T", " ").replace("+00:00", ""),
                r["tool"], r["tier"], "YES" if r["confirmed"] else "auto",
                r["outcome"], (r["result"] or "")[:200].replace("\n", " "),
            ))
        if hasattr(self, "audit_summary"):
            denied = sum(1 for r in rows if r["outcome"] == "denied")
            self.audit_summary.configure(
                text=f"{len(rows)} calls logged   ·   {denied} denied")

    # ---- pipeline control ------------------------------------------------

    def _set_engaged(self, engaged: bool):
        """Keep the button honest when the loop stops on its own (or errors)."""
        self.listen_btn.configure(text="DISENGAGE" if engaged else "ENGAGE")
        self.substate_label.configure(
            text="say “hey jarvis”" if engaged else "press  ENGAGE  to listen")

    def toggle_listening(self):
        if self.worker and self.worker.is_alive():
            self.stop_event.set()
            self.hush()                    # stop mid-sentence, not after it
            self.listen_btn.configure(text="ENGAGE")
            self.substate_label.configure(text="stopping…")
            self.log("system", "Disengaging.")
            return
        if self.pipeline_factory is None:
            self.log("system", "No pipeline attached (UI-only mode).")
            return
        self.stop_event.clear()
        self.worker = threading.Thread(target=self._run_pipeline, daemon=True)
        self.worker.start()
        self._set_engaged(True)

    def fill_and_send(self, prompt):
        self.entry.delete(0, "end")
        self.entry.insert(0, prompt)
        self.submit_typed()

    def submit_typed(self):
        text = self.entry.get().strip()
        if not text:
            return
        self.entry.delete(0, "end")
        self._entry_history.append(text)
        self._entry_pos = len(self._entry_history)
        self.log("user", text)
        if self.pipeline_factory is None:
            self.log("system", "No pipeline attached (UI-only mode).")
            return
        # One turn at a time. llama.cpp has a single context, so a second
        # question sent mid-answer would corrupt the first rather than queue --
        # say so instead of silently dropping it.
        pipe = self._cached_pipeline
        if pipe is not None and pipe.busy:
            self.log("system", "Still working on the previous request — one moment.")
            return
        threading.Thread(target=self._handle_text, args=(text,), daemon=True).start()

    def _handle_text(self, text):
        try:
            pipeline = self._pipeline()
            self.set_state("thinking")
            reply = pipeline.ask(text, on_token=self.stream_token)
            self.end_stream()
            pipeline.speak(reply)
        except Exception as exc:  # noqa: BLE001
            self.end_stream()
            self.log("denied", f"{type(exc).__name__}: {exc}")
            self.set_state("error")
        else:
            # Do NOT hard-code "idle": a typed question asked while the voice
            # loop is running would otherwise leave the ring reading STANDBY
            # while the microphone is in fact still live.
            self.set_state("listening" if self._voice_active() else "idle")
        self.post("audit")

    def _voice_active(self) -> bool:
        return bool(self.worker and self.worker.is_alive() and not self.stop_event.is_set())

    def _pipeline(self):
        """Build the pipeline once, however many threads ask for it at once.

        Without this lock, typing a message and then pressing ENGAGE while the
        models were still loading built TWO pipelines -- two llama.cpp contexts
        of ~4.5GB each on a 6GB card. The second load dies with a CUDA OOM that
        looks like a model problem and is not.
        """
        with self._pipeline_lock:
            if self._cached_pipeline is None:
                self.log("system", "Loading models — first run takes a few seconds…")
                self._cached_pipeline = self.pipeline_factory(self)
                self.log("system", "Models online.")
                self.post("status")
            return self._cached_pipeline

    def _run_pipeline(self):
        try:
            pipeline = self._pipeline()
            pipeline.run_voice_loop(self.stop_event)
        except Exception as exc:  # noqa: BLE001
            self.log("denied", f"pipeline stopped: {type(exc).__name__}: {exc}")
            self.set_state("error")
        finally:
            self.set_state("idle")
            self.post("engaged", False)

    # ---- system tray ------------------------------------------------------

    def _start_tray(self):
        """Tray icon in its own thread. Never touches tkinter directly."""
        try:
            import pystray
            from PIL import Image, ImageDraw
        except Exception:  # noqa: BLE001 - tray is a convenience, not a requirement
            return

        img = Image.new("RGBA", (64, 64), (4, 8, 15, 255))
        d = ImageDraw.Draw(img)
        d.ellipse((6, 6, 58, 58), outline=(0, 229, 255, 255), width=3)
        d.arc((14, 14, 50, 50), start=210, end=110, fill=(0, 229, 255, 255), width=3)
        d.ellipse((26, 26, 38, 38), fill=(234, 255, 255, 255))

        menu = pystray.Menu(
            pystray.MenuItem("Show", lambda: self.root.after(0, self._show_window),
                             default=True),
            pystray.MenuItem("Toggle listening",
                             lambda: self.root.after(0, self.toggle_listening)),
            pystray.MenuItem("Quit", lambda: self.root.after(0, self.on_close)),
        )
        self._tray_icon = pystray.Icon("jarvis", img, "J.A.R.V.I.S.", menu)
        threading.Thread(target=self._tray_icon.run, daemon=True).start()

    def _show_window(self):
        self.root.deiconify()
        self.root.lift()
        self.root.focus_force()

    def _raise_to_front(self):
        """Force the window above other apps, once, at startup.

        lift() alone loses to Windows' foreground-lock. Setting -topmost and
        immediately clearing it is the standard way through: it wins the raise
        without leaving the window permanently pinned over everything else.
        """
        try:
            self.root.deiconify()
            self.root.lift()
            self.root.attributes("-topmost", True)
            self.root.after(200, lambda: self.root.attributes("-topmost", False))
            self.root.focus_force()
        except tk.TclError:
            pass                          # window already gone; nothing to raise

    # ---- lifecycle -------------------------------------------------------

    def on_close(self):
        self.stop_event.set()
        if self._tray_icon is not None:
            try:
                self._tray_icon.stop()
            except Exception:  # noqa: BLE001
                pass
        if self._cached_pipeline is not None:
            try:
                self._cached_pipeline.shutdown()
            except Exception:  # noqa: BLE001
                pass
        self.reactor.stop()
        self.audit.close()
        self.root.destroy()

    def run(self):
        self.root.mainloop()
