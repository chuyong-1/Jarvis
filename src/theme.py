"""HUD theme: palette, fonts, and the angular/glow drawing helpers.

Kept separate from app.py so the look can be changed without touching the
threading logic, which is the part that must not be casually edited.

tkinter has no alpha channel, so "glow" is faked by drawing the same shape
several times at decreasing widths in colours interpolated toward the
background. That is why _blend() exists.
"""
import sys

BG = "#04080f"          # near-black with a blue cast
PANEL = "#07131c"       # panel fill
PANEL_EDGE = "#0d2a3a"  # panel border
GRID = "#0a1a26"        # faint grid lines

CYAN = "#00e5ff"        # primary HUD accent
CYAN_DIM = "#0a7a92"
AMBER = "#ffb000"       # tool activity / caution
GREEN = "#00ff9c"       # assistant output / ok
RED = "#ff3b30"         # denial / destructive
VIOLET = "#b07cff"      # thinking
TEXT = "#c8e8f4"
TEXT_DIM = "#5b7f8f"

MONO = "Consolas"
UI_FONT = "Segoe UI"

# state -> (accent colour, label, ring speed multiplier)
STATES = {
    "idle":         (CYAN_DIM, "STANDBY",     0.25),
    "listening":    (CYAN,     "LISTENING",   0.6),
    "recording":    (RED,      "RECORDING",   1.6),
    "transcribing": (AMBER,    "TRANSCRIBING", 1.2),
    "thinking":     (VIOLET,   "PROCESSING",  2.0),
    "speaking":     (GREEN,    "SPEAKING",    1.0),
    "error":        (RED,      "FAULT",       0.15),
}


def _hex(rgb):
    return "#%02x%02x%02x" % rgb


def _rgb(colour):
    colour = colour.lstrip("#")
    return tuple(int(colour[i:i + 2], 16) for i in (0, 2, 4))


def blend(colour, background, t):
    """Interpolate `colour` toward `background`. t=0 -> colour, t=1 -> background."""
    c, b = _rgb(colour), _rgb(background)
    return _hex(tuple(round(c[i] + (b[i] - c[i]) * t) for i in range(3)))


def mix(a, b, t):
    """Interpolate between two arbitrary colours. t=0 -> a, t=1 -> b.

    Used to EASE the reactor between state colours instead of snapping. A hard
    cut from cyan to red reads as a glitch; a 200ms fade reads as a transition.
    """
    return blend(a, b, max(0.0, min(1.0, t)))


def setup_dpi():
    """Declare DPI awareness. MUST be called before the Tk root is created.

    Without this, Windows renders the window at 96dpi and bitmap-stretches the
    result to the display's scale factor -- 125% on the machine this targets.
    Stretching is fatal to this particular look: a HUD is drawn almost entirely
    from one-pixel lines, and every one of them comes out soft. Tk then reports
    real pixels and scales point-sized fonts itself, so only explicit pixel
    dimensions need the factor from scale_factor().
    """
    if sys.platform != "win32":
        return
    try:
        import ctypes
        # 2 = per-monitor v2; correct when a window is dragged between a laptop
        # panel and an external display at a different scale.
        ctypes.windll.shcore.SetProcessDpiAwareness(2)
    except Exception:  # noqa: BLE001 - older Windows has no shcore
        try:
            ctypes.windll.user32.SetProcessDPIAware()
        except Exception:  # noqa: BLE001
            pass


def scale_factor(root) -> float:
    """Display scale, 1.0 at 96dpi. 1.25 on a 125% display."""
    try:
        return max(1.0, root.winfo_fpixels("1i") / 96.0)
    except Exception:  # noqa: BLE001
        return 1.0


def draw_brackets(canvas, x0, y0, x1, y1, colour=CYAN_DIM, size=14, width=2, tags=()):
    """Angular corner brackets — the cheapest way to read as 'HUD panel'."""
    for (cx, cy, dx, dy) in ((x0, y0, 1, 1), (x1, y0, -1, 1),
                             (x0, y1, 1, -1), (x1, y1, -1, -1)):
        canvas.create_line(cx, cy, cx + dx * size, cy, fill=colour, width=width, tags=tags)
        canvas.create_line(cx, cy, cx, cy + dy * size, fill=colour, width=width, tags=tags)


def apply_ttk_theme(style, scale: float = 1.0):
    """Force the 'clam' theme so widget colours are actually controllable.

    The native Windows ttk theme ignores most colour options, which is why a
    HUD look is impossible without switching themes first.
    """
    style.theme_use("clam")

    style.configure(".", background=BG, foreground=TEXT,
                    fieldbackground=PANEL, borderwidth=0, focuscolor=CYAN_DIM)
    style.configure("TFrame", background=BG)
    style.configure("Panel.TFrame", background=PANEL)
    style.configure("TLabel", background=BG, foreground=TEXT, font=(MONO, 9))
    style.configure("Dim.TLabel", background=BG, foreground=TEXT_DIM, font=(MONO, 8))
    style.configure("Head.TLabel", background=BG, foreground=CYAN,
                    font=(MONO, 11, "bold"))

    style.configure("TButton", background=PANEL, foreground=CYAN,
                    bordercolor=CYAN_DIM, lightcolor=PANEL, darkcolor=PANEL,
                    borderwidth=1, focusthickness=1, padding=(14, 6),
                    font=(MONO, 9, "bold"))
    style.map("TButton",
              background=[("active", "#0d2b38"), ("pressed", "#123a3f")],
              foreground=[("active", "#9df3ff")],
              bordercolor=[("active", CYAN)])

    # Quick-action chips: same shape, quieter, so they read as suggestions
    # rather than four commands competing with SEND for attention.
    style.configure("Chip.TButton", background=BG, foreground=TEXT_DIM,
                    bordercolor=PANEL_EDGE, lightcolor=BG, darkcolor=BG,
                    padding=(10, 3), font=(MONO, 8))
    style.map("Chip.TButton",
              background=[("active", PANEL)],
              foreground=[("active", CYAN)],
              bordercolor=[("active", CYAN_DIM)])

    style.configure("Danger.TButton", foreground=RED, bordercolor="#5c1a17")
    style.map("Danger.TButton", foreground=[("active", "#ff8a82")],
              bordercolor=[("active", RED)])
    style.configure("Go.TButton", foreground=GREEN, bordercolor="#125c42")
    style.map("Go.TButton", foreground=[("active", "#7dffcd")],
              bordercolor=[("active", GREEN)])

    style.configure("TNotebook", background=BG, borderwidth=0, tabmargins=(0, 4, 0, 0))
    # clam draws tabs with a raised 3-D border by default, which made the
    # UNSELECTED tab look like the active one. Flattening light/dark/border and
    # driving colour purely from state fixes the inversion.
    style.configure("TNotebook.Tab", background=BG, foreground=TEXT_DIM,
                    padding=(20, 8), font=(MONO, 9, "bold"), borderwidth=0,
                    lightcolor=BG, darkcolor=BG, bordercolor=BG)
    style.map("TNotebook.Tab",
              background=[("selected", PANEL), ("active", "#0a1d28")],
              foreground=[("selected", CYAN), ("active", TEXT)],
              lightcolor=[("selected", PANEL)],
              darkcolor=[("selected", PANEL)],
              bordercolor=[("selected", PANEL_EDGE)],
              expand=[("selected", (0, 0, 0, 0))])

    # rowheight is in PIXELS, so unlike the point-sized fonts it does not scale
    # itself: on a 125% display the text outgrows the row and is clipped.
    style.configure("Treeview", background=PANEL, fieldbackground=PANEL,
                    foreground=TEXT, rowheight=int(24 * scale), borderwidth=0,
                    font=(MONO, 9))
    style.configure("Treeview.Heading", background="#0b2130", foreground=CYAN,
                    font=(MONO, 9, "bold"), borderwidth=0, padding=(6, 5))
    style.map("Treeview.Heading", background=[("active", "#123245")])
    style.map("Treeview", background=[("selected", "#0f3444")],
              foreground=[("selected", "#ffffff")])

    # clam draws an Entry as a 3-D sunken box: fieldbackground alone colours the
    # text area but leaves a light bevel around it, which on this palette reads
    # as a white border. light/dark/border must all be set too.
    style.configure("TEntry", fieldbackground=PANEL, foreground=TEXT,
                    insertcolor=CYAN, bordercolor=PANEL_EDGE,
                    lightcolor=PANEL_EDGE, darkcolor=PANEL_EDGE,
                    selectbackground=CYAN_DIM, selectforeground=BG,
                    borderwidth=1, padding=8)
    style.map("TEntry",
              bordercolor=[("focus", CYAN_DIM)],
              lightcolor=[("focus", CYAN_DIM)],
              darkcolor=[("focus", CYAN_DIM)],
              fieldbackground=[("readonly", PANEL), ("disabled", PANEL)])

    # Same story for the scrollbar: without light/dark the thumb keeps clam's
    # default pale bevel and the arrows sit on a white square.
    style.configure("Vertical.TScrollbar", background=PANEL_EDGE, troughcolor=BG,
                    bordercolor=BG, arrowcolor=CYAN_DIM, darkcolor=PANEL,
                    lightcolor=PANEL, borderwidth=0, arrowsize=int(12 * scale),
                    width=int(12 * scale))
    style.map("Vertical.TScrollbar",
              background=[("active", CYAN_DIM), ("pressed", CYAN)],
              arrowcolor=[("active", CYAN)])
