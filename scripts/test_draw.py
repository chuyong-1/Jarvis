"""draw_image: rendering, validation, containment, and the open_in allowlist.

Before this tool existed, "open paint and draw a rocket" launched Paint and told
the user to draw it themselves -- and "draw me a house and open it in paint"
claimed a house had been created while only launching Paint on a .txt that was
never written. A drawing tool has to actually produce the file it reports.
"""
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from audit import AuditLog  # noqa: E402
from drawing import DrawError, render  # noqa: E402
from tools import ToolExecutor  # noqa: E402

fails = 0


def check(label, ok):
    global fails
    if not ok:
        fails += 1
    print(f"  [{'ok ' if ok else 'FAIL'}] {label}")


ROCKET = [
    {"kind": "polygon", "points": [[450, 80], [390, 240], [510, 240]], "fill": "red"},
    {"kind": "rectangle", "box": [390, 240, 510, 520], "fill": "white",
     "outline": "black", "width": 3},
    {"kind": "ellipse", "box": [425, 300, 475, 350], "fill": "skyblue"},
    {"kind": "polygon", "points": [[420, 520], [450, 650], [480, 520]], "fill": "orange"},
    {"kind": "text", "at": [360, 60], "text": "ROCKET", "fill": "black"},
]

tmp = Path(tempfile.mkdtemp(prefix="jarvis_draw_"))
audit = AuditLog(tmp / "audit.db")
ex = ToolExecutor([tmp], audit, confirm_fn=lambda *a, **k: True)

print("rendering")
img = render(ROCKET, 900, 700, "white")
check("canvas is the requested size", (img.width, img.height) == (900, 700))
check("something was actually drawn (not a blank canvas)",
      len(img.getcolors(maxcolors=1 << 20) or []) > 1)

print("\nvalidation — errors the model can act on")
for label, shapes in [
    ("unknown kind", [{"kind": "sunburst", "box": [0, 0, 5, 5]}]),
    ("polygon with too few points", [{"kind": "polygon", "points": [[1, 1], [2, 2]]}]),
    ("rectangle with no box", [{"kind": "rectangle", "fill": "red"}]),
    ("bad colour", [{"kind": "rectangle", "box": [0, 0, 5, 5], "fill": "octarine"}]),
    ("non-numeric point", [{"kind": "polygon", "points": [[1, 1], [2, "x"], [3, 3]]}]),
    ("empty shape list", []),
]:
    try:
        render(shapes, 100, 100)
        check(f"{label} is rejected", False)
    except DrawError as exc:
        check(f"{label} -> {str(exc)[:52]}", True)
    except Exception as exc:  # noqa: BLE001
        check(f"{label} raised {type(exc).__name__}, not DrawError", False)

print("\nno shape may be silently invisible")
for kind, geom in [("rectangle", {"box": [10, 10, 90, 90]}),
                   ("ellipse", {"box": [10, 10, 90, 90]}),
                   ("polygon", {"points": [[10, 10], [90, 10], [50, 90]]})]:
    blank = render([{"kind": kind, **geom}], 100, 100, "white")
    drawn = len(blank.getcolors(maxcolors=1 << 20) or [])
    check(f"{kind} with no fill and no outline still draws", drawn > 1)

check("inverted box is normalised, not an error",
      render([{"kind": "rectangle", "box": [80, 80, 10, 10], "fill": "red"}], 100, 100)
      is not None)
try:
    render(ROCKET, 99999, 99999)
    check("absurd canvas is rejected", False)
except DrawError:
    check("absurd canvas is rejected", True)

print("\nthe tool end to end")
out = tmp / "rocket.png"
r = ex.execute("draw_image", {"path": str(out), "shapes": ROCKET})
check("call succeeded", r.ok)
check("the PNG really exists", out.is_file())
check("the PNG is not empty", out.is_file() and out.stat().st_size > 1000)
check("result names the file", out.name in r.output)

r = ex.execute("draw_image", {"path": str(tmp / "bad.png"),
                              "shapes": [{"kind": "nope"}]})
check("a bad shape list fails the call", not r.ok)
check("and writes no file", not (tmp / "bad.png").exists())

r = ex.execute("draw_image", {"path": str(tmp / "notes.txt"), "shapes": ROCKET})
check("refuses a non-image extension", not r.ok)

print("\ncontainment and the app allowlist still apply")
r = ex.execute("draw_image", {"path": r"C:\Windows\Temp\escape.png", "shapes": ROCKET})
check("cannot draw outside the allowed roots", not r.ok)
check("denial says so", "denied" in r.output.lower())

r = ex.execute("draw_image", {"path": str(tmp / "x.png"), "shapes": ROCKET,
                              "open_in": "Some Unapproved App"})
check("open_in cannot name an app outside the allowlist", not r.ok)
check("and the picture is not written when the app is refused",
      not (tmp / "x.png").exists())

audit.close()
print(f"\n{'PASS' if not fails else 'FAIL'}: {fails} failing")
sys.exit(1 if fails else 0)
