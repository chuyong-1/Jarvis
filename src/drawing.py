"""Turn a list of shape primitives into a PNG.

The assistant could launch Paint but had no way to put anything IN it, so asked
to "draw a rocket" it either handed the job back ("please draw your rocket using
the tools available") or, worse, claimed to have drawn one and opened a .txt
that was never written. Both are failures of capability dressed up as answers.

Primitives rather than pixels, because a 7B cannot emit an image but can emit
"triangle here, rectangle below it, two fins, a flame" -- and that composes into
a recognisable drawing. The output is a normal PNG in an allowed root, so Paint
opens it as an ordinary file and the user can keep drawing on top.
"""
from PIL import Image, ImageColor, ImageDraw

# A drawing is a document, not a render farm: these bounds keep a malformed or
# runaway shape list from allocating gigabytes or hanging the turn.
MAX_DIMENSION = 4096
MAX_SHAPES = 250
DEFAULT_SIZE = (900, 700)

KINDS = ("rectangle", "ellipse", "polygon", "line", "text")


class DrawError(ValueError):
    """A shape list that cannot be drawn. The message goes back to the model,
    so it says what was wrong and what was expected."""


def _colour(value, default=None):
    if value is None or value == "":
        return default
    try:
        ImageColor.getrgb(str(value))
    except ValueError:
        raise DrawError(f"unknown colour {value!r} — use a name like 'red' or a "
                        f"hex value like '#ff8800'") from None
    return str(value)


def _points(shape, key="points", minimum=2):
    raw = shape.get(key)
    if not isinstance(raw, (list, tuple)) or len(raw) < minimum:
        raise DrawError(f"{shape.get('kind')} needs {key!r}: a list of at least "
                        f"{minimum} [x, y] pairs")
    out = []
    for pt in raw:
        if not isinstance(pt, (list, tuple)) or len(pt) != 2:
            raise DrawError(f"each point must be [x, y]; got {pt!r}")
        try:
            out.append((float(pt[0]), float(pt[1])))
        except (TypeError, ValueError):
            raise DrawError(f"point coordinates must be numbers; got {pt!r}") from None
    return out


def _box(shape):
    raw = shape.get("box")
    if not isinstance(raw, (list, tuple)) or len(raw) != 4:
        raise DrawError(f"{shape.get('kind')} needs 'box': [x0, y0, x1, y1]")
    try:
        x0, y0, x1, y1 = (float(v) for v in raw)
    except (TypeError, ValueError):
        raise DrawError(f"box values must be numbers; got {raw!r}") from None
    # PIL raises on an inverted box, which the model produces often enough that
    # normalising is kinder than failing the whole drawing over corner order.
    return (min(x0, x1), min(y0, y1), max(x0, x1), max(y0, y1))


def render(shapes, width=None, height=None, background="white"):
    """Draw `shapes` onto a new image and return it.

    Raises DrawError with a message aimed at the model, never a bare PIL error.
    """
    width = int(width or DEFAULT_SIZE[0])
    height = int(height or DEFAULT_SIZE[1])
    if not (1 <= width <= MAX_DIMENSION and 1 <= height <= MAX_DIMENSION):
        raise DrawError(f"canvas must be between 1x1 and "
                        f"{MAX_DIMENSION}x{MAX_DIMENSION}; got {width}x{height}")
    if not isinstance(shapes, (list, tuple)) or not shapes:
        raise DrawError("'shapes' must be a non-empty list of shape objects")
    if len(shapes) > MAX_SHAPES:
        raise DrawError(f"too many shapes ({len(shapes)}); the limit is {MAX_SHAPES}")

    img = Image.new("RGB", (width, height), _colour(background, "white"))
    d = ImageDraw.Draw(img)

    for i, shape in enumerate(shapes):
        if not isinstance(shape, dict):
            raise DrawError(f"shape {i} is not an object: {shape!r}")
        kind = str(shape.get("kind", "")).lower().strip()
        if kind not in KINDS:
            raise DrawError(f"shape {i} has unknown kind {kind!r}; "
                            f"use one of {', '.join(KINDS)}")
        fill = _colour(shape.get("fill"))
        outline = _colour(shape.get("outline"))
        # A shape with neither fill nor outline is INVISIBLE in PIL -- it draws
        # nothing at all, silently. The model omits both often (it wrote a
        # house as an unfilled roof and body and produced a near-blank canvas),
        # so an uncoloured shape gets a black outline instead of vanishing.
        # Drawing something plain beats drawing nothing and reporting success.
        if fill is None and outline is None and kind in ("rectangle", "ellipse", "polygon"):
            outline = "black"
        try:
            wide = max(1, int(shape.get("width", 1) or 1))
        except (TypeError, ValueError):
            wide = 1

        if kind == "rectangle":
            d.rectangle(_box(shape), fill=fill, outline=outline, width=wide)
        elif kind == "ellipse":
            d.ellipse(_box(shape), fill=fill, outline=outline, width=wide)
        elif kind == "polygon":
            d.polygon(_points(shape, minimum=3), fill=fill, outline=outline)
        elif kind == "line":
            d.line(_points(shape, minimum=2), fill=fill or "black", width=wide,
                   joint="curve")
        elif kind == "text":
            raw = shape.get("at")
            if not isinstance(raw, (list, tuple)) or len(raw) != 2:
                raise DrawError("text needs 'at': [x, y]")
            try:
                xy = (float(raw[0]), float(raw[1]))
            except (TypeError, ValueError):
                raise DrawError(f"text 'at' must be numbers; got {raw!r}") from None
            body = str(shape.get("text", ""))
            if not body:
                raise DrawError("text shape needs a non-empty 'text'")
            d.text(xy, body, fill=fill or "black")

    return img
