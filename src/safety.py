"""Path containment -- the security boundary every file tool must pass through.

The rule: a path is allowed only if, AFTER fully resolving symlinks/junctions
and normalising case, it is the same as, or lives underneath, one of
ALLOWED_ROOTS. Everything else is rejected.

Windows makes this harder than it looks. Handled here:
  - ".." traversal
  - symlinks AND directory junctions that point outside a root
  - case-insensitivity (C:\\Users vs c:\\users are the same file)
  - sibling-prefix attacks ("...\\docs_evil" must not pass a "...\\docs" root)
  - alternate data streams ("secret.txt:hidden")
  - reserved device names (CON, NUL, COM1, ...)
  - paths that do not exist yet (write_file targets) -- the existing part of
    the chain is still resolved, so a symlinked parent cannot be used to escape

KNOWN LIMITATION (TOCTOU): a path validated here could in principle be replaced
by a symlink before the caller opens it. For a single-user local assistant that
is an acceptable risk; a hostile local process is already game over. Do not
reuse this module in a multi-tenant context without handle-based revalidation.
"""
import os
from pathlib import Path

WINDOWS_RESERVED = {
    "CON", "PRN", "AUX", "NUL",
    *(f"COM{i}" for i in range(1, 10)),
    *(f"LPT{i}" for i in range(1, 10)),
}


class PathNotAllowed(Exception):
    """Raised when a path fails containment. Message is safe to show the user."""


def _norm(p: Path) -> str:
    """Case-folded string form for comparison (Windows paths are case-insensitive)."""
    return os.path.normcase(str(p))


def _reject_windows_traps(candidate: str) -> None:
    # Alternate data streams: "file.txt:hidden" hides content from naive readers.
    # A drive letter colon at index 1 is legitimate; any other colon is not.
    tail = candidate[2:] if len(candidate) > 1 and candidate[1] == ":" else candidate
    if ":" in tail:
        raise PathNotAllowed("alternate data streams are not permitted")

    for part in Path(candidate).parts:
        stem = part.split(".")[0].upper().strip()
        if stem in WINDOWS_RESERVED:
            raise PathNotAllowed(f"reserved device name not permitted: {part!r}")


def resolve_within(candidate: str | Path, allowed_roots: list[Path]) -> Path:
    """Resolve `candidate` and confirm it is inside one of `allowed_roots`.

    Returns the fully-resolved absolute Path on success.
    Raises PathNotAllowed on any violation. Fails closed: an empty root list
    rejects everything.
    """
    if not allowed_roots:
        raise PathNotAllowed(
            "no allowed roots are configured -- refusing all filesystem access"
        )

    if candidate is None or str(candidate).strip() == "":
        raise PathNotAllowed("empty path")

    _reject_windows_traps(str(candidate))

    # resolve() follows symlinks/junctions and normalises "..". With
    # strict=False it still resolves the parts that DO exist, so a symlinked
    # parent directory cannot be used to smuggle a not-yet-created file out.
    try:
        resolved = Path(candidate).resolve()
    except (OSError, ValueError) as exc:
        raise PathNotAllowed(f"unresolvable path: {exc}") from exc

    for root in allowed_roots:
        try:
            resolved_root = Path(root).resolve()
        except (OSError, ValueError):
            continue

        # Compare on normalised PARTS, never on string prefixes: a plain
        # startswith() would let "D:\docs_evil" pass a "D:\docs" root.
        root_parts = Path(_norm(resolved_root)).parts
        cand_parts = Path(_norm(resolved)).parts
        if cand_parts[: len(root_parts)] == root_parts:
            return resolved

    raise PathNotAllowed(
        f"path is outside every allowed root: {resolved}"
    )


def is_within(candidate: str | Path, allowed_roots: list[Path]) -> bool:
    """Boolean form of resolve_within(), for filtering rather than enforcing."""
    try:
        resolve_within(candidate, allowed_roots)
        return True
    except PathNotAllowed:
        return False
