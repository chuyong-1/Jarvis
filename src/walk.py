"""One bounded, pruning filesystem walk, shared by the indexer and the tools.

WHY THIS IS ITS OWN MODULE: the moment ALLOWED_ROOTS grew past a toy sandbox to
include Downloads, every `rglob("*")` in the codebase became a performance bug.
Measured on this machine:

    D:\\jarvis-sandbox            7 entries
    Documents                 1,544
    Desktop                   1,182
    Downloads               229,746   <-- checked-out repositories
    -------------------------------
    total                   232,479   -> find_files took 25.6 SECONDS

Filtering AFTER the walk, which is what the old code did, does not help: the
cost is in descending into node_modules and .git, not in the filtering. os.walk
allows pruning `dirnames` in place so those subtrees are never entered at all.

It also carries a hard cap. Pruning fixes the ordinary case; the cap is what
stops one pathological directory from hanging a turn.
"""
import os
from pathlib import Path

# Never worth walking: build output, dependency trees, VCS metadata, caches.
# None of it is a document the user would ask about, and collectively it is
# ~99% of the entries under a developer's Downloads folder.
SKIP_DIRS = {
    ".git", ".hg", ".svn", "node_modules", "__pycache__", "venv", ".venv",
    "site-packages", "bower_components", "vendor", "Pods",
    "dist", "build", "target", "out", ".next", ".nuxt", ".output",
    ".cache", ".gradle", ".idea", ".vs", ".mypy_cache", ".pytest_cache", ".tox",
    "$RECYCLE.BIN", "System Volume Information", ".chroma", "chroma",
}

# Enough to cover any realistic document collection; small enough that the
# worst case is a pause rather than a hang.
MAX_ENTRIES = 50_000


def walk_files(roots, skip_dirs=SKIP_DIRS, max_entries=MAX_ENTRIES):
    """Yield files under `roots`, never descending into `skip_dirs`.

    Bounded at `max_entries` files in total across all roots.
    """
    seen = 0
    for root in roots:
        root = Path(root)
        if not root.is_dir():
            continue
        for dirpath, dirnames, filenames in os.walk(root, onerror=None):
            # In-place, so os.walk itself skips them -- this is the whole point.
            dirnames[:] = [d for d in dirnames if d not in skip_dirs]
            for name in filenames:
                yield Path(dirpath) / name
                seen += 1
                if seen >= max_entries:
                    return


def walk_dirs(directory: Path, skip_dirs=SKIP_DIRS):
    """Immediate subdirectories of `directory`, minus the noise. Not recursive."""
    try:
        return sorted((p for p in Path(directory).iterdir()
                       if p.is_dir() and p.name not in skip_dirs),
                      key=lambda p: p.name.lower())
    except OSError:
        return []
