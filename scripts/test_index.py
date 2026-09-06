"""Stage 3 verification: index the sandbox, then run semantic searches.

Run:
    venv\\Scripts\\python.exe scripts\\test_index.py
    venv\\Scripts\\python.exe scripts\\test_index.py --force
    venv\\Scripts\\python.exe scripts\\test_index.py --live

Runs against the SANDBOX ONLY by default, in a throwaway Chroma directory.
Once the roots grew to include real documents, pointing this at the live index
stopped being a regression test: the expected answers depended on whichever
files the machine happened to hold, so retrieval quality appeared to drift with
the user's Desktop rather than with the code. --live inspects the real index.
"""
import argparse
import sys
import tempfile
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from config import INDEX_ROOTS  # noqa: E402
from index_store import FileIndex  # noqa: E402

SANDBOX = Path(r"D:\jarvis-sandbox")

# Queries phrased so the answer is NOT a literal keyword match -- that is the
# whole point of embedding search over grep.
QUERIES = [
    ("why did the warehouse project slip",        "customs"),
    ("which carrier contract needs renewing",     "contoso"),
    ("what is over budget",                       "helsinki"),
    ("proposed price per pallet",                 "36.90"),
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--force", action="store_true", help="reindex even if unchanged")
    ap.add_argument("--live", action="store_true",
                    help="use the real index over every INDEX_ROOTS folder")
    args = ap.parse_args()

    if args.live:
        roots, chroma = INDEX_ROOTS, None
    else:
        roots = [SANDBOX]
        chroma = Path(tempfile.mkdtemp(prefix="jarvis_idx_"))
    print(f"Roots: {[str(r) for r in roots]}")
    idx = FileIndex(roots) if chroma is None else FileIndex(roots, chroma_dir=chroma)

    t = time.time()
    stats = idx.reindex_all(force=args.force)
    print(f"\nIndex run in {time.time()-t:.1f}s: {stats}")
    print(f"Total chunks in collection: {idx.count()}")

    print("\n--- semantic search ---")
    failures = 0
    for query, expect in QUERIES:
        hits = idx.search(query, n_results=3)
        if not hits:
            print(f"  [BAD] {query!r} -> no hits")
            failures += 1
            continue
        top = hits[0]
        blob = " ".join(h["text"].lower() + " " + h["name"].lower() for h in hits)
        ok = expect.lower() in blob
        failures += not ok
        print(f"  [{'ok ' if ok else 'BAD'}] {query!r}")
        print(f"        top: {top['name']} (dist {top['distance']:.3f}) expect {expect!r} in top-3")
        if not ok:
            print(f"        got: {top['text'][:120]!r}")

    print(f"\n{'FAIL' if failures else 'PASS'}: {failures} failing quer(y/ies)")
    sys.exit(1 if failures else 0)


if __name__ == "__main__":
    main()
