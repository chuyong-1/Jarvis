"""Chroma-backed index over files inside ALLOWED_ROOTS.

Embeddings use Chroma's built-in ONNX all-MiniLM-L6-v2 rather than the
sentence-transformers package: same model and same vectors, but it runs on
onnxruntime (already installed) instead of dragging in ~2.5GB of PyTorch.

Containment is enforced HERE as well as at the tool layer. The walker
validates every path it discovers, so a directory junction planted inside a
root cannot be used to pull outside content into the index. Defence in depth:
the index is a read path into file contents, so it is part of the boundary.
"""
import re
import sys
from pathlib import Path

import chromadb
from chromadb.utils import embedding_functions

sys.path.insert(0, str(Path(__file__).resolve().parent))

from extract import SUPPORTED_SUFFIXES, ExtractionError, chunk_text, extract_text  # noqa: E402
from safety import is_within  # noqa: E402
from walk import SKIP_DIRS, walk_files  # noqa: E402

CHROMA_DIR = Path(__file__).resolve().parent.parent / "chroma"
COLLECTION = "files"

# Directories that are never worth indexing live in walk.SKIP_DIRS, shared with
# the file tools so both agree on what is noise.

# Chroma refuses a single add() above its max batch size. The value is a server
# property (5461 here), so stay comfortably under it rather than probing for it.
ADD_BATCH = 2000
# A cap per file, not per run: one 12,715-chunk document would otherwise crowd
# out every other result and consume the whole embedding budget on its own.
MAX_CHUNKS_PER_FILE = 400


class FileIndex:
    def __init__(self, allowed_roots: list[Path], chroma_dir: Path = CHROMA_DIR):
        self.allowed_roots = allowed_roots
        self.client = chromadb.PersistentClient(path=str(chroma_dir))
        self.collection = self.client.get_or_create_collection(
            name=COLLECTION,
            embedding_function=embedding_functions.ONNXMiniLM_L6_V2(),
            metadata={"hnsw:space": "cosine"},
        )

    # ---- walking -------------------------------------------------------

    def iter_files(self):
        """Yield indexable files under the allowed roots, never leaving them."""
        for root in self.allowed_roots:
            root = Path(root)
            if not root.is_dir():
                continue
            # walk_files PRUNES skipped directories rather than walking them and
            # filtering after: descending into node_modules was the entire cost.
            for path in walk_files([root]):
                if path.suffix.lower() not in SUPPORTED_SUFFIXES:
                    continue
                # A junction inside the root can point anywhere; re-validate.
                if not is_within(path, self.allowed_roots):
                    print(f"  SKIP (escapes root): {path}")
                    continue
                yield path

    # ---- indexing ------------------------------------------------------

    def _stored_signature(self, path: Path) -> str | None:
        got = self.collection.get(where={"source": str(path)}, limit=1, include=["metadatas"])
        metas = got.get("metadatas") or []
        return metas[0].get("signature") if metas else None

    @staticmethod
    def _signature(path: Path) -> str:
        st = path.stat()
        return f"{st.st_mtime_ns}:{st.st_size}"

    def remove_file(self, path: Path) -> None:
        self.collection.delete(where={"source": str(path)})

    def index_file(self, path: Path, force: bool = False) -> tuple[str, int]:
        """Index one file. Returns (status, n_chunks)."""
        if not is_within(path, self.allowed_roots):
            return "denied", 0

        sig = self._signature(path)
        if not force and self._stored_signature(path) == sig:
            return "unchanged", 0

        try:
            text = extract_text(path)
        except ExtractionError as exc:
            # One bad file must never abort a whole index run.
            print(f"  extract failed {path.name}: {exc}")
            return "error", 0

        chunks = chunk_text(text)
        if not chunks:
            return "empty", 0
        # One enormous file must not dominate the index or the embedding budget.
        # Observed as soon as the roots grew past the sandbox: a single document
        # produced 12,715 chunks, more than every other file combined.
        if len(chunks) > MAX_CHUNKS_PER_FILE:
            print(f"  truncating {path.name}: {len(chunks)} chunks -> {MAX_CHUNKS_PER_FILE}")
            chunks = chunks[:MAX_CHUNKS_PER_FILE]

        # Drop old chunks first: an edited file may produce fewer chunks than
        # before, and the leftovers would otherwise linger as phantom results.
        self.remove_file(path)

        # Prefix each chunk with the filename before embedding. Without it,
        # filename-shaped queries ("my meeting notes") match on content only and
        # can rank an unrelated file above the obvious one. The prefix also gives
        # the LLM provenance inline when the chunk is fed back to it.
        #
        # Added in BATCHES: Chroma rejects any single add() larger than its max
        # batch size (5461 here), and that rejection aborted the whole run with
        # an InternalError rather than skipping the one oversized file.
        for start in range(0, len(chunks), ADD_BATCH):
            batch = chunks[start:start + ADD_BATCH]
            self.collection.add(
                ids=[f"{path}::{start + i}" for i in range(len(batch))],
                documents=[f"[{path.name}]\n{c}" for c in batch],
                metadatas=[
                    {"source": str(path), "name": path.name, "suffix": path.suffix.lower(),
                     "chunk": start + i, "signature": sig}
                    for i in range(len(batch))
                ],
            )
        # The status vocabulary is fixed: reindex_all counts by it.
        return "indexed", len(chunks)

    def reindex_all(self, force: bool = False) -> dict:
        stats = {"indexed": 0, "unchanged": 0, "error": 0, "empty": 0, "denied": 0, "chunks": 0}
        seen = set()
        for path in self.iter_files():
            status, n = self.index_file(path, force=force)
            stats[status] += 1
            stats["chunks"] += n
            seen.add(str(path))

        # Purge chunks whose source file no longer exists / is no longer eligible.
        got = self.collection.get(include=["metadatas"])
        stale = {m["source"] for m in (got.get("metadatas") or []) if m["source"] not in seen}
        for src in stale:
            self.collection.delete(where={"source": src})
        stats["purged"] = len(stale)
        return stats

    # ---- retrieval -----------------------------------------------------

    @staticmethod
    def _tokens(text: str) -> set[str]:
        return {t for t in re.split(r"[^a-z0-9]+", text.lower()) if len(t) > 2}

    def _lexical_ranking(self, query: str) -> list[str]:
        """Rank chunk ids by literal term overlap with filename and content.

        Dense embeddings are content-dominated and systematically underweight
        exact lexical matches: querying "meeting notes" ranked ideas.md ABOVE
        meeting_notes.txt, because MiniLM sees two topically similar notes and
        the filename is a small share of the chunk. A lexical pass restores the
        signal that an exact name match is strong evidence.

        This scans the collection, which is fine at personal scale (thousands of
        chunks). Swap in a real BM25 index if it ever grows past that.
        """
        qt = self._tokens(query)
        if not qt:
            return []
        got = self.collection.get(include=["metadatas", "documents"])
        scored = []
        for cid, meta, doc in zip(got["ids"], got["metadatas"], got["documents"]):
            name_tokens = self._tokens(meta["name"])
            body_tokens = self._tokens(doc)
            # Filename matches weigh far more than body matches: naming a file
            # is an intentional act, a word appearing in prose is not.
            score = 3.0 * len(qt & name_tokens) + len(qt & body_tokens) / max(len(qt), 1)
            if score > 0:
                scored.append((score, cid))
        scored.sort(reverse=True)
        return [cid for _, cid in scored]

    def search(self, query: str, n_results: int = 5, k_rrf: int = 60) -> list[dict]:
        """Hybrid dense + lexical retrieval fused with Reciprocal Rank Fusion.

        RRF is used rather than a weighted score blend because cosine distances
        and lexical scores live on incomparable scales; fusing on RANK avoids
        having to normalise them against each other.
        """
        pool = max(n_results * 4, 20)
        dense = self.collection.query(query_texts=[query], n_results=pool)
        dense_ids = dense["ids"][0]
        records = {
            cid: {"text": doc, "source": meta["source"], "name": meta["name"],
                  "chunk": meta["chunk"], "distance": dist}
            for cid, doc, meta, dist in zip(dense_ids, dense["documents"][0],
                                            dense["metadatas"][0], dense["distances"][0])
        }

        scores: dict[str, float] = {}
        for rank, cid in enumerate(dense_ids):
            scores[cid] = scores.get(cid, 0.0) + 1.0 / (k_rrf + rank + 1)
        for rank, cid in enumerate(self._lexical_ranking(query)[:pool]):
            scores[cid] = scores.get(cid, 0.0) + 1.0 / (k_rrf + rank + 1)

        # Lexical hits that dense retrieval missed entirely need fetching.
        missing = [cid for cid in scores if cid not in records]
        if missing:
            got = self.collection.get(ids=missing, include=["metadatas", "documents"])
            for cid, doc, meta in zip(got["ids"], got["documents"], got["metadatas"]):
                records[cid] = {"text": doc, "source": meta["source"], "name": meta["name"],
                                "chunk": meta["chunk"], "distance": None}

        out = []
        for cid in sorted(scores, key=lambda c: -scores[c]):
            rec = records.get(cid)
            # Never surface a hit outside the roots, even if one got indexed
            # under an older, wider configuration.
            if rec is None or not is_within(rec["source"], self.allowed_roots):
                continue
            rec = dict(rec, rrf=scores[cid])
            out.append(rec)
            if len(out) >= n_results:
                break
        return out

    def count(self) -> int:
        return self.collection.count()
