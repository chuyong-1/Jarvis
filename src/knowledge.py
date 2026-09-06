"""Stage 5: a local personal knowledge graph.

NetworkX rather than Neo4j Community: Neo4j needs a JVM and a running server,
which contradicts shipping a single standalone app and turns packaging into
bundling a database daemon. A personal KG is thousands of nodes, not millions,
so Cypher and a query planner buy nothing here. If this ever outgrows memory,
the persistence format below is a plain node-link JSON that any graph store
can ingest.

Facts are stored as (subject, predicate, object) triples on a MultiDiGraph, so
the same pair of entities can be linked by several different relations.

Provenance is kept on every edge: what said it, when, and how confident. A
personal assistant that learns things about you WILL learn some of them wrong,
and without provenance there is no way to review or revoke a bad fact.
"""
import json
import re
from datetime import datetime, timezone
from pathlib import Path

import networkx as nx

GRAPH_PATH = Path(__file__).resolve().parent.parent / "knowledge.json"

STOPWORDS = {
    "the", "and", "for", "with", "that", "this", "what", "when", "where", "who",
    "how", "why", "are", "was", "were", "does", "did", "has", "have", "had",
    "you", "your", "yours", "our", "their", "his", "her", "its", "about", "from",
    "into", "over", "under", "than", "then", "them", "they", "she", "him", "which",
    "there", "here", "been", "being", "will", "would", "could", "should", "can",
    "tell", "know", "say", "said", "get", "got", "use", "used", "using",
}


def _norm(text: str) -> str:
    return re.sub(r"\s+", " ", str(text).strip().lower())


def _tokens(text: str) -> set[str]:
    return {t for t in re.split(r"[^a-z0-9]+", str(text).lower())
            if len(t) > 2 and t not in STOPWORDS}


class KnowledgeGraph:
    def __init__(self, path: Path = GRAPH_PATH):
        self.path = Path(path)
        self.g = nx.MultiDiGraph()
        if self.path.exists():
            self.load()

    # ---- persistence ----------------------------------------------------

    def load(self) -> None:
        try:
            data = json.loads(self.path.read_text(encoding="utf-8"))
            self.g = nx.node_link_graph(data, directed=True, multigraph=True, edges="links")
        except Exception:  # noqa: BLE001 - a corrupt graph must not brick startup
            self.g = nx.MultiDiGraph()

    def save(self) -> None:
        data = nx.node_link_data(self.g, edges="links")
        self.path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")

    # ---- writing --------------------------------------------------------

    def add_fact(self, subject: str, predicate: str, obj: str,
                 source: str = "conversation", confidence: float = 0.8) -> bool:
        """Add or update a (subject, predicate, object) triple. Returns True if new."""
        s, p, o = _norm(subject), _norm(predicate), _norm(obj)
        if not (s and p and o):
            return False

        for node, label in ((s, subject), (o, obj)):
            if node not in self.g:
                self.g.add_node(node, label=str(label).strip(), mentions=0)
            self.g.nodes[node]["mentions"] += 1

        existing = self.g.get_edge_data(s, o, key=p)
        if existing is not None:
            # Re-asserting a known fact raises confidence rather than duplicating it.
            existing["confidence"] = min(1.0, existing["confidence"] + 0.05)
            existing["last_seen"] = datetime.now(timezone.utc).isoformat(timespec="seconds")
            return False

        self.g.add_edge(s, o, key=p, predicate=p, source=source, confidence=confidence,
                        created=datetime.now(timezone.utc).isoformat(timespec="seconds"),
                        last_seen=datetime.now(timezone.utc).isoformat(timespec="seconds"))
        return True

    def forget(self, subject: str = None, predicate: str = None, obj: str = None) -> int:
        """Remove matching facts. At least one filter must be given."""
        if not any((subject, predicate, obj)):
            return 0
        s, p, o = _norm(subject or ""), _norm(predicate or ""), _norm(obj or "")
        doomed = [
            (u, v, k) for u, v, k in self.g.edges(keys=True)
            if (not s or u == s) and (not p or k == p) and (not o or v == o)
        ]
        self.g.remove_edges_from(doomed)
        # Drop entities that no longer participate in any fact.
        for node in [n for n in list(self.g.nodes) if self.g.degree(n) == 0]:
            self.g.remove_node(node)
        return len(doomed)

    # ---- reading --------------------------------------------------------

    def _fact_text(self, u: str, v: str, k: str) -> str:
        lu = self.g.nodes[u].get("label", u)
        lv = self.g.nodes[v].get("label", v)
        return f"{lu} {k} {lv}"

    def all_facts(self) -> list[dict]:
        return [
            {"subject": u, "predicate": k, "object": v, "text": self._fact_text(u, v, k),
             **{a: d[a] for a in ("confidence", "source", "created") if a in d}}
            for u, v, k, d in self.g.edges(keys=True, data=True)
        ]

    def query(self, text: str, limit: int = 8, hops: int = 1) -> list[dict]:
        """Return facts relevant to `text`, expanded `hops` edges from matches.

        Matching is token overlap against entity labels and predicates. The
        expansion matters: asking about "Dana" should surface what Dana owns,
        not merely the single edge whose label contains "dana".
        """
        qt = _tokens(text)
        if not qt or self.g.number_of_edges() == 0:
            return []

        seeds = {n for n in self.g.nodes if _tokens(self.g.nodes[n].get("label", n)) & qt}
        # Predicate matches ("who owns what") count as seeds via their endpoints.
        for u, v, k in self.g.edges(keys=True):
            if _tokens(k) & qt:
                seeds.update((u, v))
        if not seeds:
            return []

        frontier = set(seeds)
        for _ in range(hops):
            nxt = set()
            for n in frontier:
                nxt.update(self.g.successors(n))
                nxt.update(self.g.predecessors(n))
            frontier |= nxt

        scored = []
        for u, v, k, d in self.g.edges(keys=True, data=True):
            if u not in frontier and v not in frontier:
                continue
            direct = len(_tokens(self._fact_text(u, v, k)) & qt)
            seed_bonus = 1.0 if (u in seeds or v in seeds) else 0.0
            score = direct + seed_bonus + d.get("confidence", 0.5)
            scored.append((score, {"subject": u, "predicate": k, "object": v,
                                   "text": self._fact_text(u, v, k),
                                   "confidence": d.get("confidence", 0.5),
                                   "source": d.get("source", "?")}))
        scored.sort(key=lambda x: -x[0])
        return [f for _, f in scored[:limit]]

    USER_ALIASES = {"the user", "user", "i", "me", "my"}

    def core_facts(self, limit: int = 6) -> list[dict]:
        """Facts about the user themselves, regardless of the current question.

        Token overlap alone is not enough for preferences: "the user prefers CSV"
        shares NO tokens with "how should you export my spreadsheets?", so a
        purely lexical query returns nothing and the assistant asks a question it
        was explicitly told the answer to. Personal preferences are few and
        almost always relevant, so they are injected unconditionally.
        """
        facts = [
            {"subject": u, "predicate": k, "object": v, "text": self._fact_text(u, v, k),
             "confidence": d.get("confidence", 0.5), "source": d.get("source", "?")}
            for u, v, k, d in self.g.edges(keys=True, data=True)
            if u in self.USER_ALIASES
        ]
        facts.sort(key=lambda f: -f["confidence"])
        return facts[:limit]

    def context_for(self, text: str, limit: int = 8) -> list[dict]:
        """Facts to put in front of the model: query-matched PLUS core user facts."""
        seen, out = set(), []
        for fact in self.core_facts() + self.query(text, limit=limit):
            key = (fact["subject"], fact["predicate"], fact["object"])
            if key not in seen:
                seen.add(key)
                out.append(fact)
        return out[:limit]

    def stats(self) -> dict:
        return {"entities": self.g.number_of_nodes(), "facts": self.g.number_of_edges()}
