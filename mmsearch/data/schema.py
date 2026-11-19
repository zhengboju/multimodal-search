"""Unified retrieval data model (plan §3.2).

A dataset on disk (storage/processed/<name>/):
  corpus.jsonl    {"doc_id", "text"?, "image"? (path relative to dataset root), "meta"}
  queries.jsonl   {"query_id", "text"?, "image"?, "lang", "meta"}   meta may hold answers / domain / source
  qrels.tsv       query_id \t doc_id \t relevance
  info.json       {"name", "split", "domain", "corpus_from"?}
`corpus_from` lets a derived dataset (e.g. translated queries) reuse another dataset's corpus.
"""

from __future__ import annotations

import csv
import random
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Iterator

from PIL import Image

from ..utils.misc import read_json, read_jsonl, write_json, write_jsonl
from ..utils.paths import processed_dir


@dataclass
class RetrievalSample:
    """One (query, doc, relevance) triple — the plan's canonical sample format."""

    query_id: str
    query: str | Image.Image
    doc_id: str
    doc: str | Image.Image
    relevance: int
    lang: str = "en"
    domain: str = ""


def _open_image(path: str) -> Image.Image:
    img = Image.open(path)
    return img.convert("RGB")


@dataclass
class Doc:
    doc_id: str
    text: str | None = None
    image: str | None = None  # absolute path once loaded
    meta: dict = field(default_factory=dict)

    def load_image(self) -> Image.Image:
        if self.image is None:
            raise ValueError(f"doc {self.doc_id} has no image")
        return _open_image(self.image)

    @property
    def content(self) -> str | Image.Image:
        return self.load_image() if self.image else (self.text or "")


@dataclass
class Query:
    query_id: str
    text: str | None = None
    image: str | None = None
    lang: str = "en"
    meta: dict = field(default_factory=dict)

    def load_image(self) -> Image.Image:
        if self.image is None:
            raise ValueError(f"query {self.query_id} has no image")
        return _open_image(self.image)

    @property
    def answers(self) -> list[str]:
        a = self.meta.get("answers")
        if a is None:
            return []
        return [str(x) for x in (a if isinstance(a, list) else [a])]


@dataclass
class RetrievalDataset:
    name: str
    corpus: dict[str, Doc]
    queries: dict[str, Query]
    qrels: dict[str, dict[str, int]]
    info: dict = field(default_factory=dict)
    root: Path | None = None

    # ------------------------------------------------------------------ views
    def __repr__(self) -> str:
        return f"RetrievalDataset({self.name}: {len(self.corpus)} docs, {len(self.queries)} queries)"

    @property
    def corpus_name(self) -> str:
        """Identity of the corpus (derived datasets share their source's corpus -> shared embedding caches)."""
        return Path(str(self.info.get("corpus_from") or self.name)).name

    @property
    def doc_ids(self) -> list[str]:
        return list(self.corpus.keys())

    def to_samples(self) -> Iterator[RetrievalSample]:
        for qid, rels in self.qrels.items():
            q = self.queries[qid]
            for did, rel in rels.items():
                d = self.corpus[did]
                yield RetrievalSample(
                    query_id=qid,
                    query=q.text if q.text is not None else q.load_image(),
                    doc_id=did,
                    doc=d.content,
                    relevance=rel,
                    lang=q.lang,
                    domain=self.info.get("domain", ""),
                )

    def invert(self) -> "RetrievalDataset":
        """Swap roles: docs become queries (e.g. COCO text->image  =>  image->text)."""
        queries = {d.doc_id: Query(query_id=d.doc_id, text=d.text, image=d.image, meta=d.meta) for d in self.corpus.values()}
        corpus = {q.query_id: Doc(doc_id=q.query_id, text=q.text, image=q.image, meta=q.meta) for q in self.queries.values()}
        qrels: dict[str, dict[str, int]] = {}
        for qid, rels in self.qrels.items():
            for did, rel in rels.items():
                qrels.setdefault(did, {})[qid] = rel
        return RetrievalDataset(self.name + "_inv", corpus, queries, qrels, dict(self.info), self.root)

    def subset_queries(self, n: int | None, seed: int = 0) -> "RetrievalDataset":
        if n is None or n >= len(self.queries):
            return self
        qids = sorted(self.queries)
        random.Random(seed).shuffle(qids)
        keep = set(qids[:n])
        return RetrievalDataset(
            self.name,
            self.corpus,
            {k: v for k, v in self.queries.items() if k in keep},
            {k: v for k, v in self.qrels.items() if k in keep},
            dict(self.info),
            self.root,
        )

    def with_queries(self, queries: dict[str, Query], name_suffix: str = "") -> "RetrievalDataset":
        """Same corpus & qrels, transformed query texts (rewrites, translations...)."""
        return RetrievalDataset(
            self.name + name_suffix,
            self.corpus,
            queries,
            {k: v for k, v in self.qrels.items() if k in queries},
            dict(self.info),
            self.root,
        )

    # ------------------------------------------------------------------ io
    def save(self, root: str | Path, save_corpus: bool = True) -> Path:
        root = Path(root)
        root.mkdir(parents=True, exist_ok=True)

        def rel(p: str | None) -> str | None:
            if p is None:
                return None
            try:
                return str(Path(p).resolve().relative_to(root.resolve())).replace("\\", "/")
            except ValueError:
                return str(p)

        if save_corpus:
            write_jsonl(
                ({"doc_id": d.doc_id, "text": d.text, "image": rel(d.image), "meta": d.meta} for d in self.corpus.values()),
                root / "corpus.jsonl",
            )
        write_jsonl(
            (
                {"query_id": q.query_id, "text": q.text, "image": rel(q.image), "lang": q.lang, "meta": q.meta}
                for q in self.queries.values()
            ),
            root / "queries.jsonl",
        )
        with open(root / "qrels.tsv", "w", encoding="utf-8", newline="") as f:
            w = csv.writer(f, delimiter="\t")
            for qid, rels in self.qrels.items():
                for did, r in rels.items():
                    w.writerow([qid, did, r])
        write_json({"name": self.name, **self.info}, root / "info.json")
        return root

    @classmethod
    def load(cls, root: str | Path) -> "RetrievalDataset":
        root = Path(root)
        info = read_json(root / "info.json") if (root / "info.json").exists() else {}
        corpus_root = root
        if info.get("corpus_from"):
            corpus_root = Path(info["corpus_from"])
            if not corpus_root.is_absolute():
                corpus_root = processed_dir() / corpus_root

        def absp(base: Path, p: str | None) -> str | None:
            if p is None:
                return None
            pp = Path(p)
            return str(pp if pp.is_absolute() else base / pp)

        corpus = {}
        for r in read_jsonl(corpus_root / "corpus.jsonl"):
            corpus[r["doc_id"]] = Doc(r["doc_id"], r.get("text"), absp(corpus_root, r.get("image")), r.get("meta") or {})
        # optional OCR sidecar produced by data/ocr.py
        ocr_path = corpus_root / "ocr.jsonl"
        if ocr_path.exists():
            for r in read_jsonl(ocr_path):
                if r["doc_id"] in corpus and not corpus[r["doc_id"]].text:
                    corpus[r["doc_id"]].text = r["text"]
        queries = {}
        for r in read_jsonl(root / "queries.jsonl"):
            queries[r["query_id"]] = Query(
                r["query_id"], r.get("text"), absp(root, r.get("image")), r.get("lang", "en"), r.get("meta") or {}
            )
        qrels: dict[str, dict[str, int]] = {}
        with open(root / "qrels.tsv", encoding="utf-8") as f:
            for row in csv.reader(f, delimiter="\t"):
                if len(row) >= 3 and row[0] in queries:
                    qrels.setdefault(row[0], {})[row[1]] = int(row[2])
        name = info.pop("name", root.name)
        return cls(name, corpus, queries, qrels, info, root)


def dataset_path(name_or_path: str | Path) -> Path:
    p = Path(name_or_path)
    if p.exists():
        return p
    return processed_dir() / str(name_or_path)


def load_dataset(name_or_path: str | Path) -> RetrievalDataset:
    """Load a processed dataset by name (storage/processed/<name>) or path."""
    p = dataset_path(name_or_path)
    if not (p / "queries.jsonl").exists():
        raise FileNotFoundError(f"{p} is not a processed dataset; run `python -m mmsearch.data.preprocess ...` first")
    return RetrievalDataset.load(p)


def sample_to_dict(s: RetrievalSample) -> dict:
    d = asdict(s)
    for k in ("query", "doc"):
        if not isinstance(d[k], str):
            d[k] = "<image>"
    return d
