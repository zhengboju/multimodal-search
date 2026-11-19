"""Index factory + CLI to encode a corpus, cache its multi-vector store and report index statistics.

python -m mmsearch.index.build_index --dataset vidore_docvqa --encoder vidore/colqwen2-v1.0-hf \
    --index '{type: plaid, nprobe: 4}' --compress '[{type: kmeans, k: 64}]'
"""

from __future__ import annotations

import argparse
import time
from typing import Any

import yaml

from ..utils.misc import get_logger
from .compress import compress
from .indexes import ExhaustiveIndex, HierarchicalIndex, MultiVectorIndex, PLAIDLiteIndex
from .store import MultiVectorStore

log = get_logger(__name__)


def build_index(store: MultiVectorStore, cfg: dict | None, device: str | None = None) -> MultiVectorIndex:
    cfg = dict(cfg or {"type": "exhaustive"})
    t = cfg.pop("type", "exhaustive")
    if t == "exhaustive":
        return ExhaustiveIndex(store, device=device, **cfg)
    if t == "plaid":
        return PLAIDLiteIndex(store, device=device, **cfg)
    if t == "hierarchical":
        return HierarchicalIndex(store, device=device, **cfg)
    if t == "faiss":
        from .faiss_index import FaissTokenIndex

        return FaissTokenIndex(store, device=device, **cfg)
    raise ValueError(f"unknown index type {t}")


def build_with_compression(store: MultiVectorStore, index_cfg: Any, compress_cfg: Any, device: str | None = None):
    """Returns (index, compressor, build_stats)."""
    t0 = time.perf_counter()
    cstore, comp = compress(store, compress_cfg)
    t1 = time.perf_counter()
    index = build_index(cstore, index_cfg, device)
    t2 = time.perf_counter()
    stats = {"compressor": comp.describe(), **{f"store_{k}": v for k, v in cstore.stats().items()},
             **index.stats(), "compress_s": t1 - t0, "build_s": t2 - t1}
    return index, comp, stats


def main() -> None:
    from ..data.schema import load_dataset
    from ..retrieve.multivector import encode_corpus_cached, get_encoder
    from ..models.colpali import EncoderConfig

    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--encoder", required=True, help="model id (…-hf => hf backend) or custom checkpoint dir")
    ap.add_argument("--index", default="{type: exhaustive}", help="YAML/JSON index config")
    ap.add_argument("--compress", default="null", help="YAML/JSON compressor config")
    args = ap.parse_args()
    ds = load_dataset(args.dataset)
    spec = EncoderConfig.from_any(args.encoder)
    store = encode_corpus_cached(spec, ds, lambda: get_encoder(spec))
    log.info("store: %s", store.stats())
    _, _, stats = build_with_compression(store, yaml.safe_load(args.index), yaml.safe_load(args.compress))
    log.info("index: %s", stats)


if __name__ == "__main__":
    main()
