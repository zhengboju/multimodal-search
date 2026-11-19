"""OCR page images with a vLLM-served VLM; writes <dataset>/ocr.jsonl which RetrievalDataset.load merges as doc text.

Used by the BM25 (sparse) retriever and hybrid fusion — the "OCR text" baseline that visual retrieval is compared with.

python -m mmsearch.data.ocr --datasets vidore_docvqa vidore_tatdqa --max-pixels 1003520
"""

from __future__ import annotations

import argparse

from ..llm.client import VLMClient, user_message
from ..utils.misc import get_logger, read_jsonl, write_jsonl
from .schema import dataset_path, load_dataset

log = get_logger(__name__)

OCR_PROMPT = (
    "Transcribe all text on this document page in natural reading order. "
    "Include titles, table cells (row by row), chart labels and numbers. Output plain text only."
)


def ocr_dataset(name: str, client: VLMClient, max_pixels: int, max_tokens: int, batch: int = 256) -> None:
    ds = load_dataset(name)
    root = ds.root if not ds.info.get("corpus_from") else dataset_path(ds.info["corpus_from"])
    out = root / "ocr.jsonl"
    done = {r["doc_id"] for r in read_jsonl(out)} if out.exists() else set()
    todo = [d for d in ds.corpus.values() if d.image and d.doc_id not in done]
    log.info("%s: %d pages to OCR (%d cached)", name, len(todo), len(done))
    for i in range(0, len(todo), batch):
        chunk = todo[i : i + batch]
        msgs = [[user_message(OCR_PROMPT, [d.image], max_pixels)] for d in chunk]
        texts = client.chat_text(msgs, desc=f"ocr {name} {i}/{len(todo)}", max_tokens=max_tokens)
        write_jsonl(({"doc_id": d.doc_id, "text": t} for d, t in zip(chunk, texts) if t), out, append=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", nargs="+", required=True)
    ap.add_argument("--base-url", default=None)
    ap.add_argument("--model", default=None)
    ap.add_argument("--max-pixels", type=int, default=1280 * 28 * 28)
    ap.add_argument("--max-tokens", type=int, default=2048)
    ap.add_argument("--concurrency", type=int, default=64)
    args = ap.parse_args()
    client = VLMClient(args.base_url, args.model, max_concurrency=args.concurrency)
    for name in args.datasets:
        ocr_dataset(name, client, args.max_pixels, args.max_tokens)


if __name__ == "__main__":
    main()
