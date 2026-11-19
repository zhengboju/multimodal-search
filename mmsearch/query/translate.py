"""Translate a processed dataset's queries (multilingual robustness: ViDoRe -> zh / ja).

python -m mmsearch.query.translate --datasets vidore_all --langs zh ja
Creates storage/processed/<name>__<lang> with corpus_from=<name> (same pages & qrels, translated queries).
"""

from __future__ import annotations

import argparse

from ..data.schema import Query, RetrievalDataset, dataset_path, load_dataset
from ..llm.client import VLMClient
from ..utils.misc import get_logger
from ..utils.paths import processed_dir
from .prompts import LANG_NAMES, TRANSLATE_PROMPT

log = get_logger(__name__)


def translated_name(name: str, lang: str) -> str:
    return f"{name}__{lang}"


def translate_texts(client: VLMClient, texts: list[str], lang: str, model: str | None = None) -> list[str | None]:
    msgs = [[{"role": "user", "content": TRANSLATE_PROMPT.format(lang_name=LANG_NAMES.get(lang, lang), query=t)}] for t in texts]
    outs = client.chat_text(msgs, desc=f"translate->{lang}", model=model, max_tokens=256)
    return [o.strip().strip('"').split("\n")[0] if o else None for o in outs]


def translate_dataset(name: str, lang: str, client: VLMClient, model: str | None = None, overwrite: bool = False) -> RetrievalDataset:
    out_name = translated_name(name, lang)
    if (dataset_path(out_name) / "queries.jsonl").exists() and not overwrite:
        return load_dataset(out_name)
    ds = load_dataset(name)
    qids = list(ds.queries)
    trans = translate_texts(client, [ds.queries[q].text or "" for q in qids], lang, model)
    queries = {}
    for q, t in zip(qids, trans):
        if t:
            src = ds.queries[q]
            queries[q] = Query(q, t, None, lang, {**src.meta, "orig_text": src.text, "orig_lang": src.lang})
    new = RetrievalDataset(out_name, ds.corpus, queries, {q: ds.qrels[q] for q in queries},
                           {**ds.info, "lang": lang, "corpus_from": ds.info.get("corpus_from", name)})
    new.save(processed_dir() / out_name, save_corpus=False)
    log.info("translated %s -> %s (%d/%d)", name, out_name, len(queries), len(qids))
    return load_dataset(out_name)


def main() -> None:
    from ..eval.run_benchmark import expand_datasets

    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", nargs="+", required=True)
    ap.add_argument("--langs", nargs="+", default=["zh", "ja"])
    ap.add_argument("--base-url", default=None)
    ap.add_argument("--model", default=None)
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()
    client = VLMClient(args.base_url, args.model, max_concurrency=64)
    for d in expand_datasets(args.datasets):
        for lang in args.langs:
            translate_dataset(d, lang, client, overwrite=args.overwrite)


if __name__ == "__main__":
    main()
