"""Query rewriter inference + query-composition modes + length-matched random-expansion control."""

from __future__ import annotations

import json
import random
import re
from dataclasses import asdict, dataclass, field
from typing import Sequence

from ..llm.client import VLMClient, user_message
from .prompts import REWRITE_SYSTEM

_JSON = re.compile(r"\{.*\}", re.S)


@dataclass
class RewriteResult:
    rewrite: str
    intent: str = "other"
    entities: list[str] = field(default_factory=list)
    lang: str = "en"

    def to_json(self) -> str:
        return json.dumps(asdict(self), ensure_ascii=False)


def parse_rewrite(text: str | None) -> RewriteResult | None:
    if not text:
        return None
    m = _JSON.search(text)
    if not m:
        return None
    try:
        obj = json.loads(m.group(0))
    except json.JSONDecodeError:
        return None
    rw = str(obj.get("rewrite") or "").strip()
    if not rw:
        return None
    ents = obj.get("entities") or []
    return RewriteResult(rw, str(obj.get("intent") or "other"), [str(e) for e in ents if e], str(obj.get("lang") or "en"))


def rewrite_messages(query: str, image: str | None = None) -> list[dict]:
    user = user_message(f"Query: {query}", [image] if image else [], 768 * 28 * 28) if image else {"role": "user", "content": f"Query: {query}"}
    return [{"role": "system", "content": REWRITE_SYSTEM}, user]


class QueryRewriter:
    def __init__(self, client: VLMClient | None = None, model: str | None = None, max_tokens: int = 256):
        self.client = client or VLMClient()
        self.model = model or self.client.model
        self.max_tokens = max_tokens

    def rewrite_many(self, queries: Sequence[str], images: Sequence[str | None] | None = None, n: int = 1,
                     temperature: float = 0.0, desc: str | None = "rewrite") -> list[list[RewriteResult | None]]:
        images = images or [None] * len(queries)
        msgs = [rewrite_messages(q, im) for q, im in zip(queries, images)]
        res = self.client.chat_many(msgs, desc=desc, model=self.model, n=n, temperature=temperature, max_tokens=self.max_tokens)
        out = []
        for r in res:
            if r is None:
                out.append([None] * n)
            else:
                out.append([parse_rewrite(c.message.content) for c in r.choices])
        return out


def compose(original: str, rr: RewriteResult | None, mode: str = "replace") -> str:
    """replace: rewrite only | concat: original + rewrite | entities: original + extracted entities."""
    if rr is None:
        return original
    if mode == "replace":
        return rr.rewrite
    if mode == "concat":
        return f"{original} {rr.rewrite}"
    if mode == "entities":
        return f"{original} {' '.join(rr.entities)}"
    raise ValueError(mode)


def random_expansion(original: str, target_words: int, vocab: Sequence[str], rng: random.Random) -> str:
    """Control for E6: append random words (drawn from other queries' rewrites) until the query reaches the same
    length as its rewrite — isolates 'longer query' from 'semantically completed query'."""
    n_add = max(0, target_words - len(original.split()))
    return original + (" " + " ".join(rng.choice(vocab) for _ in range(n_add)) if n_add and vocab else "")
