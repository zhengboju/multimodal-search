"""Multimodal RAG: retrieved page images go straight into the VLM prompt under a visual-token budget."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Sequence

from ..llm.budget import allocate_tokens, tokens_to_pixels
from ..llm.client import VLMClient, image_to_data_url

SYSTEM_PROMPT = (
    "You answer questions about documents using only the provided page images. "
    "Reply with the shortest possible answer (a word, number or short phrase) copied from the page when possible. "
    "No explanation."
)
CLOSED_BOOK_SYSTEM = "Answer the question with the shortest possible answer (a word, number or short phrase). No explanation."


@dataclass
class RAGConfig:
    k: int = 3
    total_tokens: int = 6144  # visual-token budget shared by all pages in the prompt
    strategy: str = "uniform"  # uniform | rank_decay
    min_tokens: int = 256
    max_tokens: int = 1280
    max_new_tokens: int = 64
    extra: dict = field(default_factory=dict)


class RAGGenerator:
    def __init__(self, client: VLMClient | None = None, cfg: RAGConfig | None = None):
        self.client = client or VLMClient()
        self.cfg = cfg or RAGConfig()

    def pixel_budget(self, n_images: int) -> list[int]:
        c = self.cfg
        min_t = min(c.min_tokens, max(64, c.total_tokens // max(1, n_images)))
        return [tokens_to_pixels(t) for t in allocate_tokens(n_images, c.total_tokens, c.strategy, min_t, c.max_tokens)]

    def messages(self, question: str, images: Sequence[str]) -> list[dict]:
        if not images:
            return [{"role": "system", "content": CLOSED_BOOK_SYSTEM},
                    {"role": "user", "content": f"Question: {question}\nAnswer:"}]
        parts: list[dict] = []
        for i, (img, px) in enumerate(zip(images, self.pixel_budget(len(images)))):
            parts.append({"type": "text", "text": f"Page {i + 1}:"})
            parts.append({"type": "image_url", "image_url": {"url": image_to_data_url(img, px)}})
        parts.append({"type": "text", "text": f"Question: {question}\nAnswer:"})
        return [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": parts}]

    def answer_many(self, items: Sequence[tuple[str, Sequence[str]]], desc: str | None = None) -> list[str]:
        msgs = [self.messages(q, imgs) for q, imgs in items]
        outs = self.client.chat_text(msgs, desc=desc, max_tokens=self.cfg.max_new_tokens)
        return [(o or "").strip().split("\n")[0] for o in outs]

    async def aanswer(self, question: str, images: Sequence[str]) -> str:
        r = await self.client.achat(self.messages(question, images), max_tokens=self.cfg.max_new_tokens)
        return (r.choices[0].message.content or "").strip().split("\n")[0]
