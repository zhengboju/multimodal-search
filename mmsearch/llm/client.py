"""OpenAI-compatible client for vLLM-served (V)LMs, shared by OCR / rerank / RAG / judge / query synthesis.

Start a server first, e.g. `bash scripts/serve_vlm.sh Qwen/Qwen2.5-VL-7B-Instruct 8000`.
Defaults come from env: MMS_VLM_BASE_URL (http://127.0.0.1:8000/v1), MMS_VLM_MODEL (first served model).
"""

from __future__ import annotations

import asyncio
import base64
import io
import os
from pathlib import Path
from typing import Any, Sequence

from PIL import Image
from tqdm import tqdm

from ..utils.misc import get_logger

log = get_logger(__name__)

ImageLike = str | Path | Image.Image


def load_pil(img: ImageLike) -> Image.Image:
    if isinstance(img, Image.Image):
        return img.convert("RGB")
    return Image.open(img).convert("RGB")


def image_to_data_url(img: ImageLike, max_pixels: int | None = None, quality: int = 90) -> str:
    """Downscale to <= max_pixels (keeps aspect ratio) and encode as a JPEG data URL."""
    im = load_pil(img)
    if max_pixels and im.width * im.height > max_pixels:
        s = (max_pixels / (im.width * im.height)) ** 0.5
        im = im.resize((max(28, int(im.width * s)), max(28, int(im.height * s))), Image.LANCZOS)
    buf = io.BytesIO()
    im.save(buf, format="JPEG", quality=quality)
    return "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode()


def user_message(text: str, images: Sequence[ImageLike] = (), max_pixels: int | Sequence[int] | None = None,
                 image_labels: Sequence[str] | None = None) -> dict:
    """One user turn with interleaved images (optionally labelled, e.g. 'Page 1:') followed by text."""
    parts: list[dict] = []
    for i, im in enumerate(images):
        mp = max_pixels[i] if isinstance(max_pixels, (list, tuple)) else max_pixels
        if image_labels:
            parts.append({"type": "text", "text": image_labels[i]})
        parts.append({"type": "image_url", "image_url": {"url": image_to_data_url(im, mp)}})
    parts.append({"type": "text", "text": text})
    return {"role": "user", "content": parts}


class VLMClient:
    def __init__(
        self,
        base_url: str | None = None,
        model: str | None = None,
        api_key: str | None = None,
        max_concurrency: int = 32,
        timeout: float = 600.0,
        max_retries: int = 3,
    ):
        from openai import AsyncOpenAI, OpenAI

        self.base_url = base_url or os.environ.get("MMS_VLM_BASE_URL", "http://127.0.0.1:8000/v1")
        api_key = api_key or os.environ.get("MMS_VLM_API_KEY", "EMPTY")
        self._sync = OpenAI(base_url=self.base_url, api_key=api_key, timeout=timeout, max_retries=max_retries)
        self._async = AsyncOpenAI(base_url=self.base_url, api_key=api_key, timeout=timeout, max_retries=max_retries)
        self.model = model or os.environ.get("MMS_VLM_MODEL") or self._first_model()
        self.max_concurrency = max_concurrency

    def _first_model(self) -> str:
        models = self._sync.models.list().data
        if not models:
            raise RuntimeError(f"no model served at {self.base_url}")
        return models[0].id

    # ------------------------------------------------------------------ async API
    async def achat(self, messages: list[dict], model: str | None = None, **kw: Any):
        kw.setdefault("temperature", 0.0)
        return await self._async.chat.completions.create(model=model or self.model, messages=messages, **kw)

    async def _run_all(self, batch: list[list[dict]], desc: str | None, **kw: Any) -> list[Any]:
        sem = asyncio.Semaphore(self.max_concurrency)
        pbar = tqdm(total=len(batch), desc=desc, disable=desc is None)

        async def one(i: int, msgs: list[dict]):
            async with sem:
                try:
                    return i, await self.achat(msgs, **kw)
                except Exception as e:  # noqa: BLE001 - keep the batch going, record failure
                    log.warning("request %d failed: %r", i, e)
                    return i, None
                finally:
                    pbar.update(1)

        out: list[Any] = [None] * len(batch)
        for i, r in await asyncio.gather(*(one(i, m) for i, m in enumerate(batch))):
            out[i] = r
        pbar.close()
        return out

    # ------------------------------------------------------------------ sync helpers (offline jobs)
    def chat_many(self, batch: list[list[dict]], desc: str | None = None, **kw: Any) -> list[Any]:
        return asyncio.run(self._run_all(batch, desc, **kw))

    def chat_text(self, batch: list[list[dict]], desc: str | None = None, **kw: Any) -> list[str | None]:
        res = self.chat_many(batch, desc, **kw)
        return [r.choices[0].message.content if r is not None else None for r in res]


def yes_probability(completion: Any, yes: Sequence[str] = ("yes", "Yes", "YES"), no: Sequence[str] = ("no", "No", "NO")) -> float:
    """P(yes) from the first generated token's top logprobs (normalised over yes/no mass)."""
    import math

    if completion is None:
        return 0.0
    try:
        top = completion.choices[0].logprobs.content[0].top_logprobs
    except (AttributeError, IndexError, TypeError):
        txt = (completion.choices[0].message.content or "").strip().lower()
        return 1.0 if txt.startswith("yes") else 0.0
    py = sum(math.exp(t.logprob) for t in top if t.token.strip() in yes)
    pn = sum(math.exp(t.logprob) for t in top if t.token.strip() in no)
    if py + pn == 0:
        return 0.0
    return py / (py + pn)
