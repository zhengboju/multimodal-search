"""ColPali-style multi-vector document encoder on a Qwen2-VL backbone (ColQwen2) + MaxSim helpers.

Two backends behind one interface:
  hf      transformers' native ColQwen2ForRetrieval checkpoints (e.g. vidore/colqwen2-v1.0-hf) — the
          reference / upper bound used to validate the evaluation pipeline against the paper numbers.
  custom  our reproduction: Qwen2-VL(-2B) backbone + LoRA + 128-d projection head, trained with
          train/train_colpali.py. Checkpoints are dirs with encoder_config.json + adapter/ + proj.pt.

Output of encode_*: list of per-item tensors [L_i, dim] (fp16, CPU, padding removed, L2-normalised).
"""

from __future__ import annotations

import functools
import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Sequence

import torch
import torch.nn as nn
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from ..utils.hub import resolve_model
from ..utils.misc import get_logger
from .compat import from_pretrained, parse_dtype, pick_attn_impl, text_hidden_size
from .losses import maxsim_scores
from .projector import ProjectionHead

log = get_logger(__name__)

PATCH = 28
DOC_PROMPT = "<|im_start|>user\n<|vision_start|><|image_pad|><|vision_end|>Describe the image.<|im_end|><|endoftext|>"
QUERY_PREFIX = "Query: "
QUERY_AUG_TOKEN = "<|endoftext|>"
# LoRA on the language model only (the negative lookahead keeps the vision tower frozen)
DEFAULT_LORA_TARGET = r"^(?!.*visual).*\.(q_proj|k_proj|v_proj|o_proj|gate_proj|up_proj|down_proj)$"


@dataclass
class EncoderConfig:
    model: str = "Qwen/Qwen2-VL-2B-Instruct"  # base VLM (custom) or ColQwen2 checkpoint (hf)
    backend: str = "custom"  # custom | hf
    checkpoint: str | None = None  # custom-backend training output dir (overrides fields from its encoder_config.json)
    dim: int = 128
    max_pixels: int | None = 768 * PATCH * PATCH  # ~768 visual tokens / page (ColQwen2 setting)
    min_pixels: int | None = 4 * PATCH * PATCH
    query_augmentation: int = 10
    dtype: str = "bf16"
    attn_implementation: str = "auto"
    quantization: str | None = None  # None | int8 | nf4 (bitsandbytes)
    extra: dict = field(default_factory=dict)

    @classmethod
    def from_any(cls, spec: dict | "EncoderConfig" | str) -> "EncoderConfig":
        if isinstance(spec, EncoderConfig):
            return spec
        if isinstance(spec, str):
            spec = {"checkpoint": spec} if Path(spec).is_dir() and (Path(spec) / "encoder_config.json").exists() else {"model": spec}
        spec = dict(spec)
        if spec.get("checkpoint"):
            saved = json.loads((Path(spec["checkpoint"]) / "encoder_config.json").read_text())
            saved.update({k: v for k, v in spec.items() if v is not None})
            spec = saved
        if "backend" not in spec:
            spec["backend"] = "hf" if str(spec.get("model", "")).endswith("-hf") else "custom"
        known = {f for f in cls.__dataclass_fields__}
        return cls(**{k: v for k, v in spec.items() if k in known})

    def tag(self) -> str:
        """Stable identifier for embedding caches."""
        base = Path(self.checkpoint).name if self.checkpoint else self.model.replace("/", "__")
        q = f"_{self.quantization}" if self.quantization else ""
        return f"{self.backend}_{base}_px{self.max_pixels}{q}"


def _set_pixel_bounds(image_processor: Any, min_pixels: int | None, max_pixels: int | None) -> None:
    if min_pixels is None and max_pixels is None:
        return
    for attr, v in (("min_pixels", min_pixels), ("max_pixels", max_pixels)):
        if v is not None and hasattr(image_processor, attr):
            setattr(image_processor, attr, v)
    size = getattr(image_processor, "size", None)
    if isinstance(size, dict):
        new = dict(size)
        if min_pixels is not None:
            new["shortest_edge"] = min_pixels
        if max_pixels is not None:
            new["longest_edge"] = max_pixels
        image_processor.size = new


def _quant_config(q: str | None):
    if not q:
        return None
    from transformers import BitsAndBytesConfig

    if q == "int8":
        return BitsAndBytesConfig(load_in_8bit=True)
    if q == "nf4":
        return BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.bfloat16)
    raise ValueError(f"unknown quantization {q}")


def build_doc_inputs(processor: Any, cfg: EncoderConfig, images: Sequence[Image.Image]) -> dict[str, torch.Tensor]:
    """Module-level (picklable via functools.partial) so DataLoader workers never capture the GPU model."""
    if cfg.backend == "hf":
        fn = getattr(processor, "process_images", None)
        out = fn(images=list(images)) if fn else processor(images=list(images), return_tensors="pt")
    else:
        out = processor(text=[DOC_PROMPT] * len(images), images=list(images), padding="longest", return_tensors="pt")
    return dict(out)


def build_query_inputs(processor: Any, cfg: EncoderConfig, texts: Sequence[str]) -> dict[str, torch.Tensor]:
    if cfg.backend == "hf":
        fn = getattr(processor, "process_queries", None)
        out = fn(text=list(texts)) if fn else processor(text=list(texts), return_tensors="pt")
    else:
        texts = [QUERY_PREFIX + t + QUERY_AUG_TOKEN * cfg.query_augmentation for t in texts]
        out = processor(text=texts, padding="longest", return_tensors="pt")
    return dict(out)


class _Images(Dataset):
    def __init__(self, images: Sequence[str | Image.Image]):
        self.images = images

    def __len__(self) -> int:
        return len(self.images)

    def __getitem__(self, i: int) -> Image.Image:
        im = self.images[i]
        return im.convert("RGB") if isinstance(im, Image.Image) else Image.open(im).convert("RGB")


class ColQwen2Encoder(nn.Module):
    def __init__(self, spec: dict | EncoderConfig | str, device: str | torch.device | None = None, for_training: bool = False):
        super().__init__()
        from transformers import AutoProcessor

        self.cfg = EncoderConfig.from_any(spec)
        cfg = self.cfg
        self.dtype = parse_dtype(cfg.dtype)
        dev = torch.device(device) if device else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        attn = pick_attn_impl(cfg.attn_implementation)
        qconf = _quant_config(cfg.quantization)
        load_kw: dict[str, Any] = {"attn_implementation": attn}
        if qconf is not None:
            load_kw.update(quantization_config=qconf, device_map={"": dev.index or 0})
        path = resolve_model(cfg.model)
        self.processor = AutoProcessor.from_pretrained(path)
        self._peft = False

        if cfg.backend == "hf":
            from transformers import ColQwen2ForRetrieval

            self.model = from_pretrained(ColQwen2ForRetrieval, path, self.dtype, **load_kw)
            if cfg.max_pixels is not None and cfg.extra.get("override_pixels", False):
                _set_pixel_bounds(self.processor.image_processor, cfg.min_pixels, cfg.max_pixels)
            self.proj = None
        elif cfg.backend == "custom":
            from transformers import AutoModelForImageTextToText

            _set_pixel_bounds(self.processor.image_processor, cfg.min_pixels, cfg.max_pixels)
            self.processor.tokenizer.padding_side = "right"
            self.vlm = from_pretrained(AutoModelForImageTextToText, path, self.dtype, **load_kw)
            self.vlm.config.use_cache = False
            self.proj = ProjectionHead(text_hidden_size(self.vlm.config), cfg.dim)
            ckpt = Path(cfg.checkpoint) if cfg.checkpoint else None
            if ckpt is not None:
                if (ckpt / "adapter").exists():
                    from peft import PeftModel

                    self.vlm = PeftModel.from_pretrained(self.vlm, str(ckpt / "adapter"), is_trainable=for_training)
                    if for_training:
                        self._peft = True
                    else:
                        self.vlm = self.vlm.merge_and_unload()
                self.proj.load_state_dict(torch.load(ckpt / "proj.pt", map_location="cpu"))
            elif not for_training:
                log.warning("custom backend without checkpoint: projection head is randomly initialised")
            if not for_training:
                self.proj.to(self.dtype)
        else:
            raise ValueError(f"unknown backend {cfg.backend}")
        if qconf is None:
            self.to(dev)
        elif self.proj is not None:
            self.proj.to(dev)
        if not for_training:
            self.eval()

    # ------------------------------------------------------------------ training setup (custom backend)
    def enable_lora(self, r: int = 32, alpha: int = 32, dropout: float = 0.1, target: str = DEFAULT_LORA_TARGET) -> None:
        from peft import LoraConfig, get_peft_model

        assert self.cfg.backend == "custom"
        lcfg = LoraConfig(r=r, lora_alpha=alpha, lora_dropout=dropout, target_modules=target, bias="none")
        self.vlm = get_peft_model(self.vlm, lcfg)
        self._peft = True
        self.vlm.print_trainable_parameters()

    def enable_gradient_checkpointing(self) -> None:
        base = self.vlm.get_base_model() if self._peft else self.vlm
        base.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})

    def trainable_parameters(self) -> list[nn.Parameter]:
        return [p for p in self.parameters() if p.requires_grad]

    @property
    def backbone(self) -> nn.Module:
        m = self.vlm.get_base_model() if self._peft else self.vlm
        return m.model

    @property
    def device(self) -> torch.device:
        return next(self.parameters()).device

    # ------------------------------------------------------------------ preprocessing
    def doc_inputs(self, images: Sequence[Image.Image]) -> dict[str, torch.Tensor]:
        return build_doc_inputs(self.processor, self.cfg, images)

    def query_inputs(self, texts: Sequence[str]) -> dict[str, torch.Tensor]:
        return build_query_inputs(self.processor, self.cfg, texts)

    # ------------------------------------------------------------------ forward
    def forward_inputs(self, inputs: dict[str, torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
        inputs = {k: v.to(self.device) for k, v in inputs.items() if isinstance(v, torch.Tensor)}
        if "pixel_values" in inputs and self.dtype is not None and inputs["pixel_values"].is_floating_point():
            inputs["pixel_values"] = inputs["pixel_values"].to(self.dtype)
        mask = inputs["attention_mask"]
        if self.cfg.backend == "hf":
            emb = self.model(**inputs).embeddings
            return emb, mask
        out = self.backbone(
            input_ids=inputs["input_ids"],
            attention_mask=mask,
            pixel_values=inputs.get("pixel_values"),
            image_grid_thw=inputs.get("image_grid_thw"),
            use_cache=False,
            return_dict=True,
        )
        return self.proj(out.last_hidden_state, mask), mask

    def forward(self, query_inputs: dict, doc_inputs: dict):
        """Single DDP forward computing both sides."""
        q, qm = self.forward_inputs(query_inputs)
        d, dm = self.forward_inputs(doc_inputs)
        return q, qm, d, dm

    # ------------------------------------------------------------------ inference
    @staticmethod
    def _unpad(emb: torch.Tensor, mask: torch.Tensor) -> list[torch.Tensor]:
        return [e[m.bool()].to(torch.float16).cpu() for e, m in zip(emb, mask)]

    @torch.no_grad()
    def encode_images(self, images: Sequence[str | Image.Image], batch_size: int = 16, num_workers: int = 4,
                      desc: str | None = "encode pages") -> list[torch.Tensor]:
        dl = DataLoader(_Images(images), batch_size=batch_size, num_workers=num_workers,
                        collate_fn=functools.partial(build_doc_inputs, self.processor, self.cfg))
        out: list[torch.Tensor] = []
        for batch in tqdm(dl, desc=desc, disable=desc is None):
            emb, mask = self.forward_inputs(batch)
            out.extend(self._unpad(emb, mask))
        return out

    @torch.no_grad()
    def encode_queries(self, texts: Sequence[str], batch_size: int = 64, desc: str | None = None) -> list[torch.Tensor]:
        out: list[torch.Tensor] = []
        for i in tqdm(range(0, len(texts), batch_size), desc=desc, disable=desc is None):
            emb, mask = self.forward_inputs(self.query_inputs(texts[i : i + batch_size]))
            out.extend(self._unpad(emb, mask))
        return out

    # ------------------------------------------------------------------ persistence (custom backend)
    def save(self, out_dir: str | Path) -> None:
        assert self.cfg.backend == "custom"
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        cfg = asdict(self.cfg)
        cfg["checkpoint"] = None
        (out_dir / "encoder_config.json").write_text(json.dumps(cfg, indent=2))
        if self._peft:
            self.vlm.save_pretrained(str(out_dir / "adapter"))
        torch.save({k: v.detach().float().cpu() for k, v in self.proj.state_dict().items()}, out_dir / "proj.pt")


def pad_multivectors(items: Sequence[torch.Tensor], device: torch.device | str | None = None,
                     dtype: torch.dtype | None = None) -> tuple[torch.Tensor, torch.Tensor]:
    """List of [L_i, D] -> padded [N, Lmax, D] + bool mask [N, Lmax]."""
    n = len(items)
    D = items[0].shape[-1]
    Lmax = max(int(t.shape[0]) for t in items)
    dtype = dtype or items[0].dtype
    out = torch.zeros(n, Lmax, D, dtype=dtype, device=device)
    mask = torch.zeros(n, Lmax, dtype=torch.bool, device=device)
    for i, t in enumerate(items):
        out[i, : t.shape[0]] = t.to(device=device, dtype=dtype)
        mask[i, : t.shape[0]] = True
    return out, mask


def score_multivector(queries: Sequence[torch.Tensor], docs: Sequence[torch.Tensor], device: str | torch.device = "cpu",
                      q_batch: int = 32, doc_chunk: int = 256) -> torch.Tensor:
    """Exhaustive MaxSim score matrix [Nq, Nd] (reference implementation used by tests and small corpora)."""
    d, dm = pad_multivectors(docs, device, torch.float32 if str(device) == "cpu" else torch.float16)
    rows = []
    for i in range(0, len(queries), q_batch):
        q, qm = pad_multivectors(queries[i : i + q_batch], device, d.dtype)
        rows.append(maxsim_scores(q, qm, d, dm, doc_chunk).cpu())
    return torch.cat(rows)
