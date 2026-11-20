"""CLIP dual encoder wrapper (HF transformers CLIPModel, weights via ModelScope)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms as T
from tqdm import tqdm

from ..utils.hub import resolve_model
from .compat import from_pretrained, parse_dtype

CLIP_MEAN = (0.48145466, 0.4578275, 0.40821073)
CLIP_STD = (0.26862954, 0.26130258, 0.27577711)


class ImageListDataset(Dataset):
    def __init__(self, images: Sequence[str | Image.Image], transform):
        self.images = images
        self.transform = transform

    def __len__(self) -> int:
        return len(self.images)

    def __getitem__(self, i: int) -> torch.Tensor:
        im = self.images[i]
        im = im.convert("RGB") if isinstance(im, Image.Image) else Image.open(im).convert("RGB")
        return self.transform(im)


class CLIPWrapper(nn.Module):
    def __init__(self, model: str = "openai/clip-vit-large-patch14", dtype: str | torch.dtype | None = None,
                 gradient_checkpointing: bool = False):
        super().__init__()
        from transformers import AutoTokenizer, CLIPModel

        self.model_name = model
        path = resolve_model(model)
        self.model = from_pretrained(CLIPModel, path, parse_dtype(dtype))
        self.tokenizer = AutoTokenizer.from_pretrained(path)
        self.image_size = int(self.model.config.vision_config.image_size)
        mean, std = CLIP_MEAN, CLIP_STD
        pp = Path(path) / "preprocessor_config.json"
        if pp.exists():  # stay faithful to the checkpoint's normalisation (plan: wrong mean/std is the #1 cause of low baselines)
            cfg = json.loads(pp.read_text())
            mean, std = tuple(cfg.get("image_mean", mean)), tuple(cfg.get("image_std", std))
        self.mean, self.std = mean, std
        if gradient_checkpointing:
            self.model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})

    # ---------------------------------------------------------------- preprocessing
    def transform(self, train: bool = False):
        s = self.image_size
        norm = [T.ToTensor(), T.Normalize(self.mean, self.std)]
        if train:
            return T.Compose([T.RandomResizedCrop(s, scale=(0.6, 1.0), interpolation=T.InterpolationMode.BICUBIC), *norm])
        return T.Compose([T.Resize(s, interpolation=T.InterpolationMode.BICUBIC), T.CenterCrop(s), *norm])

    def tokenize(self, texts: Sequence[str]) -> dict[str, torch.Tensor]:
        return self.tokenizer(list(texts), padding=True, truncation=True, max_length=77, return_tensors="pt")

    # ---------------------------------------------------------------- towers
    @property
    def logit_scale(self) -> nn.Parameter:
        return self.model.logit_scale

    def encode_image(self, pixel_values: torch.Tensor, normalize: bool = True) -> torch.Tensor:
        out = self.model.vision_model(pixel_values=pixel_values)
        feat = self.model.visual_projection(out.pooler_output)
        return F.normalize(feat, dim=-1) if normalize else feat

    def encode_text(self, input_ids: torch.Tensor, attention_mask: torch.Tensor, normalize: bool = True) -> torch.Tensor:
        out = self.model.text_model(input_ids=input_ids, attention_mask=attention_mask)
        feat = self.model.text_projection(out.pooler_output)
        return F.normalize(feat, dim=-1) if normalize else feat

    def forward(self, pixel_values: torch.Tensor, input_ids: torch.Tensor, attention_mask: torch.Tensor,
                extra_pixel_values: torch.Tensor | None = None):
        """Single forward for DDP: images (+ optional hard-negative images) and texts."""
        if extra_pixel_values is not None and extra_pixel_values.numel() > 0:
            n = pixel_values.shape[0]
            feats = self.encode_image(torch.cat([pixel_values, extra_pixel_values]))
            img, extra = feats[:n], feats[n:]
        else:
            img, extra = self.encode_image(pixel_values), None
        txt = self.encode_text(input_ids, attention_mask)
        return img, txt, extra

    # ---------------------------------------------------------------- inference helpers
    @property
    def device(self) -> torch.device:
        return next(self.parameters()).device

    @torch.no_grad()
    def encode_images(self, images: Sequence[str | Image.Image], batch_size: int = 256, num_workers: int = 8,
                      desc: str | None = "encode images") -> torch.Tensor:
        dl = DataLoader(ImageListDataset(images, self.transform(False)), batch_size=batch_size,
                        num_workers=num_workers, pin_memory=True)
        dtype = next(self.parameters()).dtype
        out = []
        for px in tqdm(dl, desc=desc, disable=desc is None):
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=self.device.type == "cuda"):
                out.append(self.encode_image(px.to(self.device, dtype=dtype, non_blocking=True)).float().cpu())
        return torch.cat(out) if out else torch.empty(0)

    @torch.no_grad()
    def encode_texts(self, texts: Sequence[str], batch_size: int = 1024, desc: str | None = "encode texts") -> torch.Tensor:
        out = []
        rng = range(0, len(texts), batch_size)
        for i in tqdm(rng, desc=desc, disable=desc is None):
            tok = self.tokenize(texts[i : i + batch_size]).to(self.device)
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=self.device.type == "cuda"):
                out.append(self.encode_text(tok["input_ids"], tok["attention_mask"]).float().cpu())
        return torch.cat(out) if out else torch.empty(0)

    def save(self, out_dir: str | Path) -> None:
        out_dir = Path(out_dir)
        self.model.save_pretrained(out_dir)
        self.tokenizer.save_pretrained(out_dir)
        pp = out_dir / "preprocessor_config.json"  # merge, so the dir can also be loaded by AutoProcessor
        cfg = json.loads(pp.read_text()) if pp.exists() else {"image_size": self.image_size, "crop_size": self.image_size}
        cfg.update({"image_mean": list(self.mean), "image_std": list(self.std)})
        pp.write_text(json.dumps(cfg, indent=2))
