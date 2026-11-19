"""Week-1 D4: modality gap in CLIP embedding space.

python -m mmsearch.eval.modality_gap --dataset coco_test

Reports: centroid distance between image and text embeddings, matched vs. unmatched cosine distributions,
and what happens to retrieval when the gap is removed (subtract each modality's mean, renormalise).
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
import torch.nn.functional as F

from ..data.schema import load_dataset
from ..utils.misc import get_logger, write_json
from ..utils.paths import runs_dir
from .eval_clip import bidirectional_recall, dataset_tensors

log = get_logger(__name__)


def gap_stats(img: torch.Tensor, txt: torch.Tensor, t2i: torch.Tensor) -> dict:
    mi, mt = img.mean(0), txt.mean(0)
    matched = (txt * img[t2i]).sum(-1)
    g = torch.Generator().manual_seed(0)
    rand_img = img[torch.randint(0, img.shape[0], (txt.shape[0],), generator=g)]
    unmatched = (txt * rand_img).sum(-1)
    return {
        "centroid_distance": float((mi - mt).norm()),
        "centroid_cosine": float(F.cosine_similarity(mi, mt, dim=0)),
        "img_self_cos_mean": float((img @ mi / mi.norm()).mean()),
        "txt_self_cos_mean": float((txt @ mt / mt.norm()).mean()),
        "matched_cos_mean": float(matched.mean()),
        "unmatched_cos_mean": float(unmatched.mean()),
        "_matched": matched,
        "_unmatched": unmatched,
    }


def main() -> None:
    from ..models.clip_wrapper import CLIPWrapper

    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="openai/clip-vit-large-patch14")
    ap.add_argument("--dataset", default="coco_test")
    args = ap.parse_args()
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    m = CLIPWrapper(args.model, dtype="fp16" if dev == "cuda" else None).to(dev).eval()
    ds = load_dataset(args.dataset)
    images, texts, t2i = dataset_tensors(ds)
    img, txt = m.encode_images(images), m.encode_texts(texts)
    s = gap_stats(img, txt, t2i)
    base = bidirectional_recall(img, txt, t2i)
    img_c, txt_c = F.normalize(img - img.mean(0), dim=-1), F.normalize(txt - txt.mean(0), dim=-1)
    closed = bidirectional_recall(img_c, txt_c, t2i)
    out = runs_dir() / "analysis" / f"modality_gap_{Path(args.model).name}_{ds.name}"
    res = {k: v for k, v in s.items() if not k.startswith("_")}
    res.update({"retrieval_original": base, "retrieval_gap_removed": closed})
    write_json(res, out / "gap.json")
    print(res)
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(6, 4))
        ax.hist(s["_matched"].numpy(), bins=80, alpha=0.6, label="matched (image, caption)")
        ax.hist(s["_unmatched"].numpy(), bins=80, alpha=0.6, label="random pairs")
        ax.set_xlabel("cosine similarity")
        ax.set_title(f"CLIP modality gap — centroid dist {s['centroid_distance']:.3f}")
        ax.legend()
        fig.tight_layout()
        fig.savefig(out / "cosine_hist.png", dpi=150)
    except ImportError:
        pass


if __name__ == "__main__":
    main()
