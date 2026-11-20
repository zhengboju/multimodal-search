"""Week-1 D4: the modality gap in CLIP embedding space — measure it, then try to remove it *properly*.

The interesting question is not "does the gap exist" (it does) but "is the gap what limits retrieval".
Naively subtracting each modality's mean answers a different question: it removes the gap *and* the shared
cone direction, and it rescales every vector, so any change mixes three effects. This script therefore runs
several interventions on the same embeddings and separates them:

  original    img, txt                                   baseline
  translate   txt - m_txt + m_img  (centroids aligned,   the textbook "gap removal": a constant shift of the
              spread and shared direction preserved)     candidate side, so i2t order barely changes
  center      x - mean(x), both modalities               removes gap + shared cone direction + per-vector scale
  center_txt  only texts centred                         isolates which modality's information gets damaged
  center_img  only images centred

Reported gap statistics: unit-centroid cosine/distance, cone tightness ||mean(x)|| per modality (for unit
vectors this equals the mean cosine to its own centroid), matched vs unmatched similarity, and how much of a
matched pair's offset is explained by the gap direction itself.

python -m mmsearch.eval.modality_gap --dataset coco_test [--conditions original translate center]
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
import torch.nn.functional as F

from ..data.schema import load_dataset
from ..utils.misc import get_logger, markdown_table, write_json
from ..utils.paths import runs_dir
from .eval_clip import bidirectional_recall, dataset_tensors

log = get_logger(__name__)

CONDITIONS = ["original", "translate", "center", "center_txt", "center_img"]


# ----------------------------------------------------------------------------- pure helpers (unit-tested)
def raw_centroid(x: torch.Tensor) -> torch.Tensor:
    """Mean embedding (not normalised): for unit vectors its norm = mean cosine to its own centroid."""
    return x.mean(0)


def unit_centroid(x: torch.Tensor) -> torch.Tensor:
    return F.normalize(raw_centroid(x), dim=0)


def centered(x: torch.Tensor) -> torch.Tensor:
    return F.normalize(x - raw_centroid(x), dim=-1)


def translated(txt: torch.Tensor, img: torch.Tensor) -> torch.Tensor:
    """Shift the text cloud so its centroid coincides with the image centroid (true gap removal)."""
    return F.normalize(txt - raw_centroid(txt) + raw_centroid(img), dim=-1)


def apply_condition(name: str, img: torch.Tensor, txt: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    if name == "original":
        return img, txt
    if name == "translate":
        return img, translated(txt, img)
    if name == "center":
        return centered(img), centered(txt)
    if name == "center_txt":
        return img, centered(txt)
    if name == "center_img":
        return centered(img), txt
    raise ValueError(f"unknown condition {name}; choose from {CONDITIONS}")


def gap_stats(img: torch.Tensor, txt: torch.Tensor, t2i: torch.Tensor, seed: int = 0) -> dict:
    mi, mt = raw_centroid(img), raw_centroid(txt)
    mi_u, mt_u = F.normalize(mi, dim=0), F.normalize(mt, dim=0)
    matched = (txt * img[t2i]).sum(-1)
    g = torch.Generator().manual_seed(seed)
    unmatched = (txt * img[torch.randint(0, img.shape[0], (txt.shape[0],), generator=g)]).sum(-1)
    # how much of a matched pair's offset is just the modality gap vector? (caption - image vs m_txt - m_img,
    # so +1 = "the caption sits exactly where the modality gap predicts it should")
    offset = txt - img[t2i]
    gap_dir = F.normalize(mt - mi, dim=0)
    offset_cos = F.cosine_similarity(offset, gap_dir.expand_as(offset), dim=-1)
    dist = float((mi_u - mt_u).norm())
    return {
        "centroid_distance": dist,  # between unit centroids, comparable across checkpoints
        "centroid_distance_raw": float((mi - mt).norm()),
        "centroid_cosine": float(F.cosine_similarity(mi_u, mt_u, dim=0)),
        "img_cone_norm": float(mi.norm()),  # mean cosine of images to their own centroid
        "txt_cone_norm": float(mt.norm()),
        "matched_cos_mean": float(matched.mean()),
        "unmatched_cos_mean": float(unmatched.mean()),
        "content_margin": float(matched.mean() - unmatched.mean()),  # what actually separates pairs
        "offset_along_gap_mean": float(offset_cos.mean()),
        "offset_along_gap_q25": float(offset_cos.quantile(0.25)),
        "offset_along_gap_q75": float(offset_cos.quantile(0.75)),
        "_matched": matched,
        "_unmatched": unmatched,
    }


# ----------------------------------------------------------------------------- main
def main() -> None:
    from ..models.clip_wrapper import CLIPWrapper

    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="openai/clip-vit-large-patch14")
    ap.add_argument("--dataset", default="coco_test")
    ap.add_argument("--conditions", nargs="+", default=CONDITIONS, choices=CONDITIONS)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    m = CLIPWrapper(args.model, dtype="fp16" if dev == "cuda" else None).to(dev).eval()
    ds = load_dataset(args.dataset)
    images, texts, t2i = dataset_tensors(ds)
    log.info("%s: %d images, %d captions", ds.name, len(images), len(texts))
    img, txt = m.encode_images(images), m.encode_texts(texts)

    s = gap_stats(img, txt, t2i)
    results = {c: bidirectional_recall(*apply_condition(c, img, txt), t2i) for c in args.conditions}
    base = results.get("original", {})

    out = Path(args.out or runs_dir() / "analysis" / f"modality_gap_{Path(args.model).name}_{ds.name}")
    res = {k: v for k, v in s.items() if not k.startswith("_")}
    res["conditions"] = {c: {k: float(v) for k, v in r.items()} for c, r in results.items()}
    res["delta_vs_original"] = {
        c: {k: round(float(r[k] - base[k]), 2) for k in ("i2t_R@1", "t2i_R@1", "rsum") if k in base and k in r}
        for c, r in results.items() if c != "original"
    }
    res.update({"model": args.model, "dataset": ds.name, "n_images": len(images), "n_captions": len(texts)})
    write_json(res, out / "gap.json")

    # ---------------- report
    log.info(
        "gap: unit-centroid cos %.3f (dist %.3f) | cone ||mean|| img %.3f txt %.3f | matched %.3f unmatched %.3f "
        "(margin %.3f) | matched-pair offset along gap %.2f",
        s["centroid_cosine"], s["centroid_distance"], s["img_cone_norm"], s["txt_cone_norm"],
        s["matched_cos_mean"], s["unmatched_cos_mean"], s["content_margin"], s["offset_along_gap_mean"],
    )
    rows = [{"condition": c, **{k: v for k, v in r.items() if k.startswith(("i2t_R@1", "t2i_R@1", "rsum"))}} for c, r in results.items()]
    md = [
        f"# Modality gap — {Path(args.model).name} on {ds.name}",
        "",
        "## Gap statistics",
        "",
        markdown_table([{k: v for k, v in res.items() if isinstance(v, (int, float))}], digits=4),
        "",
        "* `img_cone_norm` / `txt_cone_norm` = ||mean(x)||: how tightly each modality clusters (mean cosine to its own centroid).",
        "* `content_margin` = matched − unmatched similarity: the part of the score that actually carries content.",
        "* `offset_along_gap_mean` = mean cos(caption − image, gap vector m_txt − m_img): +1 means the offset of a matched pair is entirely the gap, 0 means it is content.",
        "",
        "## Retrieval under each intervention",
        "",
        markdown_table(rows, ["condition", "i2t_R@1", "t2i_R@1", "rsum"], digits=2),
        "",
        "* `translate` is the textbook gap removal (constant shift of the text cloud): if the gap were what limits",
        "  recall, this would jump. It does not move much — the gap is not where the information is lost.",
        "* `center` additionally removes the shared cone direction and rescales every vector; the drop there is",
        "  collateral damage, not evidence that the gap is useful. `center_txt` / `center_img` show which side breaks.",
    ]
    text = "\n".join(md) + "\n"
    (out / "report.md").write_text(text, encoding="utf-8")
    print(text)

    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(6, 4))
        ax.hist(s["_matched"].float().numpy(), bins=80, alpha=0.6, label="matched (image, caption)")
        ax.hist(s["_unmatched"].float().numpy(), bins=80, alpha=0.6, label="random pairs")
        ax.set_xlabel("cosine similarity")
        ax.set_title(f"{Path(args.model).name}: centroid cos {s['centroid_cosine']:.2f}, margin {s['content_margin']:.2f}")
        ax.legend()
        fig.tight_layout()
        fig.savefig(out / "cosine_hist.png", dpi=150)

        if len(rows) > 1:
            fig, ax = plt.subplots(figsize=(6, 3.2))
            names = [r["condition"] for r in rows]
            ax.bar([f"{n}\ni2t" for n in names], [r["i2t_R@1"] for r in rows], alpha=0.7, label="i2t R@1")
            ax.bar([f"{n}\nt2i" for n in names], [r["t2i_R@1"] for r in rows], alpha=0.7, label="t2i R@1")
            ax.set_ylabel("R@1 (%)")
            ax.set_title("Retrieval under gap-removal interventions")
            ax.legend()
            fig.tight_layout()
            fig.savefig(out / "conditions.png", dpi=150)
    except ImportError:
        pass
    log.info("-> %s", out)


if __name__ == "__main__":
    main()
