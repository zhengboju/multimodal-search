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


def pair_similarities(img: torch.Tensor, txt: torch.Tensor, t2i: torch.Tensor, seed: int = 0) -> tuple[torch.Tensor, torch.Tensor]:
    """Cosine of matched (caption, its image) pairs and of random pairs — the discrimination budget."""
    matched = (txt * img[t2i]).sum(-1)
    g = torch.Generator().manual_seed(seed)
    unmatched = (txt * img[torch.randint(0, img.shape[0], (txt.shape[0],), generator=g)]).sum(-1)
    return matched, unmatched


def gap_stats(img: torch.Tensor, txt: torch.Tensor, t2i: torch.Tensor, seed: int = 0) -> dict:
    mi, mt = raw_centroid(img), raw_centroid(txt)
    mi_u, mt_u = F.normalize(mi, dim=0), F.normalize(mt, dim=0)
    matched, unmatched = pair_similarities(img, txt, t2i, seed)
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


def build_verdict(by: dict[str, dict], s: dict, eps: float = 2.0) -> list[str]:
    """Turn the measurements into the conclusion, instead of asserting one in advance."""
    base = by.get("original")
    if base is None:
        return []
    out: list[str] = []
    tr = by.get("translate")
    if tr is not None:
        di, dt = tr["i2t_R@1"] - base["i2t_R@1"], tr["t2i_R@1"] - base["t2i_R@1"]
        if min(di, dt) < -eps:
            out.append(
                f"**Post-hoc gap removal is destructive here: `translate` costs {di:+.1f} i2t / {dt:+.1f} t2i R@1 and "
                f"shrinks the matched−unmatched margin from {base['margin']:.3f} to {tr['margin']:.3f}.** "
                f"Matched pairs are offset {s['offset_along_gap_mean']:.2f} (cos) along the gap axis, so that axis carries "
                "the match signal — subtracting it is not post-processing, it is deleting information. "
                "Gap/cone must be addressed in the training objective, not in embedding space."
            )
        elif max(di, dt) > eps:
            out.append(
                f"`translate` gains {di:+.1f} i2t / {dt:+.1f} t2i R@1 (margin {base['margin']:.3f} -> {tr['margin']:.3f}): "
                "the gap is a removable nuisance in this checkpoint, so simple post-processing is worth keeping."
            )
        else:
            out.append(
                f"`translate` moves recall by only {di:+.1f} / {dt:+.1f} points: the gap is not what limits retrieval; "
                f"the loss happens inside each modality (cone tightness {s['img_cone_norm']:.2f} vs {s['txt_cone_norm']:.2f})."
            )
    ct, ci = by.get("center_txt"), by.get("center_img")
    if ct is not None and ci is not None:
        asym = (ct["i2t_R@1"] - base["i2t_R@1"]) - (ci["i2t_R@1"] - base["i2t_R@1"])
        out.append(
            f"Side asymmetry: centring only the texts changes i2t by {ct['i2t_R@1'] - base['i2t_R@1']:+.1f} "
            f"(t2i {ct['t2i_R@1'] - base['t2i_R@1']:+.1f}); centring only the images changes i2t by "
            f"{ci['i2t_R@1'] - base['i2t_R@1']:+.1f} (t2i {ci['t2i_R@1'] - base['t2i_R@1']:+.1f}). "
            f"The two sides are {'not' if abs(asym) > eps else 'roughly equally'} symmetric under surgery — "
            "the text cloud is the more spread one and behaves differently as candidate vs query."
        )
    out.append(
        f"Content budget: margin {s['content_margin']:.3f} against a centroid offset of {s['centroid_distance']:.3f}. "
        "That ratio, not the gap itself, is the headroom a single-vector dual encoder has on this data "
        "(and the reason late-interaction / multi-vector retrieval is worth its storage cost)."
    )
    return out


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
    def cond_row(name: str, r: dict) -> dict:
        mi, mt = pair_similarities(*apply_condition(name, img, txt), t2i)
        return {"condition": name, "i2t_R@1": r["i2t_R@1"], "t2i_R@1": r["t2i_R@1"], "rsum": r["rsum"],
                "matched": float(mi.mean()), "unmatched": float(mt.mean()), "margin": float((mi - mt).mean())}

    rows = [cond_row(c, r) for c, r in results.items()]
    by = {r["condition"]: r for r in rows}
    verdict = build_verdict(by, s)
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
        markdown_table(rows, ["condition", "i2t_R@1", "t2i_R@1", "rsum", "matched", "unmatched", "margin"], digits=3),
        "",
        "* `translate` = the textbook gap removal (constant shift of the text cloud onto the image centroid).",
        "* `margin` = matched − unmatched similarity under that intervention: what the ranking actually has to work with.",
        "",
        *verdict,
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
