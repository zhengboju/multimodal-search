"""Image-text retrieval evaluation, standard COCO/Flickr protocol (Karpathy test split).

  i2t R@K : fraction of images whose best-ranked GT caption (any of ~5) is within top-K
  t2i R@K : fraction of captions whose GT image is within top-K
  COCO 1K : mean over 5 folds of 1000 images; COCO 5K: full set.

python -m mmsearch.eval.eval_clip --model openai/clip-vit-large-patch14 --dataset coco_test --folds 5
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch

from ..data.schema import RetrievalDataset, load_dataset
from ..utils.misc import get_logger, markdown_table, write_json
from ..utils.paths import runs_dir
from .ledger import record_test_eval

log = get_logger(__name__)


def bidirectional_recall(img: torch.Tensor, txt: torch.Tensor, txt2img: torch.Tensor, ks=(1, 5, 10)) -> dict[str, float]:
    """img [N, D], txt [M, D] (normalised), txt2img [M] image index of each caption."""
    sims = img.float() @ txt.float().t()  # [N, M]
    N = img.shape[0]
    # i2t: rank of the best GT caption for each image
    order = sims.argsort(dim=1, descending=True)  # [N, M]
    gt = txt2img[order] == torch.arange(N).unsqueeze(1)  # [N, M] bool at ranked positions
    i2t_rank = gt.float().argmax(dim=1)  # first True position
    # t2i
    order_t = sims.t().argsort(dim=1, descending=True)  # [M, N]
    t2i_rank = (order_t == txt2img.unsqueeze(1)).float().argmax(dim=1)
    out = {}
    for k in ks:
        out[f"i2t_R@{k}"] = float((i2t_rank < k).float().mean() * 100)
        out[f"t2i_R@{k}"] = float((t2i_rank < k).float().mean() * 100)
    out["rsum"] = sum(v for kk, v in out.items() if "R@" in kk)
    return out


def dataset_tensors(ds: RetrievalDataset) -> tuple[list[str], list[str], torch.Tensor]:
    img_ids = list(ds.corpus)
    pos = {d: i for i, d in enumerate(img_ids)}
    qids = list(ds.queries)
    texts = [ds.queries[q].text or "" for q in qids]
    txt2img = torch.tensor([pos[next(iter(ds.qrels[q]))] for q in qids])
    return [ds.corpus[d].image for d in img_ids], texts, txt2img


def evaluate_encoder(encode_images, encode_texts, ds: RetrievalDataset, folds: int = 1) -> dict[str, float]:
    images, texts, txt2img = dataset_tensors(ds)
    img = encode_images(images)
    txt = encode_texts(texts)
    if folds <= 1:
        return bidirectional_recall(img, txt, txt2img)
    N = img.shape[0]
    size = N // folds
    agg: dict[str, list[float]] = {}
    for f in range(folds):
        lo, hi = f * size, (f + 1) * size
        sel = ((txt2img >= lo) & (txt2img < hi)).nonzero(as_tuple=True)[0]
        r = bidirectional_recall(img[lo:hi], txt[sel], txt2img[sel] - lo)
        for k, v in r.items():
            agg.setdefault(k, []).append(v)
    return {k: sum(v) / len(v) for k, v in agg.items()}


def main() -> None:
    from ..models.clip_wrapper import CLIPWrapper

    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="openai/clip-vit-large-patch14", help="hub id or fine-tuned checkpoint dir")
    ap.add_argument("--dataset", default="coco_test")
    ap.add_argument("--folds", type=int, nargs="+", default=[1, 5], help="1 = 5K (COCO) / 1K (Flickr); 5 = COCO 1K protocol")
    ap.add_argument("--out", default=None)
    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument("--num-workers", type=int, default=8, help="image-decode threads (no /dev/shm needed)")
    args = ap.parse_args()

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    model = CLIPWrapper(args.model, dtype="fp16" if dev == "cuda" else None).to(dev).eval()
    ds = load_dataset(args.dataset)
    images, texts, _ = dataset_tensors(ds)
    img = model.encode_images(images, args.batch_size, num_workers=args.num_workers)
    txt = model.encode_texts(texts, args.batch_size)
    rows = []
    for f in args.folds:
        res = evaluate_encoder(lambda _: img, lambda _: txt, ds, f)
        protocol = f"{len(images) // f}" + ("-img" if f == 1 else f"-img x{f} folds")
        rows.append({"model": Path(args.model).name, "dataset": ds.name, "protocol": protocol, **res})
        record_test_eval(ds.name, f"clip:{args.model}", res)
    out = Path(args.out or runs_dir() / "clip_eval" / f"{Path(args.model).name}_{ds.name}")
    write_json(rows, out / "results.json")
    md = markdown_table(rows, ["model", "dataset", "protocol", "i2t_R@1", "i2t_R@5", "i2t_R@10", "t2i_R@1", "t2i_R@5", "t2i_R@10", "rsum"])
    (out / "results.md").write_text(md, encoding="utf-8")
    print(md)


if __name__ == "__main__":
    main()
