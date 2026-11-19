"""Model / dataset resolution. ModelScope first (HF is unreachable on the training server).

Order is controlled by MMS_HUB_ORDER (default "modelscope,hf"). The HF fallback honours HF_ENDPOINT
(e.g. https://hf-mirror.com). Downloads land in storage/models or storage/raw and are marked complete
with a sentinel file so later calls are offline.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Sequence

from .misc import get_logger
from .paths import models_dir, raw_dir

log = get_logger(__name__)

# Canonical (HF-style) id -> ModelScope id when they differ. Same-id repos need no entry.
MODELSCOPE_ALIASES: dict[str, str] = {
    "openai/clip-vit-large-patch14": "AI-ModelScope/clip-vit-large-patch14",
    "openai/clip-vit-base-patch32": "AI-ModelScope/clip-vit-base-patch32",
    "openai/clip-vit-large-patch14-336": "AI-ModelScope/clip-vit-large-patch14-336",
}

_SENTINEL = ".mms_complete"
# never needed by this project (PyTorch + safetensors only); CLIP repos ship flax/tf/bin copies of the same weights
MODEL_IGNORE = ["*.msgpack", "*.h5", "*.ot", "*.onnx", "onnx/*", "*.tflite", "coreml/*"]
SAFETENSORS_ONLY = {"openai/clip-vit-large-patch14", "openai/clip-vit-base-patch32", "openai/clip-vit-large-patch14-336"}


def _order() -> list[str]:
    return [s.strip() for s in os.environ.get("MMS_HUB_ORDER", "modelscope,hf").split(",") if s.strip()]


def _download(repo_id: str, local: Path, repo_type: str, allow_patterns: Sequence[str] | None, revision: str | None) -> None:
    errors = []
    for source in _order():
        try:
            if source == "modelscope":
                from modelscope.hub.snapshot_download import snapshot_download

                ms_id = MODELSCOPE_ALIASES.get(repo_id, repo_id)
                kwargs = dict(local_dir=str(local), allow_patterns=list(allow_patterns) if allow_patterns else None)
                if repo_type == "dataset":
                    kwargs["repo_type"] = "dataset"
                else:
                    kwargs["ignore_patterns"] = MODEL_IGNORE + (["pytorch_model*.bin"] if repo_id in SAFETENSORS_ONLY else [])
                if revision:
                    kwargs["revision"] = revision
                log.info("downloading %s %s from ModelScope -> %s", repo_type, ms_id, local)
                snapshot_download(ms_id, **kwargs)
            elif source == "hf":
                from huggingface_hub import snapshot_download

                log.info("downloading %s %s from HF(%s) -> %s", repo_type, repo_id, os.environ.get("HF_ENDPOINT", "default"), local)
                snapshot_download(
                    repo_id,
                    repo_type=repo_type,
                    local_dir=str(local),
                    allow_patterns=list(allow_patterns) if allow_patterns else None,
                    ignore_patterns=None if repo_type == "dataset" else MODEL_IGNORE,
                    revision=revision,
                )
            else:
                raise ValueError(f"unknown hub source {source}")
            return
        except Exception as e:  # noqa: BLE001 - try next source
            log.warning("%s download failed for %s: %r", source, repo_id, e)
            errors.append(f"{source}: {e!r}")
    raise RuntimeError(f"could not download {repo_type} {repo_id}:\n" + "\n".join(errors))


def _resolve(repo_id: str, root: Path, repo_type: str, allow_patterns, revision, tag: str | None) -> str:
    p = Path(os.path.expanduser(repo_id))
    if p.exists():
        return str(p)
    local = root / (repo_id.replace("/", "__") + (f"__{tag}" if tag else ""))
    sentinel = local / _SENTINEL
    if sentinel.exists():
        return str(local)
    local.mkdir(parents=True, exist_ok=True)
    _download(repo_id, local, repo_type, allow_patterns, revision)
    sentinel.write_text("ok")
    return str(local)


def resolve_model(model_id: str, allow_patterns: Sequence[str] | None = None, revision: str | None = None) -> str:
    """Return a local directory for `model_id` (local path, or hub id downloaded ModelScope-first)."""
    return _resolve(model_id, models_dir(), "model", allow_patterns, revision, None)


def resolve_dataset(
    repo_id: str, allow_patterns: Sequence[str] | None = None, revision: str | None = None, tag: str | None = None
) -> str:
    """Return a local directory with the dataset's raw files (parquet / tar ...).

    `tag` distinguishes partial downloads with different allow_patterns (e.g. "test" vs "train").
    """
    return _resolve(repo_id, raw_dir(), "dataset", allow_patterns, revision, tag)


def main() -> None:
    import argparse

    ap = argparse.ArgumentParser(description="Pre-download models/datasets (ModelScope first).")
    ap.add_argument("--model", nargs="*", default=[])
    ap.add_argument("--dataset", nargs="*", default=[])
    args = ap.parse_args()
    for m in args.model:
        print(m, "->", resolve_model(m))
    for d in args.dataset:
        print(d, "->", resolve_dataset(d))


if __name__ == "__main__":
    main()
