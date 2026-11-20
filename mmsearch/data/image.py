"""Image decoding helpers."""

from __future__ import annotations

from PIL import Image

_JPEG = {"JPEG", "MPO"}


def load_rgb(path: str, size: int | None = None) -> Image.Image:
    """Open an image as RGB.

    `size` enables JPEG draft decoding: libjpeg decodes at the smallest scale whose *both* sides still cover
    `size` (1/2, 1/4, 1/8 ...). For COCO/Flickr (~640x480 -> 320x240 at 1/2) that is a measured ~20-25% saving
    on a 224² target — worth having, but the big win for an idle GPU is parallel decode (see
    utils/dataloader.py), not draft. Note the target must be <= min(w,h)/2 to trigger any reduction: asking for
    256 on a 640x480 image is a no-op because 240 < 256. Leave `size=None` for evaluation so numbers stay
    comparable with full-decode baselines.
    """
    im = Image.open(path)
    if size and (im.format or "").upper() in _JPEG:
        im.draft("RGB", (int(size), int(size)))
    return im.convert("RGB")
