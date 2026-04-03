from __future__ import annotations

import os
import tempfile
from pathlib import Path

import numpy as np
from PIL import Image, UnidentifiedImageError


def inspect_image(path: Path) -> tuple[bool, int, int, str]:
    try:
        with Image.open(path) as image:
            image.verify()
        with Image.open(path) as image:
            width, height = image.size
        return True, width, height, ""
    except (UnidentifiedImageError, OSError, ValueError) as exc:
        return False, 0, 0, str(exc)


def grayscale_rgb(image: Image.Image) -> Image.Image:
    """
    Photoreal grayscale conversion:
    1) convert sRGB -> linear RGB
    2) compute luminance with Rec.709 coefficients
    3) convert linear luminance -> sRGB
    4) replicate channel to RGB
    """
    rgb = np.asarray(image.convert("RGB"), dtype=np.float32) / 255.0

    # sRGB -> linear RGB
    linear = np.where(
        rgb <= 0.04045,
        rgb / 12.92,
        ((rgb + 0.055) / 1.055) ** 2.4,
    )

    # Rec.709 luminance in linear space
    y_linear = (
        0.2126 * linear[..., 0]
        + 0.7152 * linear[..., 1]
        + 0.0722 * linear[..., 2]
    )

    # linear -> sRGB
    y_srgb = np.where(
        y_linear <= 0.0031308,
        12.92 * y_linear,
        1.055 * np.power(y_linear, 1.0 / 2.4) - 0.055,
    )

    y_u8 = np.clip(np.round(y_srgb * 255.0), 0, 255).astype(np.uint8)
    gray_rgb = np.stack((y_u8, y_u8, y_u8), axis=-1)
    return Image.fromarray(gray_rgb)


def save_grayscale_rgb(src: Path, dst: Path, quality: int = 95) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(dir=str(dst.parent), prefix=f"{dst.name}.tmp.")
    os.close(fd)
    tmp_path = Path(tmp_name)
    try:
        with Image.open(src) as image:
            gray_rgb = grayscale_rgb(image)
            gray_rgb.save(tmp_path, format="JPEG", quality=quality)
        os.replace(tmp_path, dst)
    finally:
        if tmp_path.exists():
            tmp_path.unlink(missing_ok=True)
