from __future__ import annotations

from PIL import Image

from lbd.utils.image_utils import grayscale_rgb


def test_grayscale_rgb_channels_are_equal() -> None:
    image = Image.new("RGB", (3, 1))
    image.putdata([(255, 0, 0), (0, 255, 0), (0, 0, 255)])

    gray_image = grayscale_rgb(image)
    assert gray_image.mode == "RGB"

    for pixel in gray_image.getdata():
        r, g, b = pixel
        assert r == g == b

