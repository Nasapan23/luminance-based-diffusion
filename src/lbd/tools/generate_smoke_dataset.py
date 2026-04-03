from __future__ import annotations

import argparse
import csv
import json
import random
from pathlib import Path

from PIL import Image, ImageDraw


def _rand_color(rng: random.Random) -> tuple[int, int, int]:
    return (rng.randint(0, 255), rng.randint(0, 255), rng.randint(0, 255))


def _draw_pattern(image: Image.Image, rng: random.Random) -> None:
    draw = ImageDraw.Draw(image)
    width, height = image.size
    for _ in range(6):
        x0 = rng.randint(0, width - 1)
        y0 = rng.randint(0, height - 1)
        x1 = rng.randint(x0, width - 1)
        y1 = rng.randint(y0, height - 1)
        draw.rectangle([x0, y0, x1, y1], outline=_rand_color(rng), width=1)


def _make_image(path: Path, size: tuple[int, int], seed: int) -> None:
    rng = random.Random(seed)
    image = Image.new("RGB", size, color=_rand_color(rng))
    _draw_pattern(image, rng)
    path.parent.mkdir(parents=True, exist_ok=True)
    image.save(path, quality=92)


def _build_coco(root: Path, count: int, size: tuple[int, int], seed: int) -> None:
    rng = random.Random(seed)
    base = root / "data/raw_smoke/coco"
    images_dir = base / "images/train2017"
    annotations_dir = base / "annotations"
    annotations_dir.mkdir(parents=True, exist_ok=True)

    images = []
    annotations = []
    for idx in range(count):
        image_id = idx + 1
        file_name = f"{image_id:012d}.jpg"
        _make_image(images_dir / file_name, size, seed=rng.randint(0, 10_000_000))
        images.append({"id": image_id, "file_name": f"images/train2017/{file_name}"})
        annotations.append({"image_id": image_id, "caption": f"smoke coco scene {image_id}"})

    payload = {"images": images, "annotations": annotations}
    (annotations_dir / "captions_train2017.json").write_text(
        json.dumps(payload),
        encoding="utf-8",
    )


def _build_places2(root: Path, count: int, size: tuple[int, int], seed: int) -> None:
    rng = random.Random(seed)
    base = root / "data/raw_smoke/places2/images/train"
    for idx in range(count):
        stem = f"places_{idx:06d}"
        image_path = base / f"{stem}.jpg"
        _make_image(image_path, size, seed=rng.randint(0, 10_000_000))
        image_path.with_suffix(".txt").write_text(
            f"smoke places scene {idx}",
            encoding="utf-8",
        )


def _build_openimages(root: Path, count: int, size: tuple[int, int], seed: int) -> None:
    rng = random.Random(seed)
    images_dir = root / "data/raw_smoke/openimages/images/train"
    metadata_dir = root / "data/raw_smoke/openimages/metadata"
    metadata_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    for idx in range(count):
        image_id = f"open_{idx:06d}"
        _make_image(images_dir / f"{image_id}.jpg", size, seed=rng.randint(0, 10_000_000))
        rows.append({"ImageID": image_id, "caption": f"smoke open image {idx}"})

    csv_path = metadata_dir / "captions.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["ImageID", "caption"])
        writer.writeheader()
        writer.writerows(rows)


def generate_smoke_dataset(
    repo_root: Path,
    per_source: int = 400,
    width: int = 128,
    height: int = 128,
    seed: int = 2026,
) -> None:
    size = (width, height)
    _build_coco(repo_root, count=per_source, size=size, seed=seed + 11)
    _build_places2(repo_root, count=per_source, size=size, seed=seed + 23)
    _build_openimages(repo_root, count=per_source, size=size, seed=seed + 37)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="lbd-smoke-data",
        description="Generate a synthetic 3-source dataset for smoke tests.",
    )
    parser.add_argument("--repo-root", type=Path, default=Path.cwd())
    parser.add_argument("--per-source", type=int, default=400)
    parser.add_argument("--width", type=int, default=128)
    parser.add_argument("--height", type=int, default=128)
    parser.add_argument("--seed", type=int, default=2026)
    args = parser.parse_args(argv)

    generate_smoke_dataset(
        repo_root=args.repo_root.resolve(),
        per_source=args.per_source,
        width=args.width,
        height=args.height,
        seed=args.seed,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

