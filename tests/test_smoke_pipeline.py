from __future__ import annotations

import csv
import json
from pathlib import Path

from PIL import Image

from lbd.config import dump_yaml
from lbd.data.build_base100k import run_build_base100k
from lbd.data.ingest import run_ingest


def _write_image(path: Path, color: tuple[int, int, int]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    image = Image.new("RGB", (32, 24), color=color)
    image.save(path)


def test_smoke_ingest_and_build(tmp_path: Path) -> None:
    repo = tmp_path

    coco_dir = repo / "data/raw/coco"
    places_dir = repo / "data/raw/places2"
    open_dir = repo / "data/raw/openimages"

    # COCO mini data
    coco_images = coco_dir / "images/train2017"
    for idx in range(12):
        file_name = f"{idx:012d}.jpg"
        _write_image(coco_images / file_name, (idx, 40, 80))
    coco_annotations = {
        "images": [{"id": idx, "file_name": f"images/train2017/{idx:012d}.jpg"} for idx in range(12)],
        "annotations": [{"image_id": idx, "caption": f"coco caption {idx}"} for idx in range(12)],
    }
    ann_path = coco_dir / "annotations/captions_train2017.json"
    ann_path.parent.mkdir(parents=True, exist_ok=True)
    ann_path.write_text(json.dumps(coco_annotations), encoding="utf-8")

    # Places2 mini data
    places_images = places_dir / "images/train"
    for idx in range(8):
        image_path = places_images / f"places_{idx}.jpg"
        _write_image(image_path, (40, idx * 10, 60))
        image_path.with_suffix(".txt").write_text(f"places caption {idx}", encoding="utf-8")

    # OpenImages mini data
    open_images = open_dir / "images/train"
    for idx in range(10):
        _write_image(open_images / f"open_{idx}.jpg", (50, 90, idx * 7))
    captions_csv = open_dir / "metadata/captions.csv"
    captions_csv.parent.mkdir(parents=True, exist_ok=True)
    with captions_csv.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["ImageID", "caption"])
        writer.writeheader()
        for idx in range(10):
            writer.writerow({"ImageID": f"open_{idx}", "caption": f"open caption {idx}"})

    configs_dir = repo / "configs"
    configs_dir.mkdir(parents=True, exist_ok=True)

    ingest_config = {
        "output_index_csv": "data/base100k/meta/ingest_index.csv",
        "threads": 4,
        "sources": [
            {
                "id": "coco",
                "type": "coco",
                "enabled": True,
                "raw_dir": "data/raw/coco",
                "local_globs": ["images/**/*.jpg"],
                "captions_json": "annotations/captions_train2017.json",
                "fallback_caption": "fallback coco",
            },
            {
                "id": "places2",
                "type": "places2",
                "enabled": True,
                "raw_dir": "data/raw/places2",
                "local_globs": ["images/**/*.jpg"],
                "fallback_caption": "fallback places",
            },
            {
                "id": "openimages",
                "type": "openimages",
                "enabled": True,
                "raw_dir": "data/raw/openimages",
                "local_globs": ["images/**/*.jpg"],
                "captions_csv": "metadata/captions.csv",
                "fallback_caption": "fallback open",
            },
        ],
    }
    ingest_config_path = configs_dir / "sources.yaml"
    dump_yaml(ingest_config, ingest_config_path)

    build_config = {
        "run_name": "smoke",
        "runs_root": "runs",
        "input_index_csv": "data/base100k/meta/ingest_index.csv",
        "output_root": "data/base100k",
        "target_total": 20,
        "split_counts": {"train": 14, "val": 3, "test": 3},
        "seed": 123,
        "threads": 4,
        "batch_size": 5,
        "copy_mode": "copy",
        "dedupe_by_sha256": True,
    }
    build_config_path = configs_dir / "build_base100k.yaml"
    dump_yaml(build_config, build_config_path)

    output_index = run_ingest(ingest_config_path, repo_root=repo)
    assert output_index.exists()

    run_id = run_build_base100k(build_config_path, repo_root=repo)
    assert run_id

    final_index = repo / "data/base100k/meta/index.csv"
    assert final_index.exists()
    with final_index.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == 20
    assert all(row["status"] == "done" for row in rows)

    split_counts = {"train": 0, "val": 0, "test": 0}
    for row in rows:
        split_counts[row["split"]] += 1
        assert (repo / row["color_path"]).exists()
        assert (repo / row["gray_path"]).exists()
        assert row["caption"]
    assert split_counts == {"train": 14, "val": 3, "test": 3}

