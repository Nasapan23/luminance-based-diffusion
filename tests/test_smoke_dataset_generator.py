from __future__ import annotations

import csv
import json
from pathlib import Path

from lbd.tools.generate_smoke_dataset import generate_smoke_dataset


def test_generate_smoke_dataset_layout(tmp_path: Path) -> None:
    generate_smoke_dataset(tmp_path, per_source=10, width=32, height=32, seed=1)

    coco_images = list((tmp_path / "data/raw_smoke/coco/images/train2017").glob("*.jpg"))
    places_images = list((tmp_path / "data/raw_smoke/places2/images/train").glob("*.jpg"))
    open_images = list((tmp_path / "data/raw_smoke/openimages/images/train").glob("*.jpg"))
    assert len(coco_images) == 10
    assert len(places_images) == 10
    assert len(open_images) == 10

    coco_ann = tmp_path / "data/raw_smoke/coco/annotations/captions_train2017.json"
    payload = json.loads(coco_ann.read_text(encoding="utf-8"))
    assert len(payload["images"]) == 10
    assert len(payload["annotations"]) == 10

    open_csv = tmp_path / "data/raw_smoke/openimages/metadata/captions.csv"
    with open_csv.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == 10

