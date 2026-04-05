from __future__ import annotations

import csv
from pathlib import Path

from PIL import Image

from lbd.config import dump_yaml
from lbd.data.ingest import run_ingest


def _write_image(path: Path, color: tuple[int, int, int]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    image = Image.new("RGB", (32, 24), color=color)
    image.save(path)


def test_local_source_prefers_sidecar_then_manifest_then_fallback(tmp_path: Path) -> None:
    repo = tmp_path
    image_dir = repo / "data" / "raw" / "local_objects"

    sidecar_image = image_dir / "artifact_sidecar.jpg"
    manifest_image = image_dir / "artifact_manifest.jpg"
    fallback_image = image_dir / "artifact_fallback.jpg"

    _write_image(sidecar_image, (10, 20, 30))
    _write_image(manifest_image, (40, 50, 60))
    _write_image(fallback_image, (70, 80, 90))

    sidecar_image.with_suffix(".txt").write_text(
        "two-handled ceramic vessel on white plinth",
        encoding="utf-8",
    )

    captions_csv = image_dir / "captions.csv"
    with captions_csv.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["relative_path", "caption"])
        writer.writeheader()
        writer.writerow(
            {
                "relative_path": "artifact_manifest.jpg",
                "caption": "small carved ceramic cup",
            }
        )

    ingest_config = {
        "output_index_csv": "data/local/meta/ingest_index.csv",
        "threads": 2,
        "sources": [
            {
                "id": "local_objects",
                "type": "local",
                "enabled": True,
                "raw_dir": "data/raw/local_objects",
                "local_globs": ["*.jpg"],
                "captions_csv": "captions.csv",
                "fallback_caption": "generic ceramic object",
            }
        ],
    }
    config_path = repo / "configs" / "local_sources.yaml"
    config_path.parent.mkdir(parents=True, exist_ok=True)
    dump_yaml(ingest_config, config_path)

    output_index = run_ingest(config_path, repo_root=repo)
    with output_index.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))

    captions = {row["relative_path"]: row["caption"] for row in rows}
    assert captions["artifact_sidecar.jpg"] == "two-handled ceramic vessel on white plinth"
    assert captions["artifact_manifest.jpg"] == "small carved ceramic cup"
    assert captions["artifact_fallback.jpg"] == "generic ceramic object"
