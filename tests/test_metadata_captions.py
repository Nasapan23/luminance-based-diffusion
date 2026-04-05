from __future__ import annotations

import json

from lbd.data.build_base100k import _write_metadata_jsonl


def test_color_metadata_uses_amphora_keyword(tmp_path) -> None:
    rows = [
        {
            "source_id": "amphora",
            "caption": "ceramic vessel",
            "color_path": "data/vaze_bw/color/train/amphora_001.jpg",
            "gray_path": "data/vaze_bw/gray/train/amphora_001.jpg",
        }
    ]

    color_dir = tmp_path / "color" / "train"
    gray_dir = tmp_path / "gray" / "train"

    _write_metadata_jsonl(rows, color_dir, "color_path")
    _write_metadata_jsonl(rows, gray_dir, "gray_path")

    color_payload = json.loads((color_dir / "metadata.jsonl").read_text(encoding="utf-8").strip())
    gray_payload = json.loads((gray_dir / "metadata.jsonl").read_text(encoding="utf-8").strip())

    assert color_payload == {
        "file_name": "amphora_001.jpg",
        "text": "x23_amphoras",
    }
    assert gray_payload == {
        "file_name": "amphora_001.jpg",
        "text": "ceramic vessel",
    }
