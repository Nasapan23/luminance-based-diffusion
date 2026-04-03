from __future__ import annotations

from pathlib import Path

from lbd.tracking.history import HistoryDB


def test_history_state_transitions_and_retry_counter(tmp_path: Path) -> None:
    db = HistoryDB(tmp_path / "history.sqlite")
    run_id = "run_1"
    db.create_or_update_run(
        run_id=run_id,
        stage="build_base100k",
        config_hash="abc",
        config_path="runs/run_1/config.resolved.yaml",
    )
    db.upsert_items(
        run_id,
        [
            {
                "item_id": "item_1",
                "sample_id": "sample_1",
                "source_id": "coco",
                "split": "train",
                "raw_path": "data/raw/coco/a.jpg",
                "color_path": "data/base100k/color/train/item_1.jpg",
                "gray_path": "data/base100k/gray/train/item_1.jpg",
                "status": "pending",
            },
            {
                "item_id": "item_2",
                "sample_id": "sample_2",
                "source_id": "openimages",
                "split": "val",
                "raw_path": "data/raw/openimages/b.jpg",
                "color_path": "data/base100k/color/val/item_2.jpg",
                "gray_path": "data/base100k/gray/val/item_2.jpg",
                "status": "pending",
            },
        ],
    )

    pending = db.get_items_by_status(run_id, ["pending"])
    assert len(pending) == 2

    db.mark_item_status(run_id, "item_1", "in_progress")
    db.mark_item_status(run_id, "item_1", "failed", "broken image", increment_retry=True)

    failed = db.get_items_by_status(run_id, ["failed"])
    assert len(failed) == 1
    assert failed[0]["item_id"] == "item_1"
    assert failed[0]["retries"] == 1
    assert failed[0]["last_error"] == "broken image"

    db.close()

