from __future__ import annotations

from lbd.data.build_base100k import _assign_splits, _dedupe_by_hash, allocate_proportional


def test_allocate_proportional_matches_target_exactly() -> None:
    capacities = {"coco": 800, "openimages": 400, "places2": 300}
    target = 1000
    allocations = allocate_proportional(capacities, target)

    assert sum(allocations.values()) == target
    assert allocations["coco"] >= allocations["openimages"] >= allocations["places2"]


def test_assign_splits_exact_counts() -> None:
    rows = []
    for idx in range(60):
        rows.append({"sample_id": f"coco_{idx}", "source_id": "coco"})
    for idx in range(30):
        rows.append({"sample_id": f"openimages_{idx}", "source_id": "openimages"})
    for idx in range(10):
        rows.append({"sample_id": f"places2_{idx}", "source_id": "places2"})

    out = _assign_splits(rows, split_counts={"train": 80, "val": 10, "test": 10}, seed=42)
    counts = {"train": 0, "val": 0, "test": 0}
    for row in out:
        counts[row["split"]] += 1

    assert counts == {"train": 80, "val": 10, "test": 10}


def test_dedupe_by_hash_uses_sha256_then_raw_path() -> None:
    rows = [
        {"sample_id": "a", "raw_path": "a.jpg", "sha256": "same"},
        {"sample_id": "b", "raw_path": "b.jpg", "sha256": "same"},
        {"sample_id": "c", "raw_path": "c.jpg", "sha256": ""},
        {"sample_id": "d", "raw_path": "c.jpg", "sha256": ""},
        {"sample_id": "e", "raw_path": "e.jpg", "sha256": "unique"},
    ]

    deduped = _dedupe_by_hash(rows)

    assert [row["sample_id"] for row in deduped] == ["a", "c", "e"]
