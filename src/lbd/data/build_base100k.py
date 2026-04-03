from __future__ import annotations

import csv
import logging
import math
import random
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path

from lbd.config import dump_yaml, load_yaml, make_config_hash
from lbd.tracking.history import HistoryDB
from lbd.utils.fs_utils import as_repo_relative, atomic_copy, atomic_hardlink_or_copy
from lbd.utils.image_utils import save_grayscale_rgb


LOGGER = logging.getLogger(__name__)
ACTIVE_STATUSES = ["pending", "failed", "in_progress", "skipped"]


def allocate_proportional(capacities: dict[str, int], total_target: int) -> dict[str, int]:
    if total_target < 0:
        raise ValueError("total_target must be >= 0")
    if not capacities:
        if total_target == 0:
            return {}
        raise ValueError("No capacities provided for non-zero target.")

    total_capacity = sum(capacities.values())
    if total_target > total_capacity:
        raise ValueError(
            f"Cannot allocate target {total_target}; only {total_capacity} items available."
        )

    if total_target == 0:
        return {key: 0 for key in capacities}

    allocations: dict[str, int] = {}
    remainders: list[tuple[float, int, str]] = []
    for key in sorted(capacities):
        capacity = capacities[key]
        share = (capacity / total_capacity) * total_target
        base = min(capacity, math.floor(share))
        allocations[key] = base
        remainders.append((share - base, capacity, key))

    remaining = total_target - sum(allocations.values())
    remainders.sort(key=lambda item: (-item[0], -item[1], item[2]))

    while remaining > 0:
        progressed = False
        for _, _, key in remainders:
            if allocations[key] >= capacities[key]:
                continue
            allocations[key] += 1
            remaining -= 1
            progressed = True
            if remaining == 0:
                break
        if not progressed:
            break

    if sum(allocations.values()) != total_target:
        raise RuntimeError("Allocation failed to satisfy target exactly.")
    return allocations


def _is_true(value: str) -> bool:
    return str(value).strip().lower() in {"1", "true", "yes", "y"}


def _read_ingest_index(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        rows = list(reader)
    return rows


def _dedupe_by_hash(rows: list[dict[str, str]]) -> list[dict[str, str]]:
    seen: set[str] = set()
    output: list[dict[str, str]] = []
    for row in rows:
        key = row.get("sha256", "").strip() or f"path::{row['raw_path']}"
        if key in seen:
            continue
        seen.add(key)
        output.append(row)
    return output


def _sample_rows_by_source(
    rows: list[dict[str, str]], target_total: int, seed: int
) -> tuple[list[dict[str, str]], dict[str, int]]:
    grouped: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        grouped[row["source_id"]].append(row)

    capacities = {source_id: len(items) for source_id, items in grouped.items()}
    quotas = allocate_proportional(capacities, target_total)

    selected: list[dict[str, str]] = []
    for index, source_id in enumerate(sorted(grouped)):
        source_rows = list(grouped[source_id])
        rng = random.Random(seed + (index * 9973))
        rng.shuffle(source_rows)
        selected.extend(source_rows[: quotas[source_id]])

    if len(selected) != target_total:
        raise RuntimeError("Sampled row count does not match target.")

    return selected, quotas


def _assign_splits(
    selected_rows: list[dict[str, str]],
    split_counts: dict[str, int],
    seed: int,
) -> list[dict[str, str]]:
    train_target = int(split_counts["train"])
    val_target = int(split_counts["val"])
    test_target = int(split_counts["test"])
    total_target = train_target + val_target + test_target
    if total_target != len(selected_rows):
        raise ValueError("Split targets must sum to selected row count.")

    grouped: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in selected_rows:
        grouped[row["source_id"]].append(row)

    capacities = {source_id: len(items) for source_id, items in grouped.items()}
    train_by_source = allocate_proportional(capacities, train_target)

    remaining_for_val = {
        source_id: capacities[source_id] - train_by_source[source_id] for source_id in capacities
    }
    val_by_source = allocate_proportional(remaining_for_val, val_target)
    test_by_source = {
        source_id: remaining_for_val[source_id] - val_by_source[source_id] for source_id in capacities
    }

    output: list[dict[str, str]] = []
    for index, source_id in enumerate(sorted(grouped)):
        source_rows = list(grouped[source_id])
        rng = random.Random(seed + 50000 + (index * 9973))
        rng.shuffle(source_rows)

        n_train = train_by_source[source_id]
        n_val = val_by_source[source_id]
        n_test = test_by_source[source_id]

        train_rows = source_rows[:n_train]
        val_rows = source_rows[n_train : n_train + n_val]
        test_rows = source_rows[n_train + n_val : n_train + n_val + n_test]

        for row in train_rows:
            row_copy = dict(row)
            row_copy["split"] = "train"
            output.append(row_copy)
        for row in val_rows:
            row_copy = dict(row)
            row_copy["split"] = "val"
            output.append(row_copy)
        for row in test_rows:
            row_copy = dict(row)
            row_copy["split"] = "test"
            output.append(row_copy)

    if len(output) != len(selected_rows):
        raise RuntimeError("Split assignment failed to preserve row count.")

    split_counter = defaultdict(int)
    for row in output:
        split_counter[row["split"]] += 1

    expected = {"train": train_target, "val": val_target, "test": test_target}
    if dict(split_counter) != expected:
        raise RuntimeError(f"Split mismatch. expected={expected}, actual={dict(split_counter)}")

    return output


def _resolve_repo_path(path_str: str, repo_root: Path) -> Path:
    candidate = Path(path_str)
    if candidate.is_absolute():
        return candidate
    return (repo_root / candidate).resolve()


def _write_csv(path: Path, rows: list[dict[str, str]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _build_items(
    rows: list[dict[str, str]],
    output_root: Path,
    repo_root: Path,
) -> list[dict[str, str]]:
    items: list[dict[str, str]] = []
    for row in rows:
        raw_abs = _resolve_repo_path(row["raw_path"], repo_root)
        split = row["split"]
        ext = raw_abs.suffix.lower()
        if ext not in {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tif", ".tiff"}:
            ext = ".jpg"

        item_id = row["sample_id"]
        color_abs = output_root / "color" / split / f"{item_id}{ext}"
        gray_abs = output_root / "gray" / split / f"{item_id}.jpg"
        color_path = as_repo_relative(color_abs, repo_root)
        gray_path = as_repo_relative(gray_abs, repo_root)

        items.append(
            {
                "item_id": item_id,
                "sample_id": row["sample_id"],
                "source_id": row["source_id"],
                "split": split,
                "caption": row["caption"],
                "raw_path": row["raw_path"],
                "color_path": color_path,
                "gray_path": gray_path,
                "raw_path_abs": raw_abs.as_posix(),
                "color_path_abs": color_abs.resolve().as_posix(),
                "gray_path_abs": gray_abs.resolve().as_posix(),
                "width": row.get("width", ""),
                "height": row.get("height", ""),
                "sha256": row.get("sha256", ""),
            }
        )
    return items


def _process_item(item: dict[str, str], copy_mode: str) -> tuple[bool, str]:
    raw_path = Path(item["raw_path_abs"])
    color_path = Path(item["color_path_abs"])
    gray_path = Path(item["gray_path_abs"])

    if not raw_path.exists():
        return False, f"Missing source image: {raw_path}"

    if color_path.exists() and gray_path.exists():
        return True, "already_exists"

    if not color_path.exists():
        if copy_mode == "hardlink":
            atomic_hardlink_or_copy(raw_path, color_path)
        else:
            atomic_copy(raw_path, color_path)

    if not gray_path.exists():
        save_grayscale_rgb(raw_path, gray_path)

    return True, ""


def _run_processing_batches(
    history: HistoryDB,
    run_id: str,
    items: list[dict[str, str]],
    threads: int,
    batch_size: int,
    copy_mode: str,
) -> None:
    if not items:
        LOGGER.info("No pending items to process for run %s", run_id)
        return

    LOGGER.info(
        "Processing %s pending items with threads=%s batch_size=%s",
        len(items),
        threads,
        batch_size,
    )

    for batch_idx, start in enumerate(range(0, len(items), batch_size), start=1):
        batch_items = items[start : start + batch_size]
        history.record_batch_start(run_id, batch_idx, len(batch_items))
        history.record_event(
            run_id,
            event_type="batch_started",
            message=f"Processing batch {batch_idx} with {len(batch_items)} items",
            batch_id=batch_idx,
        )

        done_count = 0
        failed_count = 0
        with ThreadPoolExecutor(max_workers=threads) as executor:
            future_map = {}
            for item in batch_items:
                history.mark_item_status(run_id, item["item_id"], "in_progress")
                future = executor.submit(_process_item, item, copy_mode)
                future_map[future] = item

            for future in as_completed(future_map):
                item = future_map[future]
                item_id = item["item_id"]
                try:
                    success, message = future.result()
                except Exception as exc:  # pragma: no cover - defensive
                    success = False
                    message = str(exc)

                if success:
                    history.mark_item_status(run_id, item_id, "done", last_error="")
                    done_count += 1
                else:
                    history.mark_item_status(
                        run_id,
                        item_id,
                        "failed",
                        last_error=message,
                        increment_retry=True,
                    )
                    history.record_event(
                        run_id,
                        event_type="item_failed",
                        message=message,
                        item_id=item_id,
                        batch_id=batch_idx,
                    )
                    failed_count += 1

        history.record_batch_finish(
            run_id=run_id,
            batch_id=batch_idx,
            done_items=done_count,
            failed_items=failed_count,
        )
        history.record_event(
            run_id,
            event_type="batch_finished",
            message=f"Batch {batch_idx} complete: done={done_count} failed={failed_count}",
            batch_id=batch_idx,
        )


def _compute_split_counts(rows: list[dict[str, str]]) -> dict[str, int]:
    counts = {"train": 0, "val": 0, "test": 0}
    for row in rows:
        counts[row["split"]] += 1
    return counts


def _finalize_manifests(
    history: HistoryDB,
    run_id: str,
    selected_items: list[dict[str, str]],
    output_root: Path,
    run_dir: Path,
) -> None:
    item_status_lookup: dict[str, str] = {}
    item_error_lookup: dict[str, str] = {}
    for record in history.get_all_items(run_id):
        item_status_lookup[record["item_id"]] = record["status"]
        item_error_lookup[record["item_id"]] = record["last_error"] or ""

    index_rows: list[dict[str, str]] = []
    split_counts: dict[str, dict[str, int]] = {
        "train": {"total": 0, "done": 0, "failed": 0},
        "val": {"total": 0, "done": 0, "failed": 0},
        "test": {"total": 0, "done": 0, "failed": 0},
    }
    source_counts: dict[tuple[str, str], dict[str, int]] = defaultdict(
        lambda: {"total": 0, "done": 0, "failed": 0}
    )

    for item in selected_items:
        item_id = item["item_id"]
        status = item_status_lookup.get(item_id, "pending")
        error = item_error_lookup.get(item_id, "")
        split = item["split"]
        split_counts[split]["total"] += 1
        source_counts[(item["source_id"], split)]["total"] += 1
        if status == "done":
            split_counts[split]["done"] += 1
            source_counts[(item["source_id"], split)]["done"] += 1
        elif status == "failed":
            split_counts[split]["failed"] += 1
            source_counts[(item["source_id"], split)]["failed"] += 1

        row = dict(item)
        row["status"] = status
        row["last_error"] = error
        index_rows.append(row)

    meta_dir = output_root / "meta"
    meta_dir.mkdir(parents=True, exist_ok=True)

    index_fieldnames = [
        "item_id",
        "sample_id",
        "source_id",
        "split",
        "caption",
        "raw_path",
        "color_path",
        "gray_path",
        "status",
        "last_error",
        "width",
        "height",
        "sha256",
    ]
    _write_csv(meta_dir / "index.csv", index_rows, index_fieldnames)

    split_rows = []
    for split, values in split_counts.items():
        split_rows.append(
            {
                "split": split,
                "total": str(values["total"]),
                "done": str(values["done"]),
                "failed": str(values["failed"]),
                "pending": str(values["total"] - values["done"] - values["failed"]),
            }
        )
    _write_csv(
        meta_dir / "split_counts.csv",
        split_rows,
        ["split", "total", "done", "failed", "pending"],
    )

    source_rows = []
    for (source_id, split), values in sorted(source_counts.items()):
        source_rows.append(
            {
                "source_id": source_id,
                "split": split,
                "total": str(values["total"]),
                "done": str(values["done"]),
                "failed": str(values["failed"]),
            }
        )
    _write_csv(
        meta_dir / "source_contributions.csv",
        source_rows,
        ["source_id", "split", "total", "done", "failed"],
    )

    captions_rows = [
        {
            "sample_id": item["sample_id"],
            "caption": item["caption"],
        }
        for item in selected_items
    ]
    _write_csv(meta_dir / "captions.csv", captions_rows, ["sample_id", "caption"])

    selected_path = run_dir / "selected_items.csv"
    selected_fieldnames = [
        "item_id",
        "sample_id",
        "source_id",
        "split",
        "caption",
        "raw_path",
        "color_path",
        "gray_path",
        "width",
        "height",
        "sha256",
    ]
    _write_csv(selected_path, selected_items, selected_fieldnames)
    _write_csv(meta_dir / "selected_items.csv", selected_items, selected_fieldnames)


def _load_selected_items(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        return list(reader)


def _make_run_id(run_name: str) -> str:
    ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    return f"{ts}_{run_name}"


def run_build_base100k(config_path: Path, repo_root: Path | None = None) -> str:
    root = (repo_root or Path.cwd()).resolve()
    config = load_yaml(config_path)

    run_name = str(config.get("run_name", "base100k"))
    runs_root_value = config.get("runs_root", "runs")
    runs_root = _resolve_repo_path(str(runs_root_value), root)
    runs_root.mkdir(parents=True, exist_ok=True)

    run_id = _make_run_id(run_name)
    run_dir = runs_root / run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    resolved_config = dict(config)
    resolved_config["run_id"] = run_id
    resolved_config["config_path"] = as_repo_relative(Path(config_path), root)
    resolved_config_path = run_dir / "config.resolved.yaml"
    dump_yaml(resolved_config, resolved_config_path)

    config_hash = make_config_hash(resolved_config)
    history = HistoryDB(run_dir / "history.sqlite")
    history.create_or_update_run(
        run_id=run_id,
        stage="build_base100k",
        config_hash=config_hash,
        config_path=as_repo_relative(resolved_config_path, root),
        status="running",
    )

    try:
        input_index_value = config.get("input_index_csv", "data/base100k/meta/ingest_index.csv")
        input_index_path = _resolve_repo_path(str(input_index_value), root)
        if not input_index_path.exists():
            raise FileNotFoundError(f"Ingest index not found: {input_index_path}")

        all_rows = _read_ingest_index(input_index_path)
        valid_rows = [row for row in all_rows if _is_true(row.get("valid", "0"))]
        if bool(config.get("dedupe_by_sha256", True)):
            valid_rows = _dedupe_by_hash(valid_rows)

        target_total = int(config.get("target_total", 100000))
        split_counts = config.get("split_counts", {"train": 80000, "val": 10000, "test": 10000})
        split_counts = {
            "train": int(split_counts["train"]),
            "val": int(split_counts["val"]),
            "test": int(split_counts["test"]),
        }
        if sum(split_counts.values()) != target_total:
            raise ValueError("split_counts must sum exactly to target_total.")

        if len(valid_rows) < target_total:
            raise ValueError(
                f"Insufficient valid rows: have {len(valid_rows)}, require {target_total}."
            )

        seed = int(config.get("seed", 42))
        sampled_rows, source_quotas = _sample_rows_by_source(valid_rows, target_total, seed)
        assigned_rows = _assign_splits(sampled_rows, split_counts=split_counts, seed=seed)

        output_root_value = config.get("output_root", "data/base100k")
        output_root = _resolve_repo_path(str(output_root_value), root)
        output_root.mkdir(parents=True, exist_ok=True)

        selected_items = _build_items(assigned_rows, output_root=output_root, repo_root=root)
        history.upsert_items(
            run_id,
            [
                {
                    "item_id": item["item_id"],
                    "sample_id": item["sample_id"],
                    "source_id": item["source_id"],
                    "split": item["split"],
                    "raw_path": item["raw_path"],
                    "color_path": item["color_path"],
                    "gray_path": item["gray_path"],
                    "status": "pending",
                }
                for item in selected_items
            ],
        )

        pending_rows = history.get_items_by_status(run_id, ACTIVE_STATUSES)
        pending_lookup = {row["item_id"]: dict(row) for row in pending_rows}
        pending_items = [
            item for item in selected_items if item["item_id"] in pending_lookup
        ]

        threads = int(config.get("threads", 16))
        batch_size = int(config.get("batch_size", 512))
        copy_mode = str(config.get("copy_mode", "copy")).lower()
        if copy_mode not in {"copy", "hardlink"}:
            raise ValueError("copy_mode must be either 'copy' or 'hardlink'.")

        _run_processing_batches(
            history=history,
            run_id=run_id,
            items=pending_items,
            threads=threads,
            batch_size=batch_size,
            copy_mode=copy_mode,
        )

        _finalize_manifests(history, run_id, selected_items, output_root, run_dir)

        final_rows = history.get_all_items(run_id)
        failed = sum(1 for row in final_rows if row["status"] == "failed")
        run_status = "completed" if failed == 0 else "partial_failed"
        history.finish_run(run_id, run_status, note=f"failed_items={failed}")
        history.record_event(
            run_id,
            event_type="run_finished",
            message=f"Run finished with status={run_status}. Source quotas={source_quotas}",
        )
        LOGGER.info("Build run complete: %s status=%s", run_id, run_status)
        return run_id
    except Exception as exc:
        history.finish_run(run_id, "failed", note=str(exc))
        history.record_event(run_id, event_type="run_failed", message=str(exc))
        raise
    finally:
        history.close()


def resume_build_base100k(
    run_id: str,
    runs_root: Path | None = None,
    repo_root: Path | None = None,
) -> str:
    root = (repo_root or Path.cwd()).resolve()
    runs_root_path = _resolve_repo_path(str(runs_root or "runs"), root)
    run_dir = runs_root_path / run_id
    if not run_dir.exists():
        raise FileNotFoundError(f"Run directory not found: {run_dir}")

    resolved_config_path = run_dir / "config.resolved.yaml"
    selected_items_path = run_dir / "selected_items.csv"
    if not resolved_config_path.exists():
        raise FileNotFoundError(f"Resolved config missing for run {run_id}: {resolved_config_path}")
    if not selected_items_path.exists():
        raise FileNotFoundError(
            f"Selected items manifest missing for run {run_id}: {selected_items_path}"
        )

    config = load_yaml(resolved_config_path)
    config_hash = make_config_hash(config)
    history = HistoryDB(run_dir / "history.sqlite")
    history.create_or_update_run(
        run_id=run_id,
        stage="build_base100k",
        config_hash=config_hash,
        config_path=as_repo_relative(resolved_config_path, root),
        status="running",
        note="resume",
    )

    try:
        selected_items = _load_selected_items(selected_items_path)
        history.upsert_items(
            run_id,
            [
                {
                    "item_id": item["item_id"],
                    "sample_id": item["sample_id"],
                    "source_id": item["source_id"],
                    "split": item["split"],
                    "raw_path": item["raw_path"],
                    "color_path": item["color_path"],
                    "gray_path": item["gray_path"],
                    "status": "pending",
                }
                for item in selected_items
            ],
        )

        pending_rows = history.get_items_by_status(run_id, ACTIVE_STATUSES)
        pending_lookup = {row["item_id"] for row in pending_rows}
        pending_items = []
        for item in selected_items:
            if item["item_id"] not in pending_lookup:
                continue
            item_copy = dict(item)
            item_copy["raw_path_abs"] = _resolve_repo_path(item["raw_path"], root).as_posix()
            item_copy["color_path_abs"] = _resolve_repo_path(item["color_path"], root).as_posix()
            item_copy["gray_path_abs"] = _resolve_repo_path(item["gray_path"], root).as_posix()
            pending_items.append(item_copy)

        threads = int(config.get("threads", 16))
        batch_size = int(config.get("batch_size", 512))
        copy_mode = str(config.get("copy_mode", "copy")).lower()
        _run_processing_batches(
            history=history,
            run_id=run_id,
            items=pending_items,
            threads=threads,
            batch_size=batch_size,
            copy_mode=copy_mode,
        )

        output_root = _resolve_repo_path(str(config.get("output_root", "data/base100k")), root)
        _finalize_manifests(history, run_id, selected_items, output_root, run_dir)

        final_rows = history.get_all_items(run_id)
        failed = sum(1 for row in final_rows if row["status"] == "failed")
        run_status = "completed" if failed == 0 else "partial_failed"
        history.finish_run(run_id, run_status, note=f"failed_items={failed}")
        history.record_event(
            run_id,
            event_type="run_finished",
            message=f"Resume finished with status={run_status}",
        )
        LOGGER.info("Resume complete: %s status=%s", run_id, run_status)
        return run_id
    except Exception as exc:
        history.finish_run(run_id, "failed", note=str(exc))
        history.record_event(run_id, event_type="run_failed", message=str(exc))
        raise
    finally:
        history.close()
