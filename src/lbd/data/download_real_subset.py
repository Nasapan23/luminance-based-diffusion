from __future__ import annotations

import csv
import hashlib
import io
import json
import logging
import random
import tarfile
import zipfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from time import monotonic, sleep
from typing import Any
from urllib.parse import urlparse

import requests

from lbd.config import load_yaml


LOGGER = logging.getLogger(__name__)
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}


def _resolve_path(value: str | Path, repo_root: Path) -> Path:
    path = Path(value)
    if path.is_absolute():
        return path
    return (repo_root / path).resolve()


def _download_file(
    url: str,
    dst: Path,
    timeout_sec: float = 120.0,
    *,
    log_label: str | None = None,
    log_progress: bool = False,
    progress_interval_sec: float = 15.0,
) -> Path:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists() and dst.stat().st_size > 0:
        if log_progress:
            size_mb = dst.stat().st_size / (1024 * 1024)
            label = log_label or dst.name
            LOGGER.info("%s already exists (%.1f MiB), skipping.", label, size_mb)
        return dst

    label = log_label or dst.name
    if log_progress:
        LOGGER.info("Download start: %s <- %s", label, url)

    with requests.get(url, stream=True, timeout=timeout_sec) as response:
        response.raise_for_status()
        total_bytes = int(response.headers.get("content-length", "0") or "0")
        downloaded_bytes = 0
        last_log = monotonic()
        with dst.open("wb") as handle:
            for chunk in response.iter_content(chunk_size=1024 * 256):
                if chunk:
                    handle.write(chunk)
                    downloaded_bytes += len(chunk)
                if log_progress and monotonic() - last_log >= progress_interval_sec:
                    if total_bytes > 0:
                        pct = (downloaded_bytes / total_bytes) * 100.0
                        LOGGER.info(
                            "Download progress: %s %.1f%% (%d/%d MiB)",
                            label,
                            pct,
                            downloaded_bytes // (1024 * 1024),
                            total_bytes // (1024 * 1024),
                        )
                    else:
                        LOGGER.info(
                            "Download progress: %s %d MiB",
                            label,
                            downloaded_bytes // (1024 * 1024),
                        )
                    last_log = monotonic()
    if log_progress:
        LOGGER.info(
            "Download complete: %s (%d MiB)",
            label,
            downloaded_bytes // (1024 * 1024),
        )
    return dst


def _safe_extract_zip_member(zip_path: Path, member_name: str, output_dir: Path) -> Path:
    with zipfile.ZipFile(zip_path, "r") as archive:
        names = set(archive.namelist())
        if member_name not in names:
            raise FileNotFoundError(f"Member '{member_name}' not found in {zip_path}")
        target = (output_dir / member_name).resolve()
        target.parent.mkdir(parents=True, exist_ok=True)
        with archive.open(member_name, "r") as src, target.open("wb") as dst:
            dst.write(src.read())
    return target


def _download_many(
    jobs: list[tuple[str, Path]],
    threads: int,
    timeout_sec: float,
    *,
    progress_label: str = "downloads",
    progress_interval_sec: float = 15.0,
) -> tuple[int, int]:
    if not jobs:
        return 0, 0

    total = len(jobs)
    done = 0
    failed = 0
    started = monotonic()
    last_log = started
    LOGGER.info("%s start: total=%s threads=%s", progress_label, total, threads)
    with ThreadPoolExecutor(max_workers=threads) as executor:
        future_map = {
            executor.submit(_download_file, url, dst, timeout_sec): (url, dst)
            for url, dst in jobs
        }
        for completed in as_completed(future_map):
            future = completed
            url, dst = future_map[future]
            try:
                future.result()
                done += 1
            except Exception as exc:
                failed += 1
                LOGGER.warning("Download failed: %s -> %s (%s)", url, dst, exc)
            now = monotonic()
            finished = done + failed
            if now - last_log >= progress_interval_sec or finished == total:
                LOGGER.info(
                    "%s progress: %s/%s (done=%s failed=%s elapsed=%.1fs)",
                    progress_label,
                    finished,
                    total,
                    done,
                    failed,
                    now - started,
                )
                last_log = now
    LOGGER.info(
        "%s complete: total=%s done=%s failed=%s elapsed=%.1fs",
        progress_label,
        total,
        done,
        failed,
        monotonic() - started,
    )
    return done, failed


def _run_coco(config: dict[str, Any], repo_root: Path, seed: int, threads: int) -> None:
    if not bool(config.get("enabled", True)):
        return

    count = int(config.get("count", 334))
    LOGGER.info("COCO source enabled. target_count=%s", count)
    output_dir = _resolve_path(config.get("output_dir", "data/raw/coco"), repo_root)
    images_dir = output_dir / "images" / "train2017"
    annotations_dir = output_dir / "annotations"
    downloads_dir = output_dir / "downloads"
    images_dir.mkdir(parents=True, exist_ok=True)
    annotations_dir.mkdir(parents=True, exist_ok=True)
    downloads_dir.mkdir(parents=True, exist_ok=True)

    annotations_url = str(
        config.get(
            "annotations_url",
            "http://images.cocodataset.org/annotations/annotations_trainval2017.zip",
        )
    )
    base_image_url = str(
        config.get("base_image_url", "http://images.cocodataset.org/train2017")
    ).rstrip("/")

    zip_name = Path(urlparse(annotations_url).path).name or "annotations_trainval2017.zip"
    zip_path = downloads_dir / zip_name
    _download_file(
        annotations_url,
        zip_path,
        log_label="COCO annotations archive",
        log_progress=True,
    )

    captions_path = annotations_dir / "captions_train2017.json"
    if not captions_path.exists():
        _safe_extract_zip_member(
            zip_path=zip_path,
            member_name="annotations/captions_train2017.json",
            output_dir=output_dir,
        )

    with captions_path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)

    images = payload.get("images", [])
    if not images:
        raise RuntimeError("COCO captions file has no image entries.")

    rng = random.Random(seed)
    rng.shuffle(images)
    selected = images[:count]

    jobs: list[tuple[str, Path]] = []
    for row in selected:
        file_name = str(row.get("file_name", "")).strip()
        if not file_name:
            continue
        url = f"{base_image_url}/{Path(file_name).name}"
        dst = images_dir / Path(file_name).name
        if dst.exists() and dst.stat().st_size > 0:
            continue
        jobs.append((url, dst))

    done, failed = _download_many(
        jobs,
        threads=threads,
        timeout_sec=120.0,
        progress_label="COCO images",
    )
    LOGGER.info("COCO download complete. requested=%s done=%s failed=%s", count, done, failed)


def _run_openimages(config: dict[str, Any], repo_root: Path, seed: int, threads: int) -> None:
    if not bool(config.get("enabled", True)):
        return

    count = int(config.get("count", 333))
    LOGGER.info("OpenImages source enabled. target_count=%s", count)
    output_dir = _resolve_path(config.get("output_dir", "data/raw/openimages"), repo_root)
    images_dir = output_dir / "images" / "train"
    metadata_dir = output_dir / "metadata"
    downloads_dir = output_dir / "downloads"
    images_dir.mkdir(parents=True, exist_ok=True)
    metadata_dir.mkdir(parents=True, exist_ok=True)
    downloads_dir.mkdir(parents=True, exist_ok=True)

    index_tsv_url = str(
        config.get(
            "index_tsv_url",
            "https://storage.googleapis.com/cvdf-datasets/oid/open-images-dataset-validation.tsv",
        )
    )
    index_tsv_path = downloads_dir / (Path(urlparse(index_tsv_url).path).name or "openimages.tsv")
    _download_file(
        index_tsv_url,
        index_tsv_path,
        log_label="OpenImages index TSV",
        log_progress=True,
    )

    rows: list[dict[str, str]] = []
    with index_tsv_path.open("r", encoding="utf-8", newline="") as handle:
        first_line = handle.readline().strip()
        handle.seek(0)

        if first_line.startswith("TsvHttpData-"):
            # Legacy OpenImages TSV format:
            # line0 marker, then tab-separated rows with URL, size, hash
            # no explicit ImageID column.
            _ = handle.readline()  # skip marker
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                parts = line.split("\t")
                if not parts:
                    continue
                image_url = parts[0].strip()
                if not image_url:
                    continue
                image_id = "oi_" + hashlib.sha1(image_url.encode("utf-8")).hexdigest()[:20]
                rows.append({"ImageID": image_id, "OriginalURL": image_url})
        else:
            reader = csv.DictReader(handle, delimiter="\t")
            for row in reader:
                image_id = str(row.get("ImageID", "")).strip()
                image_url = str(row.get("OriginalURL", "")).strip()
                if not image_id or not image_url:
                    continue
                rows.append({"ImageID": image_id, "OriginalURL": image_url})

    if not rows:
        raise RuntimeError(f"No valid OpenImages rows found in {index_tsv_path}")

    rng = random.Random(seed + 1000)
    rng.shuffle(rows)

    id_to_dst: dict[str, Path] = {}
    for row in rows:
        image_id = row["ImageID"]
        image_url = row["OriginalURL"]
        suffix = Path(urlparse(image_url).path).suffix.lower()
        if suffix not in IMAGE_EXTENSIONS:
            suffix = ".jpg"
        id_to_dst[image_id] = images_dir / f"{image_id}{suffix}"

    def existing_count() -> int:
        return sum(
            1
            for dst in id_to_dst.values()
            if dst.exists() and dst.stat().st_size > 0
        )

    current = existing_count()
    done_total = 0
    failed_total = 0
    attempted_jobs_total = 0
    cursor = 0
    batch_count = 0
    no_progress_batches = 0
    max_batches = int(config.get("max_batches", 40))
    max_no_progress_batches = int(config.get("max_no_progress_batches", 8))
    max_attempt_jobs = int(config.get("max_attempt_jobs", max(count * 20, count)))
    cooldown_sec = float(config.get("cooldown_sec", 0.5))
    stop_reason = "target_reached_or_rows_exhausted"

    while current < count and cursor < len(rows):
        if batch_count >= max_batches:
            stop_reason = f"max_batches_reached({max_batches})"
            LOGGER.warning("OpenImages early-stop: %s", stop_reason)
            break
        if attempted_jobs_total >= max_attempt_jobs:
            stop_reason = f"max_attempt_jobs_reached({max_attempt_jobs})"
            LOGGER.warning("OpenImages early-stop: %s", stop_reason)
            break

        current_before = current
        needed = count - current
        batch_size = max(needed * 3, threads * 4)
        batch = rows[cursor : cursor + batch_size]
        cursor += batch_size
        batch_count += 1

        jobs: list[tuple[str, Path]] = []
        for row in batch:
            image_id = row["ImageID"]
            image_url = row["OriginalURL"]
            dst = id_to_dst[image_id]
            if dst.exists() and dst.stat().st_size > 0:
                continue
            jobs.append((image_url, dst))

        if not jobs:
            continue

        attempted_jobs_total += len(jobs)
        done, failed = _download_many(
            jobs,
            threads=threads,
            timeout_sec=120.0,
            progress_label=f"OpenImages batch {batch_count}",
        )
        done_total += done
        failed_total += failed
        current = existing_count()

        if current > current_before:
            no_progress_batches = 0
        else:
            no_progress_batches += 1
            if no_progress_batches >= max_no_progress_batches:
                stop_reason = f"no_progress_batches_reached({max_no_progress_batches})"
                LOGGER.warning("OpenImages early-stop: %s", stop_reason)
                break

        LOGGER.info(
            "OpenImages progress: current=%s target=%s attempted_done=%s attempted_failed=%s attempted_jobs=%s batch=%s",
            current,
            count,
            done_total,
            failed_total,
            attempted_jobs_total,
            batch_count,
        )
        if cooldown_sec > 0:
            sleep(cooldown_sec)

    current = existing_count()
    LOGGER.info(
        "OpenImages download complete. requested=%s available=%s attempted_done=%s attempted_failed=%s attempted_jobs=%s stop_reason=%s",
        count,
        current,
        done_total,
        failed_total,
        attempted_jobs_total,
        stop_reason,
    )

    captions_rows: list[dict[str, str]] = []
    for row in rows:
        image_id = row["ImageID"]
        dst = id_to_dst[image_id]
        if dst.exists() and dst.stat().st_size > 0:
            captions_rows.append({"ImageID": image_id, "caption": "open images photo"})
        if len(captions_rows) >= count:
            break

    captions_csv = metadata_dir / "captions.csv"
    with captions_csv.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["ImageID", "caption"])
        writer.writeheader()
        writer.writerows(captions_rows)


def _is_safe_extract_path(base_dir: Path, target: Path) -> bool:
    try:
        target.resolve().relative_to(base_dir.resolve())
        return True
    except ValueError:
        return False


def _run_places2(config: dict[str, Any], repo_root: Path, count: int) -> None:
    if not bool(config.get("enabled", True)):
        return

    LOGGER.info("Places2 source enabled. target_count=%s", count)
    output_dir = _resolve_path(config.get("output_dir", "data/raw/places2"), repo_root)
    images_dir = output_dir / "images" / "val"
    downloads_dir = output_dir / "downloads"
    images_dir.mkdir(parents=True, exist_ok=True)
    downloads_dir.mkdir(parents=True, exist_ok=True)

    tar_url = str(
        config.get("tar_url", "http://data.csail.mit.edu/places/places365/val_256.tar")
    )
    tar_path = downloads_dir / (Path(urlparse(tar_url).path).name or "val_256.tar")
    _download_file(
        tar_url,
        tar_path,
        timeout_sec=300.0,
        log_label="Places2 val_256.tar",
        log_progress=True,
    )

    existing = list(images_dir.glob("**/*.jpg"))
    if len(existing) >= count:
        LOGGER.info("Places2 already has >= %s images, skipping extraction.", count)
        return

    extracted = len(existing)
    with tarfile.open(tar_path, "r:*") as archive:
        for member in archive:
            if extracted >= count:
                break
            if not member.isfile():
                continue
            name = member.name.replace("\\", "/")
            if not name.lower().endswith(".jpg"):
                continue

            # Keep category structure under images/val/<category>/<file>.jpg
            rel_parts = Path(name).parts
            if len(rel_parts) >= 2:
                rel_path = Path(*rel_parts[-2:])
            else:
                rel_path = Path(Path(name).name)

            dst = images_dir / rel_path
            if dst.exists() and dst.stat().st_size > 0:
                extracted += 1
                continue
            if not _is_safe_extract_path(images_dir, dst):
                continue

            dst.parent.mkdir(parents=True, exist_ok=True)
            src = archive.extractfile(member)
            if src is None:
                continue
            with src, dst.open("wb") as handle:
                handle.write(src.read())

            # Sidecar text caption with category for places adapter.
            category = dst.parent.name.replace("_", " ")
            dst.with_suffix(".txt").write_text(
                f"{category} scene photograph",
                encoding="utf-8",
            )
            extracted += 1

    LOGGER.info("Places2 extraction complete. target=%s extracted=%s", count, extracted)


def run_download_real_subset(config_path: Path, repo_root: Path | None = None) -> None:
    root = (repo_root or Path.cwd()).resolve()
    config = load_yaml(config_path)
    seed = int(config.get("seed", 42))
    threads = int(config.get("threads", 16))
    LOGGER.info(
        "Starting real subset download: config=%s seed=%s threads=%s root=%s",
        config_path,
        seed,
        threads,
        root,
    )

    sources = dict(config.get("sources") or {})
    if not sources:
        raise ValueError("No sources defined in download config.")

    coco_cfg = dict(sources.get("coco") or {})
    places_cfg = dict(sources.get("places2") or {})
    open_cfg = dict(sources.get("openimages") or {})

    _run_coco(coco_cfg, root, seed=seed, threads=threads)
    _run_places2(places_cfg, root, count=int(places_cfg.get("count", 333)))
    _run_openimages(open_cfg, root, seed=seed, threads=threads)
