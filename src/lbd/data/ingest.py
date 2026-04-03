from __future__ import annotations

import csv
import logging
import tarfile
import zipfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from urllib.parse import urlparse
from urllib.request import urlretrieve

from lbd.config import load_yaml
from lbd.data.adapters import BaseSourceAdapter, create_source_adapter
from lbd.utils.fs_utils import as_repo_relative
from lbd.utils.hash_utils import sha256_file, stable_id
from lbd.utils.image_utils import inspect_image


LOGGER = logging.getLogger(__name__)


def _download_if_enabled(source_cfg: dict, raw_dir: Path) -> None:
    if not bool(source_cfg.get("download_enabled", False)):
        return

    download_urls = source_cfg.get("download_urls") or []
    if not download_urls:
        return

    download_dir = raw_dir / "downloads"
    download_dir.mkdir(parents=True, exist_ok=True)
    extract_archives = bool(source_cfg.get("extract_archives", False))

    for url in download_urls:
        parsed = urlparse(url)
        file_name = Path(parsed.path).name
        if not file_name:
            LOGGER.warning("Skipping URL with empty filename: %s", url)
            continue
        target = download_dir / file_name
        if target.exists():
            LOGGER.info("Download exists, skipping: %s", target)
        else:
            LOGGER.info("Downloading %s -> %s", url, target)
            urlretrieve(url, target)

        if extract_archives:
            _extract_archive(target, raw_dir)


def _extract_archive(archive_path: Path, output_dir: Path) -> None:
    suffixes = [s.lower() for s in archive_path.suffixes]
    if ".zip" in suffixes:
        LOGGER.info("Extracting zip archive: %s", archive_path)
        with zipfile.ZipFile(archive_path, "r") as archive:
            archive.extractall(output_dir)
        return
    if ".tar" in suffixes or ".gz" in suffixes or ".tgz" in suffixes:
        LOGGER.info("Extracting tar archive: %s", archive_path)
        with tarfile.open(archive_path, "r:*") as archive:
            archive.extractall(output_dir)
        return
    LOGGER.warning("Unsupported archive type, skipped extraction: %s", archive_path)


def _build_ingest_row(
    path: Path,
    adapter: BaseSourceAdapter,
    source_id: str,
    repo_root: Path,
) -> dict[str, str]:
    relative_in_source = path.relative_to(adapter.raw_dir).as_posix()
    sample_id = f"{source_id}_{stable_id(source_id, relative_in_source)}"
    caption = adapter.caption_for(path)
    valid, width, height, error = inspect_image(path)
    sha256 = sha256_file(path) if valid else ""

    return {
        "sample_id": sample_id,
        "source_id": source_id,
        "raw_path": as_repo_relative(path, repo_root),
        "relative_path": relative_in_source,
        "caption": caption,
        "width": str(width),
        "height": str(height),
        "sha256": sha256,
        "valid": "1" if valid else "0",
        "error": error,
    }


def _write_index_csv(rows: list[dict[str, str]], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "sample_id",
        "source_id",
        "raw_path",
        "relative_path",
        "caption",
        "width",
        "height",
        "sha256",
        "valid",
        "error",
    ]
    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def run_ingest(config_path: Path, repo_root: Path | None = None) -> Path:
    root = (repo_root or Path.cwd()).resolve()
    config = load_yaml(config_path)
    output_index = config.get("output_index_csv", "data/base100k/meta/ingest_index.csv")
    output_index_path = Path(output_index)
    if not output_index_path.is_absolute():
        output_index_path = (root / output_index_path).resolve()

    threads = int(config.get("threads", 16))
    sources = config.get("sources", [])
    if not sources:
        raise ValueError("No sources defined in ingest config.")

    rows: list[dict[str, str]] = []

    for source_cfg in sources:
        if not bool(source_cfg.get("enabled", True)):
            continue

        adapter = create_source_adapter(source_cfg, root)
        _download_if_enabled(source_cfg, adapter.raw_dir)
        adapter.refresh_metadata()

        image_paths = adapter.collect_image_paths()
        if not image_paths:
            LOGGER.warning("No image files found for source '%s'.", adapter.source_id)
            continue

        LOGGER.info(
            "Source %s: indexing %s files using %s threads",
            adapter.source_id,
            len(image_paths),
            threads,
        )
        with ThreadPoolExecutor(max_workers=threads) as executor:
            futures = [
                executor.submit(
                    _build_ingest_row,
                    image_path,
                    adapter,
                    adapter.source_id,
                    root,
                )
                for image_path in image_paths
            ]
            for future in as_completed(futures):
                rows.append(future.result())

    rows.sort(key=lambda item: (item["source_id"], item["raw_path"]))
    _write_index_csv(rows, output_index_path)
    LOGGER.info("Wrote ingest index: %s (%s rows)", output_index_path, len(rows))
    return output_index_path

