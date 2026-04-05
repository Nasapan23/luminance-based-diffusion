from __future__ import annotations

import csv
import json
import logging
from pathlib import Path


LOGGER = logging.getLogger(__name__)

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tif", ".tiff"}


class BaseSourceAdapter:
    def __init__(self, source_cfg: dict, repo_root: Path):
        self.source_cfg = source_cfg
        self.repo_root = repo_root
        self.source_id = str(source_cfg["id"])
        raw_dir_value = source_cfg.get("raw_dir", f"data/raw/{self.source_id}")
        raw_dir = Path(raw_dir_value)
        self.raw_dir = raw_dir if raw_dir.is_absolute() else (repo_root / raw_dir).resolve()
        self.fallback_caption = source_cfg.get("fallback_caption", "photo")
        self.local_globs = source_cfg.get("local_globs") or ["**/*"]

    def refresh_metadata(self) -> None:
        """Load optional source metadata for captions."""

    def collect_image_paths(self) -> list[Path]:
        seen: dict[str, Path] = {}
        for pattern in self.local_globs:
            for path in self.raw_dir.glob(pattern):
                if not path.is_file():
                    continue
                if path.suffix.lower() not in IMAGE_EXTENSIONS:
                    continue
                seen[path.resolve().as_posix()] = path.resolve()
        paths = sorted(seen.values())
        LOGGER.info("Source %s: discovered %s candidate images", self.source_id, len(paths))
        return paths

    def caption_for(self, path: Path) -> str:
        return self.fallback_caption


class LocalSourceAdapter(BaseSourceAdapter):
    def __init__(self, source_cfg: dict, repo_root: Path):
        super().__init__(source_cfg, repo_root)
        self.sidecar_caption_ext = str(source_cfg.get("sidecar_caption_ext", ".txt")).strip() or ".txt"
        if not self.sidecar_caption_ext.startswith("."):
            self.sidecar_caption_ext = f".{self.sidecar_caption_ext}"
        self.captions_csv = self._resolve_optional_path(source_cfg.get("captions_csv"))
        self.captions_jsonl = self._resolve_optional_path(source_cfg.get("captions_jsonl"))
        self.caption_lookup: dict[str, str] = {}

    def _resolve_optional_path(self, value: str | None) -> Path | None:
        if not value:
            return None
        candidate = Path(str(value))
        return candidate if candidate.is_absolute() else (self.raw_dir / candidate).resolve()

    def _register_caption(self, key: str, caption: str) -> None:
        normalized_caption = str(caption).strip()
        normalized_key = str(key).strip()
        if not normalized_caption or not normalized_key:
            return

        posix_key = Path(normalized_key).as_posix()
        self.caption_lookup.setdefault(posix_key, normalized_caption)
        self.caption_lookup.setdefault(Path(posix_key).name, normalized_caption)
        self.caption_lookup.setdefault(Path(posix_key).stem, normalized_caption)

    def _row_to_caption(self, row: dict[str, str]) -> tuple[str, str]:
        key = (
            row.get("relative_path")
            or row.get("path")
            or row.get("file_name")
            or row.get("filename")
            or row.get("image")
            or row.get("ImageID")
            or row.get("image_id")
            or row.get("stem")
            or ""
        ).strip()
        caption = (
            row.get("caption")
            or row.get("text")
            or row.get("prompt")
            or row.get("description")
            or row.get("title")
            or ""
        ).strip()
        return key, caption

    def refresh_metadata(self) -> None:
        self.caption_lookup = {}

        if self.captions_csv and self.captions_csv.exists():
            with self.captions_csv.open("r", encoding="utf-8", newline="") as handle:
                reader = csv.DictReader(handle)
                for row in reader:
                    key, caption = self._row_to_caption(row)
                    self._register_caption(key, caption)
            LOGGER.info(
                "Loaded local CSV caption entries for %s: %s",
                self.source_id,
                len(self.caption_lookup),
            )
        elif self.captions_csv:
            LOGGER.warning(
                "Local captions CSV not found for %s: %s",
                self.source_id,
                self.captions_csv,
            )

        if self.captions_jsonl and self.captions_jsonl.exists():
            with self.captions_jsonl.open("r", encoding="utf-8") as handle:
                for line in handle:
                    text = line.strip()
                    if not text:
                        continue
                    payload = json.loads(text)
                    if not isinstance(payload, dict):
                        continue
                    key, caption = self._row_to_caption(payload)
                    self._register_caption(key, caption)
            LOGGER.info(
                "Loaded local JSONL caption entries for %s: %s",
                self.source_id,
                len(self.caption_lookup),
            )
        elif self.captions_jsonl:
            LOGGER.warning(
                "Local captions JSONL not found for %s: %s",
                self.source_id,
                self.captions_jsonl,
            )

    def _sidecar_caption_for(self, path: Path) -> str:
        candidates = [
            path.with_suffix(self.sidecar_caption_ext),
            Path(f"{path}{self.sidecar_caption_ext}"),
        ]
        seen: set[str] = set()
        for candidate in candidates:
            candidate_key = candidate.resolve().as_posix() if candidate.exists() else candidate.as_posix()
            if candidate_key in seen:
                continue
            seen.add(candidate_key)
            if not candidate.exists():
                continue
            text = candidate.read_text(encoding="utf-8", errors="ignore").strip()
            if text:
                return text
        return ""

    def caption_for(self, path: Path) -> str:
        sidecar_caption = self._sidecar_caption_for(path)
        if sidecar_caption:
            return sidecar_caption

        rel = path.relative_to(self.raw_dir).as_posix()
        return (
            self.caption_lookup.get(rel)
            or self.caption_lookup.get(path.name)
            or self.caption_lookup.get(path.stem)
            or self.fallback_caption
        )


class CocoSourceAdapter(BaseSourceAdapter):
    def __init__(self, source_cfg: dict, repo_root: Path):
        super().__init__(source_cfg, repo_root)
        captions_json_value = source_cfg.get("captions_json", "annotations/captions_train2017.json")
        captions_json_path = Path(captions_json_value)
        self.captions_json = (
            captions_json_path
            if captions_json_path.is_absolute()
            else (self.raw_dir / captions_json_path).resolve()
        )
        self.caption_lookup: dict[str, str] = {}

    def refresh_metadata(self) -> None:
        if not self.captions_json.exists():
            LOGGER.warning(
                "COCO captions file not found for %s: %s",
                self.source_id,
                self.captions_json,
            )
            return

        with self.captions_json.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)

        image_id_to_name: dict[int, str] = {}
        for item in payload.get("images", []):
            image_id = item.get("id")
            file_name = item.get("file_name")
            if image_id is None or not file_name:
                continue
            image_id_to_name[int(image_id)] = str(file_name)

        for ann in payload.get("annotations", []):
            image_id = ann.get("image_id")
            caption = ann.get("caption")
            if image_id is None or not caption:
                continue
            file_name = image_id_to_name.get(int(image_id))
            if not file_name:
                continue
            self.caption_lookup.setdefault(file_name, str(caption).strip())
            self.caption_lookup.setdefault(Path(file_name).name, str(caption).strip())

        LOGGER.info(
            "Loaded COCO caption entries for %s: %s", self.source_id, len(self.caption_lookup)
        )

    def caption_for(self, path: Path) -> str:
        rel = path.relative_to(self.raw_dir).as_posix()
        return (
            self.caption_lookup.get(rel)
            or self.caption_lookup.get(path.name)
            or self.fallback_caption
        )


class Places2SourceAdapter(BaseSourceAdapter):
    def caption_for(self, path: Path) -> str:
        sidecar = path.with_suffix(".txt")
        if sidecar.exists():
            text = sidecar.read_text(encoding="utf-8", errors="ignore").strip()
            if text:
                return text
        return self.fallback_caption


class OpenImagesSourceAdapter(BaseSourceAdapter):
    def __init__(self, source_cfg: dict, repo_root: Path):
        super().__init__(source_cfg, repo_root)
        captions_csv_value = source_cfg.get("captions_csv", "metadata/captions.csv")
        captions_csv_path = Path(captions_csv_value)
        self.captions_csv = (
            captions_csv_path
            if captions_csv_path.is_absolute()
            else (self.raw_dir / captions_csv_path).resolve()
        )
        self.caption_lookup: dict[str, str] = {}

    def refresh_metadata(self) -> None:
        if not self.captions_csv.exists():
            LOGGER.warning(
                "OpenImages captions/labels CSV not found for %s: %s",
                self.source_id,
                self.captions_csv,
            )
            return

        with self.captions_csv.open("r", encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle)
            for row in reader:
                image_id = (row.get("ImageID") or row.get("image_id") or "").strip()
                if not image_id:
                    continue
                candidate_caption = (
                    row.get("caption")
                    or row.get("Caption")
                    or row.get("title")
                    or row.get("Title")
                    or row.get("description")
                    or row.get("Description")
                    or row.get("LabelName")
                    or ""
                ).strip()
                if not candidate_caption:
                    continue
                self.caption_lookup.setdefault(image_id, candidate_caption)

        LOGGER.info(
            "Loaded OpenImages caption entries for %s: %s",
            self.source_id,
            len(self.caption_lookup),
        )

    def caption_for(self, path: Path) -> str:
        image_id = path.stem
        return self.caption_lookup.get(image_id, self.fallback_caption)


def create_source_adapter(source_cfg: dict, repo_root: Path) -> BaseSourceAdapter:
    source_type = str(source_cfg.get("type", "local")).lower().strip()
    if source_type in {"local", "base"}:
        return LocalSourceAdapter(source_cfg, repo_root)
    if source_type == "coco":
        return CocoSourceAdapter(source_cfg, repo_root)
    if source_type == "places2":
        return Places2SourceAdapter(source_cfg, repo_root)
    if source_type == "openimages":
        return OpenImagesSourceAdapter(source_cfg, repo_root)
    LOGGER.warning("Unknown source type '%s'. Falling back to base adapter.", source_type)
    return BaseSourceAdapter(source_cfg, repo_root)
