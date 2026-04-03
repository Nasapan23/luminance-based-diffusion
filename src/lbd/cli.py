from __future__ import annotations

import argparse
import logging
from pathlib import Path

from lbd.data.build_base100k import resume_build_base100k, run_build_base100k
from lbd.data.ingest import run_ingest
from lbd.infer.comfyui import run_comfyui_stage
from lbd.logging_utils import setup_logging
from lbd.train.launch import run_training_command


LOGGER = logging.getLogger(__name__)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="lbd",
        description="Luminance-based diffusion data preparation and training orchestration CLI.",
    )
    subparsers = parser.add_subparsers(dest="namespace", required=True)

    data_parser = subparsers.add_parser("data", help="Dataset ingest/build commands.")
    data_sub = data_parser.add_subparsers(dest="data_command", required=True)

    ingest_parser = data_sub.add_parser("ingest", help="Ingest source datasets into unified index.")
    ingest_parser.add_argument("--config", required=True, type=Path)

    build_parser = data_sub.add_parser(
        "build-base100k",
        help="Build the combined color+grayscale dataset with resumable history.",
    )
    build_parser.add_argument("--config", required=True, type=Path)

    resume_parser = data_sub.add_parser("resume", help="Resume a failed/interrupted build run.")
    resume_parser.add_argument("--run-id", required=True)
    resume_parser.add_argument("--runs-root", default="runs")

    train_parser = subparsers.add_parser("train", help="Training launch commands.")
    train_sub = train_parser.add_subparsers(dest="train_command", required=True)

    train_sdxl = train_sub.add_parser("sdxl", help="Launch SDXL training command.")
    train_sdxl.add_argument("--config", required=True, type=Path)
    train_sdxl.add_argument("--dry-run", action="store_true")

    train_lora = train_sub.add_parser("lora", help="Launch LoRA training command.")
    train_lora.add_argument("--config", required=True, type=Path)
    train_lora.add_argument("--dry-run", action="store_true")

    infer_parser = subparsers.add_parser("infer", help="Inference orchestration commands.")
    infer_sub = infer_parser.add_subparsers(dest="infer_command", required=True)

    infer_graygen = infer_sub.add_parser(
        "graygen",
        help="Run grayscale structure generation pipeline via ComfyUI.",
    )
    infer_graygen.add_argument("--config", required=True, type=Path)
    infer_graygen.add_argument("--dry-run", action="store_true")

    infer_recolor = infer_sub.add_parser(
        "recolor",
        help="Run prompt-based recolor img2img pipeline via ComfyUI.",
    )
    infer_recolor.add_argument("--config", required=True, type=Path)
    infer_recolor.add_argument("--dry-run", action="store_true")

    infer_refine = infer_sub.add_parser(
        "refine",
        help="Run low-denoise refinement img2img pipeline via ComfyUI.",
    )
    infer_refine.add_argument("--config", required=True, type=Path)
    infer_refine.add_argument("--dry-run", action="store_true")

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    setup_logging()

    if args.namespace == "data":
        if args.data_command == "ingest":
            output = run_ingest(config_path=args.config)
            LOGGER.info("Ingest complete: %s", output)
            return 0
        if args.data_command == "build-base100k":
            run_id = run_build_base100k(config_path=args.config)
            LOGGER.info("Build complete. run_id=%s", run_id)
            return 0
        if args.data_command == "resume":
            run_id = resume_build_base100k(
                run_id=args.run_id,
                runs_root=Path(args.runs_root),
            )
            LOGGER.info("Resume complete. run_id=%s", run_id)
            return 0
        raise ValueError(f"Unsupported data command: {args.data_command}")

    if args.namespace == "train":
        if args.train_command in {"sdxl", "lora"}:
            cmd = run_training_command(config_path=args.config, dry_run=args.dry_run)
            LOGGER.info("Resolved command: %s", " ".join(cmd))
            return 0
        raise ValueError(f"Unsupported train command: {args.train_command}")

    if args.namespace == "infer":
        if args.infer_command in {"graygen", "recolor", "refine"}:
            run_id = run_comfyui_stage(
                config_path=args.config,
                stage=args.infer_command,
                dry_run=args.dry_run,
            )
            LOGGER.info("Inference complete. run_id=%s", run_id)
            return 0
        raise ValueError(f"Unsupported infer command: {args.infer_command}")

    raise ValueError(f"Unsupported namespace: {args.namespace}")


if __name__ == "__main__":
    raise SystemExit(main())
