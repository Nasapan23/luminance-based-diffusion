from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

from lbd.config import dump_yaml, load_yaml
from lbd.infer.comfyui import run_comfyui_stage


def _resolve_repo_path(path_value: str | Path, repo_root: Path) -> Path:
    candidate = Path(path_value)
    if candidate.is_absolute():
        return candidate
    return (repo_root / candidate).resolve()


def _stage_run_dir(config: dict[str, Any], run_id: str, repo_root: Path) -> Path:
    runs_root_value = config.get("runs_root", "runs/infer")
    runs_root = _resolve_repo_path(str(runs_root_value), repo_root)
    return runs_root / run_id


def _load_resolved_jobs(run_dir: Path) -> list[dict[str, Any]]:
    jobs_path = run_dir / "jobs.resolved.json"
    with jobs_path.open("r", encoding="utf-8") as handle:
        return list(json.load(handle))


def _load_results(run_dir: Path) -> dict[str, dict[str, str]]:
    results_path = run_dir / "results.csv"
    with results_path.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    return {str(row["job_id"]): row for row in rows}


def _first_output_path(
    result_row: dict[str, str],
    run_dir: Path,
    job_id: str,
    repo_root: Path,
    dry_run: bool,
) -> str:
    output_files = str(result_row.get("output_files", "")).strip()
    if output_files:
        first = output_files.split("|", 1)[0].strip()
        if first:
            return first
    expected = run_dir / "outputs" / f"{job_id}_000.png"
    if dry_run:
        return expected.as_posix()
    return expected.relative_to(repo_root).as_posix()


def _make_stage_jobs_from_previous(
    previous_run_dir: Path,
    repo_root: Path,
    dry_run: bool,
    job_prefix: str,
) -> list[dict[str, Any]]:
    resolved_jobs = _load_resolved_jobs(previous_run_dir)
    results = _load_results(previous_run_dir)

    jobs: list[dict[str, Any]] = []
    for index, job in enumerate(resolved_jobs, start=1):
        prev_job_id = str(job["job_id"])
        result_row = results.get(prev_job_id, {})
        input_image = _first_output_path(
            result_row=result_row,
            run_dir=previous_run_dir,
            job_id=prev_job_id,
            repo_root=repo_root,
            dry_run=dry_run,
        )
        jobs.append(
            {
                "job_id": f"{job_prefix}_{index:03d}",
                "input_image": input_image,
            }
        )
    return jobs


def _write_derived_config(
    base_config: dict[str, Any],
    jobs: list[dict[str, Any]],
    path: Path,
    run_name_suffix: str,
) -> Path:
    config = dict(base_config)
    config["run_name"] = f"{config.get('run_name', path.stem)}_{run_name_suffix}"
    config["jobs"] = jobs
    dump_yaml(config, path)
    return path


def run_chained_comfyui_pipeline(
    gray_config_path: Path,
    recolor_config_path: Path,
    refine_config_path: Path | None = None,
    dry_run: bool = False,
    repo_root: Path | None = None,
) -> dict[str, str]:
    root = (repo_root or Path.cwd()).resolve()

    gray_config = load_yaml(gray_config_path)
    recolor_config = load_yaml(recolor_config_path)
    refine_config = load_yaml(refine_config_path) if refine_config_path else None

    gray_run_id = run_comfyui_stage(
        config_path=gray_config_path,
        stage="graygen",
        dry_run=dry_run,
        repo_root=root,
    )
    gray_run_dir = _stage_run_dir(gray_config, gray_run_id, root)

    recolor_jobs = _make_stage_jobs_from_previous(
        previous_run_dir=gray_run_dir,
        repo_root=root,
        dry_run=dry_run,
        job_prefix="recolor",
    )
    recolor_cfg_path = gray_run_dir / "recolor.derived.yaml"
    _write_derived_config(recolor_config, recolor_jobs, recolor_cfg_path, "derived")
    recolor_run_id = run_comfyui_stage(
        config_path=recolor_cfg_path,
        stage="recolor",
        dry_run=dry_run,
        repo_root=root,
    )

    outputs = {"graygen_run_id": gray_run_id, "recolor_run_id": recolor_run_id}

    if refine_config is not None:
        recolor_run_dir = _stage_run_dir(recolor_config, recolor_run_id, root)
        refine_jobs = _make_stage_jobs_from_previous(
            previous_run_dir=recolor_run_dir,
            repo_root=root,
            dry_run=dry_run,
            job_prefix="refine",
        )
        refine_cfg_path = recolor_run_dir / "refine.derived.yaml"
        _write_derived_config(refine_config, refine_jobs, refine_cfg_path, "derived")
        refine_run_id = run_comfyui_stage(
            config_path=refine_cfg_path,
            stage="refine",
            dry_run=dry_run,
            repo_root=root,
        )
        outputs["refine_run_id"] = refine_run_id

    return outputs


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m lbd.infer.pipeline",
        description="Run graygen -> recolor -> optional refine pipeline.",
    )
    parser.add_argument("--gray-config", type=Path, required=True)
    parser.add_argument("--recolor-config", type=Path, required=True)
    parser.add_argument("--refine-config", type=Path)
    parser.add_argument("--dry-run", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    outputs = run_chained_comfyui_pipeline(
        gray_config_path=args.gray_config,
        recolor_config_path=args.recolor_config,
        refine_config_path=args.refine_config,
        dry_run=args.dry_run,
    )
    print(json.dumps(outputs, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
