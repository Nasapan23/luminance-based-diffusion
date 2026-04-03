from __future__ import annotations

import csv
from pathlib import Path

from PIL import Image

from lbd.config import dump_yaml
from lbd.infer.comfyui import run_comfyui_stage


def test_infer_graygen_dry_run_writes_run_artifacts(tmp_path: Path) -> None:
    repo = tmp_path
    configs_dir = repo / "configs"
    configs_dir.mkdir(parents=True, exist_ok=True)

    config = {
        "backend": "comfyui",
        "run_name": "dryrun_graygen",
        "runs_root": "runs/infer",
        "seed": 7,
        "comfyui": {
            "base_url": "http://127.0.0.1:8188",
            "timeout_sec": 5,
            "prompt_timeout_sec": 30,
            "poll_interval_sec": 1,
        },
        "model": {
            "checkpoint_name": "gray_sdxl.safetensors",
        },
        "defaults": {
            "prompt": "grayscale vase",
            "negative_prompt": "blurry",
            "width": 512,
            "height": 512,
            "steps": 20,
            "cfg": 5.5,
            "denoise": 1.0,
            "filename_prefix": "lbd/graygen",
        },
        "jobs": [
            {"job_id": "jobA", "seed": 111},
            {"job_id": "jobB", "seed": 112},
        ],
    }
    path = configs_dir / "infer_graygen.yaml"
    dump_yaml(config, path)

    run_id = run_comfyui_stage(path, stage="graygen", dry_run=True, repo_root=repo)
    run_dir = repo / "runs/infer" / run_id
    assert run_dir.exists()
    assert (run_dir / "config.resolved.yaml").exists()
    assert (run_dir / "jobs.resolved.json").exists()
    assert (run_dir / "workflows/jobA.json").exists()
    assert (run_dir / "workflows/jobB.json").exists()

    results_path = run_dir / "results.csv"
    with results_path.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == 2
    assert all(row["status"] == "dry_run" for row in rows)


def test_infer_recolor_dry_run_requires_existing_input_image(tmp_path: Path) -> None:
    repo = tmp_path
    input_path = repo / "data/experiments/inputs/sample_gray.jpg"
    input_path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (16, 16), color=(128, 128, 128)).save(input_path)

    config = {
        "backend": "comfyui",
        "run_name": "dryrun_recolor",
        "runs_root": "runs/infer",
        "seed": 3,
        "comfyui": {"base_url": "http://127.0.0.1:8188"},
        "model": {
            "checkpoint_name": "gray_sdxl.safetensors",
        },
        "defaults": {
            "prompt": "colored amphora",
            "input_image": "data/experiments/inputs/sample_gray.jpg",
            "steps": 20,
            "cfg": 6.0,
            "denoise": 0.25,
            "filename_prefix": "lbd/recolor",
        },
        "jobs": [{"job_id": "recolorA"}],
    }
    config_path = repo / "configs/infer_recolor.yaml"
    config_path.parent.mkdir(parents=True, exist_ok=True)
    dump_yaml(config, config_path)

    run_id = run_comfyui_stage(config_path, stage="recolor", dry_run=True, repo_root=repo)
    run_dir = repo / "runs/infer" / run_id
    assert run_dir.exists()
    assert (run_dir / "workflows/recolorA.json").exists()
