from __future__ import annotations

import json
from pathlib import Path

from lbd.config import dump_yaml
from lbd.infer.pipeline import run_chained_comfyui_pipeline


def test_chained_pipeline_dry_run_writes_derived_stage_configs(tmp_path: Path) -> None:
    repo = tmp_path
    configs_dir = repo / "configs"
    configs_dir.mkdir(parents=True, exist_ok=True)

    gray_cfg = {
        "backend": "comfyui",
        "run_name": "gray_chain",
        "runs_root": "runs/infer",
        "seed": 11,
        "comfyui": {"base_url": "http://127.0.0.1:8188"},
        "model": {"checkpoint_name": "gray_sdxl.safetensors", "lora_name": "gray_lora.safetensors"},
        "defaults": {
            "prompt": "grayscale amphora",
            "negative_prompt": "bad",
            "width": 512,
            "height": 512,
            "steps": 10,
            "cfg": 5.5,
            "filename_prefix": "lbd/graygen",
        },
        "jobs": [{"job_id": "grayA"}, {"job_id": "grayB"}],
    }
    recolor_cfg = {
        "backend": "comfyui",
        "run_name": "recolor_chain",
        "runs_root": "runs/infer",
        "seed": 22,
        "comfyui": {"base_url": "http://127.0.0.1:8188"},
        "model": {"checkpoint_name": "color_sdxl.safetensors"},
        "defaults": {
            "prompt": "black figure amphora with red clay body",
            "negative_prompt": "bad",
            "steps": 10,
            "cfg": 5.5,
            "denoise": 0.25,
            "filename_prefix": "lbd/recolor",
        },
        "jobs": [],
    }
    refine_cfg = {
        "backend": "comfyui",
        "run_name": "refine_chain",
        "runs_root": "runs/infer",
        "seed": 33,
        "comfyui": {"base_url": "http://127.0.0.1:8188"},
        "model": {"checkpoint_name": "refiner.safetensors"},
        "defaults": {
            "prompt": "refined amphora studio photograph",
            "negative_prompt": "bad",
            "steps": 8,
            "cfg": 5.0,
            "denoise": 0.18,
            "filename_prefix": "lbd/refine",
        },
        "jobs": [],
    }

    gray_path = configs_dir / "gray.yaml"
    recolor_path = configs_dir / "recolor.yaml"
    refine_path = configs_dir / "refine.yaml"
    dump_yaml(gray_cfg, gray_path)
    dump_yaml(recolor_cfg, recolor_path)
    dump_yaml(refine_cfg, refine_path)

    outputs = run_chained_comfyui_pipeline(
        gray_config_path=gray_path,
        recolor_config_path=recolor_path,
        refine_config_path=refine_path,
        dry_run=True,
        repo_root=repo,
    )

    gray_run_dir = repo / "runs/infer" / outputs["graygen_run_id"]
    recolor_run_dir = repo / "runs/infer" / outputs["recolor_run_id"]
    refine_run_dir = repo / "runs/infer" / outputs["refine_run_id"]

    assert (gray_run_dir / "recolor.derived.yaml").exists()
    assert (recolor_run_dir / "refine.derived.yaml").exists()

    recolor_jobs = json.loads((gray_run_dir / "jobs.resolved.json").read_text(encoding="utf-8"))
    assert len(recolor_jobs) == 2

    derived_recolor = (gray_run_dir / "recolor.derived.yaml").read_text(encoding="utf-8")
    assert "input_image:" in derived_recolor
    assert "grayA_000.png" in derived_recolor

    assert (refine_run_dir / "results.csv").exists()
