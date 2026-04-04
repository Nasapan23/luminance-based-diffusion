from __future__ import annotations

from pathlib import Path

from lbd.train import launch


def _norm(value: str) -> str:
    return value.replace("\\", "/")


def _base_config() -> dict:
    return {
        "command": {
            "accelerate_binary": "accelerate",
            "script": "external/diffusers/examples/text_to_image/train_text_to_image_sdxl.py",
            "args": {
                "train_data_dir": "data/base1k/gray/train",
                "mixed_precision": "fp16",
                "max_train_steps": 10,
            },
        },
        "runtime": {
            "device": "auto",
            "prefer_cuda": True,
            "allow_cpu_fallback": True,
            "mixed_precision": {"cuda": "fp16", "cpu": "no"},
            "accelerate_launch_args": [],
            "env": {},
        },
    }


def test_runtime_auto_falls_back_to_cpu_when_cuda_missing(monkeypatch) -> None:
    monkeypatch.setattr(
        launch,
        "_detect_cuda_state",
        lambda: {
            "torch_available": True,
            "cuda_available": False,
            "cuda_device_count": 0,
            "cuda_device_name": "",
            "reason": "",
        },
    )
    cfg = _base_config()
    cmd, env, plan = launch._build_train_command(cfg, Path.cwd())

    assert plan["selected_device"] == "cpu"
    assert plan["use_cpu"] is True
    assert "--cpu" in cmd
    assert _norm(cmd[3]).endswith("src/lbd/train/accelerate_entry.py")
    assert _norm(cmd[4]).endswith("external/diffusers/examples/text_to_image/train_text_to_image_sdxl.py")
    assert "--mixed_precision" in cmd
    assert "no" in cmd
    assert env == {}


def test_runtime_auto_uses_cuda_when_available(monkeypatch) -> None:
    monkeypatch.setattr(
        launch,
        "_detect_cuda_state",
        lambda: {
            "torch_available": True,
            "cuda_available": True,
            "cuda_device_count": 1,
            "cuda_device_name": "NVIDIA RTX 4090",
            "reason": "",
        },
    )
    cfg = _base_config()
    cfg["runtime"]["cuda_visible_devices"] = "0"
    cmd, env, plan = launch._build_train_command(cfg, Path.cwd())

    assert plan["selected_device"] == "cuda"
    assert plan["use_cpu"] is False
    assert "--cpu" not in cmd
    assert _norm(cmd[2]).endswith("src/lbd/train/accelerate_entry.py")
    assert _norm(cmd[3]).endswith("external/diffusers/examples/text_to_image/train_text_to_image_sdxl.py")
    assert env["CUDA_VISIBLE_DEVICES"] == "0"
