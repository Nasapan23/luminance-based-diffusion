from __future__ import annotations

import logging
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

from lbd.config import load_yaml


LOGGER = logging.getLogger(__name__)


def _flag_name(key: str) -> str:
    return f"--{key}"


def _append_arg(cmd: list[str], key: str, value) -> None:
    flag = _flag_name(key)
    if isinstance(value, bool):
        if value:
            cmd.append(flag)
        return
    if value is None:
        return
    if isinstance(value, list):
        for entry in value:
            cmd.extend([flag, str(entry)])
        return
    cmd.extend([flag, str(value)])


def _resolve_script_path(script_value: str, repo_root: Path) -> Path:
    script_path = Path(script_value)
    if not script_path.is_absolute():
        script_path = (repo_root / script_path).resolve()
    return script_path


def _detect_cuda_state() -> dict[str, Any]:
    try:
        import torch  # type: ignore
    except Exception as exc:  # pragma: no cover - depends on environment
        return {
            "torch_available": False,
            "cuda_available": False,
            "cuda_device_count": 0,
            "cuda_device_name": "",
            "reason": str(exc),
        }

    try:
        cuda_available = bool(torch.cuda.is_available())
        cuda_device_count = int(torch.cuda.device_count()) if cuda_available else 0
        cuda_device_name = (
            str(torch.cuda.get_device_name(0))
            if cuda_available and cuda_device_count > 0
            else ""
        )
    except Exception as exc:  # pragma: no cover - defensive
        cuda_available = False
        cuda_device_count = 0
        cuda_device_name = ""
        return {
            "torch_available": True,
            "cuda_available": False,
            "cuda_device_count": 0,
            "cuda_device_name": "",
            "reason": str(exc),
        }

    return {
        "torch_available": True,
        "cuda_available": cuda_available,
        "cuda_device_count": cuda_device_count,
        "cuda_device_name": cuda_device_name,
        "reason": "",
    }


def _resolve_runtime_policy(config: dict) -> dict[str, Any]:
    runtime_cfg = dict(config.get("runtime") or {})
    device_policy = str(runtime_cfg.get("device", "auto")).strip().lower()
    prefer_cuda = bool(runtime_cfg.get("prefer_cuda", True))
    allow_cpu_fallback = bool(runtime_cfg.get("allow_cpu_fallback", True))

    if device_policy not in {"auto", "cuda", "cpu"}:
        raise ValueError("runtime.device must be one of: auto, cuda, cpu")

    cuda_state = _detect_cuda_state()
    cuda_available = bool(cuda_state.get("cuda_available", False))

    if device_policy == "cpu":
        use_cpu = True
        selected_device = "cpu"
    elif device_policy == "cuda":
        if not cuda_available:
            raise RuntimeError(
                "runtime.device=cuda requested but CUDA is unavailable. "
                f"Details: {cuda_state.get('reason', 'unknown')}"
            )
        use_cpu = False
        selected_device = "cuda"
    else:
        if prefer_cuda and cuda_available:
            use_cpu = False
            selected_device = "cuda"
        else:
            if prefer_cuda and not allow_cpu_fallback:
                raise RuntimeError(
                    "CUDA preferred but unavailable and CPU fallback disabled. "
                    "Set runtime.allow_cpu_fallback=true or runtime.device=cpu."
                )
            use_cpu = True
            selected_device = "cpu"

    mixed_precision_cfg = dict(runtime_cfg.get("mixed_precision") or {})
    selected_mixed_precision = mixed_precision_cfg.get(selected_device)

    launch_args = [str(item) for item in runtime_cfg.get("accelerate_launch_args", [])]
    if use_cpu and "--cpu" not in launch_args:
        launch_args = ["--cpu", *launch_args]

    env_overrides = {str(k): str(v) for k, v in dict(runtime_cfg.get("env") or {}).items()}
    cuda_visible_devices = runtime_cfg.get("cuda_visible_devices")
    if cuda_visible_devices not in (None, ""):
        env_overrides["CUDA_VISIBLE_DEVICES"] = str(cuda_visible_devices)

    return {
        "selected_device": selected_device,
        "use_cpu": use_cpu,
        "launch_args": launch_args,
        "env_overrides": env_overrides,
        "selected_mixed_precision": selected_mixed_precision,
        "cuda_state": cuda_state,
    }


def _build_train_command(config: dict, repo_root: Path) -> tuple[list[str], dict[str, str], dict[str, Any]]:
    command_cfg = config.get("command", {})
    args_cfg = dict(command_cfg.get("args", {}))
    runtime_plan = _resolve_runtime_policy(config)

    accelerate_binary = str(command_cfg.get("accelerate_binary", "accelerate"))
    script_value = command_cfg.get("script")
    if not script_value:
        raise ValueError("Training config missing command.script")

    script_path = _resolve_script_path(str(script_value), repo_root)
    wrapper_path = (repo_root / "src" / "lbd" / "train" / "accelerate_entry.py").resolve()
    if not wrapper_path.exists():
        raise FileNotFoundError(f"Training wrapper missing: {wrapper_path}")

    selected_mixed_precision = runtime_plan.get("selected_mixed_precision")
    if selected_mixed_precision is not None:
        args_cfg["mixed_precision"] = str(selected_mixed_precision)

    cmd = [accelerate_binary, "launch", *runtime_plan["launch_args"], str(wrapper_path), str(script_path)]
    for key, value in args_cfg.items():
        _append_arg(cmd, key, value)

    for extra in command_cfg.get("extra_cli", []):
        cmd.append(str(extra))
    return cmd, runtime_plan["env_overrides"], runtime_plan


def _prepend_pythonpath(env: dict[str, str], path: Path) -> None:
    candidate = str(path.resolve())
    current = env.get("PYTHONPATH", "")
    if not current:
        env["PYTHONPATH"] = candidate
        return

    parts = [p for p in current.split(os.pathsep) if p]
    if candidate in parts:
        return
    env["PYTHONPATH"] = os.pathsep.join([candidate, *parts])


def _resolve_output_dir(config: dict, repo_root: Path) -> Path | None:
    command_cfg = dict(config.get("command", {}))
    args_cfg = dict(command_cfg.get("args", {}))
    output_dir = args_cfg.get("output_dir")
    if output_dir in (None, ""):
        return None
    return _resolve_script_path(str(output_dir), repo_root)


def _run_with_logfile(cmd: list[str], env: dict[str, str], log_path: Path) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("a", encoding="utf-8") as handle:
        handle.write(f"$ {' '.join(cmd)}\n")
        handle.flush()
        with subprocess.Popen(
            cmd,
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        ) as process:
            assert process.stdout is not None
            for line in process.stdout:
                sys.stdout.write(line)
                handle.write(line)
            return_code = process.wait()
            if return_code != 0:
                raise subprocess.CalledProcessError(return_code, cmd)


def run_training_command(
    config_path: Path,
    dry_run: bool = False,
    repo_root: Path | None = None,
) -> list[str]:
    root = (repo_root or Path.cwd()).resolve()
    config = load_yaml(config_path)
    cmd, env_overrides, runtime_plan = _build_train_command(config, root)
    output_dir = _resolve_output_dir(config, root)
    cuda_state = runtime_plan.get("cuda_state", {})
    LOGGER.info(
        "Training runtime: device=%s use_cpu=%s cuda_available=%s device_name=%s",
        runtime_plan.get("selected_device"),
        runtime_plan.get("use_cpu"),
        cuda_state.get("cuda_available"),
        cuda_state.get("cuda_device_name", ""),
    )
    if env_overrides:
        LOGGER.info("Training env overrides: %s", env_overrides)
    LOGGER.info("Training command: %s", " ".join(cmd))
    if dry_run:
        return cmd

    env = os.environ.copy()
    env.update(env_overrides)
    external_diffusers_src = (root / "external" / "diffusers" / "src").resolve()
    if external_diffusers_src.exists():
        _prepend_pythonpath(env, external_diffusers_src)
        LOGGER.info("Prepended external diffusers to PYTHONPATH: %s", external_diffusers_src)
    if output_dir is not None:
        log_path = output_dir / "train.log"
        LOGGER.info("Streaming training output to log file: %s", log_path)
        _run_with_logfile(cmd, env=env, log_path=log_path)
    else:
        subprocess.run(cmd, check=True, env=env)
    return cmd
