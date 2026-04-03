from __future__ import annotations

import csv
import json
import logging
import random
import re
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import requests

from lbd.config import dump_yaml, load_yaml, make_config_hash
from lbd.utils.fs_utils import as_repo_relative


LOGGER = logging.getLogger(__name__)
DEFAULT_NEGATIVE_PROMPT = "low quality, blurry, deformed, artifacts"


def _utc_timestamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")


def _resolve_repo_path(path_value: str | Path, repo_root: Path) -> Path:
    candidate = Path(path_value)
    if candidate.is_absolute():
        return candidate
    return (repo_root / candidate).resolve()


def _safe_token(value: str) -> str:
    return re.sub(r"[^a-zA-Z0-9._-]+", "_", value).strip("_") or "run"


def _to_int(value: Any, field_name: str) -> int:
    try:
        return int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Expected integer for '{field_name}', got: {value!r}") from exc


def _to_float(value: Any, field_name: str) -> float:
    try:
        return float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Expected float for '{field_name}', got: {value!r}") from exc


class WorkflowBuilder:
    def __init__(self):
        self._next_id = 1
        self.graph: dict[str, dict[str, Any]] = {}

    def add(self, class_type: str, inputs: dict[str, Any], title: str | None = None) -> str:
        node_id = str(self._next_id)
        self._next_id += 1
        node = {"inputs": inputs, "class_type": class_type}
        if title:
            node["_meta"] = {"title": title}
        self.graph[node_id] = node
        return node_id


def _attach_model_and_clip(
    builder: WorkflowBuilder,
    checkpoint_name: str,
    lora_name: str,
    lora_strength_model: float,
    lora_strength_clip: float,
) -> tuple[list[Any], list[Any], list[Any]]:
    ckpt = builder.add(
        "CheckpointLoaderSimple",
        {"ckpt_name": checkpoint_name},
        title="Checkpoint Loader",
    )
    model_ref: list[Any] = [ckpt, 0]
    clip_ref: list[Any] = [ckpt, 1]
    vae_ref: list[Any] = [ckpt, 2]

    if lora_name:
        lora = builder.add(
            "LoraLoader",
            {
                "model": model_ref,
                "clip": clip_ref,
                "lora_name": lora_name,
                "strength_model": lora_strength_model,
                "strength_clip": lora_strength_clip,
            },
            title="LoRA Loader",
        )
        model_ref = [lora, 0]
        clip_ref = [lora, 1]

    return model_ref, clip_ref, vae_ref


def build_workflow_for_job(stage: str, job: dict[str, Any]) -> dict[str, Any]:
    stage_name = stage.lower().strip()
    if stage_name not in {"graygen", "recolor", "refine"}:
        raise ValueError(f"Unsupported stage: {stage}")

    checkpoint_name = str(job.get("checkpoint_name", "")).strip()
    if not checkpoint_name:
        raise ValueError("checkpoint_name is required for ComfyUI inference.")

    lora_name = str(job.get("lora_name", "")).strip()
    lora_strength_model = _to_float(job.get("lora_strength_model", 1.0), "lora_strength_model")
    lora_strength_clip = _to_float(job.get("lora_strength_clip", 1.0), "lora_strength_clip")
    prompt = str(job.get("prompt", "")).strip()
    if not prompt:
        raise ValueError("prompt is required for inference job.")
    negative_prompt = str(job.get("negative_prompt", DEFAULT_NEGATIVE_PROMPT)).strip()

    steps = _to_int(job.get("steps", 35), "steps")
    cfg = _to_float(job.get("cfg", 6.0), "cfg")
    seed = _to_int(job.get("seed", 0), "seed")
    sampler_name = str(job.get("sampler_name", "euler"))
    scheduler = str(job.get("scheduler", "normal"))
    denoise = _to_float(job.get("denoise", 1.0), "denoise")
    filename_prefix = str(job.get("filename_prefix", f"lbd/{stage_name}")).strip()

    builder = WorkflowBuilder()
    model_ref, clip_ref, vae_ref = _attach_model_and_clip(
        builder=builder,
        checkpoint_name=checkpoint_name,
        lora_name=lora_name,
        lora_strength_model=lora_strength_model,
        lora_strength_clip=lora_strength_clip,
    )

    positive = builder.add(
        "CLIPTextEncode",
        {"text": prompt, "clip": clip_ref},
        title="Positive Prompt",
    )
    negative = builder.add(
        "CLIPTextEncode",
        {"text": negative_prompt, "clip": clip_ref},
        title="Negative Prompt",
    )

    if stage_name == "graygen":
        width = _to_int(job.get("width", 1024), "width")
        height = _to_int(job.get("height", 1024), "height")
        batch_size = _to_int(job.get("batch_size", 1), "batch_size")
        latent = builder.add(
            "EmptyLatentImage",
            {"width": width, "height": height, "batch_size": batch_size},
            title="Empty Latent",
        )
    else:
        input_image = str(job.get("comfy_input_image", "")).strip()
        if not input_image:
            raise ValueError(
                f"comfy_input_image is required for stage '{stage_name}'."
            )
        load_image = builder.add(
            "LoadImage",
            {"image": input_image, "upload": "image"},
            title="Load Input Image",
        )
        latent = builder.add(
            "VAEEncode",
            {"pixels": [load_image, 0], "vae": vae_ref},
            title="VAE Encode",
        )

    sampler = builder.add(
        "KSampler",
        {
            "seed": seed,
            "steps": steps,
            "cfg": cfg,
            "sampler_name": sampler_name,
            "scheduler": scheduler,
            "denoise": denoise,
            "model": model_ref,
            "positive": [positive, 0],
            "negative": [negative, 0],
            "latent_image": [latent, 0],
        },
        title="KSampler",
    )
    decode = builder.add(
        "VAEDecode",
        {"samples": [sampler, 0], "vae": vae_ref},
        title="VAE Decode",
    )
    builder.add(
        "SaveImage",
        {"filename_prefix": filename_prefix, "images": [decode, 0]},
        title="Save Output",
    )

    return builder.graph


class ComfyUIClient:
    def __init__(self, base_url: str, timeout_sec: float = 60.0):
        self.base_url = base_url.rstrip("/")
        self.timeout_sec = timeout_sec
        self.session = requests.Session()

    def _url(self, path: str) -> str:
        return f"{self.base_url}{path}"

    def ping(self) -> None:
        response = self.session.get(self._url("/system_stats"), timeout=self.timeout_sec)
        response.raise_for_status()

    def upload_image(self, image_path: Path) -> str:
        with image_path.open("rb") as handle:
            files = {"image": (image_path.name, handle, "application/octet-stream")}
            data = {"type": "input", "overwrite": "false"}
            response = self.session.post(
                self._url("/upload/image"),
                files=files,
                data=data,
                timeout=self.timeout_sec,
            )
        response.raise_for_status()
        payload = response.json()
        uploaded_name = payload.get("name") or payload.get("filename")
        if not uploaded_name:
            raise RuntimeError(f"Upload succeeded but no filename returned: {payload}")
        return str(uploaded_name)

    def queue_prompt(self, prompt_graph: dict[str, Any], client_id: str) -> str:
        payload = {"prompt": prompt_graph, "client_id": client_id}
        response = self.session.post(
            self._url("/prompt"),
            json=payload,
            timeout=self.timeout_sec,
        )
        response.raise_for_status()
        data = response.json()
        prompt_id = data.get("prompt_id")
        if not prompt_id:
            raise RuntimeError(f"ComfyUI did not return prompt_id: {data}")
        return str(prompt_id)

    def wait_for_prompt(
        self,
        prompt_id: str,
        timeout_sec: float,
        poll_interval_sec: float,
    ) -> dict[str, Any]:
        deadline = time.time() + timeout_sec
        last_payload: dict[str, Any] = {}
        while time.time() <= deadline:
            response = self.session.get(
                self._url(f"/history/{prompt_id}"),
                timeout=self.timeout_sec,
            )
            if response.status_code == 200:
                payload = response.json()
                if isinstance(payload, dict):
                    last_payload = payload
                    record = payload.get(prompt_id)
                    if not record and len(payload) == 1:
                        record = next(iter(payload.values()))
                    if isinstance(record, dict) and record.get("outputs"):
                        return record
            time.sleep(poll_interval_sec)
        raise TimeoutError(
            f"Timed out waiting for prompt {prompt_id}. Last history payload keys: "
            f"{list(last_payload.keys())[:8]}"
        )

    def _download_image_file(
        self,
        filename: str,
        subfolder: str,
        image_type: str,
        dst_path: Path,
    ) -> None:
        params = {
            "filename": filename,
            "subfolder": subfolder,
            "type": image_type,
        }
        response = self.session.get(
            self._url("/view"),
            params=params,
            timeout=self.timeout_sec,
            stream=True,
        )
        response.raise_for_status()
        dst_path.parent.mkdir(parents=True, exist_ok=True)
        with dst_path.open("wb") as handle:
            for chunk in response.iter_content(chunk_size=1024 * 64):
                if chunk:
                    handle.write(chunk)

    def download_outputs(
        self,
        prompt_record: dict[str, Any],
        output_dir: Path,
        file_prefix: str,
    ) -> list[Path]:
        output_dir.mkdir(parents=True, exist_ok=True)
        downloaded: list[Path] = []

        outputs = prompt_record.get("outputs", {})
        index = 0
        for node_output in outputs.values():
            if not isinstance(node_output, dict):
                continue
            images = node_output.get("images", [])
            if not isinstance(images, list):
                continue
            for image_item in images:
                if not isinstance(image_item, dict):
                    continue
                filename = str(image_item.get("filename", "")).strip()
                if not filename:
                    continue
                subfolder = str(image_item.get("subfolder", "")).strip()
                image_type = str(image_item.get("type", "output")).strip() or "output"
                suffix = Path(filename).suffix or ".png"
                dst = output_dir / f"{file_prefix}_{index:03d}{suffix}"
                self._download_image_file(filename, subfolder, image_type, dst)
                downloaded.append(dst)
                index += 1
        return downloaded


def _default_denoise(stage: str) -> float:
    if stage == "graygen":
        return 1.0
    if stage == "recolor":
        return 0.25
    if stage == "refine":
        return 0.18
    return 1.0


def _normalize_jobs(
    config: dict[str, Any],
    stage: str,
    repo_root: Path,
    validate_inputs: bool = True,
) -> list[dict[str, Any]]:
    model_cfg = dict(config.get("model") or {})
    defaults = dict(config.get("defaults") or {})
    jobs_cfg = config.get("jobs") or [{}]
    base_seed = _to_int(config.get("seed", 42), "seed")
    randomize_missing_seed = bool(config.get("randomize_missing_seed", False))

    jobs: list[dict[str, Any]] = []
    for idx, row in enumerate(jobs_cfg, start=1):
        if not isinstance(row, dict):
            raise ValueError(f"Each jobs[] entry must be a mapping. Got: {type(row)}")
        job: dict[str, Any] = {}
        job.update(model_cfg)
        job.update(defaults)
        job.update(row)

        job_id = str(job.get("job_id", f"{stage}_{idx:03d}")).strip()
        job["job_id"] = _safe_token(job_id)

        prompt = str(job.get("prompt") or job.get("positive_prompt") or "").strip()
        if not prompt:
            raise ValueError(f"Job {job['job_id']} is missing 'prompt'.")
        job["prompt"] = prompt
        job["negative_prompt"] = str(
            job.get("negative_prompt", DEFAULT_NEGATIVE_PROMPT)
        ).strip()

        if "seed" in job and job["seed"] not in (None, ""):
            job["seed"] = _to_int(job["seed"], "seed")
        else:
            if randomize_missing_seed:
                job["seed"] = random.randint(1, 2_147_483_647)
            else:
                job["seed"] = base_seed + (idx - 1)

        job["steps"] = _to_int(job.get("steps", 35), "steps")
        job["cfg"] = _to_float(job.get("cfg", 6.0), "cfg")
        job["denoise"] = _to_float(job.get("denoise", _default_denoise(stage)), "denoise")
        job["sampler_name"] = str(job.get("sampler_name", "euler")).strip()
        job["scheduler"] = str(job.get("scheduler", "normal")).strip()
        job["batch_size"] = _to_int(job.get("batch_size", 1), "batch_size")
        job["width"] = _to_int(job.get("width", 1024), "width")
        job["height"] = _to_int(job.get("height", 1024), "height")
        job["filename_prefix"] = str(
            job.get("filename_prefix", f"lbd/{stage}/{job['job_id']}")
        ).strip()

        checkpoint_name = str(job.get("checkpoint_name", "")).strip()
        if not checkpoint_name:
            raise ValueError(f"Job {job['job_id']} is missing model.checkpoint_name.")
        job["checkpoint_name"] = checkpoint_name

        lora_name = str(job.get("lora_name", "")).strip()
        job["lora_name"] = lora_name
        job["lora_strength_model"] = _to_float(job.get("lora_strength_model", 1.0), "lora_strength_model")
        job["lora_strength_clip"] = _to_float(job.get("lora_strength_clip", 1.0), "lora_strength_clip")

        if stage in {"recolor", "refine"}:
            input_value = str(job.get("input_image", "")).strip()
            if not input_value:
                raise ValueError(f"Job {job['job_id']} is missing required 'input_image'.")
            input_abs = _resolve_repo_path(input_value, repo_root)
            if validate_inputs and not input_abs.exists():
                raise FileNotFoundError(
                    f"Job {job['job_id']} input_image not found: {input_abs}"
                )
            job["input_image"] = as_repo_relative(input_abs, repo_root)
            job["input_image_abs"] = input_abs.as_posix()

        jobs.append(job)

    return jobs


def _write_csv(path: Path, rows: list[dict[str, str]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _make_run_dir(config: dict[str, Any], stage: str, repo_root: Path) -> tuple[str, Path]:
    run_name = _safe_token(str(config.get("run_name", stage)))
    runs_root_value = config.get("runs_root", "runs/infer")
    runs_root = _resolve_repo_path(str(runs_root_value), repo_root)
    runs_root.mkdir(parents=True, exist_ok=True)
    run_id = f"{_utc_timestamp()}_{stage}_{run_name}"
    run_dir = runs_root / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "workflows").mkdir(parents=True, exist_ok=True)
    (run_dir / "outputs").mkdir(parents=True, exist_ok=True)
    return run_id, run_dir


def run_comfyui_stage(
    config_path: Path,
    stage: str,
    dry_run: bool = False,
    repo_root: Path | None = None,
) -> str:
    stage_name = stage.lower().strip()
    if stage_name not in {"graygen", "recolor", "refine"}:
        raise ValueError(f"Unsupported inference stage: {stage}")

    root = (repo_root or Path.cwd()).resolve()
    config = load_yaml(config_path)
    backend = str(config.get("backend", "comfyui")).strip().lower()
    if backend != "comfyui":
        raise ValueError(f"Unsupported backend '{backend}'. Only 'comfyui' is implemented.")

    run_id, run_dir = _make_run_dir(config, stage_name, root)
    resolved_config = dict(config)
    resolved_config["run_id"] = run_id
    resolved_config["stage"] = stage_name
    resolved_config["config_path"] = as_repo_relative(Path(config_path), root)
    resolved_config["config_hash"] = make_config_hash(resolved_config)
    dump_yaml(resolved_config, run_dir / "config.resolved.yaml")

    jobs = _normalize_jobs(config, stage_name, root, validate_inputs=not dry_run)
    jobs_path = run_dir / "jobs.resolved.json"
    jobs_path.write_text(json.dumps(jobs, indent=2), encoding="utf-8")

    result_rows: list[dict[str, str]] = []
    event_rows: list[dict[str, str]] = []
    event_rows.append(
        {
            "ts_utc": datetime.now(timezone.utc).isoformat(),
            "event": "run_started",
            "message": f"stage={stage_name} jobs={len(jobs)} dry_run={dry_run}",
        }
    )

    comfy_cfg = dict(config.get("comfyui") or {})
    client_id = str(comfy_cfg.get("client_id") or f"lbd-{uuid.uuid4().hex[:12]}")
    timeout_sec = _to_float(comfy_cfg.get("timeout_sec", 60), "comfyui.timeout_sec")
    prompt_timeout_sec = _to_float(
        comfy_cfg.get("prompt_timeout_sec", 1800), "comfyui.prompt_timeout_sec"
    )
    poll_interval_sec = _to_float(
        comfy_cfg.get("poll_interval_sec", 2), "comfyui.poll_interval_sec"
    )
    base_url = str(comfy_cfg.get("base_url", "http://127.0.0.1:8188")).strip()
    upload_inputs = bool(comfy_cfg.get("upload_inputs", True))

    client: ComfyUIClient | None = None
    if not dry_run:
        client = ComfyUIClient(base_url=base_url, timeout_sec=timeout_sec)
        client.ping()
        event_rows.append(
            {
                "ts_utc": datetime.now(timezone.utc).isoformat(),
                "event": "comfyui_ping_ok",
                "message": f"Connected to {base_url}",
            }
        )

    failed = 0
    for job in jobs:
        job_id = str(job["job_id"])
        workflow_job = dict(job)
        prompt_id = ""
        output_paths: list[Path] = []
        error = ""

        try:
            if stage_name in {"recolor", "refine"}:
                if dry_run:
                    workflow_job["comfy_input_image"] = Path(job["input_image"]).name
                else:
                    assert client is not None
                    input_abs = Path(job["input_image_abs"])
                    if upload_inputs:
                        uploaded_name = client.upload_image(input_abs)
                        workflow_job["comfy_input_image"] = uploaded_name
                    else:
                        workflow_job["comfy_input_image"] = input_abs.name

            workflow = build_workflow_for_job(stage_name, workflow_job)
            workflow_path = run_dir / "workflows" / f"{job_id}.json"
            workflow_path.write_text(json.dumps(workflow, indent=2), encoding="utf-8")

            if dry_run:
                status = "dry_run"
            else:
                assert client is not None
                prompt_id = client.queue_prompt(workflow, client_id=client_id)
                record = client.wait_for_prompt(
                    prompt_id=prompt_id,
                    timeout_sec=prompt_timeout_sec,
                    poll_interval_sec=poll_interval_sec,
                )
                output_paths = client.download_outputs(
                    prompt_record=record,
                    output_dir=run_dir / "outputs",
                    file_prefix=job_id,
                )
                status = "done" if output_paths else "done_no_files"

            event_rows.append(
                {
                    "ts_utc": datetime.now(timezone.utc).isoformat(),
                    "event": "job_finished",
                    "message": f"job_id={job_id} status={status} outputs={len(output_paths)}",
                }
            )
        except Exception as exc:
            failed += 1
            status = "failed"
            error = str(exc)
            event_rows.append(
                {
                    "ts_utc": datetime.now(timezone.utc).isoformat(),
                    "event": "job_failed",
                    "message": f"job_id={job_id} error={error}",
                }
            )
            LOGGER.exception("Inference job failed: %s", job_id)

        result_rows.append(
            {
                "job_id": job_id,
                "status": status,
                "prompt_id": prompt_id,
                "seed": str(job["seed"]),
                "steps": str(job["steps"]),
                "cfg": str(job["cfg"]),
                "denoise": str(job["denoise"]),
                "input_image": str(job.get("input_image", "")),
                "output_count": str(len(output_paths)),
                "output_files": "|".join(as_repo_relative(p, root) for p in output_paths),
                "prompt": str(job["prompt"]),
                "negative_prompt": str(job["negative_prompt"]),
                "error": error,
            }
        )

    event_rows.append(
        {
            "ts_utc": datetime.now(timezone.utc).isoformat(),
            "event": "run_finished",
            "message": f"status={'completed' if failed == 0 else 'partial_failed'} failed={failed}",
        }
    )

    result_fields = [
        "job_id",
        "status",
        "prompt_id",
        "seed",
        "steps",
        "cfg",
        "denoise",
        "input_image",
        "output_count",
        "output_files",
        "prompt",
        "negative_prompt",
        "error",
    ]
    _write_csv(run_dir / "results.csv", result_rows, result_fields)
    _write_csv(run_dir / "events.csv", event_rows, ["ts_utc", "event", "message"])

    LOGGER.info(
        "Inference %s finished. run_id=%s failed_jobs=%s run_dir=%s",
        stage_name,
        run_id,
        failed,
        run_dir,
    )
    return run_id
