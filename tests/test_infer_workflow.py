from __future__ import annotations

from lbd.infer.comfyui import build_workflow_for_job


def _class_types(workflow: dict) -> set[str]:
    return {node["class_type"] for node in workflow.values()}


def test_build_graygen_workflow_contains_expected_nodes() -> None:
    workflow = build_workflow_for_job(
        "graygen",
        {
            "checkpoint_name": "gray_sdxl.safetensors",
            "lora_name": "vase_lora.safetensors",
            "lora_strength_model": 1.0,
            "lora_strength_clip": 1.0,
            "prompt": "grayscale vase",
            "negative_prompt": "blurry",
            "width": 1024,
            "height": 1024,
            "batch_size": 1,
            "seed": 123,
            "steps": 35,
            "cfg": 6.0,
            "sampler_name": "euler",
            "scheduler": "normal",
            "denoise": 1.0,
            "filename_prefix": "lbd/graygen/test",
        },
    )

    classes = _class_types(workflow)
    assert "CheckpointLoaderSimple" in classes
    assert "LoraLoader" in classes
    assert "EmptyLatentImage" in classes
    assert "KSampler" in classes
    assert "SaveImage" in classes


def test_build_recolor_workflow_contains_loadimage_and_vaeencode() -> None:
    workflow = build_workflow_for_job(
        "recolor",
        {
            "checkpoint_name": "gray_sdxl.safetensors",
            "prompt": "colored vase",
            "negative_prompt": "bad",
            "comfy_input_image": "input_image.png",
            "seed": 123,
            "steps": 35,
            "cfg": 6.0,
            "sampler_name": "euler",
            "scheduler": "normal",
            "denoise": 0.25,
            "filename_prefix": "lbd/recolor/test",
        },
    )

    classes = _class_types(workflow)
    assert "LoadImage" in classes
    assert "VAEEncode" in classes
    assert "KSampler" in classes

