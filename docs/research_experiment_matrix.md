# Research Experiment Matrix

This document centralizes the experiment parameters currently configured in the repository, in a paper-friendly format.

Use this file as the canonical reference for:
- training setup summaries
- inference setup summaries
- table-ready hyperparameter values
- identifying which values are fixed in config versus which must be reported from actual run logs

## Reporting Conventions

- Estimated epochs are computed as:
  `estimated_epochs = max_train_steps * train_batch_size * gradient_accumulation_steps / train_samples_used`
- If `max_train_samples` is set, then `train_samples_used = min(actual_train_split_size, max_train_samples)`.
- If neither `max_train_steps` nor `num_train_epochs` is pinned in the repo config, the exact epoch count is not fixed by configuration and must be taken from run metadata or logs.
- All values below reflect the repository state as of `2026-04-05`.

## Dataset Splits Used by Config

| Dataset config | Total samples | Train | Val | Test | Notes |
| --- | ---: | ---: | ---: | ---: | --- |
| `data/base20k` | 20000 | 16000 | 2000 | 2000 | Defined in `configs/build_base100k.yaml` |
| `data/vaze_bw` | not fixed globally | repo uses train split only | n/a | n/a | LoRA configs additionally cap training to `max_train_samples=250` |

## Training Matrix

| Experiment | Mode / Method | Base model | Train data | Resolution | Train batch | Grad accum | Effective batch | LR | Rank | Max train steps | Train samples used | Estimated epochs | Output |
| --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| SDXL grayscale base | full SDXL text-to-image fine-tuning | `stabilityai/stable-diffusion-xl-base-1.0` | `data/base20k/gray/train` | 1024 | 1 | 4 | 4 | `1e-6` | n/a | 1800 | 16000 | 0.45 | `runs/train_sdxl_grayscale` |
| Amphora LoRA grayscale | SDXL LoRA fine-tuning | `stabilityai/stable-diffusion-xl-base-1.0` | `data/vaze_bw/gray/train` | 384 | 1 | 1 | 1 | `1e-4` | 8 | 600 | 250 | 2.40 | `runs/train_lora_amphora_bw` |
| Amphora LoRA color | SDXL LoRA fine-tuning | `stabilityai/stable-diffusion-xl-base-1.0` | `data/vaze_bw/color/train` | 384 | 1 | 1 | 1 | `1e-4` | 8 | 600 | 250 | 2.40 | `runs/train_lora_amphora_color` |
| Base20k grayscale LoRA | SDXL LoRA fine-tuning | `runs/train_sdxl_grayscale/checkpoint-last` | `data/base20k/gray/train` | 1024 | 1 | not set in config | not fixed | `1e-4` | 32 | not set in config | 16000 | not fixed | `runs/train_lora_vase_grayscale` |

## Training Notes by Experiment

### 1) SDXL grayscale base

- Script: `external/diffusers/examples/text_to_image/train_text_to_image_sdxl.py`
- Mixed precision: `fp16`
- Gradient checkpointing: enabled
- Checkpointing cadence: every `200` optimizer steps
- Checkpoint retention: `3`
- Paper-safe statement:
  `The grayscale SDXL model was fine-tuned for 1,800 optimizer steps on a 16k-image training split at 1024x1024 resolution, with effective batch size 4 and learning rate 1e-6, corresponding to approximately 0.45 epochs.`

### 2) Amphora LoRA grayscale

- Script: `external/diffusers/examples/text_to_image/train_text_to_image_lora_sdxl.py`
- Mixed precision: `fp16`
- Gradient checkpointing: enabled
- `max_train_samples=250`
- `dataloader_num_workers=0`
- Environment overrides:
  - `LBD_DISABLE_CUDNN=1`
  - `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`
- Paper-safe statement:
  `The grayscale amphora LoRA was trained for 600 steps at 384x384 resolution with rank 8, learning rate 1e-4, batch size 1, and a capped 250-image training subset, corresponding to approximately 2.4 epochs.`

### 3) Amphora LoRA color

- Script: `external/diffusers/examples/text_to_image/train_text_to_image_lora_sdxl.py`
- Same training policy as the grayscale amphora LoRA, but on the color train split
- Paper-safe statement:
  `The color amphora LoRA was trained for 600 steps at 384x384 resolution with rank 8, learning rate 1e-4, batch size 1, and a capped 250-image training subset, corresponding to approximately 2.4 epochs.`

### 4) Base20k grayscale LoRA

- Script: `external/diffusers/examples/text_to_image/train_text_to_image_lora_sdxl.py`
- This config does not pin `max_train_steps`
- This config does not pin `num_train_epochs`
- This means the exact step count and epoch count are not reproducibly defined by the repo config alone
- Paper guidance:
  report this experiment only from actual run logs, or pin the config first before using it in a formal experimental table

## Inference Matrix

| Stage | Mode / Method | Checkpoint type | LoRA | Resolution | Steps | CFG | Sampler | Scheduler | Denoise | Batch | Seed policy | Output intent |
| --- | --- | --- | --- | --- | ---: | ---: | --- | --- | ---: | ---: | --- | --- |
| `graygen` | text-to-image generation | grayscale SDXL checkpoint | grayscale amphora LoRA | 1024x1024 | 35 | 6.0 | `euler` | `normal` | 1.0 | 1 | fixed seeds (`1337`, `1338`) | generate grayscale vase structure |
| `recolor` | img2img recolor | color SDXL checkpoint | recolor LoRA | input-driven | 35 | 6.0 | `euler` | `normal` | 0.25 | 1 | fixed seed (`2026`) | recolor grayscale structure from prompt |
| `refine` | img2img refinement | color refiner checkpoint | recolor LoRA | input-driven | 30 | 5.5 | `euler` | `normal` | 0.18 | 1 | fixed seed (`4040`) | refine recolored output |

## Inference Notes

- Backend: ComfyUI
- Upload policy: `upload_inputs=true`
- Poll interval: `2s`
- Prompt timeout: `1800s`
- Recommended pipeline order:
  `prompt -> graygen -> recolor -> refine`

## Values Safe to Cite Directly in a Paper

These values are fixed by repo config and can be cited directly:
- SDXL grayscale base: `1800` steps, `1024x1024`, effective batch `4`, LR `1e-6`
- Amphora LoRA grayscale: `600` steps, `384x384`, rank `8`, LR `1e-4`, about `2.4` epochs on `250` samples
- Amphora LoRA color: `600` steps, `384x384`, rank `8`, LR `1e-4`, about `2.4` epochs on `250` samples
- Inference presets:
  - graygen: `35` steps, CFG `6.0`
  - recolor: `35` steps, CFG `6.0`, denoise `0.25`
  - refine: `30` steps, CFG `5.5`, denoise `0.18`

## Values That Should Be Taken From Run Artifacts

Do not cite these from config alone unless you pin them first:
- exact step count for `configs/train_lora.yaml`
- exact epoch count for `configs/train_lora.yaml`
- realized wall-clock time per run
- realized GPU memory usage
- actual final train-set size for `data/vaze_bw` before the `250`-sample cap, if you want to describe the full source collection

## Recommended Table Fragments for a Paper

### Training Table

| Model | Method | Data | Resolution | Steps | Effective batch | LR | Rank | Epochs |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| SDXL grayscale base | full fine-tune | Base20k gray train | 1024 | 1800 | 4 | `1e-6` | n/a | 0.45 |
| Amphora LoRA grayscale | LoRA | Vaze gray train | 384 | 600 | 1 | `1e-4` | 8 | 2.40 |
| Amphora LoRA color | LoRA | Vaze color train | 384 | 600 | 1 | `1e-4` | 8 | 2.40 |

### Inference Table

| Stage | Method | Steps | CFG | Denoise | Sampler |
| --- | --- | ---: | ---: | ---: | --- |
| graygen | text-to-image | 35 | 6.0 | 1.0 | Euler |
| recolor | img2img recolor | 35 | 6.0 | 0.25 | Euler |
| refine | img2img refinement | 30 | 5.5 | 0.18 | Euler |
