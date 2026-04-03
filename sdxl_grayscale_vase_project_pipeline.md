# PROJECT SETUP PROMPT — SDXL GRAYSCALE → VASE LORA → PROMPT COLORIZATION

This document follows **your exact logic pipeline**:

```
SDXL grayscale base model
→ grayscale vase LoRA
→ SDXL img2img recolor from prompt
→ optional refinement
```

The goal is:

- Stable grayscale composition
- Vase structure realism
- Prompt-driven color control
- Historically plausible color from description

Not automatic random colorization.

---

# STAGE 0 — SYSTEM PREPARATION

Before downloading datasets or training anything.

## Required hardware (recommended minimum)

Local GPU:

- RTX 3090 or RTX 4090 recommended
- 24GB VRAM preferred

OR Cloud GPU:

- RTX 4090 (best price/performance)
- A100 80GB (very comfortable)

---

## Required software

Install:

```
Python 3.10
CUDA 11.8+
Git
```

Then install environment:

```
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
pip install diffusers transformers accelerate datasets pillow opencv-python
pip install xformers
```

Then initialize accelerate:

```
accelerate config
```

Choose:

```
GPU training
fp16 precision
single GPU (initially)
```

---

# STAGE 1 — PREPARE GRAYSCALE DATASET

You said you already have:

```
100k hyperrealistic RGB images
```

Perfect.

Now convert them to grayscale.

---

## Folder structure

Create:

```
dataset/

images_rgb/
images_gray/
captions/
```

---

## Convert RGB → Grayscale

Run this script:

```python
from PIL import Image
from pathlib import Path

src = Path("dataset/images_rgb")
dst = Path("dataset/images_gray")
dst.mkdir(exist_ok=True)

for p in src.glob("*"):
    if p.suffix.lower() not in [".jpg", ".png", ".jpeg", ".webp"]:
        continue

    img = Image.open(p).convert("RGB")

    gray = img.convert("L")

    gray_rgb = Image.merge("RGB", (gray, gray, gray))

    gray_rgb.save(dst / p.name)
```

Important:

**Use grayscale replicated to RGB channels.**

SDXL expects 3 channels.

---

# STAGE 2 — TRAIN SDXL GRAYSCALE MODEL

Base model:

```
stabilityai/stable-diffusion-xl-base-1.0
```

Download automatically through Diffusers.

---

## Training goal

Train SDXL to produce:

```
structure
lighting
shadows
composition
material depth
```

Without color influence.

---

## Training command template

```bash
accelerate launch train_text_to_image_sdxl.py \
 --pretrained_model_name_or_path="stabilityai/stable-diffusion-xl-base-1.0" \
 --train_data_dir="dataset/images_gray" \
 --resolution=1024 \
 --train_batch_size=1 \
 --gradient_accumulation_steps=4 \
 --gradient_checkpointing \
 --mixed_precision="fp16" \
 --learning_rate=1e-6 \
 --max_train_steps=100000
```

Recommended dataset size:

```
50k–100k grayscale images
```

---

# STAGE 3 — TRAIN VASE GRAYSCALE LORA

Now train domain specialization.

Only vase types.

Still grayscale.

---

## Vase dataset structure

```
vase_dataset/

images_gray/
captions/
```

Captions example:

```
ancient greek amphora
roman ceramic vase
byzantine ornamental vase
large decorative amphora with handles
```

Focus on:

- shape
- structure
- ornament

Not color.

---

## Train LoRA

Use Kohya or Diffusers LoRA training.

Template:

```bash
accelerate launch train_text_to_image_lora_sdxl.py \
 --pretrained_model_name_or_path="YOUR_GRAYSCALE_SDXL" \
 --train_data_dir="vase_dataset" \
 --resolution=1024 \
 --train_batch_size=1 \
 --learning_rate=1e-4 \
 --rank=32
```

Output:

```
vase_grayscale_lora.safetensors
```

---

# STAGE 4 — GRAYSCALE GENERATION (STRUCTURE LOCK)

Generate grayscale vase compositions.

Load:

```
Grayscale SDXL
+ Vase LoRA
```

Prompt example:

```
ancient greek amphora,
symmetric ceramic vase,
detailed ornamental bands,
strong studio lighting,
high contrast grayscale photograph
```

Generate:

```
8–16 variations
```

Select best.

---

# STAGE 5 — STRUCTURE REFINEMENT (IMG2IMG GRAYSCALE)

Fix geometry errors.

Use:

```
img2img grayscale
```

Parameters:

```
denoise: 0.25–0.35
steps: 30–40
CFG: 5–7
```

Goal:

```
perfect vase structure
```

Still grayscale.

---

# STAGE 6 — COLORIZATION USING PROMPT (CRITICAL STAGE)

This is where color is applied.

Not DDColor.

Use:

```
SDXL img2img recolor
```

---

## Load pipeline

```
Grayscale output image
+ color prompt
```

---

## Color prompt structure (VERY IMPORTANT)

Always describe:

```
object
material
base color
ornament color
finish type
lighting
```

---

## Example HIGH QUALITY color prompt

```
ancient greek amphora,
ceramic material,
deep black glazed body,
red ochre painted figures,
golden decorative accents,
glossy ceramic finish,
studio lighting,
realistic reflections
```

---

## Img2img parameters

```
denoise: 0.20–0.30
CFG: 6
steps: 35
```

This preserves structure.

Adds color.

---

# STAGE 7 — FINAL REFINEMENT (OPTIONAL BUT RECOMMENDED)

Small polish pass.

Use:

```
SDXL img2img
low denoise
```

Parameters:

```
denoise: 0.15–0.20
```

Goal:

```
improve realism
fix color transitions
sharpen details
```

---

# DATA PREPARATION CHECKLIST

Before training verify:

```
✔ Images cleaned
✔ No corrupted files
✔ Resolution normalized
✔ Captions present
✔ Dataset balanced
✔ Duplicate removal done
```

---

# TRAINING ORDER SUMMARY

Follow strictly:

```
1. Convert RGB → grayscale
2. Train SDXL grayscale
3. Train vase grayscale LoRA
4. Generate grayscale images
5. Fix structure
6. Recolor using prompt
7. Optional refinement
```

---

# MOST IMPORTANT PARAMETERS SUMMARY

Grayscale generation:

```
denoise: N/A (txt2img)
CFG: 5–7
steps: 30–40
```

Color recolor:

```
denoise: 0.20–0.30
CFG: 6
steps: 35
```

Refinement:

```
denoise: 0.15–0.20
```

---

# WHAT TO DOWNLOAD

You will need:

## Base model

```
stabilityai/stable-diffusion-xl-base-1.0
```

---

## Tools

Install repos:

```
git clone https://github.com/huggingface/diffusers
```

Optional (LoRA training):

```
git clone https://github.com/kohya-ss/sd-scripts
```

---

# FUTURE IMPROVEMENTS (AFTER FIRST SUCCESS)

Later you may add:

```
ControlNet edges
Material LoRA
Lighting LoRA
Color-specific LoRA
```

But not initially.

---

# FINAL GOAL PIPELINE

This is your intended architecture:

```
TEXT PROMPT

↓

SDXL GRAYSCALE
+ VASE LORA

↓

GRAYSCALE IMAGE
(structure locked)

↓

SDXL IMG2IMG COLOR
(prompt-driven color)

↓

FINAL REALISTIC COLORED VASE
```

---

# THIS PLAN IS CONSISTENT WITH YOUR LOGIC

You are doing:

```
structure first
color second
```

Not mixing them.

That is correct engineering logic.

