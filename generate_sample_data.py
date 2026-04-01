"""
==============================================================================
DICE — Sample Data Generator
==============================================================================
PURPOSE:
    Picks a small number of images from VisDrone_Clean and runs the full
    degradation pipeline on them, producing a self-contained sample_data/
    folder that can be committed to GitHub as a visual proof-of-concept.

USAGE:
    python generate_sample_data.py

OUTPUT:
    sample_data/
    ├── originals/        ← Copies of the raw VisDrone source frames
    ├── Class_0_Clean/    ← Pixel-perfect copy (no degradation)
    ├── Class_1_Blurred/  ← Gaussian Blur applied
    └── Class_2_Noisy/    ← Gaussian Noise applied
==============================================================================
"""

import os
import cv2
import numpy as np
import random
from pathlib import Path

# ── Configuration ─────────────────────────────────────────────────────────────
SOURCE_DIR   = Path("./VisDrone_Clean")   # Raw image source
SAMPLE_DIR   = Path("./sample_data")       # Output root (committed to GitHub)
NUM_SAMPLES  = 5                           # Keep this tiny — GitHub is not a DAS
RANDOM_SEED  = 42

# Degradation parameters (mirror dice_data_generator.py exactly)
BLUR_KERNEL  = (15, 15)
BLUR_SIGMA_X = 0
NOISE_MEAN   = 0.0
NOISE_STD    = 25.0

VALID_EXTENSIONS = {".jpg", ".jpeg", ".png"}

# ── Output sub-folders ────────────────────────────────────────────────────────
DIRS = {
    "originals"       : SAMPLE_DIR / "originals",
    "Class_0_Clean"   : SAMPLE_DIR / "Class_0_Clean",
    "Class_1_Blurred" : SAMPLE_DIR / "Class_1_Blurred",
    "Class_2_Noisy"   : SAMPLE_DIR / "Class_2_Noisy",
}

for d in DIRS.values():
    d.mkdir(parents=True, exist_ok=True)

# ── Collect and sample source images ─────────────────────────────────────────
all_images = [
    f for f in SOURCE_DIR.iterdir()
    if f.is_file() and f.suffix.lower() in VALID_EXTENSIONS
]

if not all_images:
    raise FileNotFoundError(f"No images found in {SOURCE_DIR}. "
                            "Please ensure VisDrone_Clean/ exists and is populated.")

random.seed(RANDOM_SEED)
selected = random.sample(all_images, min(NUM_SAMPLES, len(all_images)))
print(f"Selected {len(selected)} images from {SOURCE_DIR}:\n")

# ── Process each selected image ───────────────────────────────────────────────
np.random.seed(RANDOM_SEED)

for img_path in selected:
    image = cv2.imread(str(img_path))
    if image is None:
        print(f"  [SKIP] Could not read: {img_path.name}")
        continue

    fname = img_path.name
    print(f"  Processing: {fname}")

    # Original — save to originals/
    cv2.imwrite(str(DIRS["originals"] / fname), image)

    # Class 0 — Clean (exact copy)
    cv2.imwrite(str(DIRS["Class_0_Clean"] / fname), image)

    # Class 1 — Gaussian Blur
    blurred = cv2.GaussianBlur(image, BLUR_KERNEL, BLUR_SIGMA_X)
    cv2.imwrite(str(DIRS["Class_1_Blurred"] / fname), blurred)

    # Class 2 — Gaussian Noise
    float_img      = image.astype(np.float32)
    noise          = np.random.normal(NOISE_MEAN, NOISE_STD, float_img.shape)
    noisy          = np.clip(float_img + noise, 0, 255).astype(np.uint8)
    cv2.imwrite(str(DIRS["Class_2_Noisy"] / fname), noisy)

print(f"\nDone. Sample data written to: {SAMPLE_DIR.resolve()}")
print("You can now 'git add sample_data/' and push to GitHub.")
