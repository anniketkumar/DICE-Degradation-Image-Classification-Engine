"""
==============================================================================
DICE Dataset Generator
Degradation & Image Classification Engine — Data Preparation Script
==============================================================================
PURPOSE:
    Reads raw, high-resolution drone images from a source folder and produces
    a structured, three-class dataset ready for use with TensorFlow's
    `tf.keras.utils.image_dataset_from_directory()`.

OUTPUT FOLDER STRUCTURE:
    dataset/
    ├── Class_0_Clean/      ← Direct copies of original images
    ├── Class_1_Blurred/    ← Gaussian-blurred (out-of-focus simulation)
    └── Class_2_Noisy/      ← Gaussian-noise-added (sensor noise simulation)

USAGE:
    python generate_dice_dataset.py

DEPENDENCIES:
    pip install opencv-python numpy tqdm
==============================================================================
"""

import os
import cv2
import numpy as np
from pathlib import Path
from tqdm import tqdm  # pip install tqdm  →  rich progress bar in the terminal


# ==============================================================================
# ── CONFIGURATION ─────────────────────────────────────────────────────────────
# ==============================================================================

# Root directory of THIS script (so paths are always relative to where it lives)
BASE_DIR = Path(__file__).parent

# Source folder that holds all raw VisDrone .jpg images
SOURCE_DIR = BASE_DIR / "VisDrone_Clean"

# Destination root — TensorFlow's image_dataset_from_directory expects every
# class to be a SUBDIRECTORY inside a single parent folder.
DEST_DIR = BASE_DIR / "dataset"

# ── Class sub-folders (names become the class labels TensorFlow reads) ─────────
CLASS_0_CLEAN   = DEST_DIR / "Class_0_Clean"
CLASS_1_BLURRED = DEST_DIR / "Class_1_Blurred"
CLASS_2_NOISY   = DEST_DIR / "Class_2_Noisy"

# ── Gaussian Blur parameters ───────────────────────────────────────────────────
# cv2.GaussianBlur(src, ksize, sigmaX)
#
#   ksize  : Kernel size (width, height). MUST be odd positive integers.
#            A larger kernel = a stronger, wider blur.
#            (15, 15) gives a moderately strong, realistic lens-blur effect.
#            Try (7, 7) for light defocus or (25, 25) for heavy blur.
#
#   sigmaX : Standard deviation of the Gaussian in the X direction.
#            When set to 0, OpenCV calculates it automatically from ksize
#            using the formula:  sigma = 0.3 * ((ksize - 1) * 0.5 - 1) + 0.8
#            For (15, 15) ksize, auto-sigma ≈ 2.15 pixels.
#            You can override this with a fixed value, e.g. 3.0, for more
#            control over the blur spread.
BLUR_KERNEL  = (15, 15)   # (width, height) — both odd
BLUR_SIGMA_X = 0          # 0 = auto-calculate from kernel size

# ── Gaussian Noise parameters ──────────────────────────────────────────────────
# We generate noise using numpy's random normal distribution:
#   np.random.normal(mean, std_dev, shape)
#
#   NOISE_MEAN   : Centre of the noise distribution. 0.0 means the noise has
#                  no directional bias (it won't consistently brighten or
#                  darken the image). Almost always 0.
#
#   NOISE_STD    : Standard deviation (spread) of the noise.
#                  Controls how "strong" the static appears.
#                  ≈  5–10  → barely visible grain (clean sensor)
#                  ≈ 15–25  → moderate sensor noise (realistic hand-held/drone)
#                  ≈ 30–50  → heavy noise (low-light or damaged sensor)
#                  We use 25 to create a clearly visible but realistic effect.
#
# After adding floating-point noise we MUST:
#   1. Clip back to [0, 255]   so no pixel goes out of the uint8 range.
#   2. Cast to uint8           so the image can be saved correctly.
NOISE_MEAN = 0.0
NOISE_STD  = 25.0

# ── Supported image file extensions (lowercase) ────────────────────────────────
VALID_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif", ".webp"}


# ==============================================================================
# ── HELPER FUNCTIONS ──────────────────────────────────────────────────────────
# ==============================================================================

def create_output_dirs(*dirs: Path) -> None:
    """Create all destination directories (and any missing parents)."""
    for d in dirs:
        d.mkdir(parents=True, exist_ok=True)
    print(f"[✓] Output directories ready under: {DEST_DIR}\n")


def apply_gaussian_blur(image: np.ndarray) -> np.ndarray:
    """
    Simulate an out-of-focus drone camera by convolving the image with a
    Gaussian kernel.

    A Gaussian kernel is a 2-D bell-curve weight matrix.  Each output pixel
    becomes a weighted average of its neighbourhood, where pixels closer to
    the centre contribute more — this smooths edges and fine detail just like
    optical defocus.

    Args:
        image: BGR uint8 numpy array read by cv2.imread().

    Returns:
        Blurred BGR uint8 numpy array (same shape as input).
    """
    return cv2.GaussianBlur(image, BLUR_KERNEL, BLUR_SIGMA_X)


def apply_gaussian_noise(image: np.ndarray) -> np.ndarray:
    """
    Simulate sensor / thermal / ISO noise by adding random values sampled
    from a Gaussian (normal) distribution to every pixel channel.

    Real camera sensors produce random electron fluctuations that appear as
    'static' in the captured image.  This is well-modelled by additive white
    Gaussian noise (AWGN).

    Pipeline:
        1. Convert image to float32 so arithmetic doesn't overflow uint8.
        2. Generate a noise array of the same shape (H × W × C) with values
           drawn from N(NOISE_MEAN, NOISE_STD²).
        3. Add noise to the float image.
        4. Clip to [0, 255] — negative or >255 values are clamped.
        5. Cast back to uint8 for saving.

    Args:
        image: BGR uint8 numpy array read by cv2.imread().

    Returns:
        Noisy BGR uint8 numpy array (same shape as input).
    """
    # Step 1 — work in float32 to avoid integer wrap-around
    float_image = image.astype(np.float32)

    # Step 2 — sample noise: shape matches (height, width, channels)
    noise = np.random.normal(loc=NOISE_MEAN, scale=NOISE_STD, size=float_image.shape)

    # Step 3 — add noise to every pixel
    noisy_float = float_image + noise

    # Step 4 — clip: any value < 0 → 0, any value > 255 → 255
    noisy_clipped = np.clip(noisy_float, 0.0, 255.0)

    # Step 5 — restore uint8 dtype expected by cv2.imwrite()
    return noisy_clipped.astype(np.uint8)


# ==============================================================================
# ── MAIN PIPELINE ─────────────────────────────────────────────────────────────
# ==============================================================================

def main() -> None:
    print("=" * 62)
    print("  DICE Dataset Generator")
    print("  Degradation & Image Classification Engine")
    print("=" * 62)
    print(f"\n[→] Source  : {SOURCE_DIR}")
    print(f"[→] Output  : {DEST_DIR}\n")

    # ── Collect all valid image paths from the source directory ───────────────
    # Using pathlib.Path.iterdir() to list files; filter by extension.
    all_files = [
        f for f in SOURCE_DIR.iterdir()
        if f.is_file() and f.suffix.lower() in VALID_EXTENSIONS
    ]

    total = len(all_files)
    if total == 0:
        print("[✗] No valid image files found in SOURCE_DIR. Exiting.")
        return

    print(f"[✓] Found {total} image(s) to process.\n")

    # ── Ensure all destination class folders exist ────────────────────────────
    create_output_dirs(CLASS_0_CLEAN, CLASS_1_BLURRED, CLASS_2_NOISY)

    # ── Counters for the summary report ──────────────────────────────────────
    success_count = 0
    skip_count    = 0

    # ── Main loop with tqdm progress bar ─────────────────────────────────────
    # tqdm wraps any iterable and renders a real-time progress bar:
    #   bar_format controls the visual style of the bar.
    #   unit="img" labels each iteration step as "img" in the display.
    with tqdm(
        total=total,
        desc="Processing images",
        unit="img",
        colour="cyan",
        dynamic_ncols=True,
    ) as pbar:

        for idx, img_path in enumerate(all_files):

            # ── Error handling: imread can return None for corrupt files ──────
            image = cv2.imread(str(img_path))

            if image is None:
                # Log the bad file but continue so one corrupt image doesn't
                # stop the entire job.
                tqdm.write(f"  [WARN] Skipping unreadable file: {img_path.name}")
                skip_count += 1
                pbar.update(1)
                continue

            # Reuse the original filename so images stay traceable
            filename = img_path.name

            # ── Class 0: Save a pixel-perfect copy of the original ────────────
            # cv2.imwrite preserves the original quality; we use the exact
            # source bytes by writing the array that imread loaded.
            cv2.imwrite(str(CLASS_0_CLEAN / filename), image)

            # ── Class 1: Apply Gaussian Blur ──────────────────────────────────
            blurred_image = apply_gaussian_blur(image)
            cv2.imwrite(str(CLASS_1_BLURRED / filename), blurred_image)

            # ── Class 2: Apply Gaussian Noise ─────────────────────────────────
            noisy_image = apply_gaussian_noise(image)
            cv2.imwrite(str(CLASS_2_NOISY / filename), noisy_image)

            success_count += 1

            # ── Optional: extra verbose print every 100 images ────────────────
            # tqdm already shows progress, but this adds a timestamped log
            # line that's useful when redirecting stdout to a log file.
            if (idx + 1) % 100 == 0:
                tqdm.write(f"  [INFO] Processed {idx + 1}/{total} images...")

            pbar.update(1)

    # ── Final summary ─────────────────────────────────────────────────────────
    print("\n" + "=" * 62)
    print("  Processing Complete!")
    print("=" * 62)
    print(f"  ✓  Successfully processed : {success_count} image(s)")
    print(f"  ⚠  Skipped (unreadable)  : {skip_count} file(s)")
    print(f"\n  Dataset saved to: {DEST_DIR}")
    print("\n  Folder layout:")
    print(f"    Class_0_Clean/    → {success_count} clean images")
    print(f"    Class_1_Blurred/  → {success_count} blurred images")
    print(f"    Class_2_Noisy/    → {success_count} noisy images")
    print("\n  Ready for tf.keras.utils.image_dataset_from_directory()")
    print("=" * 62)


# ==============================================================================
# ── ENTRY POINT ───────────────────────────────────────────────────────────────
# ==============================================================================

if __name__ == "__main__":
    # Set a fixed random seed so noise generation is reproducible across runs.
    # Remove or change this if you want different noise each time.
    np.random.seed(42)
    main()
