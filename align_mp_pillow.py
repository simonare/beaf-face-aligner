#!/usr/bin/env python3
"""
Before/After face aligner for BEAF WordPress slider.

Both images are transformed independently to a shared canonical face position
(same eye coords, same canvas size) so they overlay with zero drift.

Inputs:  samples/before.png  samples/after.png
Outputs: tmp/before_<id>.png  tmp/after_<id>.png  tmp/overlay_<id>.png  tmp/meta_<id>.json
"""

from pathlib import Path
import json
import uuid
import urllib.request
import math
import numpy as np
from PIL import Image
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision as mp_vision

BASE = Path(__file__).resolve().parent
SAMPLES_DIR = BASE / "samples"
TMP_DIR = BASE / "tmp"

BEFORE_PATH = SAMPLES_DIR / "before.png"
AFTER_PATH = SAMPLES_DIR / "after.png"

MODEL_PATH = BASE / "face_landmarker.task"
MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/"
    "face_landmarker/face_landmarker/float16/1/face_landmarker.task"
)

# ── Output canvas settings ─────────────────────────────────────────────────
CANVAS_WIDTH = 1200   # intermediate canvas width — final size is smaller after crop
CANVAS_ASPECT = 3 / 4  # width / height  (portrait 3:4)
FACE_CENTER_Y_RATIO = 0.40   # midpoint-between-eyes vertical position
EYE_DISTANCE_RATIO = 0.36   # inter-eye span as fraction of canvas width


# ── Model download ─────────────────────────────────────────────────────────

def _ensure_model() -> None:
    if not MODEL_PATH.exists():
        print("Downloading face landmarker model (~30 MB)…")
        urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)
        print("Model downloaded.")


# ── Landmark detection ─────────────────────────────────────────────────────

def detect_landmarks(img: Image.Image) -> np.ndarray:
    """Returns (478, 2) array of landmark pixel coords for the first face."""
    _ensure_model()
    rgb = np.array(img.convert("RGB"))
    h, w = rgb.shape[:2]

    base_options = mp_python.BaseOptions(model_asset_path=str(MODEL_PATH))
    options = mp_vision.FaceLandmarkerOptions(
        base_options=base_options,
        num_faces=1,
    )
    with mp_vision.FaceLandmarker.create_from_options(options) as det:
        result = det.detect(
            mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb))

    if not result.face_landmarks:
        raise RuntimeError("No face detected in image.")

    lm = result.face_landmarks[0]
    return np.array([(p.x * w, p.y * h) for p in lm], dtype=np.float64)


# MediaPipe FaceLandmarker iris centres: 468 = left iris, 473 = right iris
def eye_centers(pts: np.ndarray):
    return pts[468].copy(), pts[473].copy()


# ── Similarity transform ───────────────────────────────────────────────────

def similarity_from_eyes(
    src_left: np.ndarray, src_right: np.ndarray,
    dst_left: np.ndarray, dst_right: np.ndarray,
) -> np.ndarray:
    """
    2×3 forward affine matrix (similarity: scale + rotation + translation)
    that maps src eye positions exactly onto dst eye positions.
    No shear, no independent x/y scaling — face proportions are preserved.
    """
    sl = complex(*src_left)
    sr = complex(*src_right)
    dl = complex(*dst_left)
    dr = complex(*dst_right)

    ratio = (dr - dl) / (sr - sl)
    scale = abs(ratio)
    angle = math.atan2(ratio.imag, ratio.real)
    cos_a, sin_a = math.cos(angle), math.sin(angle)

    src_mid = (sl + sr) / 2
    dst_mid = (dl + dr) / 2
    tx = dst_mid.real - scale * (cos_a * src_mid.real - sin_a * src_mid.imag)
    ty = dst_mid.imag - scale * (sin_a * src_mid.real + cos_a * src_mid.imag)

    return np.array([
        [scale * cos_a, -scale * sin_a, tx],
        [scale * sin_a,  scale * cos_a, ty],
    ])


def pillow_inverse_coeffs(M: np.ndarray):
    """PIL Image.transform(AFFINE) needs the *inverse* mapping (output→input)."""
    full = np.vstack([M, [0.0, 0.0, 1.0]])
    inv = np.linalg.inv(full)
    a, b, c = inv[0]
    d, e, f = inv[1]
    return (float(a), float(b), float(c), float(d), float(e), float(f))


# ── Main ───────────────────────────────────────────────────────────────────

def main():
    for p in (BEFORE_PATH, AFTER_PATH):
        if not p.exists():
            raise FileNotFoundError(f"Missing input file: {p}")

    TMP_DIR.mkdir(parents=True, exist_ok=True)
    job_id = uuid.uuid4().hex[:8]

    OUT_BEFORE = TMP_DIR / f"before_{job_id}.png"
    OUT_AFTER = TMP_DIR / f"after_{job_id}.png"
    OUT_OVERLAY = TMP_DIR / f"overlay_{job_id}.png"
    OUT_META = TMP_DIR / f"meta_{job_id}.json"

    before = Image.open(BEFORE_PATH).convert("RGB")
    after = Image.open(AFTER_PATH).convert("RGB")

    print("Detecting landmarks in before…")
    pts_before = detect_landmarks(before)
    print("Detecting landmarks in after…")
    pts_after = detect_landmarks(after)

    le_b, re_b = eye_centers(pts_before)
    le_a, re_a = eye_centers(pts_after)

    # ── canonical target eye positions — same for BOTH images ───────────────
    W = CANVAS_WIDTH
    H = round(W / CANVAS_ASPECT)

    half_eye = (W * EYE_DISTANCE_RATIO) / 2
    eye_mid_x = W / 2
    eye_mid_y = H * FACE_CENTER_Y_RATIO

    dst_left = np.array([eye_mid_x - half_eye, eye_mid_y])
    dst_right = np.array([eye_mid_x + half_eye, eye_mid_y])

    # ── transform each image independently to canonical position ────────────
    M_before = similarity_from_eyes(le_b, re_b, dst_left, dst_right)
    M_after = similarity_from_eyes(le_a, re_a, dst_left, dst_right)

    canvas = (W, H)

    aligned_before = before.transform(
        canvas, method=Image.AFFINE, data=pillow_inverse_coeffs(M_before),
        resample=Image.Resampling.BICUBIC, fillcolor=(255, 255, 255))
    aligned_after = after.transform(
        canvas, method=Image.AFFINE, data=pillow_inverse_coeffs(M_after),
        resample=Image.Resampling.BICUBIC, fillcolor=(255, 255, 255))

    # ── Find the region where BOTH images have real pixels ────────────────────
    # Transform an all-white mask with black fill → 0 = canvas fill, 255 = content.
    mask_b = Image.new("L", before.size, 255).transform(
        canvas, method=Image.AFFINE, data=pillow_inverse_coeffs(M_before),
        resample=Image.Resampling.NEAREST, fillcolor=0)
    mask_a = Image.new("L", after.size, 255).transform(
        canvas, method=Image.AFFINE, data=pillow_inverse_coeffs(M_after),
        resample=Image.Resampling.NEAREST, fillcolor=0)

    combined = np.minimum(np.array(mask_b), np.array(mask_a))

    # ── Find the largest axis-aligned rectangle inside the intersection ──────
    rows_any = np.any(combined == 255, axis=1)
    cols_any = np.any(combined == 255, axis=0)
    if not rows_any.any() or not cols_any.any():
        raise RuntimeError(
            "No overlapping content region found between the two images.")

    Y1 = int(np.where(rows_any)[0][0])
    Y2 = int(np.where(rows_any)[0][-1]) + 1
    X1 = int(np.where(cols_any)[0][0])
    X2 = int(np.where(cols_any)[0][-1]) + 1
    roi = combined[Y1:Y2, X1:X2]   # every row & col has ≥1 content pixel

    # Per-row content x-span
    left_of_row = np.argmax(roi == 255, axis=1)
    right_of_row = roi.shape[1] - 1 - np.argmax((roi == 255)[:, ::-1], axis=1)
    row_widths = right_of_row - left_of_row

    # Keep only the "core" wide rows (≥95 % of the widest row).
    # Rows at the very top/bottom of a rotated rectangle are thin and skew the
    # x-range; excluding them gives a large, fill-free x estimate.
    core = row_widths >= (row_widths.max() * 0.95)
    if not core.any():
        core = row_widths >= (row_widths.max() * 0.80)

    rx1 = int(left_of_row[core].max())
    rx2 = int(right_of_row[core].min())
    if rx1 >= rx2:
        raise RuntimeError("Images don't overlap enough horizontally.")

    # Find contiguous y-range where the x-slice [rx1:rx2] is fully content
    col_slice = roi[:, rx1:rx2 + 1]
    fully_covered = np.all(col_slice == 255, axis=1)
    full_rows_idx = np.where(fully_covered)[0]
    if len(full_rows_idx) == 0:
        raise RuntimeError("Images don't overlap enough vertically.")
    ry1 = int(full_rows_idx[0])
    ry2 = int(full_rows_idx[-1]) + 1

    # Map back to canvas coords
    x1, x2 = X1 + rx1, X1 + rx2 + 1
    y1, y2 = Y1 + ry1, Y1 + ry2 + 1

    # Crop both to the intersection — no fill borders, same dimensions
    aligned_before = aligned_before.crop((x1, y1, x2, y2))
    aligned_after = aligned_after.crop((x1, y1, x2, y2))

    overlay = Image.blend(aligned_before, aligned_after, alpha=0.5)

    aligned_before.save(OUT_BEFORE,  quality=95)
    aligned_after.save(OUT_AFTER,   quality=95)
    overlay.save(OUT_OVERLAY, quality=95)

    final_w, final_h = aligned_before.size
    meta = {
        "canvas": {"width": W, "height": H},
        "output": {"width": final_w, "height": final_h, "crop": [x1, y1, x2, y2]},
        "canonical_eye_targets": {
            "left":  dst_left.tolist(),
            "right": dst_right.tolist(),
        },
        "affine_before_2x3": M_before.tolist(),
        "affine_after_2x3":  M_after.tolist(),
    }
    OUT_META.write_text(json.dumps(meta, indent=2), encoding="utf-8")

    print(
        f"\nDone. Job: {job_id}  |  Output: {final_w}×{final_h} px (cropped from {W}×{H} intermediate canvas)")
    print("Saved:", OUT_BEFORE.name)
    print("Saved:", OUT_AFTER.name)
    print("Saved:", OUT_OVERLAY.name,  " ← QC only, not for plugin")
    print("Saved:", OUT_META.name)
    print("\nUpload aligned_before.png + aligned_after.png to BEAF plugin.")


if __name__ == "__main__":
    main()
