"""
Core face-alignment logic.

Public API
----------
align_images(before, after, *, canvas_width, canvas_aspect,
             face_center_y_ratio, eye_distance_ratio) -> AlignResult

Alignment strategy (three-tier fallback)
-----------------------------------------
1. Both faces detected (frontal/mild profile)
   → Bilateral similarity transform to shared canonical eye position.
2. One face detected (side profile on one image)
   → Canonical transform for the detected face; SIFT feature matching
     to compose the transform for the undetected image.
3. Neither face detected (both side profiles)
   → Full SIFT feature matching; "after" is scale-fitted to canvas,
     "before" is composed via SIFT onto that same canvas.
"""

from __future__ import annotations

import math
import urllib.request
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision as mp_vision

# ── Model ──────────────────────────────────────────────────────────────────
_MODEL_DIR = Path(__file__).resolve().parent
MODEL_PATH = _MODEL_DIR / "face_landmarker.task"
MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/"
    "face_landmarker/face_landmarker/float16/1/face_landmarker.task"
)


def ensure_model(model_path: Path = MODEL_PATH) -> Path:
    if not model_path.exists():
        print(f"Downloading face landmarker model to {model_path} …")
        urllib.request.urlretrieve(MODEL_URL, model_path)
        print("Model downloaded.")
    return model_path


# ── Result dataclass ───────────────────────────────────────────────────────

@dataclass
class AlignResult:
    before: Image.Image   # aligned "before" — same size as after
    after:  Image.Image   # aligned "after"
    overlay: Image.Image  # 50/50 QC blend
    meta: dict            # transform metadata


# ── Internal helpers ───────────────────────────────────────────────────────

def _detect_landmarks(img: Image.Image, model_path: Path) -> np.ndarray | None:
    """
    Returns (478, 2) pixel-coord array for the first detected face, or None.
    Tries default confidence first, then a lowered threshold to catch mild profiles.
    """
    rgb = np.array(img.convert("RGB"))
    h, w = rgb.shape[:2]
    for confidence in (0.5, 0.2):
        opts = mp_vision.FaceLandmarkerOptions(
            base_options=mp_python.BaseOptions(
                model_asset_path=str(model_path)),
            num_faces=1,
            min_face_detection_confidence=confidence,
            min_face_presence_confidence=confidence,
        )
        with mp_vision.FaceLandmarker.create_from_options(opts) as det:
            result = det.detect(
                mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb))
        if result.face_landmarks:
            lm = result.face_landmarks[0]
            return np.array([(p.x * w, p.y * h) for p in lm], dtype=np.float64)
    return None


def _eye_centers(pts: np.ndarray):
    # MediaPipe iris: 468 = left, 473 = right
    return pts[468].copy(), pts[473].copy()


def _similarity_from_eyes(
    src_left: np.ndarray, src_right: np.ndarray,
    dst_left: np.ndarray, dst_right: np.ndarray,
) -> np.ndarray:
    sl, sr = complex(*src_left), complex(*src_right)
    dl, dr = complex(*dst_left), complex(*dst_right)
    ratio = (dr - dl) / (sr - sl)
    scale = abs(ratio)
    angle = math.atan2(ratio.imag, ratio.real)
    cos_a, sin_a = math.cos(angle), math.sin(angle)
    src_mid, dst_mid = (sl + sr) / 2, (dl + dr) / 2
    tx = dst_mid.real - scale * (cos_a * src_mid.real - sin_a * src_mid.imag)
    ty = dst_mid.imag - scale * (sin_a * src_mid.real + cos_a * src_mid.imag)
    return np.array([
        [scale * cos_a, -scale * sin_a, tx],
        [scale * sin_a,  scale * cos_a, ty],
    ])


def _compose(M_outer: np.ndarray, M_inner: np.ndarray) -> np.ndarray:
    """Compose two 2×3 affine matrices: result maps x → M_outer(M_inner(x))."""
    O = np.vstack([M_outer, [0.0, 0.0, 1.0]])
    I = np.vstack([M_inner, [0.0, 0.0, 1.0]])
    return (O @ I)[:2]


def _fit_to_canvas(img_w: int, img_h: int, W: int, H: int) -> np.ndarray:
    """Scale + center image to fill 90 % of the canvas (2×3 forward transform)."""
    scale = min(W / img_w, H / img_h) * 0.90
    tx = (W - img_w * scale) / 2
    ty = (H - img_h * scale) / 2
    return np.array([[scale, 0.0, tx], [0.0, scale, ty]], dtype=np.float64)


def _sift_transform(src: Image.Image, dst: Image.Image) -> np.ndarray:
    """
    Estimate a similarity (partial affine) transform that maps *src* pixel
    coordinates → *dst* pixel coordinates using SIFT + RANSAC.

    Returns a 2×3 numpy array.
    Raises ValueError when too few matches are found.
    """
    import cv2  # already installed as a mediapipe dependency

    gs = cv2.cvtColor(np.array(src.convert("RGB")), cv2.COLOR_RGB2GRAY)
    gd = cv2.cvtColor(np.array(dst.convert("RGB")), cv2.COLOR_RGB2GRAY)

    sift = cv2.SIFT_create(nfeatures=4000)
    kp_s, des_s = sift.detectAndCompute(gs, None)
    kp_d, des_d = sift.detectAndCompute(gd, None)

    if des_s is None or des_d is None or len(kp_s) < 4 or len(kp_d) < 4:
        raise ValueError("Not enough SIFT features for alignment.")

    # FLANN matcher with Lowe ratio test
    index_params = {"algorithm": 1, "trees": 5}
    search_params = {"checks": 50}
    flann = cv2.FlannBasedMatcher(index_params, search_params)
    matches = flann.knnMatch(des_s, des_d, k=2)
    good = [m for m, n in matches if m.distance < 0.75 * n.distance]

    if len(good) < 6:
        raise ValueError(
            f"Only {len(good)} SIFT matches found (need ≥ 6). "
            "Images may be from very different angles."
        )

    pts_s = np.float32([kp_s[m.queryIdx].pt for m in good])
    pts_d = np.float32([kp_d[m.trainIdx].pt for m in good])

    M, inliers = cv2.estimateAffinePartial2D(
        pts_s, pts_d, method=cv2.RANSAC, ransacReprojThreshold=5.0)
    if M is None:
        raise ValueError("RANSAC could not estimate a stable transform.")
    return M.astype(np.float64)


_MAX_EYE_SCALE = 5.0   # scale > 5 → eye detection unreliable (side profile)
_MIN_EYE_SCALE = 0.15  # scale < 0.15 → unreasonably large face


def _is_eye_transform_sane(M: np.ndarray) -> bool:
    """Return True when the similarity scale deduced from eye landmarks is plausible."""
    scale = math.sqrt(M[0, 0] ** 2 + M[1, 0] ** 2)
    return _MIN_EYE_SCALE <= scale <= _MAX_EYE_SCALE


def _pillow_inv_coeffs(M: np.ndarray):
    """PIL transform(AFFINE) needs inverse mapping (output → input)."""
    full = np.vstack([M, [0.0, 0.0, 1.0]])
    inv = np.linalg.inv(full)
    a, b, c = inv[0]
    d, e, f = inv[1]
    return (float(a), float(b), float(c), float(d), float(e), float(f))


def _safe_crop_box(
    combined: np.ndarray,
) -> tuple[int, int, int, int]:
    """
    Find the largest fill-free axis-aligned rectangle in `combined`
    (a 2-D uint8 mask where 255 = real pixel, 0 = fill).
    Returns (x1, y1, x2, y2) in combined-array coordinates.
    """
    rows_any = np.any(combined == 255, axis=1)
    cols_any = np.any(combined == 255, axis=0)
    if not rows_any.any() or not cols_any.any():
        raise ValueError(
            "No overlapping content region between the two images.")

    Y1 = int(np.where(rows_any)[0][0])
    Y2 = int(np.where(rows_any)[0][-1]) + 1
    X1 = int(np.where(cols_any)[0][0])
    X2 = int(np.where(cols_any)[0][-1]) + 1
    roi = combined[Y1:Y2, X1:X2]

    left_of_row = np.argmax(roi == 255, axis=1)
    right_of_row = roi.shape[1] - 1 - np.argmax((roi == 255)[:, ::-1], axis=1)
    row_widths = right_of_row - left_of_row

    core = row_widths >= (row_widths.max() * 0.95)
    if not core.any():
        core = row_widths >= (row_widths.max() * 0.80)

    rx1 = int(left_of_row[core].max())
    rx2 = int(right_of_row[core].min())
    if rx1 >= rx2:
        raise ValueError("Images don't overlap enough horizontally.")

    col_slice = roi[:, rx1:rx2 + 1]
    fully_covered = np.all(col_slice == 255, axis=1)
    full_rows_idx = np.where(fully_covered)[0]
    if len(full_rows_idx) == 0:
        raise ValueError("Images don't overlap enough vertically.")

    ry1 = int(full_rows_idx[0])
    ry2 = int(full_rows_idx[-1]) + 1
    return X1 + rx1, Y1 + ry1, X1 + rx2 + 1, Y1 + ry2 + 1


# ── Public API ─────────────────────────────────────────────────────────────

def align_images(
    before: Image.Image,
    after:  Image.Image,
    *,
    canvas_width: int = 1200,
    canvas_aspect: float = 3 / 4,
    face_center_y_ratio: float = 0.40,
    eye_distance_ratio: float = 0.36,
    model_path: Path | None = None,
) -> AlignResult:
    """
    Align *before* and *after* face photos to a shared canonical position.

    Both output images are identical in size and contain only real pixels
    (no fill borders), ready for the BEAF WordPress slider plugin.

    Parameters
    ----------
    before / after          : PIL RGB images to align.
    canvas_width            : Intermediate canvas width (px). Default 1200.
    canvas_aspect           : W/H ratio of the intermediate canvas. Default 3/4.
    face_center_y_ratio     : Eye midpoint Y as fraction of canvas height. Default 0.40.
    eye_distance_ratio      : Inter-eye span as fraction of canvas width. Default 0.36.
    model_path              : Path to face_landmarker.task (auto-downloaded if None).

    Returns
    -------
    AlignResult with .before, .after, .overlay (PIL Images) and .meta (dict).
    """
    mp_ = ensure_model(model_path or MODEL_PATH)

    lm_b = _detect_landmarks(before, mp_)
    lm_a = _detect_landmarks(after,  mp_)

    W = canvas_width
    H = round(W / canvas_aspect)
    half = W * eye_distance_ratio / 2
    mid_x = W / 2
    mid_y = H * face_center_y_ratio
    dst_left = np.array([mid_x - half, mid_y])
    dst_right = np.array([mid_x + half, mid_y])

    # ── Determine M_b and M_a (forward: image coords → canvas coords) ─────
    if lm_b is not None and lm_a is not None:
        # ── Tier 1: both detected — use eye alignment only when BOTH sane ───
        le_b, re_b = _eye_centers(lm_b)
        le_a, re_a = _eye_centers(lm_a)
        M_b = _similarity_from_eyes(le_b, re_b, dst_left, dst_right)
        M_a = _similarity_from_eyes(le_a, re_a, dst_left, dst_right)
        sane_b = _is_eye_transform_sane(M_b)
        sane_a = _is_eye_transform_sane(M_a)

        if sane_b and sane_a:
            method = "bilateral_eye"
            # M_b and M_a already computed above
        elif sane_a:
            # before is a side profile — fit before to canvas, SIFT after
            method = "sift_before_eye"
            M_b = _fit_to_canvas(before.width, before.height, W, H)
            M_a2b = _sift_transform(after, before)
            M_a = _compose(M_b, M_a2b)
        elif sane_b:
            # after is a side profile — fit after to canvas, SIFT before
            method = "sift_after_eye"
            M_a = _fit_to_canvas(after.width, after.height, W, H)
            M_b2a = _sift_transform(before, after)
            M_b = _compose(M_a, M_b2a)
        else:
            # both profiles — pure fit + SIFT
            method = "sift_only"
            M_a = _fit_to_canvas(after.width,  after.height,  W, H)
            M_b2a = _sift_transform(before, after)
            M_b = _compose(M_a, M_b2a)

    elif lm_a is not None:
        # ── Tier 2a: only "after" detected ────────────────────────────────
        le_a, re_a = _eye_centers(lm_a)
        M_a_eye = _similarity_from_eyes(le_a, re_a, dst_left, dst_right)
        if _is_eye_transform_sane(M_a_eye):
            method = "sift_after_eye"
            M_a = M_a_eye
        else:
            # after is a side profile — MediaPipe detected but iris unreliable
            method = "sift_after_profile"
            M_a = _fit_to_canvas(after.width, after.height, W, H)
        M_b2a = _sift_transform(before, after)   # before coords → after coords
        M_b = _compose(M_a, M_b2a)             # before coords → canvas

    elif lm_b is not None:
        # ── Tier 2b: only "before" detected ───────────────────────────────
        le_b, re_b = _eye_centers(lm_b)
        M_b_eye = _similarity_from_eyes(le_b, re_b, dst_left, dst_right)
        if _is_eye_transform_sane(M_b_eye):
            method = "sift_before_eye"
            M_b = M_b_eye
        else:
            # before is a side profile — MediaPipe detected but iris unreliable
            method = "sift_before_profile"
            M_b = _fit_to_canvas(before.width, before.height, W, H)
        M_a2b = _sift_transform(after, before)   # after coords → before coords
        M_a = _compose(M_b, M_a2b)             # after coords → canvas

    else:
        # ── Tier 3: no face detected — full SIFT + fit-to-canvas ──────────
        method = "sift_only"
        M_a = _fit_to_canvas(after.width,  after.height,  W, H)
        M_b2a = _sift_transform(before, after)
        M_b = _compose(M_a, M_b2a)

    # ── Apply transforms ───────────────────────────────────────────────────
    canvas = (W, H)
    ic_b = _pillow_inv_coeffs(M_b)
    ic_a = _pillow_inv_coeffs(M_a)

    ab = before.transform(canvas, method=Image.AFFINE, data=ic_b,
                          resample=Image.Resampling.BICUBIC, fillcolor=(255, 255, 255))
    aa = after.transform(canvas, method=Image.AFFINE, data=ic_a,
                         resample=Image.Resampling.BICUBIC, fillcolor=(255, 255, 255))

    mask_b = Image.new("L", before.size, 255).transform(
        canvas, method=Image.AFFINE, data=ic_b,
        resample=Image.Resampling.NEAREST, fillcolor=0)
    mask_a = Image.new("L", after.size,  255).transform(
        canvas, method=Image.AFFINE, data=ic_a,
        resample=Image.Resampling.NEAREST, fillcolor=0)

    combined = np.minimum(np.array(mask_b), np.array(mask_a))
    x1, y1, x2, y2 = _safe_crop_box(combined)

    ab = ab.crop((x1, y1, x2, y2))
    aa = aa.crop((x1, y1, x2, y2))
    overlay = Image.blend(ab, aa, alpha=0.5)

    final_w, final_h = ab.size
    meta = {
        "method":  method,
        "canvas":  {"width": W, "height": H},
        "output":  {"width": final_w, "height": final_h, "crop": [x1, y1, x2, y2]},
        "canonical_eye_targets": {
            "left":  dst_left.tolist(),
            "right": dst_right.tolist(),
        },
        "affine_before_2x3": M_b.tolist(),
        "affine_after_2x3":  M_a.tolist(),
    }
    return AlignResult(before=ab, after=aa, overlay=overlay, meta=meta)
