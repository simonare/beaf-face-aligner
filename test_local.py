#!/usr/bin/env python3
"""
Local test script — runs alignment directly without the HTTP layer.

Usage:
    python test_local.py                              # uses samples/before.png + samples/after.png
    python test_local.py path/to/b.jpg path/to/a.jpg
    python test_local.py ... --out /custom/dir

    # Resize output while preserving aspect ratio (0 = auto)
    python test_local.py ... --ratio 800 0    # fix width=800, height auto
    python test_local.py ... --ratio 0 1200   # fix height=1200, width auto

    # Override output format (jpg, png, webp …)
    python test_local.py ... --format webp

Outputs (in tmp/ by default):
    <original_before_stem>_before_<id>.<ext>
    <original_after_stem>_after_<id>.<ext>
    <original_before_stem>_overlay_<id>.<ext>
    meta_<id>.json
"""

from __future__ import annotations

import argparse
import json
import uuid
from pathlib import Path

from PIL import Image

from aligner import align_images

_BASE = Path(__file__).resolve().parent
_SAMPLES = _BASE / "samples"
_TMP = _BASE / "tmp"

# Pillow format keyword per file extension
_FMT_MAP = {
    ".jpg": "JPEG",
    ".jpeg": "JPEG",
    ".png": "PNG",
    ".webp": "WEBP",
    ".tif": "TIFF",
    ".tiff": "TIFF",
}


def _resolve_size(img: Image.Image, ratio: list[int]) -> tuple[int, int] | None:
    """Return (w, h) after applying --ratio, or None if no resize needed."""
    rw, rh = ratio
    if rw == 0 and rh == 0:
        return None
    ow, oh = img.size
    if rw == 0:
        rw = round(ow * rh / oh)
    elif rh == 0:
        rh = round(oh * rw / ow)
    return rw, rh


def _save(img: Image.Image, path: Path, fmt: str) -> None:
    kwargs: dict = {"optimize": True}
    if fmt == "JPEG":
        kwargs["quality"] = 92
    elif fmt == "WEBP":
        kwargs["quality"] = 90
    img.save(path, format=fmt, **kwargs)


def main() -> None:
    parser = argparse.ArgumentParser(description="Local face-align test")
    parser.add_argument("before", type=Path, nargs="?",
                        default=_SAMPLES / "before.png",
                        help="Before photo (default: samples/before.png)")
    parser.add_argument("after",  type=Path, nargs="?",
                        default=_SAMPLES / "after.png",
                        help="After photo  (default: samples/after.png)")
    parser.add_argument("--out",  type=Path, default=_TMP,
                        help="Output directory (default: tmp/)")
    parser.add_argument("--canvas-width",        type=int,   default=1200)
    parser.add_argument("--canvas-aspect",       type=float, default=0.75)
    parser.add_argument("--face-center-y-ratio", type=float, default=0.40)
    parser.add_argument("--eye-distance-ratio",  type=float, default=0.36)
    parser.add_argument(
        "--ratio", type=int, nargs=2, default=[0, 0], metavar=("W", "H"),
        help="Resize output preserving aspect ratio. Use 0 for auto axis. "
             "Examples: --ratio 800 0  --ratio 0 1200")
    parser.add_argument(
        "--format", dest="fmt", default=None,
        help="Output format override: png, jpg, webp, tiff …")
    args = parser.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)

    # ── Determine output extension & Pillow format ─────────────────────────
    if args.fmt:
        ext = "." + args.fmt.lstrip(".").lower()
        if ext in (".jpg", ".jpeg"):
            ext = ".jpg"
        pil_fmt = _FMT_MAP.get(ext)
        if pil_fmt is None:
            parser.error(f"Unsupported format: {args.fmt}")
    else:
        # Default: match the input before-file extension
        ext = args.before.suffix.lower()
        if ext in ("", "."):
            ext = ".png"
        if ext in (".jpg", ".jpeg"):
            ext = ".jpg"
        pil_fmt = _FMT_MAP.get(ext, "PNG")

    print(f"Loading  before : {args.before}")
    print(f"Loading  after  : {args.after}")
    img_before = Image.open(args.before).convert("RGB")
    img_after = Image.open(args.after).convert("RGB")

    print("Aligning …")
    result = align_images(
        img_before, img_after,
        canvas_width=args.canvas_width,
        canvas_aspect=args.canvas_aspect,
        face_center_y_ratio=args.face_center_y_ratio,
        eye_distance_ratio=args.eye_distance_ratio,
    )

    # ── Optional resize ────────────────────────────────────────────────────
    target_size = _resolve_size(result.before, args.ratio)
    if target_size:
        result.before = result.before.resize(target_size,  Image.LANCZOS)
        result.after = result.after.resize(target_size,   Image.LANCZOS)
        result.overlay = result.overlay.resize(target_size, Image.LANCZOS)

    # ── Output filenames ───────────────────────────────────────────────────
    job_id = uuid.uuid4().hex[:8]
    before_stem = args.before.stem
    after_stem = args.after.stem
    out_before = args.out / f"{before_stem}_before_{job_id}{ext}"
    out_after = args.out / f"{after_stem}_after_{job_id}{ext}"
    out_overlay = args.out / f"{before_stem}_overlay_{job_id}{ext}"
    out_meta = args.out / f"meta_{job_id}.json"

    _save(result.before,  out_before,  pil_fmt)
    _save(result.after,   out_after,   pil_fmt)
    _save(result.overlay, out_overlay, pil_fmt)
    out_meta.write_text(json.dumps(result.meta, indent=2), encoding="utf-8")

    w, h = result.before.size
    print(f"\nDone. Job: {job_id}  |  Output: {w}×{h} px  [{pil_fmt}]")
    print(f"  {out_before}")
    print(f"  {out_after}")
    print(f"  {out_overlay}  ← QC blend, not for BEAF plugin")
    print(f"  {out_meta}")


if __name__ == "__main__":
    main()
