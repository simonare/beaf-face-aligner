#!/usr/bin/env python3
"""
Local test script — runs alignment directly without the HTTP layer.

Usage:
    python test_local.py                         # uses samples/before.png + samples/after.png
    python test_local.py path/to/b.jpg path/to/a.jpg
    python test_local.py ... --out /custom/dir

Outputs (in tmp/ by default, named with a random job-id):
    before_<id>.png
    after_<id>.png
    overlay_<id>.png
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
    args = parser.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)

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

    job_id = uuid.uuid4().hex[:8]
    out_before = args.out / f"before_{job_id}.png"
    out_after = args.out / f"after_{job_id}.png"
    out_overlay = args.out / f"overlay_{job_id}.png"
    out_meta = args.out / f"meta_{job_id}.json"

    result.before.save(out_before,   optimize=True)
    result.after.save(out_after,    optimize=True)
    result.overlay.save(out_overlay, optimize=True)
    out_meta.write_text(json.dumps(result.meta, indent=2), encoding="utf-8")

    w, h = result.before.size
    print(f"\nDone. Job: {job_id}  |  Output: {w}×{h} px")
    print(f"  {out_before}")
    print(f"  {out_after}")
    print(f"  {out_overlay}  ← QC blend, not for BEAF plugin")
    print(f"  {out_meta}")


if __name__ == "__main__":
    main()
