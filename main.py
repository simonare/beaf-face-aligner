"""
FastAPI service for before/after face alignment.

Endpoints
---------
POST /align
    Body: multipart/form-data with fields:
        before  : image file (JPEG / PNG / WEBP …)
        after   : image file (JPEG / PNG / WEBP …)
        canvas_width        : int   (optional, default 1200)
        canvas_aspect       : float (optional, default 0.75)
        face_center_y_ratio : float (optional, default 0.40)
        eye_distance_ratio  : float (optional, default 0.36)
        resize_w            : int   (optional, 0 = auto)  — output resize width
        resize_h            : int   (optional, 0 = auto)  — output resize height
        output_format       : str   (optional) — png | jpg | webp | tiff
                              defaults to extension of the uploaded before file

    Returns: JSON
        {
          "job_id":      "…",
          "before_url":  "/result/{job_id}/<stem>_before.<ext>",
          "after_url":   "/result/{job_id}/<stem>_after.<ext>",
          "overlay_url": "/result/{job_id}/<stem>_overlay.<ext>",
          "meta": { … }
        }

GET /result/{job_id}/{filename}
    Serve a generated image file.

GET /health
    Liveness probe.
"""

from __future__ import annotations

import io
import re
import uuid
import tempfile
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from fastapi.responses import FileResponse, JSONResponse
from PIL import Image

from aligner import align_images, ensure_model

# ── App setup ──────────────────────────────────────────────────────────────
app = FastAPI(title="Face Aligner", version="1.0.0")

# Results are written to a temp directory that lives for the container's life.
RESULT_DIR = Path(tempfile.mkdtemp(prefix="face_aligner_"))

_FMT_MAP = {
    ".jpg":  ("JPEG", "image/jpeg"),
    ".jpeg": ("JPEG", "image/jpeg"),
    ".png":  ("PNG",  "image/png"),
    ".webp": ("WEBP", "image/webp"),
    ".tif":  ("TIFF", "image/tiff"),
    ".tiff": ("TIFF", "image/tiff"),
}
_SAFE_STEM = re.compile(r"[^\w\-]")   # keep word chars and hyphens only


def _ext_for(upload: UploadFile, override: Optional[str]) -> str:
    """Return normalised extension (.jpg / .png / .webp / .tiff)."""
    if override:
        ext = "." + override.lower().lstrip(".")
        if ext == ".jpeg":
            ext = ".jpg"
        return ext
    fname = upload.filename or ""
    ext = Path(fname).suffix.lower()
    if ext in (".jpg", ".jpeg"):
        return ".jpg"
    return ext if ext in _FMT_MAP else ".png"


def _save(img: Image.Image, path: Path, pil_fmt: str) -> None:
    kwargs: dict = {"optimize": True}
    if pil_fmt == "JPEG":
        kwargs["quality"] = 92
    elif pil_fmt == "WEBP":
        kwargs["quality"] = 90
    img.save(path, format=pil_fmt, **kwargs)


def _maybe_resize(img: Image.Image, rw: int, rh: int) -> Image.Image:
    if rw == 0 and rh == 0:
        return img
    ow, oh = img.size
    if rw == 0:
        rw = round(ow * rh / oh)
    elif rh == 0:
        rh = round(oh * rw / ow)
    return img.resize((rw, rh), Image.LANCZOS)


@app.on_event("startup")
async def _startup():
    """Pre-download the MediaPipe model so the first request isn't slow."""
    ensure_model()


# ── Endpoints ──────────────────────────────────────────────────────────────

@app.get("/health")
async def health():
    return {"status": "ok"}


@app.post("/align")
async def align(
    before: UploadFile = File(..., description="Before photo"),
    after:  UploadFile = File(..., description="After photo"),
    canvas_width:        int = Form(1200),
    canvas_aspect:       float = Form(0.75),
    face_center_y_ratio: float = Form(0.40),
    eye_distance_ratio:  float = Form(0.36),
    resize_w:            int = Form(
        0,    description="Output width (0 = keep)"),
    resize_h:            int = Form(
        0,    description="Output height (0 = keep)"),
    output_format: Optional[str] = Form(
        None, description="png | jpg | webp | tiff"),
):
    # ── load images ────────────────────────────────────────────────────────
    try:
        img_before = Image.open(io.BytesIO(await before.read())).convert("RGB")
        img_after = Image.open(io.BytesIO(await after.read())).convert("RGB")
    except Exception as exc:
        raise HTTPException(
            status_code=400, detail=f"Could not read image: {exc}")

    # ── output format ──────────────────────────────────────────────────────
    ext = _ext_for(before, output_format)
    fmt_info = _FMT_MAP.get(ext)
    if fmt_info is None:
        raise HTTPException(
            status_code=400, detail=f"Unsupported format: {ext}")
    pil_fmt, mime_type = fmt_info

    # ── align ──────────────────────────────────────────────────────────────
    try:
        result = align_images(
            img_before, img_after,
            canvas_width=canvas_width,
            canvas_aspect=canvas_aspect,
            face_center_y_ratio=face_center_y_ratio,
            eye_distance_ratio=eye_distance_ratio,
        )
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc))

    # ── optional resize ────────────────────────────────────────────────────
    result.before = _maybe_resize(result.before,  resize_w, resize_h)
    result.after = _maybe_resize(result.after,   resize_w, resize_h)
    result.overlay = _maybe_resize(result.overlay, resize_w, resize_h)

    # ── derive safe stems from original filenames ──────────────────────────
    before_stem = _SAFE_STEM.sub("_", Path(before.filename or "before").stem)[
        :64] or "before"
    after_stem = _SAFE_STEM.sub("_", Path(after.filename or "after").stem)[
        :64] or "after"

    # ── save outputs ───────────────────────────────────────────────────────
    job_id = uuid.uuid4().hex
    job_dir = RESULT_DIR / job_id
    job_dir.mkdir(parents=True)

    fn_before = f"{before_stem}_before{ext}"
    fn_after = f"{after_stem}_after{ext}"
    fn_overlay = f"{before_stem}_overlay{ext}"

    _save(result.before,  job_dir / fn_before,  pil_fmt)
    _save(result.after,   job_dir / fn_after,   pil_fmt)
    _save(result.overlay, job_dir / fn_overlay, pil_fmt)

    base = f"/result/{job_id}"
    return JSONResponse({
        "job_id":      job_id,
        "before_url":  f"{base}/{fn_before}",
        "after_url":   f"{base}/{fn_after}",
        "overlay_url": f"{base}/{fn_overlay}",
        "meta":        result.meta,
    })


@app.get("/result/{job_id}/{filename}")
async def get_result(job_id: str, filename: str):
    # Sanitise path components to prevent path traversal
    if "/" in job_id or ".." in job_id or "/" in filename or ".." in filename:
        raise HTTPException(status_code=400, detail="Invalid path.")
    # Allow any filename that ends with a known image extension
    if not any(filename.endswith(ext) for ext in _FMT_MAP):
        raise HTTPException(status_code=404, detail="Unknown file.")
    path = RESULT_DIR / job_id / filename
    if not path.exists():
        raise HTTPException(status_code=404, detail="Result not found.")
    mime = _FMT_MAP.get(Path(filename).suffix.lower(), ("PNG", "image/png"))[1]
    return FileResponse(path, media_type=mime)
