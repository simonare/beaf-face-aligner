"""
FastAPI service for before/after face alignment.

Endpoints
---------
POST /align
    Body: multipart/form-data with fields:
        before  : image file (JPEG / PNG)
        after   : image file (JPEG / PNG)
        canvas_width        : int   (optional, default 1200)
        canvas_aspect       : float (optional, default 0.75)
        face_center_y_ratio : float (optional, default 0.40)
        eye_distance_ratio  : float (optional, default 0.36)

    Returns: JSON
        {
          "before_url": "/result/{job_id}/before.png",
          "after_url":  "/result/{job_id}/after.png",
          "overlay_url":"/result/{job_id}/overlay.png",
          "meta": { … }
        }

GET /result/{job_id}/{filename}
    Serve a generated image file.

GET /health
    Liveness probe.
"""

from __future__ import annotations

import io
import uuid
import tempfile
from pathlib import Path

from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from fastapi.responses import FileResponse, JSONResponse
from PIL import Image

from aligner import align_images, ensure_model

# ── App setup ──────────────────────────────────────────────────────────────
app = FastAPI(title="Face Aligner", version="1.0.0")

# Results are written to a temp directory that lives for the container's life.
RESULT_DIR = Path(tempfile.mkdtemp(prefix="face_aligner_"))


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
):
    # ── load images ────────────────────────────────────────────────────────
    try:
        img_before = Image.open(io.BytesIO(await before.read())).convert("RGB")
        img_after = Image.open(io.BytesIO(await after.read())).convert("RGB")
    except Exception as exc:
        raise HTTPException(
            status_code=400, detail=f"Could not read image: {exc}")

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

    # ── save outputs ───────────────────────────────────────────────────────
    job_id = uuid.uuid4().hex
    job_dir = RESULT_DIR / job_id
    job_dir.mkdir(parents=True)

    result.before.save(job_dir / "before.png",  optimize=True)
    result.after.save(job_dir / "after.png",   optimize=True)
    result.overlay.save(job_dir / "overlay.png", optimize=True)

    base = f"/result/{job_id}"
    return JSONResponse({
        "job_id":      job_id,
        "before_url":  f"{base}/before.png",
        "after_url":   f"{base}/after.png",
        "overlay_url": f"{base}/overlay.png",
        "meta":        result.meta,
    })


@app.get("/result/{job_id}/{filename}")
async def get_result(job_id: str, filename: str):
    # Sanitise path components to prevent path traversal
    if "/" in job_id or ".." in job_id or "/" in filename or ".." in filename:
        raise HTTPException(status_code=400, detail="Invalid path.")
    if filename not in ("before.png", "after.png", "overlay.png"):
        raise HTTPException(status_code=404, detail="Unknown file.")
    path = RESULT_DIR / job_id / filename
    if not path.exists():
        raise HTTPException(status_code=404, detail="Result not found.")
    return FileResponse(path, media_type="image/png")
