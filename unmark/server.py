"""Local web UI backend.

Everything runs on the user's own machine: files are read from disk (or dropped
into a local uploads folder), processing happens in a worker thread, and the
browser polls a small JSON status endpoint for progress.
"""
from __future__ import annotations

import shutil
import threading
import traceback
import uuid
from dataclasses import asdict, dataclass, field
from pathlib import Path

import cv2
import numpy as np
from fastapi import FastAPI, HTTPException, UploadFile, File, Form
from fastapi.responses import FileResponse, JSONResponse, Response
from fastapi.staticfiles import StaticFiles

from . import ffmpegio, lama as lama_mod, upscale as up
from .detect import Detection, Region, detect
from .outro import Outro, detect_outro
from .pipeline import Cancelled, Options, Result, default_output, run
from .remove import ENGINES, Remover, prepare_region

ROOT = Path(__file__).resolve().parent.parent
UPLOADS = ROOT / "uploads"
OUTPUT = ROOT / "output"
WEB = Path(__file__).resolve().parent / "web"

app = FastAPI(title="unmark")

_sources: dict[str, "Source"] = {}
_jobs: dict[str, "Job"] = {}
_lock = threading.Lock()
_lama = None
_lama_lock = threading.Lock()


def _get_lama():
    """Load LaMa once and keep it. The preview has to render with the engine the
    job will actually use, or it shows a problem the output will not have."""
    global _lama
    if _lama is not None:
        return _lama
    ok, _ = lama_mod.available()
    if not ok:
        return None
    with _lama_lock:
        if _lama is None:
            try:
                _lama = lama_mod.LaMa()
            except Exception:
                return None
    return _lama


@dataclass
class Source:
    id: str
    path: str
    info: ffmpegio.VideoInfo
    detection: Detection
    outro: Outro | None = None

    def payload(self) -> dict:
        return {
            "id": self.id,
            "path": self.path,
            "name": Path(self.path).name,
            "info": asdict(self.info),
            "label": self.info.label,
            "regions": [r.to_dict() for r in self.detection.regions],
            "detected": self.detection.found,
            "outro": self.outro.to_dict() if self.outro else None,
        }


@dataclass
class Job:
    id: str
    stage: str = "queued"
    fraction: float = 0.0
    message: str = ""
    done: bool = False
    error: str = ""
    cancelled: bool = False
    result: dict | None = None
    _cancel: threading.Event = field(default_factory=threading.Event)

    def payload(self) -> dict:
        return {"id": self.id, "stage": self.stage, "fraction": round(self.fraction, 4),
                "message": self.message, "done": self.done, "error": self.error,
                "cancelled": self.cancelled, "result": self.result}


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #

def _png(image: np.ndarray) -> bytes:
    ok, buf = cv2.imencode(".png", image)
    if not ok:
        raise HTTPException(500, "failed to encode preview")
    return buf.tobytes()


def _load_source(path: str) -> Source:
    p = Path(path).expanduser()
    if not p.exists():
        raise HTTPException(400, f"file not found: {p}")
    info = ffmpegio.probe(str(p))
    det = detect(str(p), info=info)
    card = detect_outro(str(p), info)
    src = Source(id=uuid.uuid4().hex[:12], path=str(p), info=info, detection=det,
                 outro=card)
    with _lock:
        _sources[src.id] = src
    return src


def _get_source(source_id: str) -> Source:
    src = _sources.get(source_id)
    if src is None:
        raise HTTPException(404, "unknown source; open the video again")
    return src


def _preview_frame_index(info: ffmpegio.VideoInfo) -> int:
    return int((info.n_frames or 1) * 0.4)


def _annotate(frame: np.ndarray, regions: list[Region]) -> np.ndarray:
    out = frame.copy()
    for r in regions:
        pad = max(4, r.w // 4)
        cv2.rectangle(out, (r.x - pad, r.y - pad), (r.x + r.w + pad, r.y + r.h + pad),
                      (80, 230, 90), max(1, out.shape[1] // 400))
    return out


def _zoom(frame: np.ndarray, region: Region, size: int = 320) -> np.ndarray:
    half = max(region.w, region.h) * 2
    cx, cy = region.x + region.w // 2, region.y + region.h // 2
    x0 = max(0, min(cx - half, frame.shape[1] - 2 * half))
    y0 = max(0, min(cy - half, frame.shape[0] - 2 * half))
    crop = frame[y0:y0 + 2 * half, x0:x0 + 2 * half]
    if crop.size == 0:
        crop = frame
    return cv2.resize(crop, (size, size), interpolation=cv2.INTER_NEAREST)


# --------------------------------------------------------------------------- #
# routes
# --------------------------------------------------------------------------- #

@app.get("/api/env")
def env() -> dict:
    torch_info = up.torch_status()
    lama_ok, lama_why = lama_mod.available()
    return {
        "ffmpeg": ffmpegio.have_ffmpeg(),
        "encoders": ffmpegio.encoder_report(),
        "default_encoder": ffmpegio.pick_encoder(),
        "torch": torch_info,
        "lama": {"available": lama_ok, "downloaded": lama_mod.is_downloaded(),
                 "reason": lama_why},
        "engines": list(ENGINES),
        "targets": list(up.TARGETS),
        "models": {k: v["label"] for k, v in up.MODELS.items()},
        "output_dir": str(OUTPUT),
    }


@app.post("/api/open")
def open_path(path: str = Form(...)) -> dict:
    return _load_source(path).payload()


@app.post("/api/upload")
async def upload(file: UploadFile = File(...)) -> dict:
    UPLOADS.mkdir(parents=True, exist_ok=True)
    dest = UPLOADS / f"{uuid.uuid4().hex[:8]}_{Path(file.filename or 'video.mp4').name}"
    with dest.open("wb") as fh:
        shutil.copyfileobj(file.file, fh)
    return _load_source(str(dest)).payload()


@app.get("/api/preview/{source_id}")
def preview(source_id: str, engine: str = "auto", zoom: int = 1) -> Response:
    """A before/after strip for one frame, so the fix is visible before committing."""
    src = _get_source(source_id)
    frame = ffmpegio.read_frame_at(src.path, _preview_frame_index(src.info))

    regions = src.detection.regions
    prepared = [prepare_region(r, src.info.width, src.info.height) for r in regions]
    lama = _get_lama() if engine in ("auto", "ai") else None
    cleaned = (Remover(prepared, engine, lama=lama).apply(frame)
               if prepared else frame)

    if zoom and regions:
        before = _zoom(_annotate(frame, regions), regions[0])
        after = _zoom(cleaned, regions[0])
    else:
        scale = 480 / max(frame.shape[:2])
        size = (int(frame.shape[1] * scale), int(frame.shape[0] * scale))
        before = cv2.resize(_annotate(frame, regions), size)
        after = cv2.resize(cleaned, size)

    gap = np.full((before.shape[0], 8, 3), 24, dtype=np.uint8)
    return Response(content=_png(np.hstack([before, gap, after])), media_type="image/png")


@app.post("/api/process")
def process(source_id: str = Form(...), engine: str = Form("auto"),
            target: str = Form("off"), upscale_mode: str = Form("lanczos"),
            model: str = Form(up.DEFAULT_MODEL), encoder: str = Form("auto"),
            quality: int = Form(20), remove: bool = Form(True),
            trim_outro: bool = Form(True)) -> dict:
    src = _get_source(source_id)
    options = Options(remove=remove and bool(src.detection.regions), engine=engine,
                      target=target, upscale_mode=upscale_mode, model=model,
                      encoder=encoder, quality=quality, trim_outro=trim_outro,
                      regions=src.detection.regions if remove else None)
    try:
        options.validate()
    except ValueError as exc:
        raise HTTPException(400, str(exc))

    job = Job(id=uuid.uuid4().hex[:12])
    with _lock:
        _jobs[job.id] = job

    out_path = default_output(src.path, options, OUTPUT)

    def progress(stage: str, fraction: float, message: str) -> None:
        job.stage, job.fraction, job.message = stage, fraction, message

    def work() -> None:
        try:
            res: Result = run(src.path, out_path, options, on_progress=progress,
                              should_cancel=job._cancel.is_set)
            job.result = asdict(res) | {"fps": round(res.fps, 1)}
            job.stage, job.fraction, job.message = "done", 1.0, "Finished"
        except Cancelled:
            job.cancelled = True
            job.message = "Cancelled"
        except Exception as exc:                      # surface the real reason
            job.error = f"{type(exc).__name__}: {exc}"
            job.message = job.error
            traceback.print_exc()
        finally:
            job.done = True

    threading.Thread(target=work, daemon=True, name=f"job-{job.id}").start()
    return job.payload()


@app.get("/api/job/{job_id}")
def job_status(job_id: str) -> dict:
    job = _jobs.get(job_id)
    if job is None:
        raise HTTPException(404, "unknown job")
    return job.payload()


@app.post("/api/job/{job_id}/cancel")
def job_cancel(job_id: str) -> dict:
    job = _jobs.get(job_id)
    if job is None:
        raise HTTPException(404, "unknown job")
    job._cancel.set()
    return job.payload()


@app.get("/api/video/{job_id}")
def job_video(job_id: str):
    job = _jobs.get(job_id)
    if job is None or not job.result:
        raise HTTPException(404, "no output for that job yet")
    return FileResponse(job.result["output"], media_type="video/mp4")


@app.get("/api/source-video/{source_id}")
def source_video(source_id: str):
    return FileResponse(_get_source(source_id).path, media_type="video/mp4")


@app.exception_handler(ffmpegio.FFmpegError)
def ffmpeg_error(_request, exc: ffmpegio.FFmpegError):
    return JSONResponse({"detail": str(exc)}, status_code=400)


app.mount("/", StaticFiles(directory=str(WEB), html=True), name="web")
