"""ffmpeg/ffprobe wrappers: probing, raw frame streaming, encoder selection."""
from __future__ import annotations

import json
import shutil
import subprocess
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import numpy as np

FFMPEG = shutil.which("ffmpeg") or "ffmpeg"
FFPROBE = shutil.which("ffprobe") or "ffprobe"

# Windows: keep console windows from flashing on every subprocess.
_NO_WINDOW = getattr(subprocess, "CREATE_NO_WINDOW", 0)

# ffmpeg needs the comma inside select() escaped, or it reads as a filter separator.
_SELECT_STRIDE = r"select='not(mod(n\,%d))'"


class FFmpegError(RuntimeError):
    pass


@dataclass(frozen=True)
class VideoInfo:
    path: str
    width: int
    height: int
    fps: float
    n_frames: int
    duration: float
    codec: str
    pix_fmt: str
    has_audio: bool

    @property
    def label(self) -> str:
        return f"{self.width}x{self.height} @ {self.fps:g}fps, {self.duration:.1f}s"


def _run(cmd: list[str]) -> subprocess.CompletedProcess:
    return subprocess.run(cmd, capture_output=True, creationflags=_NO_WINDOW)


def have_ffmpeg() -> bool:
    return _run([FFMPEG, "-version"]).returncode == 0


def probe(path: str | Path) -> VideoInfo:
    """Read stream metadata. Falls back to counting packets when nb_frames is absent."""
    path = str(path)
    cmd = [FFPROBE, "-v", "error", "-print_format", "json",
           "-show_streams", "-show_format", path]
    res = _run(cmd)
    if res.returncode != 0:
        raise FFmpegError(res.stderr.decode("utf-8", "replace").strip() or "ffprobe failed")
    data = json.loads(res.stdout)

    video = next((s for s in data["streams"] if s.get("codec_type") == "video"), None)
    if video is None:
        raise FFmpegError("no video stream found")
    has_audio = any(s.get("codec_type") == "audio" for s in data["streams"])

    num, _, den = video.get("r_frame_rate", "0/1").partition("/")
    fps = float(num) / float(den) if float(den or 0) else 0.0
    duration = float(video.get("duration") or data["format"].get("duration") or 0.0)

    n_frames = int(video.get("nb_frames") or 0)
    if n_frames <= 0:
        n_frames = int(round(fps * duration)) if fps and duration else 0

    return VideoInfo(
        path=path,
        width=int(video["width"]),
        height=int(video["height"]),
        fps=fps or 30.0,
        n_frames=n_frames,
        duration=duration,
        codec=video.get("codec_name", "?"),
        pix_fmt=video.get("pix_fmt", "?"),
        has_audio=has_audio,
    )


def read_gray_frames(path: str | Path, stride: int = 1, limit: int | None = None,
                     scale: tuple[int, int] | None = None) -> np.ndarray:
    """Decode luma frames as a (N, H, W) uint8 array.

    `stride` keeps every Nth frame; `limit` caps how many are returned. Used by the
    detector, which needs coverage of the whole clip rather than every frame.
    """
    info = probe(path)
    w, h = scale or (info.width, info.height)

    filters = []
    if stride > 1:
        filters.append(_SELECT_STRIDE % stride)
    if scale:
        filters.append(f"scale={w}:{h}")

    cmd = [FFMPEG, "-v", "error", "-i", str(path)]
    if filters:
        cmd += ["-vf", ",".join(filters), "-fps_mode", "passthrough"]
    if limit:
        cmd += ["-frames:v", str(limit)]
    cmd += ["-f", "rawvideo", "-pix_fmt", "gray", "-"]

    res = _run(cmd)
    if res.returncode != 0:
        raise FFmpegError(res.stderr.decode("utf-8", "replace").strip() or "decode failed")

    frame_bytes = w * h
    usable = len(res.stdout) - (len(res.stdout) % frame_bytes)
    if usable == 0:
        raise FFmpegError("decoded no frames")
    return np.frombuffer(res.stdout[:usable], dtype=np.uint8).reshape(-1, h, w)


def read_frame_at(path: str | Path, index: int) -> np.ndarray:
    """Decode a single BGR frame by index, for UI previews."""
    info = probe(path)
    index = max(0, min(index, max(info.n_frames - 1, 0)))
    ts = index / info.fps if info.fps else 0.0

    cmd = [FFMPEG, "-v", "error", "-ss", f"{ts:.4f}", "-i", str(path),
           "-frames:v", "1", "-f", "rawvideo", "-pix_fmt", "bgr24", "-"]
    res = _run(cmd)
    expected = info.width * info.height * 3
    if res.returncode != 0 or len(res.stdout) < expected:
        # Seeking past a short/odd stream: fall back to the first frame.
        cmd = [FFMPEG, "-v", "error", "-i", str(path), "-frames:v", "1",
               "-f", "rawvideo", "-pix_fmt", "bgr24", "-"]
        res = _run(cmd)
        if res.returncode != 0 or len(res.stdout) < expected:
            raise FFmpegError("could not decode preview frame")
    return np.frombuffer(res.stdout[:expected], dtype=np.uint8).reshape(
        info.height, info.width, 3)


def open_bgr_reader(path: str | Path, info: VideoInfo) -> subprocess.Popen:
    """Start a process streaming full-rate bgr24 frames on stdout."""
    cmd = [FFMPEG, "-v", "error", "-i", str(path),
           "-f", "rawvideo", "-pix_fmt", "bgr24", "-"]
    return subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
                            bufsize=10 ** 8, creationflags=_NO_WINDOW)


def iter_bgr_frames(proc: subprocess.Popen, width: int, height: int):
    """Yield (H, W, 3) uint8 frames from a reader started by `open_bgr_reader`."""
    frame_bytes = width * height * 3
    stream = proc.stdout
    assert stream is not None
    while True:
        buf = stream.read(frame_bytes)
        if not buf or len(buf) < frame_bytes:
            break
        yield np.frombuffer(buf, dtype=np.uint8).reshape(height, width, 3)


def read_bgr_roi(path: str | Path, x: int, y: int, w: int, h: int,
                 stride: int = 1, limit: int | None = None) -> np.ndarray:
    """Decode only a crop of the frame, as (N, h, w, 3) uint8.

    The matte solve needs many frames of one small region; cropping in ffmpeg
    keeps that to a few megabytes regardless of the source resolution.
    """
    filters = [f"crop={w}:{h}:{x}:{y}"]
    if stride > 1:
        filters.append(_SELECT_STRIDE % stride)

    cmd = [FFMPEG, "-v", "error", "-i", str(path),
           "-vf", ",".join(filters), "-fps_mode", "passthrough"]
    if limit:
        cmd += ["-frames:v", str(limit)]
    cmd += ["-f", "rawvideo", "-pix_fmt", "bgr24", "-"]

    res = _run(cmd)
    if res.returncode != 0:
        raise FFmpegError(res.stderr.decode("utf-8", "replace").strip() or "roi decode failed")

    frame_bytes = w * h * 3
    usable = len(res.stdout) - (len(res.stdout) % frame_bytes)
    if usable == 0:
        raise FFmpegError("decoded no roi frames")
    return np.frombuffer(res.stdout[:usable], dtype=np.uint8).reshape(-1, h, w, 3)


@lru_cache(maxsize=1)
def available_encoders() -> tuple[str, ...]:
    res = _run([FFMPEG, "-hide_banner", "-encoders"])
    text = res.stdout.decode("utf-8", "replace")
    return tuple(name for name in
                 ("hevc_nvenc", "h264_nvenc", "libx265", "libx264")
                 if f" {name} " in text)


@lru_cache(maxsize=1)
def working_encoders() -> tuple[str, ...]:
    """Encoders that actually initialise here.

    Being listed by `ffmpeg -encoders` is not enough: NVENC is compiled in on
    every full build but refuses to open when the installed driver predates the
    nvenc API the build was made against. Only a real (tiny) encode tells us.
    """
    good = []
    for name in available_encoders():
        probe_cmd = [FFMPEG, "-v", "error", "-y",
                     "-f", "lavfi", "-i", "color=c=black:s=128x128:d=0.1",
                     "-c:v", name, "-frames:v", "1", "-f", "null", "-"]
        if _run(probe_cmd).returncode == 0:
            good.append(name)
    return tuple(good)


def encoder_report() -> list[dict]:
    """Per-encoder availability, for the UI to explain what it picked and why."""
    ok = set(working_encoders())
    listed = available_encoders()
    return [{"name": n, "listed": True, "usable": n in ok} for n in listed]


def pick_encoder(preference: str = "auto") -> str:
    encoders = working_encoders()
    if preference != "auto":
        if preference not in encoders:
            reason = ("it is not in this ffmpeg build"
                      if preference not in available_encoders()
                      else "it failed to initialise (often an out-of-date GPU driver)")
            raise FFmpegError(f"encoder {preference!r} is unusable: {reason}")
        return preference
    for name in ("hevc_nvenc", "h264_nvenc", "libx264"):
        if name in encoders:
            return name
    raise FFmpegError("no usable video encoder found")


def _quality_args(encoder: str, quality: int) -> list[str]:
    """`quality` is a CRF-like number: lower is better, 14-30 is the useful range."""
    if encoder.endswith("nvenc"):
        return ["-preset", "p5", "-tune", "hq", "-rc", "vbr",
                "-cq", str(quality), "-b:v", "0"]
    return ["-crf", str(quality), "-preset", "medium"]


def open_encoder(out_path: str | Path, in_w: int, in_h: int, fps: float,
                 source: str | Path | None = None, copy_audio: bool = False,
                 out_w: int | None = None, out_h: int | None = None,
                 encoder: str = "auto", quality: int = 20) -> subprocess.Popen:
    """Start an encoder reading bgr24 frames of size (in_w, in_h) on stdin."""
    enc = pick_encoder(encoder)
    cmd = [FFMPEG, "-v", "error", "-y",
           "-f", "rawvideo", "-pix_fmt", "bgr24",
           "-s", f"{in_w}x{in_h}", "-r", f"{fps:.6f}", "-i", "-"]

    if copy_audio and source is not None:
        cmd += ["-i", str(source), "-map", "0:v:0", "-map", "1:a:0",
                "-c:a", "copy", "-shortest"]

    if out_w and out_h and (out_w != in_w or out_h != in_h):
        flags = "lanczos" if out_w > in_w else "bicubic"
        cmd += ["-filter:v", f"scale={out_w}:{out_h}:flags={flags}"]

    cmd += ["-c:v", enc, *_quality_args(enc, quality),
            "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(out_path)]

    return subprocess.Popen(cmd, stdin=subprocess.PIPE, stdout=subprocess.DEVNULL,
                            stderr=subprocess.PIPE, bufsize=10 ** 8,
                            creationflags=_NO_WINDOW)
