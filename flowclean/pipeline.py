"""Orchestration: detect, remove, upscale, encode.

One pass over the video. Frames stream out of ffmpeg as raw BGR, go through the
removal engine and (optionally) the upscaler, and stream straight back into an
encoder, so nothing is ever staged to disk as image files.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Callable

import numpy as np

from . import ffmpegio, lama as lama_mod, upscale as up
from .detect import Detection, Region, detect
from .remove import ENGINES, PreparedRegion, Remover, estimate_matte, prepare_region

MATTE_SAMPLES = 96


@dataclass
class Options:
    remove: bool = True
    engine: str = "balanced"             # fast | balanced | ai
    target: str = "off"                  # off | 720p | 1080p | 1440p | 4k
    upscale_mode: str = "lanczos"        # lanczos | ai
    model: str = up.DEFAULT_MODEL
    encoder: str = "auto"
    quality: int = 20
    regions: list[Region] | None = None   # skip detection when supplied

    def validate(self) -> None:
        if self.engine not in ENGINES:
            raise ValueError(f"engine must be one of {ENGINES}")
        if self.target not in up.TARGETS:
            raise ValueError(f"target must be one of {list(up.TARGETS)}")
        if self.upscale_mode not in ("lanczos", "ai"):
            raise ValueError("upscale_mode must be 'lanczos' or 'ai'")
        if not self.remove and self.target == "off":
            raise ValueError("nothing to do: removal is off and no upscale target is set")


@dataclass
class Result:
    output: str
    info: dict
    regions: list[dict] = field(default_factory=list)
    plan: dict = field(default_factory=dict)
    engine: str = ""
    encoder: str = ""
    matte: dict | None = None
    frames: int = 0
    seconds: float = 0.0

    @property
    def fps(self) -> float:
        return self.frames / self.seconds if self.seconds else 0.0


class Cancelled(RuntimeError):
    pass


ProgressFn = Callable[[str, float, str], None]


def _noop(stage: str, fraction: float, message: str) -> None:
    pass


def build_regions(path: str, info: ffmpegio.VideoInfo, options: Options,
                  on_progress: ProgressFn) -> tuple[list[PreparedRegion], Detection | None]:
    """Detect (or accept) watermark regions and prepare them for the frame loop."""
    detection = None
    if options.regions:
        regions = options.regions
    else:
        on_progress("detect", 0.0, "Scanning for watermarks")
        detection = detect(path, info=info)
        regions = detection.regions

    prepared = [prepare_region(r, info.width, info.height) for r in regions]

    # The matte is an enhancement on top of any fill engine, not a tier of its
    # own: solve it, then keep it only if it verifies. A bad matte is visibly
    # worse than a plain inpaint, so the gate matters more than the coverage.
    if options.engine != "fast" and prepared:
        stride = max(1, info.n_frames // MATTE_SAMPLES) if info.n_frames else 1
        for i, prep in enumerate(prepared):
            on_progress("matte", i / len(prepared), "Solving watermark transparency")
            rois = ffmpegio.read_bgr_roi(path, prep.x, prep.y, prep.w, prep.h,
                                         stride=stride, limit=MATTE_SAMPLES)
            if len(rois) < 8:
                prep.matte_note = "too few frames to solve the matte"
                continue
            matte = estimate_matte(rois, prep.mask)
            if matte.usable:
                prep.matte = matte
                prep.matte_note = (f"exact recovery on {matte.coverage:.0%} of the "
                                   f"glyph (seam {matte.seam:.2f})")
            else:
                prep.matte_note = f"inpaint only - {matte.reject_reason()}"
    return prepared, detection


def run(input_path: str | Path, output_path: str | Path, options: Options | None = None,
        on_progress: ProgressFn | None = None,
        should_cancel: Callable[[], bool] | None = None) -> Result:
    options = options or Options()
    options.validate()
    on_progress = on_progress or _noop
    should_cancel = should_cancel or (lambda: False)

    input_path, output_path = str(input_path), str(output_path)
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)

    info = ffmpegio.probe(input_path)
    prepared: list[PreparedRegion] = []
    detection = None
    if options.remove:
        prepared, detection = build_regions(input_path, info, options, on_progress)

    lama = None
    if options.remove and options.engine == "ai" and prepared:
        ok, why = lama_mod.available()
        if ok:
            on_progress("model", 0.0, "Loading LaMa inpainting model")
            lama = lama_mod.LaMa(progress=lambda f: on_progress(
                "model", f, f"Downloading LaMa weights {f:.0%}"))
        else:
            on_progress("model", 0.0, f"AI removal unavailable ({why}); using balanced")

    remover = Remover(prepared, options.engine, lama=lama) if prepared else None

    plan = up.plan_upscale(info.width, info.height, options.target,
                           options.upscale_mode, options.model)

    net = None
    if plan.mode == "ai":
        on_progress("model", 0.0, "Loading upscaler")
        net = up.RealESRGAN(options.model)

    loop_w, loop_h = info.width * plan.net_scale, info.height * plan.net_scale
    encoder_name = ffmpegio.pick_encoder(options.encoder)

    reader = ffmpegio.open_bgr_reader(input_path, info)
    writer = ffmpegio.open_encoder(
        output_path, loop_w, loop_h, info.fps, source=input_path,
        copy_audio=info.has_audio, out_w=plan.out_w, out_h=plan.out_h,
        encoder=encoder_name, quality=options.quality)

    total = info.n_frames or 0
    started = time.time()
    count = 0
    try:
        for frame in ffmpegio.iter_bgr_frames(reader, info.width, info.height):
            if should_cancel():
                raise Cancelled("cancelled by user")
            if remover is not None:
                frame = remover.apply(frame)
            if net is not None:
                frame = net.upscale(frame)
            try:
                writer.stdin.write(np.ascontiguousarray(frame).tobytes())
            except (BrokenPipeError, OSError):
                # The encoder died; its stderr says why far better than we can.
                err = writer.stderr.read().decode("utf-8", "replace").strip()
                raise ffmpegio.FFmpegError(err or "encoder closed unexpectedly") from None
            count += 1
            if total:
                on_progress("encode", count / total, f"Frame {count}/{total}")
            elif count % 24 == 0:
                on_progress("encode", 0.0, f"Frame {count}")

        writer.stdin.close()
        code = writer.wait()
        if code != 0:
            err = writer.stderr.read().decode("utf-8", "replace").strip()
            raise ffmpegio.FFmpegError(err or f"encoder exited with {code}")
    except BaseException:
        for proc in (reader, writer):
            if proc.poll() is None:
                proc.kill()
        raise
    finally:
        reader.stdout and reader.stdout.close()
        if reader.poll() is None:
            reader.kill()
        reader.wait()

    elapsed = time.time() - started
    on_progress("done", 1.0, f"Wrote {Path(output_path).name}")

    matte_summary = None
    if prepared:
        matte_summary = {"note": prepared[0].matte_note}
        if prepared[0].matte is not None:
            matte_summary.update(prepared[0].matte.summary())

    return Result(
        output=output_path,
        info={k: v for k, v in asdict(info).items()},
        regions=[p.region.to_dict() for p in prepared],
        plan={"out_w": plan.out_w, "out_h": plan.out_h,
              "net_scale": plan.net_scale, "mode": plan.mode},
        engine=(remover.engine if remover else "none"),
        encoder=encoder_name,
        matte=matte_summary,
        frames=count,
        seconds=round(elapsed, 2),
    )


def default_output(input_path: str | Path, options: Options,
                   out_dir: str | Path = "output") -> Path:
    stem = Path(input_path).stem
    bits = []
    if options.remove:
        bits.append("clean")
    if options.target != "off":
        bits.append(options.target)
    suffix = "_".join(bits) or "out"
    return Path(out_dir) / f"{stem}_{suffix}.mp4"
