"""Command line entry point.

    python -m unmark                     launch the web UI
    python -m unmark 01.mp4              detect + remove, same resolution
    python -m unmark 01.mp4 --to 4k      remove and upscale
    python -m unmark --detect 01.mp4     report what it finds, change nothing
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from . import upscale as up
from .detect import detect
from .ffmpegio import FFmpegError, encoder_report, have_ffmpeg, probe
from .outro import detect_outro
from .pipeline import Options, default_output, run
from .remove import ENGINES


def _progress(stage: str, fraction: float, message: str) -> None:
    bar = int(fraction * 30)
    sys.stderr.write(f"\r{stage:<8} [{'#' * bar}{'.' * (30 - bar)}] {message[:44]:<44}")
    sys.stderr.flush()
    if stage == "done":
        sys.stderr.write("\n")


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="unmark",
                                description="Automatic watermark removal and upscaling.")
    p.add_argument("input", nargs="*", help="video files (omit to launch the web UI)")
    p.add_argument("-o", "--output", help="output file (single input only)")
    p.add_argument("--to", default="off", choices=list(up.TARGETS),
                   help="upscale target, keyed on the short side (default: off)")
    p.add_argument("--engine", default="balanced", choices=list(ENGINES),
                   help="removal engine (default: balanced)")
    p.add_argument("--upscaler", default="lanczos", choices=["lanczos", "ai"],
                   help="how to upscale when --to is set (default: lanczos)")
    p.add_argument("--model", default=up.DEFAULT_MODEL, choices=list(up.MODELS))
    p.add_argument("--encoder", default="auto")
    p.add_argument("--quality", type=int, default=20, help="CRF-like, lower is better")
    p.add_argument("--keep-watermark", action="store_true",
                   help="upscale only, leave the watermark alone")
    p.add_argument("--keep-outro", action="store_true",
                   help="keep a branded end card instead of trimming it")
    p.add_argument("--flow-preset", action="store_true",
                   help="if nothing is detected, assume Google Flow's corner sparkle")
    p.add_argument("--detect", action="store_true", help="report detection and exit")
    p.add_argument("--env", action="store_true", help="report capabilities and exit")
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=8420)
    p.add_argument("--no-browser", action="store_true")
    return p


def serve(args) -> int:
    import uvicorn

    url = f"http://{args.host}:{args.port}"
    if not args.no_browser:
        import threading
        import webbrowser
        threading.Timer(1.0, lambda: webbrowser.open(url)).start()
    print(f"unmark UI on {url}  (Ctrl+C to stop)")
    uvicorn.run("unmark.server:app", host=args.host, port=args.port, log_level="warning")
    return 0


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    if not have_ffmpeg():
        print("ffmpeg was not found on PATH.", file=sys.stderr)
        return 2

    if args.env:
        print(json.dumps({"encoders": encoder_report(), "torch": up.torch_status()},
                         indent=2))
        return 0

    if not args.input:
        return serve(args)

    if args.detect:
        for path in args.input:
            det = detect(path, fallback=args.flow_preset)
            regions = [r.to_dict() for r in det.regions]
            print(f"{path}: {json.dumps(regions) if regions else 'no watermark found'}")
            card = detect_outro(path)
            print(f"  end card: {json.dumps(card.to_dict()) if card else 'none'}")
        return 0

    if args.output and len(args.input) > 1:
        print("--output only works with a single input file.", file=sys.stderr)
        return 2

    options = Options(remove=not args.keep_watermark, engine=args.engine, target=args.to,
                      upscale_mode=args.upscaler, model=args.model,
                      encoder=args.encoder, quality=args.quality,
                      trim_outro=not args.keep_outro, flow_preset=args.flow_preset)
    try:
        options.validate()
    except ValueError as exc:
        print(exc, file=sys.stderr)
        return 2

    failures = 0
    for path in args.input:
        out = Path(args.output) if args.output else default_output(path, options)
        print(f"\n{path}  ->  {out}")
        try:
            info = probe(path)
            print(f"  source: {info.label}")
            result = run(path, out, options, on_progress=_progress)
        except (FFmpegError, OSError, ValueError) as exc:
            failures += 1
            print(f"  failed: {exc}", file=sys.stderr)
            continue
        print(f"  {result.plan['out_w']}x{result.plan['out_h']} via {result.encoder}, "
              f"{result.frames} frames in {result.seconds}s ({result.fps:.1f} fps)")
        if result.regions:
            for region in result.regions:
                print(f"  watermark: {region}")
        else:
            print("  watermark: none detected - removal skipped")
        if result.outro:
            o = result.outro
            print(f"  end card:  trimmed {o['seconds']}s from {o['start_time']}s "
                  f"(confidence {o['confidence']})")
        if result.matte and result.matte.get("note"):
            print(f"  matte: {result.matte['note']}")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
