"""Upscaling: resolution targets, and an optional Real-ESRGAN engine.

Two paths:

  lanczos   no upscaler in the frame loop; ffmpeg resamples on the way out.
            Sharp and effectively free, but invents no detail.
  ai        Real-ESRGAN runs on the GPU inside the frame loop, then ffmpeg
            resamples its output to the exact target.

The Real-ESRGAN networks are implemented here rather than pulled from basicsr,
which does not install cleanly on Python 3.12 and drags in a large dependency
tree for two model definitions.
"""
from __future__ import annotations

import urllib.request
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

MODELS_DIR = Path(__file__).resolve().parent.parent / "models"

# Target resolutions are keyed on the *short* side, so vertical video scales the
# way people expect: a 720x1280 clip at "4k" becomes 2160x3840.
TARGETS: dict[str, int | None] = {
    "off": None,
    "720p": 720,
    "1080p": 1080,
    "1440p": 1440,
    "4k": 2160,
}

MODELS = {
    "general-x4v3": {
        "arch": "compact", "scale": 4, "num_feat": 64, "num_conv": 32,
        "file": "realesr-general-x4v3.pth",
        "url": ("https://github.com/xinntao/Real-ESRGAN/releases/download/"
                "v0.2.5.0/realesr-general-x4v3.pth"),
        "label": "Real-ESRGAN general v3 (fast, 1.2 MB)",
    },
    "x4plus": {
        "arch": "rrdb", "scale": 4, "num_feat": 64, "num_block": 23,
        "file": "RealESRGAN_x4plus.pth",
        "url": ("https://github.com/xinntao/Real-ESRGAN/releases/download/"
                "v0.1.0/RealESRGAN_x4plus.pth"),
        "label": "Real-ESRGAN x4plus (slow, highest detail, 64 MB)",
    },
    "animevideo-x4": {
        "arch": "compact", "scale": 4, "num_feat": 64, "num_conv": 16,
        "file": "realesr-animevideov3.pth",
        "url": ("https://github.com/xinntao/Real-ESRGAN/releases/download/"
                "v0.2.5.0/realesr-animevideov3.pth"),
        "label": "Real-ESRGAN anime video v3 (fast, for animation)",
    },
}
DEFAULT_MODEL = "general-x4v3"


@dataclass(frozen=True)
class Plan:
    """How a clip gets from its source size to the requested target."""
    src_w: int
    src_h: int
    out_w: int
    out_h: int
    net_scale: int          # upscale factor applied inside the frame loop (1 = none)
    mode: str               # "lanczos" or "ai"

    @property
    def changes_size(self) -> bool:
        return (self.out_w, self.out_h) != (self.src_w, self.src_h)


def _even(n: float) -> int:
    return max(2, int(round(n)) // 2 * 2)


def resolve_mode(mode: str) -> str:
    """Turn "auto" into a concrete upscaler based on what this machine has.

    A dedicated GPU is used whenever one is present; the CPU path is a fallback,
    not a default. Real-ESRGAN reconstructs detail that lanczos can only blur
    into existence, and the cost is minutes rather than seconds.
    """
    if mode != "auto":
        return mode
    status = torch_status()
    return "ai" if (status["available"] and status["cuda"]) else "lanczos"


def plan_upscale(width: int, height: int, target: str, mode: str = "lanczos",
                 model: str = DEFAULT_MODEL) -> Plan:
    """Resolve a target name into concrete output dimensions and a net scale."""
    mode = resolve_mode(mode)
    if target not in TARGETS:
        raise ValueError(f"unknown target {target!r}; expected one of {list(TARGETS)}")
    short_side = TARGETS[target]

    if short_side is None:
        out_w, out_h = width, height
    else:
        factor = short_side / float(min(width, height))
        out_w, out_h = _even(width * factor), _even(height * factor)

    net_scale = 1
    if mode == "ai" and (out_w > width or out_h > height):
        net_scale = int(MODELS[model]["scale"])
    else:
        mode = "lanczos"
    return Plan(width, height, out_w, out_h, net_scale, mode)


def torch_status() -> dict:
    """Report whether the AI path can run, without importing torch eagerly."""
    try:
        import torch
    except Exception:
        return {"available": False, "cuda": False, "device": None,
                "reason": "PyTorch is not installed"}
    cuda = bool(torch.cuda.is_available())
    return {
        "available": True,
        "cuda": cuda,
        "device": torch.cuda.get_device_name(0) if cuda else "cpu",
        "reason": "" if cuda else "CUDA not available; the AI path would run on CPU",
    }


def model_path(model: str) -> Path:
    return MODELS_DIR / MODELS[model]["file"]


def ensure_model(model: str, progress=None) -> Path:
    """Download the weights on first use."""
    dest = model_path(model)
    if dest.exists() and dest.stat().st_size > 0:
        return dest
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(".part")
    url = MODELS[model]["url"]

    def hook(blocks, block_size, total):
        if progress and total > 0:
            progress(min(1.0, blocks * block_size / total))

    urllib.request.urlretrieve(url, tmp, reporthook=hook)
    tmp.replace(dest)
    return dest


# --------------------------------------------------------------------------- #
# networks
# --------------------------------------------------------------------------- #

def _build_nets():
    """Define the two Real-ESRGAN architectures. Imported lazily with torch."""
    import torch
    from torch import nn
    from torch.nn import functional as F

    class ResidualDenseBlock(nn.Module):
        def __init__(self, nf=64, gc=32):
            super().__init__()
            self.conv1 = nn.Conv2d(nf, gc, 3, 1, 1)
            self.conv2 = nn.Conv2d(nf + gc, gc, 3, 1, 1)
            self.conv3 = nn.Conv2d(nf + 2 * gc, gc, 3, 1, 1)
            self.conv4 = nn.Conv2d(nf + 3 * gc, gc, 3, 1, 1)
            self.conv5 = nn.Conv2d(nf + 4 * gc, nf, 3, 1, 1)
            self.lrelu = nn.LeakyReLU(0.2, inplace=True)

        def forward(self, x):
            x1 = self.lrelu(self.conv1(x))
            x2 = self.lrelu(self.conv2(torch.cat((x, x1), 1)))
            x3 = self.lrelu(self.conv3(torch.cat((x, x1, x2), 1)))
            x4 = self.lrelu(self.conv4(torch.cat((x, x1, x2, x3), 1)))
            x5 = self.conv5(torch.cat((x, x1, x2, x3, x4), 1))
            return x5 * 0.2 + x

    class RRDB(nn.Module):
        def __init__(self, nf, gc=32):
            super().__init__()
            self.rdb1, self.rdb2, self.rdb3 = (ResidualDenseBlock(nf, gc) for _ in range(3))

        def forward(self, x):
            return self.rdb3(self.rdb2(self.rdb1(x))) * 0.2 + x

    class RRDBNet(nn.Module):
        def __init__(self, num_feat=64, num_block=23, num_grow_ch=32, scale=4):
            super().__init__()
            self.scale = scale
            self.conv_first = nn.Conv2d(3, num_feat, 3, 1, 1)
            self.body = nn.Sequential(*[RRDB(num_feat, num_grow_ch)
                                        for _ in range(num_block)])
            self.conv_body = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
            self.conv_up1 = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
            self.conv_up2 = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
            self.conv_hr = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
            self.conv_last = nn.Conv2d(num_feat, 3, 3, 1, 1)
            self.lrelu = nn.LeakyReLU(0.2, inplace=True)

        def forward(self, x):
            feat = self.conv_first(x)
            feat = feat + self.conv_body(self.body(feat))
            feat = self.lrelu(self.conv_up1(
                F.interpolate(feat, scale_factor=2, mode="nearest")))
            feat = self.lrelu(self.conv_up2(
                F.interpolate(feat, scale_factor=2, mode="nearest")))
            return self.conv_last(self.lrelu(self.conv_hr(feat)))

    class SRVGGNetCompact(nn.Module):
        def __init__(self, num_feat=64, num_conv=32, scale=4):
            super().__init__()
            self.scale = scale
            layers: list = [nn.Conv2d(3, num_feat, 3, 1, 1), nn.PReLU(num_parameters=num_feat)]
            for _ in range(num_conv):
                layers += [nn.Conv2d(num_feat, num_feat, 3, 1, 1),
                           nn.PReLU(num_parameters=num_feat)]
            layers += [nn.Conv2d(num_feat, 3 * scale * scale, 3, 1, 1)]
            self.body = nn.Sequential(*layers)
            self.upsampler = nn.PixelShuffle(scale)

        def forward(self, x):
            out = self.upsampler(self.body(x))
            return out + F.interpolate(x, scale_factor=self.scale, mode="nearest")

    return RRDBNet, SRVGGNetCompact


class RealESRGAN:
    """Tiled Real-ESRGAN inference sized for modest VRAM (4 GB and up)."""

    def __init__(self, model: str = DEFAULT_MODEL, tile: int = 256, tile_pad: int = 16,
                 half: bool = True, device: str | None = None):
        import torch

        spec = MODELS[model]
        self.spec = spec
        self.scale = int(spec["scale"])
        self.tile = tile
        self.tile_pad = tile_pad

        self.device = torch.device(
            device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.half = bool(half and self.device.type == "cuda")

        RRDBNet, SRVGGNetCompact = _build_nets()
        if spec["arch"] == "rrdb":
            net = RRDBNet(num_feat=spec["num_feat"], num_block=spec["num_block"],
                          scale=self.scale)
        else:
            net = SRVGGNetCompact(num_feat=spec["num_feat"], num_conv=spec["num_conv"],
                                  scale=self.scale)

        state = torch.load(str(ensure_model(model)), map_location="cpu",
                           weights_only=True)
        for key in ("params_ema", "params"):
            if isinstance(state, dict) and key in state:
                state = state[key]
                break
        net.load_state_dict(state, strict=True)
        net.eval().to(self.device)
        if self.half:
            net.half()
        self.net = net

    def _infer(self, tensor):
        import torch
        with torch.inference_mode():
            return self.net(tensor.half() if self.half else tensor).float()

    def upscale(self, bgr: np.ndarray) -> np.ndarray:
        """Upscale one BGR uint8 frame by the network scale factor."""
        import torch

        img = bgr.astype(np.float32) / 255.0
        tensor = torch.from_numpy(img.transpose(2, 0, 1))[None].to(self.device)
        _, _, h, w = tensor.shape
        s = self.scale
        out = torch.zeros((1, 3, h * s, w * s), device=self.device)

        tile = self.tile if self.tile > 0 else max(h, w)
        pad = self.tile_pad
        for y0 in range(0, h, tile):
            for x0 in range(0, w, tile):
                y1, x1 = min(y0 + tile, h), min(x0 + tile, w)
                py0, px0 = max(y0 - pad, 0), max(x0 - pad, 0)
                py1, px1 = min(y1 + pad, h), min(x1 + pad, w)

                patch = self._infer(tensor[:, :, py0:py1, px0:px1])
                # Trim the padding back off, in output-resolution coordinates.
                ty0, tx0 = (y0 - py0) * s, (x0 - px0) * s
                out[:, :, y0 * s:y1 * s, x0 * s:x1 * s] = patch[
                    :, :, ty0:ty0 + (y1 - y0) * s, tx0:tx0 + (x1 - x0) * s]

        arr = out[0].clamp_(0, 1).cpu().numpy().transpose(1, 2, 0)
        return (arr * 255.0 + 0.5).astype(np.uint8)


def resize_to(frame: np.ndarray, width: int, height: int) -> np.ndarray:
    if frame.shape[1] == width and frame.shape[0] == height:
        return frame
    shrinking = width < frame.shape[1]
    interp = cv2.INTER_AREA if shrinking else cv2.INTER_LANCZOS4
    return cv2.resize(frame, (width, height), interpolation=interp)
