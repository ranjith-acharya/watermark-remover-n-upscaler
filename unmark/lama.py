"""LaMa inpainting for the AI removal tier.

Uses the TorchScript export of big-lama, so no model architecture code and no
basicsr-style dependency tree is needed - just torch.

Only the padded ROI around the watermark is ever passed through the network
(typically under 100x100 px), which is why this runs comfortably on a 4 GB
laptop GPU and stays fast enough for full-length clips.
"""
from __future__ import annotations

import urllib.request
from pathlib import Path

import numpy as np

MODEL_URL = "https://github.com/Sanster/models/releases/download/add_big_lama/big-lama.pt"
MODEL_FILE = "big-lama.pt"
MODELS_DIR = Path(__file__).resolve().parent.parent / "models"
PAD_TO = 8          # the network needs both dimensions to be a multiple of 8


def model_path() -> Path:
    return MODELS_DIR / MODEL_FILE


def is_downloaded() -> bool:
    p = model_path()
    return p.exists() and p.stat().st_size > 1_000_000


def ensure_model(progress=None) -> Path:
    dest = model_path()
    if is_downloaded():
        return dest
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(".part")

    def hook(blocks, block_size, total):
        if progress and total > 0:
            progress(min(1.0, blocks * block_size / total))

    urllib.request.urlretrieve(MODEL_URL, tmp, reporthook=hook)
    tmp.replace(dest)
    return dest


def available() -> tuple[bool, str]:
    """Whether the AI removal tier can run, and why not when it cannot."""
    try:
        import torch  # noqa: F401
    except Exception:
        return False, "PyTorch is not installed"
    return True, ""


class LaMa:
    def __init__(self, device: str | None = None, progress=None):
        import torch

        self.torch = torch
        self.device = torch.device(
            device or ("cuda" if torch.cuda.is_available() else "cpu"))
        model = torch.jit.load(str(ensure_model(progress)), map_location="cpu")
        model.eval().to(self.device)
        self.model = model

    def inpaint(self, roi_bgr: np.ndarray, mask: np.ndarray) -> np.ndarray:
        """Inpaint the masked pixels of one BGR patch. Returns float32 BGR."""
        torch = self.torch
        h, w = mask.shape
        ph = (PAD_TO - h % PAD_TO) % PAD_TO
        pw = (PAD_TO - w % PAD_TO) % PAD_TO

        rgb = np.ascontiguousarray(roi_bgr[:, :, ::-1]).astype(np.float32) / 255.0
        m = (mask > 0).astype(np.float32)
        if ph or pw:
            rgb = np.pad(rgb, ((0, ph), (0, pw), (0, 0)), mode="edge")
            m = np.pad(m, ((0, ph), (0, pw)), mode="constant")

        img_t = torch.from_numpy(rgb.transpose(2, 0, 1))[None].to(self.device)
        mask_t = torch.from_numpy(m)[None, None].to(self.device)

        with torch.inference_mode():
            out = self.model(img_t, mask_t)

        arr = out[0].permute(1, 2, 0).cpu().numpy()
        if arr.max() > 1.5:                     # some exports return 0..255
            arr = arr / 255.0
        arr = np.clip(arr, 0.0, 1.0)[:h, :w]
        return (arr[:, :, ::-1] * 255.0).astype(np.float32)
