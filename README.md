# flowclean — watermark remover & upscaler

Removes the watermark from AI-generated video and upscales it, through a local
web UI. Detection is automatic: you point it at a file and press one button.

Built for Google Flow / Veo output (the sparkle glyph in the corner), but the
detector is generic — anything static and overlaid is found the same way.

Everything runs on your own machine. Nothing is uploaded anywhere.

---

## Prerequisites

| | Required | Notes |
| --- | --- | --- |
| **Python 3.10+** | yes | 3.12 recommended |
| **ffmpeg + ffprobe** | yes | must be on your `PATH` |
| NVIDIA GPU + recent driver | no | only for the AI tiers |
| ~500 MB disk | yes | ~3.5 GB if you install the AI tiers |

### Installing ffmpeg

**Windows** — easiest is winget or Chocolatey:

```powershell
winget install Gyan.FFmpeg
# or:  choco install ffmpeg-full
```

**macOS** — `brew install ffmpeg`
**Linux** — `sudo apt install ffmpeg` (or your distro's equivalent)

Verify before going further — this must print a version:

```
ffmpeg -version
```

---

## Install

```
git clone https://github.com/ranjith-acharya/watermark-remover-n-upscaler.git
cd watermark-remover-n-upscaler
```

**Windows** — just run it. The first launch creates the virtual environment and
installs dependencies on its own:

```
start.bat
```

**Linux / macOS**:

```
chmod +x start.sh
./start.sh
```

**Manual setup**, if you would rather do it yourself or the script fails:

```
python -m venv .venv
.venv\Scripts\python.exe -m pip install -r requirements.txt     # Windows
.venv/bin/python -m pip install -r requirements.txt             # Linux/macOS
```

Your browser opens at <http://127.0.0.1:8420>. That is the whole install.

### Optional: the AI tiers

The defaults need no GPU and no PyTorch. If you want LaMa inpainting or
Real-ESRGAN upscaling, add PyTorch (~2.5 GB):

```
install_ai.bat                                                  # Windows
.venv/bin/python -m pip install torch torchvision \
    --index-url https://download.pytorch.org/whl/cu124          # Linux
```

Model weights (~210 MB) download by themselves the first time you pick an AI
option, into `models/`. Check what the app can see with:

```
start.bat --env
```

---

## Using it

### Web UI

```
start.bat                   opens http://127.0.0.1:8420
start.bat --port 9000       use a different port
start.bat --no-browser      don't auto-open the browser
```

Drop a video in (or paste its full path), and it scans for the watermark and
shows you a before/after preview of one frame. Pick an upscale target if you
want one, press **Process**, and the result lands in `output/`.

### Command line

The CLI runs the same engine with no prompts, which is what you want for
batches:

```
start.bat 01.mp4                        remove watermark, keep resolution
start.bat 01.mp4 --to 4k                remove and upscale to 4K
start.bat 01.mp4 --to 1080p --upscaler ai   Real-ESRGAN instead of lanczos
start.bat clip1.mp4 clip2.mp4 clip3.mp4     batch, unattended
start.bat --detect 01.mp4               report findings, change nothing
start.bat --keep-watermark 01.mp4 --to 4k   upscale only
start.bat --env                         what this machine can do
```

Useful flags: `--engine fast|balanced|ai`, `--quality 14..30` (lower is better),
`--encoder`, `-o OUTPUT`.

### Performance

Measured on an RTX 3050 laptop, for a 10-second 720×1280 clip (240 frames):

| Job | Time |
| --- | --- |
| Remove only | ~12 s |
| Remove + 1080p (lanczos) | ~16 s |
| Remove + 4K (Real-ESRGAN) | ~4 min |

Audio is stream-copied through untouched.

---

## How detection works

A watermark is a small overlay that stays put while the scene behind it changes.
Each sampled frame is high-passed to strip the low-frequency picture, then the
detector looks for pixels whose response stays strong in nearly *every* frame.
Moving scene content only lights up a given pixel some of the time, so it falls
away. What survives is filtered on size, shape and placement, plus two rules
that matter more than they look:

- **Isolation** — the response must stop at the region's edge. Static scene
  geometry (a girder, a doorway) is just as persistent, but carries on past the
  box the threshold cut it down to.
- **Dominance** — one generator renders all of its marks alike, so a genuine
  second watermark scores close to the first. A scene feature that merely got
  through the filters trails it by a wide margin.

Thresholds key off the noise floor rather than a percentile of the frame, so a
single very bright static object cannot drag the threshold up until the real
watermark falls below it and fragments.

If nothing is found confidently, it falls back to Flow's known sparkle position
(measured at x 0.800–0.867, y 0.888–0.925 of the frame) and says so in the UI.

## How removal works

Three engines fill the masked pixels:

| Engine | Method | Speed |
| --- | --- | --- |
| `fast` | inverse-distance interpolation from the region border | ~24 fps |
| `balanced` | Telea inpainting, per frame — **the default** | ~20 fps |
| `ai` | LaMa inpainting on the cropped patch (needs PyTorch) | ~3 fps |

Only the padded region around the watermark is ever processed, typically under
100×100 px, which is why even the AI engine fits comfortably in 4 GB of VRAM.

On top of whichever engine runs, flowclean tries to recover the watermark's
**alpha matte** and put the real pixels back instead of invented ones. The
overlay is constant, so it scales the scene's contrast by `(1-a)`: wherever the
video moves, the temporal variance under the mark is `(1-a)²` of what it should
be. Diffusing the surrounding pixels inward estimates the un-marked mean and
variance, giving alpha from the variance ratio and the colour from the mean.

That solve needs a background that actually moves, and a bad matte looks worse
than a plain inpaint — so it must clear three independent gates before it is
used at all:

- **coverage** — how much of the glyph it explains
- **stability** — both halves of the sampled frames must agree on alpha; noise
  driving the solve shows up here and nowhere else
- **seam** — the recovered patch must join its surroundings without a step

Fail any one and the matte is discarded, the fill engine stands alone, and the
result reports which happened and why. On dark, flat footage it is normally
rejected — that is the system working, not failing.

## Upscaling

Targets key on the **short** side, so vertical video scales the way you expect:
a 720×1280 clip at `4k` becomes 2160×3840, not a squashed landscape frame.

- **Lanczos** — ffmpeg resamples on the way out. Sharp, effectively free, and
  invents no detail.
- **Real-ESRGAN** — runs on the GPU inside the frame loop, tiled at 256 px.

---

## Project layout

```
flowclean/
  ffmpegio.py    probing, raw frame streaming, encoder selection
  detect.py      automatic watermark detection
  remove.py      fill engines and the alpha matte solve
  upscale.py     resolution planning and Real-ESRGAN
  lama.py        LaMa inpainting
  pipeline.py    orchestration: detect -> remove -> upscale -> encode
  server.py      web UI backend (FastAPI)
  web/           the UI itself
  __main__.py    CLI entry point
tests/           synthetic-fixture test suite
```

The core is independent of the UI, so every engine is testable headless.

## Tests

```
.venv\Scripts\python.exe -m pytest      # Windows
.venv/bin/python -m pytest              # Linux/macOS
```

The suite builds synthetic clips where the clean frames, the glyph shape, its
alpha and its colour are all known, then checks that detection lands on the box,
that removal moves the result toward the truth, and that the matte gates reject
a corrupted solve. Real footage cannot be asserted against — there is no ground
truth for what sits behind a watermark.

---

## Branching model

| Branch | Purpose |
| --- | --- |
| `main` | Production. Always working, always tested. Releases are tagged here. |
| `develop` | Integration. Finished work lands here first and settles. |
| `feature/*` | One change each, short-lived, deleted after merge. |
| `fix/*` | Bug fixes. Same lifecycle. |

Day-to-day flow:

```
git switch develop
git switch -c feature/batch-queue
# ... work, commit ...
git switch develop
git merge --no-ff feature/batch-queue
git branch -d feature/batch-queue
```

When `develop` is stable and the tests pass, release it:

```
git switch main
git merge --no-ff develop
git tag -a v0.2.0 -m "v0.2.0 - batch queue"
git push origin main --tags
```

**Merge, don't cherry-pick.** Cherry-picking copies a commit rather than moving
it, so the same change ends up in history twice with different hashes and the
branches quietly diverge. Save `git cherry-pick` for the one case it fits: an
urgent fix that has to reach `main` without waiting for everything else sitting
in `develop`. Then merge `main` back into `develop` afterwards so the two stay
in step.

Keep feature branches short. A branch that lives for weeks stops being a feature
branch and becomes a merge conflict.

## Troubleshooting

**"ffmpeg was not found on PATH"** — install it (see Prerequisites) and open a
new terminal so the updated `PATH` is picked up.

**Falls back to x264 / says NVENC is unavailable** — NVENC needs a driver new
enough for the nvenc API your ffmpeg build was compiled against. flowclean
probes each encoder with a real one-frame encode rather than trusting
`ffmpeg -encoders`, so it detects this instead of failing mid-run. Updating the
NVIDIA driver re-enables hardware encoding. Nothing else is affected; x264 is
slower but produces the same picture.

**AI options greyed out** — PyTorch is not installed. Run `install_ai.bat`.

**PyTorch installed but "no CUDA"** — you have the CPU build. Reinstall from the
CUDA index URL shown above. Note that CUDA for PyTorch generally works on older
drivers than NVENC does, so the AI tiers can run even when hardware encoding
cannot.

**Port already in use** — `start.bat --port 9000`.

**Detection picked the wrong thing** — run `start.bat --detect yourfile.mp4` to
see what it found. Detection is weakest on very short clips of a nearly static
camera, where scene features persist as convincingly as an overlay does.

---

## Scope

This removes the *visible* mark. Flow output also carries **SynthID**, an
invisible provenance watermark that survives re-encoding and upscaling, so the
result is visually clean but not stripped of provenance. Flow's paid tiers offer
watermark-free export directly.
