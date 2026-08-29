#!/usr/bin/env bash
# Launch flowclean on Linux or macOS. Windows users: use start.bat instead.
set -euo pipefail
cd "$(dirname "$0")"

PY=.venv/bin/python
if [ ! -x "$PY" ]; then
  echo "Creating virtual environment..."
  python3 -m venv .venv
  "$PY" -m pip install -q --upgrade pip
  "$PY" -m pip install -q -r requirements.txt
fi

if ! command -v ffmpeg >/dev/null 2>&1; then
  echo "ffmpeg was not found on PATH. Install it and try again." >&2
  exit 1
fi

exec "$PY" -m flowclean "$@"
