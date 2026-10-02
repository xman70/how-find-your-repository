#!/usr/bin/env bash
# Linux / macOS equivalent of START.bat
set -euo pipefail
cd "$(dirname "$0")"
echo "AI Text Analysis - Privacy mode: Local processing"
PY=""
for c in python3.12 python3.11 python3.10 python3.13 python3; do
  if command -v "$c" >/dev/null 2>&1 && "$c" -c 'import sys; sys.exit(0 if sys.version_info >= (3, 10) else 1)'; then PY="$c"; break; fi
done
[ -z "$PY" ] && { echo "[ERROR] Python 3.10+ not found"; exit 1; }
echo "[1/5] Python: $($PY --version)"
[ -x .venv/bin/python ] || { echo "[2/5] Creating .venv"; "$PY" -m venv .venv; }
if ! cmp -s requirements.txt .venv/requirements.installed || [ "${1:-}" = "full" ]; then
  echo "[3/5] Installing dependencies"
  .venv/bin/python -m pip install --upgrade pip >/dev/null
  .venv/bin/python -m pip install -r requirements.txt
  .venv/bin/python -m pip install -e . --no-deps >/dev/null
  [ "${1:-}" = "full" ] && .venv/bin/python -m pip install -r requirements-optional.txt
  cp requirements.txt .venv/requirements.installed
else
  echo "[3/5] Dependencies already installed"
fi
echo "[4/5] Verifying installation"
.venv/bin/python -m aidetect.verify
echo "[5/5] Starting the application"
exec .venv/bin/python app.py
