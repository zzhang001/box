#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

echo "=== VGGT-SLAM + Boxer Pipeline Setup ==="

# Python 3.11 matches VGGT-SLAM upstream; 3.12 usually works too.
PY=""
for candidate in python3.11 python3.12; do
    if command -v "$candidate" &>/dev/null; then
        PY="$candidate"
        break
    fi
done
if [ -z "$PY" ]; then
    echo "ERROR: Python 3.11 (preferred) or 3.12 required."
    echo "  Install via: brew install python@3.11"
    exit 1
fi
echo "Using: $PY ($(command -v $PY))"

if ! command -v ffmpeg &>/dev/null; then
    echo "WARNING: ffmpeg not found. Install via: brew install ffmpeg"
fi

echo "--- Initializing submodules ---"
git submodule update --init --recursive

echo "--- Creating virtual environment ---"
$PY -m venv .venv
# shellcheck disable=SC1091
source .venv/bin/activate

pip install --upgrade pip setuptools wheel

# VGGT-SLAM's setup.sh installs requirements + salad + VGGT_SPARK fork +
# perception-encoder + sam3 into extern/vggt_slam/third_party. We delegate
# instead of duplicating: keeps our setup in sync when upstream changes.
# (Perception-encoder and SAM3 are only used behind --run_os; they install
# unconditionally. Safe to prune later if disk-constrained.)
echo "--- Installing VGGT-SLAM deps (salad + VGGT_SPARK + gtsam + SAM3 + PE) ---"
pushd extern/vggt_slam >/dev/null
chmod +x setup.sh
./setup.sh
popd >/dev/null

echo "--- Installing Boxer ---"
pushd extern/boxer >/dev/null
if [ -f "pyproject.toml" ]; then
    pip install -e .
elif [ -f "requirements.txt" ]; then
    pip install -r requirements.txt
fi
popd >/dev/null

echo "--- Installing pipeline ---"
pip install -e .

echo ""
echo "=== Setup complete ==="
echo "Activate with: source .venv/bin/activate"
echo "Run with:      python -m pipeline.run --video YOUR_VIDEO.mov --output output/"
