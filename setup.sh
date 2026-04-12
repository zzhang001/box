#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

echo "=== VGGT + Boxer Pipeline Setup ==="

# Check Python 3.12
if ! command -v python3.12 &>/dev/null; then
    echo "ERROR: Python 3.12 is required. Install via: brew install python@3.12"
    exit 1
fi

# Check ffmpeg
if ! command -v ffmpeg &>/dev/null; then
    echo "WARNING: ffmpeg not found. Install via: brew install ffmpeg"
fi

# Init submodules
echo "--- Initializing submodules ---"
git submodule update --init --recursive

# Create venv
echo "--- Creating virtual environment ---"
python3.12 -m venv .venv
source .venv/bin/activate

# Upgrade pip
pip install --upgrade pip setuptools wheel

# Install VGGT
echo "--- Installing VGGT ---"
pip install -e extern/vggt

# Install Boxer dependencies
echo "--- Installing Boxer ---"
cd extern/boxer
if [ -f "pyproject.toml" ]; then
    pip install -e .
elif [ -f "requirements.txt" ]; then
    pip install -r requirements.txt
fi
cd "$SCRIPT_DIR"

# Install this pipeline
echo "--- Installing pipeline ---"
pip install -e .

echo ""
echo "=== Setup complete ==="
echo "Activate with: source .venv/bin/activate"
echo "Run with:      python -m pipeline.run --video YOUR_VIDEO.mov --output output/"
