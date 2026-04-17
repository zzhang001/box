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

# Upstream extern/vggt_slam/setup.sh installs requirements + salad + VGGT_SPARK
# fork + perception-encoder + sam3. We inline-reproduce only the parts our
# pipeline actually uses and skip perception-encoder + SAM3:
#   - PE and SAM3 are only exercised behind VGGT-SLAM's --run_os flag (CLIP
#     text-query over the reconstructed map). Our VGGT-SLAM → Boxer path
#     doesn't touch them — Boxer runs its own OWLv2 2D detector.
#   - PE pulls `decord==0.6.0`, which has no macOS ARM64 wheel, so installing
#     it breaks setup on Mac mini. Skipping also saves several GB of disk.
echo "--- Installing VGGT-SLAM base requirements (torch, open3d, gtsam, viser...) ---"
pip install -r extern/vggt_slam/requirements.txt

echo "--- Installing salad (DINO-based image retrieval for loop closure) ---"
pushd extern/vggt_slam >/dev/null
mkdir -p third_party
if [ ! -d "third_party/salad" ]; then
    git clone https://github.com/Dominic101/salad.git third_party/salad
fi
pip install -e third_party/salad
popd >/dev/null

echo "--- Installing VGGT_SPARK (MIT-SPARK's VGGT fork used by VGGT-SLAM) ---"
pushd extern/vggt_slam >/dev/null
if [ ! -d "third_party/vggt" ]; then
    git clone https://github.com/MIT-SPARK/VGGT_SPARK.git third_party/vggt
fi
pip install -e third_party/vggt
popd >/dev/null

echo "--- Installing VGGT-SLAM package ---"
pip install -e extern/vggt_slam

# salad/eval.py's load_model reads dino_salad.ckpt from the torch hub cache
# and does NOT auto-download. Fetch it up front so first-run SLAM isn't
# blocked by a missing checkpoint.
HUB_CACHE="${TORCH_HOME:-$HOME/.cache/torch}/hub/checkpoints"
mkdir -p "$HUB_CACHE"
if [ ! -f "$HUB_CACHE/dino_salad.ckpt" ]; then
    echo "--- Downloading dino_salad.ckpt (~335 MB) ---"
    curl -L -o "$HUB_CACHE/dino_salad.ckpt" \
        "https://github.com/serizba/salad/releases/download/v1.0.0/dino_salad.ckpt"
fi

# Boxer's pyproject.toml ships only lint config, not packaging metadata — it
# expects to be run from source. We put extern/boxer on sys.path inside
# pipeline/run_boxer.py, so no pip install needed here.

echo "--- Installing pipeline ---"
pip install -e .

echo ""
echo "=== Setup complete ==="
echo "Activate with: source .venv/bin/activate"
echo "Run with:      python -m pipeline.run --video YOUR_VIDEO.mov --output output/"
