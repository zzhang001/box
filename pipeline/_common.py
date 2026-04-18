"""Small constants + helpers shared across pipeline modules.

Keeping this minimal on purpose — only stuff we had in ≥3 files with
identical implementations. Math helpers (quat→R, K scaling, OBB corners)
stay inline in their callers because they're tiny and don't benefit
from a shared import.
"""

from __future__ import annotations

import numpy as np


# Fallback gravity-alignment rotation used when floor-plane RANSAC isn't
# available (or the override flag is off). Rx(-π/2) maps VGGT's OpenCV
# Y-down world to Boxer's Z-down world:
#
#     R @ [0, 1, 0] = [0, 0, -1]
#
# Imported by run_boxer.py, gravity.py, ui.py, visualize.py.
R_ALIGN_FALLBACK = np.array(
    [[1, 0, 0],
     [0, 0, 1],
     [0, -1, 0]],
    dtype=np.float32,
)


def detect_device() -> str:
    """Auto-detect the best available PyTorch compute device.

    Order: CUDA > MPS > CPU. All callers (run_boxer, run_vggt_slam,
    save_owl_bbs, config.PipelineConfig) agree on this order.
    """
    import torch
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"
