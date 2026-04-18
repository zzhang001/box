"""Extract the real pinhole intrinsic matrix from an iPhone .MOV's EXIF.

iPhone videos embed the physical lens focal length AND the 35mm-equivalent
focal length in the Apple-specific metadata tracks. From the 35mm equivalent
we can compute a pixel-accurate K for the native image resolution without
any calibration or learned estimation.

Why the 35mm equivalent? It encodes `focal_length_in_pixels` once the native
image dimensions are known:
    fx = (W / 2) / tan(HFOV / 2)
    HFOV = 2 * atan(W35 / (2 * f35))    where W35=36mm and f35 is the equiv.

Example (iPhone Air rear camera):
    f35 = 26 mm  → HFOV = 2*atan(18/26) = 69.4°
    At 1920×1080 → fx = fy = 960/tan(34.7°) ≈ 1386 px, cx=960, cy=540

Usage:
    from pipeline.iphone_k import extract_K_from_mov
    K, native_hw, meta = extract_K_from_mov("video.MOV")
"""

from __future__ import annotations

import json
import math
import shutil
import subprocess
from pathlib import Path
from typing import Optional

import numpy as np


def _run_exiftool(mov_path: Path) -> dict:
    """Parse exiftool JSON output — returns the first (and only) entry."""
    if shutil.which("exiftool") is None:
        raise RuntimeError(
            "exiftool not found on PATH. Install with: brew install exiftool"
        )
    proc = subprocess.run(
        ["exiftool", "-json", "-G1", str(mov_path)],
        capture_output=True, text=True, check=True,
    )
    data = json.loads(proc.stdout)
    if not data:
        raise RuntimeError(f"exiftool returned empty JSON for {mov_path}")
    return data[0]


def _run_ffprobe(mov_path: Path) -> tuple[int, int]:
    """Return (height, width) of the first video stream."""
    proc = subprocess.run(
        [
            "ffprobe", "-v", "error",
            "-select_streams", "v:0",
            "-show_entries", "stream=width,height",
            "-of", "json",
            str(mov_path),
        ],
        capture_output=True, text=True, check=True,
    )
    info = json.loads(proc.stdout)
    s = info["streams"][0]
    return int(s["height"]), int(s["width"])


def extract_K_from_mov(
    mov_path: Path,
    *,
    sensor_full_frame_width_mm: float = 36.0,
) -> tuple[np.ndarray, tuple[int, int], dict]:
    """Parse an iPhone .MOV and return (K, (H, W), metadata_dict).

    Args:
        mov_path: path to the .MOV file.
        sensor_full_frame_width_mm: reference sensor width used to interpret
            the 35mm-equivalent focal length. Standard is 36 mm (full-frame
            horizontal). Don't change unless you know why.

    Returns:
        K           : (3, 3) float32 pinhole matrix at native video resolution
        (H, W)      : native video (height, width)
        metadata    : dict with the raw EXIF fields that drove the computation
                      (lens_model, focal_length_35mm, hfov_deg, ...)

    Raises:
        RuntimeError if exiftool / ffprobe isn't installed, or if the file has
        no 35mm-equivalent focal length tag (some apps strip EXIF).
    """
    mov_path = Path(mov_path)
    exif = _run_exiftool(mov_path)

    # Try multiple tag names; Apple's Keys atom uses 'FocalLengthIn35mmFormat',
    # other tools use 'FocalLengthIn35mmFilm'.
    f35_val = None
    for key in [
        "VideoKeys:FocalLengthIn35mmFormat",
        "Keys:FocalLengthIn35mmFormat",
        "Composite:FocalLength35efl",
        "ExifIFD:FocalLengthIn35mmFormat",
        "QuickTime:FocalLengthIn35mmFormat",
    ]:
        if key in exif:
            f35_val = exif[key]
            break
    if f35_val is None:
        # Last resort: search by suffix.
        for k, v in exif.items():
            if k.endswith("FocalLengthIn35mmFormat") or k.endswith("FocalLength35efl"):
                f35_val = v
                break
    if f35_val is None:
        raise RuntimeError(
            f"Could not find 35mm-equivalent focal length in EXIF of {mov_path}. "
            f"Available keys: {sorted(exif.keys())[:20]}..."
        )

    # Value can be "26" or "26 mm" depending on the tool.
    if isinstance(f35_val, str):
        f35 = float(f35_val.strip().split()[0])
    else:
        f35 = float(f35_val)

    H, W = _run_ffprobe(mov_path)
    hfov_rad = 2.0 * math.atan(sensor_full_frame_width_mm / (2.0 * f35))
    fx = (W / 2.0) / math.tan(hfov_rad / 2.0)
    fy = fx  # iPhone pixels are square
    cx = W / 2.0
    cy = H / 2.0

    K = np.array(
        [
            [fx, 0.0, cx],
            [0.0, fy, cy],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float32,
    )

    meta = {
        "mov_path": str(mov_path),
        "native_hw": [H, W],
        "focal_length_35mm": f35,
        "hfov_deg": math.degrees(hfov_rad),
        "fx_px": fx,
        "fy_px": fy,
        "cx_px": cx,
        "cy_px": cy,
        "lens_model": exif.get("VideoKeys:LensModel") or exif.get("Composite:LensID"),
    }
    return K, (H, W), meta


def main() -> None:
    import argparse
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("mov", type=Path, help=".MOV file to inspect")
    args = parser.parse_args()
    K, hw, meta = extract_K_from_mov(args.mov)
    print("K (at native resolution):")
    print(K)
    print(f"\nNative (H, W) = {hw}")
    print(f"HFOV          = {meta['hfov_deg']:.2f}°")
    print(f"lens_model    = {meta['lens_model']}")
    print(f"f35mm-equiv   = {meta['focal_length_35mm']} mm")


if __name__ == "__main__":
    main()
