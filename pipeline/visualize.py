"""Visualize VGGT-SLAM output: per-frame world points → fused PLY point cloud.

Loads a saved `vggt_slam_output.pt`, confidence-filters the dense per-frame
point cloud, fuses across frames, and writes an .ply you can open in
macOS Preview, MeshLab, CloudCompare, or any 3D viewer.
"""

import argparse
from pathlib import Path

import numpy as np
import torch

from pipeline.run_vggt_slam import load_vggt_slam_output


def fuse_world_points(
    world_points: torch.Tensor,
    conf: torch.Tensor,
    conf_percentile: float = 50.0,
    max_points: int = 500_000,
) -> np.ndarray:
    """Fuse per-frame world points into one (N, 3) cloud, confidence-filtered.

    Args:
        world_points: [S, H, W, 3] per-frame world-space points (SL(4) world).
        conf: [S, H, W] per-pixel confidence.
        conf_percentile: drop the lowest `conf_percentile`% of confidences.
        max_points: uniformly subsample to this cap so viewers stay snappy.
    """
    pts = world_points.reshape(-1, 3).numpy().astype(np.float32)
    c = conf.reshape(-1).numpy().astype(np.float32)
    thresh = np.percentile(c, conf_percentile)
    mask = c > thresh
    pts = pts[mask]
    finite = np.isfinite(pts).all(axis=1)
    pts = pts[finite]
    if pts.shape[0] > max_points:
        idx = np.linspace(0, pts.shape[0] - 1, max_points).astype(np.int64)
        pts = pts[idx]
    return pts


def write_ply(points: np.ndarray, path: Path) -> None:
    """Minimal ASCII PLY writer — no colors, enough for structural inspection."""
    path.parent.mkdir(parents=True, exist_ok=True)
    N = points.shape[0]
    header = (
        "ply\n"
        "format ascii 1.0\n"
        f"element vertex {N}\n"
        "property float x\n"
        "property float y\n"
        "property float z\n"
        "end_header\n"
    )
    with open(path, "w") as f:
        f.write(header)
        for p in points:
            f.write(f"{p[0]:.6f} {p[1]:.6f} {p[2]:.6f}\n")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="VGGT-SLAM output → fused point cloud PLY",
    )
    parser.add_argument("--input", type=Path, required=True,
                        help="Directory containing vggt_slam_output.pt")
    parser.add_argument("--output", type=Path, default=None,
                        help="Output .ply path (default: <input>/fused.ply)")
    parser.add_argument("--conf-percentile", type=float, default=50.0,
                        help="Drop the lowest X%% of confidences (default 50)")
    parser.add_argument("--max-points", type=int, default=500_000,
                        help="Max points in the output PLY")
    args = parser.parse_args()

    out = load_vggt_slam_output(args.input)
    print(f"Loaded {out.world_points.shape[0]} keyframes, "
          f"{out.world_points.shape[1]}×{out.world_points.shape[2]} each")

    pts = fuse_world_points(
        out.world_points, out.world_points_conf,
        conf_percentile=args.conf_percentile,
        max_points=args.max_points,
    )
    out_path = args.output or (args.input / "fused.ply")
    write_ply(pts, out_path)
    print(f"Wrote {pts.shape[0]} points → {out_path}")

    # Print a bounding-box summary so you can sanity-check scale.
    print(f"bbox min: {pts.min(axis=0)}")
    print(f"bbox max: {pts.max(axis=0)}")
    print(f"bbox extent: {pts.max(axis=0) - pts.min(axis=0)} (world units)")


if __name__ == "__main__":
    main()
