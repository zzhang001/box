"""Visualize VGGT-SLAM output + optional fused Boxer boxes as a single PLY.

Loads a saved `vggt_slam_output.pt`, confidence-filters the dense per-frame
point cloud, fuses across frames, and writes a colored .ply you can open in
MeshLab / CloudCompare / 3dviewer.net.

Optional: pass `--scene-graph <scene_graph_fused.json>` to also render each
fused object's 3D bounding box as a dense wireframe (points along the 12
edges), colored per-label.
"""

import argparse
import colorsys
import json
from pathlib import Path
from typing import Optional

import numpy as np
import torch

from pipeline._common import R_ALIGN_FALLBACK as _R_FALLBACK
from pipeline.run_vggt_slam import load_vggt_slam_output


def fuse_world_points(
    world_points: torch.Tensor,
    conf: torch.Tensor,
    conf_percentile: float = 50.0,
    max_points: int = 500_000,
) -> np.ndarray:
    """Fuse per-frame world points into one (N, 3) cloud, confidence-filtered."""
    pts = world_points.reshape(-1, 3).numpy().astype(np.float32)
    c = conf.reshape(-1).numpy().astype(np.float32)
    thresh = np.percentile(c, conf_percentile)
    mask = c > thresh
    pts = pts[mask]
    finite = np.isfinite(pts).all(axis=1)
    pts = pts[finite]
    if pts.shape[0] > max_points:
        # Random (not linspace) — the filtered cloud may be ordered by
        # confidence or spatial locality, so linspace would bias sampling.
        rng = np.random.default_rng(0)
        idx = rng.choice(pts.shape[0], size=max_points, replace=False)
        pts = pts[idx]
    return pts


def quat_to_R(qx: float, qy: float, qz: float, qw: float) -> np.ndarray:
    """(qx, qy, qz, qw) → 3×3 rotation matrix."""
    n = (qx * qx + qy * qy + qz * qz + qw * qw) ** 0.5
    qx, qy, qz, qw = qx / n, qy / n, qz / n, qw / n
    return np.array(
        [
            [1 - 2 * (qy * qy + qz * qz),     2 * (qx * qy - qz * qw),     2 * (qx * qz + qy * qw)],
            [    2 * (qx * qy + qz * qw), 1 - 2 * (qx * qx + qz * qz),     2 * (qy * qz - qx * qw)],
            [    2 * (qx * qz - qy * qw),     2 * (qy * qz + qx * qw), 1 - 2 * (qx * qx + qy * qy)],
        ],
        dtype=np.float32,
    )


def box_edge_points(
    center: np.ndarray,
    size: np.ndarray,
    R: np.ndarray,
    samples_per_edge: int = 40,
) -> np.ndarray:
    """Generate (N, 3) points along the 12 edges of an oriented box."""
    sx, sy, sz = size / 2.0
    corners_local = np.array(
        [
            [-sx, -sy, -sz], [+sx, -sy, -sz], [+sx, +sy, -sz], [-sx, +sy, -sz],  # bottom
            [-sx, -sy, +sz], [+sx, -sy, +sz], [+sx, +sy, +sz], [-sx, +sy, +sz],  # top
        ],
        dtype=np.float32,
    )
    corners = corners_local @ R.T + center[None, :]

    edges = [
        (0, 1), (1, 2), (2, 3), (3, 0),  # bottom
        (4, 5), (5, 6), (6, 7), (7, 4),  # top
        (0, 4), (1, 5), (2, 6), (3, 7),  # verticals
    ]
    pts = []
    for a, b in edges:
        ts = np.linspace(0, 1, samples_per_edge, dtype=np.float32)
        seg = corners[a][None, :] * (1 - ts[:, None]) + corners[b][None, :] * ts[:, None]
        pts.append(seg)
    return np.concatenate(pts, axis=0)


def _color_for_label(label: str) -> tuple[int, int, int]:
    """Deterministic bright RGB from a label string."""
    h = (hash(label) % 360) / 360.0
    r, g, b = colorsys.hsv_to_rgb(h, 0.85, 1.0)
    return int(r * 255), int(g * 255), int(b * 255)


def write_colored_ply(
    xyz: np.ndarray,       # [N, 3]
    rgb: np.ndarray,        # [N, 3] uint8
    path: Path,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    N = xyz.shape[0]
    with open(path, "w") as f:
        f.write(
            "ply\n"
            "format ascii 1.0\n"
            f"element vertex {N}\n"
            "property float x\n"
            "property float y\n"
            "property float z\n"
            "property uchar red\n"
            "property uchar green\n"
            "property uchar blue\n"
            "end_header\n"
        )
        for p, c in zip(xyz, rgb):
            f.write(f"{p[0]:.5f} {p[1]:.5f} {p[2]:.5f} {int(c[0])} {int(c[1])} {int(c[2])}\n")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="VGGT-SLAM output (+ optional fused scene graph) → single PLY",
    )
    parser.add_argument("--input", type=Path, required=True,
                        help="Directory containing vggt_slam_output.pt")
    parser.add_argument("--output", type=Path, default=None,
                        help="Output .ply path (default: <input>/fused.ply)")
    parser.add_argument("--scene-graph", type=Path, default=None,
                        help="Optional scene_graph_fused.json to overlay as wireframes")
    parser.add_argument("--conf-percentile", type=float, default=50.0)
    parser.add_argument("--max-points", type=int, default=500_000)
    parser.add_argument("--edge-samples", type=int, default=40,
                        help="Points rendered along each box edge (higher = brighter lines)")
    parser.add_argument("--gravity-rotate", action="store_true",
                        help="Apply the fallback Rx(-π/2) to the cloud so it "
                             "lands in Boxer's Z-down frame; use when also "
                             "rendering fused boxes from scene_graph_fused.json")
    args = parser.parse_args()

    out = load_vggt_slam_output(args.input)
    print(f"Loaded {out.world_points.shape[0]} keyframes, "
          f"{out.world_points.shape[1]}×{out.world_points.shape[2]} each")

    pts = fuse_world_points(
        out.world_points, out.world_points_conf,
        conf_percentile=args.conf_percentile,
        max_points=args.max_points,
    )
    if args.gravity_rotate or args.scene_graph is not None:
        pts = pts @ _R_FALLBACK.T
    # Neutral gray for scene points.
    scene_rgb = np.full((pts.shape[0], 3), 160, dtype=np.uint8)

    box_xyz_list: list[np.ndarray] = []
    box_rgb_list: list[np.ndarray] = []
    if args.scene_graph is not None:
        with open(args.scene_graph) as f:
            sg = json.load(f)
        print(f"Loaded {len(sg['objects'])} fused objects from {args.scene_graph}")
        for obj in sg["objects"]:
            center = np.array(obj["center"], dtype=np.float32)
            size = np.array(obj["size"], dtype=np.float32)
            q = obj.get("orientation_quat")
            if q is not None and len(q) == 4:
                R = quat_to_R(*q)
            else:
                # Fallback: yaw-only rotation about Z.
                yaw = float(obj.get("yaw", 0.0))
                c, s = np.cos(yaw), np.sin(yaw)
                R = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]], dtype=np.float32)
            edge_pts = box_edge_points(center, size, R, samples_per_edge=args.edge_samples)
            color = _color_for_label(obj["label"])
            rgb = np.tile(np.array(color, dtype=np.uint8), (edge_pts.shape[0], 1))
            box_xyz_list.append(edge_pts)
            box_rgb_list.append(rgb)

    if box_xyz_list:
        all_xyz = np.concatenate([pts, *box_xyz_list], axis=0)
        all_rgb = np.concatenate([scene_rgb, *box_rgb_list], axis=0)
    else:
        all_xyz, all_rgb = pts, scene_rgb

    out_path = args.output or (args.input / "fused.ply")
    write_colored_ply(all_xyz, all_rgb, out_path)
    print(f"Wrote {all_xyz.shape[0]} points ({pts.shape[0]} scene + "
          f"{all_xyz.shape[0] - pts.shape[0]} box-edge) → {out_path}")
    print(f"bbox min: {all_xyz.min(axis=0)}")
    print(f"bbox max: {all_xyz.max(axis=0)}")


if __name__ == "__main__":
    main()
