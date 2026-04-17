"""Rerun-based UI to inspect VGGT-SLAM + Boxer results side-by-side.

What you get:
  • A time-scrubbable 2D image panel (left): per-keyframe RGB with the
    associated 3D OBBs projected back onto the image. If a 3D box is
    correctly localized, its wireframe hugs the object in the image;
    if BoxerNet placed it wrong, you'll see the offset immediately.
  • A 3D panel (right): fused world point cloud + fused OBBs as
    colored wireframes, plus the camera frustum moving with the time
    cursor.

Inputs:
  • Cached VGGT-SLAM output (vggt_slam_output.pt).
  • Per-frame Boxer CSV (boxer_3dbbs.csv).
  • Fused scene graph (scene_graph_fused.json) [optional].
  • iPhone frames on disk (referenced by `vggt_slam_output.image_names`).

Launch rerun's native viewer with `--spawn` (default) or connect to an
existing `rerun` instance with `--connect <host:port>`.

All logged data lives under the `world/` entity, so 2D+3D share a
common frame. Boxer's world frame (Z-down, gravity aligned) is used
throughout; the VGGT-SLAM world points are rotated with the same R_align
as the boxes.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Optional

# Homebrew Python 3.11's site.py aggressively skips .pth files (even ones
# without a leading underscore), which hides rerun-sdk's package root.
# Force-add site-packages/rerun_sdk to sys.path so `import rerun` works.
_RERUN_PATH = Path(__file__).resolve().parent.parent / ".venv" / "lib" / "python3.11" / "site-packages" / "rerun_sdk"
if _RERUN_PATH.exists() and str(_RERUN_PATH) not in sys.path:
    sys.path.insert(0, str(_RERUN_PATH))

import cv2
import numpy as np
import torch

from pipeline.gravity import estimate_gravity_rotation
from pipeline.run_vggt_slam import load_vggt_slam_output, VGGTSLAMOutput


# --------------------------------------------------------------------------
# Geometry helpers (local copies to keep ui.py independent of run_boxer)
# --------------------------------------------------------------------------


_R_FALLBACK = np.array(
    [[1, 0, 0],
     [0, 0, 1],
     [0, -1, 0]],
    dtype=np.float32,
)


def _invert_extrinsic_3x4(extr: np.ndarray) -> np.ndarray:
    R = extr[:3, :3]
    t = extr[:3, 3]
    T = np.eye(4, dtype=np.float32)
    T[:3, :3] = R.T
    T[:3, 3] = -R.T @ t
    return T


def _scale_K(K: np.ndarray, src_hw: tuple[int, int], dst_hw: tuple[int, int]) -> np.ndarray:
    src_h, src_w = src_hw
    dst_h, dst_w = dst_hw
    sx = dst_w / src_w
    sy = dst_h / src_h
    K2 = K.copy().astype(np.float32)
    K2[0, 0] *= sx
    K2[0, 2] *= sx
    K2[1, 1] *= sy
    K2[1, 2] *= sy
    return K2


def _quat_to_R(qx: float, qy: float, qz: float, qw: float) -> np.ndarray:
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


def _fuse_world_points(
    vggt_out: VGGTSLAMOutput,
    conf_percentile: float = 50.0,
    max_points: int = 300_000,
) -> np.ndarray:
    pts = vggt_out.world_points.reshape(-1, 3).numpy().astype(np.float32)
    conf = vggt_out.world_points_conf.reshape(-1).numpy().astype(np.float32)
    thresh = float(np.percentile(conf, conf_percentile))
    mask = (conf > thresh) & np.isfinite(pts).all(axis=1)
    pts = pts[mask]
    if pts.shape[0] > max_points:
        idx = np.random.default_rng(0).choice(pts.shape[0], size=max_points, replace=False)
        pts = pts[idx]
    return pts


# --------------------------------------------------------------------------
# CSV parsing
# --------------------------------------------------------------------------


def _load_boxer_csv(csv_path: Path) -> dict[int, list[dict]]:
    """Group per-frame CSV rows by time_ns. Returns {time_ns: [box_dict, ...]}."""
    out: dict[int, list[dict]] = {}
    with open(csv_path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            time_ns = int(row["time_ns"])
            box = {
                "center": np.array([
                    float(row["tx_world_object"]),
                    float(row["ty_world_object"]),
                    float(row["tz_world_object"]),
                ], dtype=np.float32),
                "size": np.array([
                    float(row["scale_x"]),
                    float(row["scale_y"]),
                    float(row["scale_z"]),
                ], dtype=np.float32),
                "quat_xyzw": np.array([
                    float(row["qx_world_object"]),
                    float(row["qy_world_object"]),
                    float(row["qz_world_object"]),
                    float(row["qw_world_object"]),
                ], dtype=np.float32),
                "label": row["name"],
                "prob": float(row["prob"]),
                "sem_id": int(row["sem_id"]),
            }
            out.setdefault(time_ns, []).append(box)
    return out


def _box_corners_world(center: np.ndarray, size: np.ndarray, R: np.ndarray) -> np.ndarray:
    """Return the 8 world-frame corners of an oriented box."""
    sx, sy, sz = size / 2.0
    corners_local = np.array(
        [
            [-sx, -sy, -sz], [+sx, -sy, -sz], [+sx, +sy, -sz], [-sx, +sy, -sz],
            [-sx, -sy, +sz], [+sx, -sy, +sz], [+sx, +sy, +sz], [-sx, +sy, +sz],
        ],
        dtype=np.float32,
    )
    return corners_local @ R.T + center[None, :]


_EDGES = [
    (0, 1), (1, 2), (2, 3), (3, 0),
    (4, 5), (5, 6), (6, 7), (7, 4),
    (0, 4), (1, 5), (2, 6), (3, 7),
]


def _project_box_to_image(
    corners_world: np.ndarray,      # [8, 3]
    T_world_camera: np.ndarray,     # [4, 4]
    K: np.ndarray,                  # [3, 3]
    img_hw: tuple[int, int],
) -> Optional[np.ndarray]:
    """Project 8 corners of an OBB into pixel space via K @ camera_from_world.

    Returns None if the box is entirely behind the camera or off-image.
    """
    T_cw = np.linalg.inv(T_world_camera)
    pts_h = np.concatenate([corners_world, np.ones((8, 1), dtype=np.float32)], axis=1)
    pts_cam = pts_h @ T_cw.T
    z = pts_cam[:, 2]
    if np.all(z <= 1e-3):
        return None  # fully behind camera
    # Avoid division by very small z
    z_safe = np.where(z > 1e-3, z, np.nan)
    uv = (pts_cam[:, :2] / z_safe[:, None]) @ K[:2, :2].T + K[:2, 2]
    valid = np.isfinite(uv).all(axis=1)
    if not valid.any():
        return None
    H, W = img_hw
    # Quick off-image reject (keep if at least one corner lands within image)
    on_image = valid & (uv[:, 0] >= -0.1 * W) & (uv[:, 0] < 1.1 * W) \
                     & (uv[:, 1] >= -0.1 * H) & (uv[:, 1] < 1.1 * H)
    if not on_image.any():
        return None
    return uv.astype(np.float32)


def _color_for_label(label: str) -> tuple[int, int, int]:
    import colorsys
    h = (hash(label) % 360) / 360.0
    r, g, b = colorsys.hsv_to_rgb(h, 0.85, 1.0)
    return int(r * 255), int(g * 255), int(b * 255)


# --------------------------------------------------------------------------
# Rerun logging
# --------------------------------------------------------------------------


def log_to_rerun(
    *,
    vggt_slam_dir: Path,
    boxer_csv: Path,
    scene_graph: Optional[Path],
    max_frames: Optional[int],
    image_resize: int,
    connect: Optional[str],
) -> None:
    import rerun as rr

    # A horizontal split (2D image ↔ 3D scene) is set via blueprint below if
    # the installed rerun version supports it; otherwise the user can
    # manually drag the panels — Rerun remembers the layout.
    try:
        import rerun.blueprint as rrb
        blueprint = rrb.Blueprint(
            rrb.Horizontal(
                rrb.Spatial2DView(origin="/world/camera/image", name="RGB + projected OBBs"),
                rrb.Spatial3DView(origin="/world", name="Fused scene + boxes"),
                column_shares=[1, 1],
            ),
            collapse_panels=True,
        ) if hasattr(rrb, "Blueprint") else None
    except Exception:
        blueprint = None

    if connect is not None:
        rr.init("box-pipeline")
        rr.connect(connect) if hasattr(rr, "connect") else rr.connect_tcp(connect)
    else:
        if blueprint is not None:
            rr.init("box-pipeline", spawn=True, default_blueprint=blueprint)
        else:
            rr.init("box-pipeline", spawn=True)

    # --- Load data ---
    print(f"[ui] loading VGGT-SLAM output from {vggt_slam_dir}")
    vggt_out = load_vggt_slam_output(Path(vggt_slam_dir))

    print(f"[ui] loading per-frame Boxer CSV from {boxer_csv}")
    per_frame_boxes = _load_boxer_csv(Path(boxer_csv))

    fused_objects: list[dict] = []
    if scene_graph is not None:
        with open(scene_graph) as f:
            fused_objects = json.load(f).get("objects", [])
        print(f"[ui] loaded {len(fused_objects)} fused objects from {scene_graph}")

    # CRITICAL: use the SAME R_align that run_boxer used when placing the
    # boxes, otherwise projected OBBs visibly drift. run_boxer persists it to
    # gravity.json next to the CSV; re-estimating from scratch here gives a
    # slightly different rotation (RANSAC is stochastic) → 2D↔3D misalignment.
    grav_path = Path(boxer_csv).parent / "gravity.json"
    if grav_path.exists():
        with open(grav_path) as f:
            grav_info = json.load(f)
        R_align_3 = np.array(grav_info["R_gravity"], dtype=np.float32)
        print(f"[ui] loaded gravity rotation from {grav_path} "
              f"(method={grav_info.get('method', 'unknown')}, "
              f"tilt_deg={grav_info.get('tilt_correction_deg', 0):.2f})")
    else:
        print(f"[ui] no gravity.json next to CSV — re-estimating (this may "
              f"disagree with the rotation used at box-writing time)")
        R_align_3, grav_info = estimate_gravity_rotation(vggt_out)
        print(f"[ui]   method={grav_info['method']} "
              f"tilt_deg={grav_info.get('tilt_correction_deg', 0):.2f}")
    R_align_4 = np.eye(4, dtype=np.float32)
    R_align_4[:3, :3] = R_align_3

    # --- Static world content (once, at time 0) ---
    rr.set_time_seconds("frame_time", 0.0)

    scene_pts = _fuse_world_points(vggt_out) @ R_align_3.T
    rr.log(
        "world/scene/points",
        rr.Points3D(
            positions=scene_pts,
            colors=np.full((scene_pts.shape[0], 3), 160, dtype=np.uint8),
            radii=0.005,
        ),
        static=True,
    )

    if fused_objects:
        centers = np.array([o["center"] for o in fused_objects], dtype=np.float32)
        sizes = np.array([o["size"] for o in fused_objects], dtype=np.float32)
        quats = np.array([
            o.get("orientation_quat", [0, 0, 0, 1]) for o in fused_objects
        ], dtype=np.float32)
        rotations = [
            rr.Quaternion(xyzw=[q[0], q[1], q[2], q[3]]) for q in quats
        ]
        colors = np.array(
            [_color_for_label(o["label"]) for o in fused_objects], dtype=np.uint8
        )
        labels = [f"{o['label']} ({o.get('confidence', 0):.2f})" for o in fused_objects]
        rr.log(
            "world/fused_boxes",
            rr.Boxes3D(
                centers=centers,
                half_sizes=sizes / 2.0,
                quaternions=rotations,
                colors=colors,
                labels=labels,
                radii=0.01,
            ),
            static=True,
        )

    # --- Per-keyframe content ---
    n_keyframes = vggt_out.extrinsic.shape[0]
    if max_frames is not None:
        n_keyframes = min(n_keyframes, max_frames)

    src_hw = vggt_out.image_size_hw   # (294, 518) on our test clip

    for i in range(n_keyframes):
        time_ns = int(vggt_out.frame_ids[i] * 1_000_000)
        rr.set_time_seconds("frame_time", time_ns / 1e9)
        rr.set_time_sequence("frame_idx", i)

        # Load + resize image.
        img_path = vggt_out.image_names[i]
        img_bgr = cv2.imread(img_path)
        if img_bgr is None:
            print(f"[ui] warning: could not read {img_path}")
            continue
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        img_resized = cv2.resize(img_rgb, (image_resize, image_resize))
        H, W = img_resized.shape[:2]

        # Pose + intrinsics, aligned into Boxer's world.
        extr = vggt_out.extrinsic[i].numpy().astype(np.float32)
        T_wc = _invert_extrinsic_3x4(extr)
        T_wc_aligned = R_align_4 @ T_wc

        K_scaled = _scale_K(
            vggt_out.intrinsic[i].numpy().astype(np.float32),
            src_hw=src_hw, dst_hw=(H, W),
        )

        # Log camera frustum + pose into 3D.
        rr.log(
            "world/camera",
            rr.Transform3D(
                translation=T_wc_aligned[:3, 3],
                mat3x3=T_wc_aligned[:3, :3],
                from_parent=False,
            ),
        )
        rr.log(
            "world/camera/image",
            rr.Pinhole(
                image_from_camera=K_scaled,
                resolution=[W, H],
            ),
        )
        rr.log("world/camera/image", rr.Image(img_resized))

        # Per-frame 3D boxes for this frame: project their 8 corners into the
        # image and log as 2D line strips. Also show them as 3D boxes
        # (non-static so they animate per frame).
        frame_boxes = per_frame_boxes.get(time_ns, [])
        if frame_boxes:
            # 3D log
            centers3 = np.array([b["center"] for b in frame_boxes], dtype=np.float32)
            sizes3 = np.array([b["size"] for b in frame_boxes], dtype=np.float32)
            rots3 = [
                rr.Quaternion(xyzw=[b["quat_xyzw"][0], b["quat_xyzw"][1],
                                    b["quat_xyzw"][2], b["quat_xyzw"][3]])
                for b in frame_boxes
            ]
            colors3 = np.array(
                [_color_for_label(b["label"]) for b in frame_boxes], dtype=np.uint8
            )
            labels3 = [f"{b['label']} ({b['prob']:.2f})" for b in frame_boxes]
            rr.log(
                "world/frame_boxes",
                rr.Boxes3D(
                    centers=centers3,
                    half_sizes=sizes3 / 2.0,
                    quaternions=rots3,
                    colors=colors3,
                    labels=labels3,
                    radii=0.008,
                ),
            )

            # Project each 3D box onto the image as a line strip over the 12 edges.
            strips_uv: list[np.ndarray] = []
            strip_colors: list[tuple[int, int, int]] = []
            strip_labels: list[str] = []
            for b in frame_boxes:
                R = _quat_to_R(*b["quat_xyzw"])
                corners_w = _box_corners_world(b["center"], b["size"], R)
                uv = _project_box_to_image(
                    corners_w, T_wc_aligned, K_scaled, img_hw=(H, W)
                )
                if uv is None:
                    continue
                for a, c in _EDGES:
                    strips_uv.append(np.stack([uv[a], uv[c]], axis=0))
                strip_colors.extend([_color_for_label(b["label"])] * 12)
                strip_labels.extend([b["label"]] * 12)
            if strips_uv:
                rr.log(
                    "world/camera/image/projected_obbs",
                    rr.LineStrips2D(
                        strips_uv,
                        colors=np.array(strip_colors, dtype=np.uint8),
                        radii=1.5,
                    ),
                )
            else:
                rr.log("world/camera/image/projected_obbs", rr.Clear(recursive=True))
        else:
            rr.log("world/camera/image/projected_obbs", rr.Clear(recursive=True))
            rr.log("world/frame_boxes", rr.Clear(recursive=True))

        if (i + 1) % 10 == 0 or i == n_keyframes - 1:
            print(f"[ui] logged {i+1}/{n_keyframes} frames")

    print("[ui] done; viewer is live — scrub the time cursor to replay frames")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--vggt-slam", type=Path, required=True,
                        help="Directory containing vggt_slam_output.pt")
    parser.add_argument("--boxer-csv", type=Path, required=True,
                        help="Per-frame boxer_3dbbs.csv from pipeline.run_boxer")
    parser.add_argument("--scene-graph", type=Path, default=None,
                        help="Optional scene_graph_fused.json (static fused boxes)")
    parser.add_argument("--max-frames", type=int, default=None)
    parser.add_argument("--image-resize", type=int, default=960,
                        help="Resize images to this square resolution; should match "
                             "Boxer's hw so the projected OBBs align with what BoxerNet saw")
    parser.add_argument("--connect", type=str, default=None,
                        help="host:port of an already-running rerun viewer "
                             "(default: spawn a new one)")
    args = parser.parse_args()

    log_to_rerun(
        vggt_slam_dir=args.vggt_slam,
        boxer_csv=args.boxer_csv,
        scene_graph=args.scene_graph,
        max_frames=args.max_frames,
        image_resize=args.image_resize,
        connect=args.connect,
    )


if __name__ == "__main__":
    main()
