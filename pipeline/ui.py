"""Rerun-based UI for VGGT-SLAM + Boxer debug, at NATIVE image resolution.

Left pane: the original iPhone frame (1920×1080, no stretching) with
  • OWL's 2D detections (from pipeline.save_owl_bbs.json) drawn in-color.
  • The 3D OBBs projected back onto the image (same frame).
  If pose + K + 3D position are all correct, each projected OBB wireframe
  hugs the object and overlaps the OWL 2D box that prompted it.

Right pane: the global 3D scene.
  • Fused VGGT-SLAM point cloud (gray).
  • Fused 3D OBBs (colored wireframes, labeled).
  • Camera trajectory + current frustum synced to the time cursor.

All 2D coordinates are rescaled from Boxer's 960×960 working space back to
the native image resolution before drawing, so the image is displayed at
its true aspect ratio. The K we use for projection is also rescaled to
native, so projection lines up pixel-for-pixel.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Optional

# Homebrew Python 3.11 site.py skips many .pth files. Force-add rerun_sdk.
_RERUN_PATH = Path(__file__).resolve().parent.parent / ".venv" / "lib" / "python3.11" / "site-packages" / "rerun_sdk"
if _RERUN_PATH.exists() and str(_RERUN_PATH) not in sys.path:
    sys.path.insert(0, str(_RERUN_PATH))

import cv2
import numpy as np
import torch

from pipeline.run_vggt_slam import load_vggt_slam_output, VGGTSLAMOutput


# --------------------------------------------------------------------------
# Geometry helpers
# --------------------------------------------------------------------------


from pipeline._common import R_ALIGN_FALLBACK as _R_FALLBACK  # noqa: E402


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


def _load_boxer_csv(csv_path: Path) -> dict[int, list[dict]]:
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
    corners_world: np.ndarray,
    T_world_camera: np.ndarray,
    K: np.ndarray,
    img_hw: tuple[int, int],
) -> Optional[np.ndarray]:
    T_cw = np.linalg.inv(T_world_camera)
    pts_h = np.concatenate([corners_world, np.ones((8, 1), dtype=np.float32)], axis=1)
    pts_cam = pts_h @ T_cw.T
    z = pts_cam[:, 2]
    if np.all(z <= 1e-3):
        return None
    z_safe = np.where(z > 1e-3, z, np.nan)
    uv = (pts_cam[:, :2] / z_safe[:, None]) @ K[:2, :2].T + K[:2, 2]
    H, W = img_hw
    valid = np.isfinite(uv).all(axis=1)
    on_image = valid & (uv[:, 0] >= -0.2 * W) & (uv[:, 0] < 1.2 * W) \
                     & (uv[:, 1] >= -0.2 * H) & (uv[:, 1] < 1.2 * H)
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
    owl_json: Optional[Path],
    scene_graph: Optional[Path],
    max_frames: Optional[int],
    save_rrd: Optional[Path],
) -> None:
    import rerun as rr

    try:
        import rerun.blueprint as rrb
        blueprint = rrb.Blueprint(
            rrb.Horizontal(
                rrb.Spatial2DView(origin="/world/camera/image", name="Frame + 2D/3D overlay"),
                rrb.Spatial3DView(origin="/world", name="Global scene"),
                column_shares=[1, 1],
            ),
            collapse_panels=True,
        ) if hasattr(rrb, "Blueprint") else None
    except Exception:
        blueprint = None

    if save_rrd is not None:
        rr.init("box-pipeline")
        if blueprint is not None:
            rr.save(str(save_rrd), default_blueprint=blueprint)
        else:
            rr.save(str(save_rrd))
    else:
        if blueprint is not None:
            rr.init("box-pipeline", spawn=True, default_blueprint=blueprint)
        else:
            rr.init("box-pipeline", spawn=True)

    print(f"[ui] loading VGGT-SLAM output from {vggt_slam_dir}")
    vggt_out = load_vggt_slam_output(Path(vggt_slam_dir))

    print(f"[ui] loading per-frame Boxer CSV from {boxer_csv}")
    per_frame_boxes = _load_boxer_csv(Path(boxer_csv))

    # Gravity: use the same R_align that run_boxer used (persisted in gravity.json).
    grav_path = Path(boxer_csv).parent / "gravity.json"
    if grav_path.exists():
        with open(grav_path) as f:
            grav_info = json.load(f)
        R_align_3 = np.array(grav_info["R_gravity"], dtype=np.float32)
        print(f"[ui] gravity from {grav_path.name}: {grav_info.get('method')}")
    else:
        print(f"[ui] no gravity.json — using fallback Rx(-π/2)")
        R_align_3 = _R_FALLBACK.copy()
    R_align_4 = np.eye(4, dtype=np.float32)
    R_align_4[:3, :3] = R_align_3

    # OWL 2D detections: {time_ns: [{bb2d_xyxy, score, label}, ...]}, coords in
    # `owl_image_resize × owl_image_resize` space. Newer runs also include
    # `pad_info_by_time_ns` so we can undo the pad-to-square transform back
    # to native image coords for display.
    owl_by_time: dict[int, list[dict]] = {}
    owl_pad_info: dict[int, dict] = {}
    owl_image_resize = 960
    owl_padded = False
    if owl_json is not None and Path(owl_json).exists():
        with open(owl_json) as f:
            owl_data = json.load(f)
        owl_image_resize = int(owl_data.get("image_resize", 960))
        owl_padded = bool(owl_data.get("image_pad_to_square", False))
        for k, v in owl_data.get("detections_by_time_ns", {}).items():
            owl_by_time[int(k)] = v
        for k, v in owl_data.get("pad_info_by_time_ns", {}).items():
            owl_pad_info[int(k)] = v
        print(f"[ui] loaded OWL 2D dets for {len(owl_by_time)} frames "
              f"(coords in {owl_image_resize}² {'(pad-to-square)' if owl_padded else '(stretched)'})")

    fused_objects: list[dict] = []
    if scene_graph is not None and Path(scene_graph).exists():
        with open(scene_graph) as f:
            fused_objects = json.load(f).get("objects", [])
        print(f"[ui] loaded {len(fused_objects)} fused objects")

    # --- Static 3D content ---
    rr.set_time("frame_time", duration=0.0)

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
        rotations = [rr.Quaternion(xyzw=[q[0], q[1], q[2], q[3]]) for q in quats]
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

    # --- Per-frame content ---
    n_keyframes = vggt_out.extrinsic.shape[0]
    if max_frames is not None:
        n_keyframes = min(n_keyframes, max_frames)

    src_hw = vggt_out.image_size_hw  # (294, 518) for our iPhone clip

    for i in range(n_keyframes):
        time_ns = int(vggt_out.frame_ids[i] * 1_000_000)
        rr.set_time("frame_time", duration=time_ns / 1e9)
        rr.set_time("frame_idx", sequence=i)

        # Load image at NATIVE resolution — no stretching.
        img_path = vggt_out.image_names[i]
        img_bgr = cv2.imread(img_path)
        if img_bgr is None:
            continue
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        orig_H, orig_W = img_rgb.shape[:2]

        # Pose + intrinsics. K is scaled from VGGT's internal (294, 518) to the
        # native (orig_H, orig_W) so projection happens in native pixel space.
        extr = vggt_out.extrinsic[i].numpy().astype(np.float32)
        T_wc = _invert_extrinsic_3x4(extr)
        T_wc_aligned = R_align_4 @ T_wc

        K_native = _scale_K(
            vggt_out.intrinsic[i].numpy().astype(np.float32),
            src_hw=src_hw, dst_hw=(orig_H, orig_W),
        )

        rr.log(
            "world/camera",
            rr.Transform3D(
                translation=T_wc_aligned[:3, 3],
                mat3x3=T_wc_aligned[:3, :3],
                relation=rr.TransformRelation.ParentFromChild,
            ),
        )
        rr.log(
            "world/camera/image",
            rr.Pinhole(
                image_from_camera=K_native,
                resolution=[orig_W, orig_H],
            ),
        )
        rr.log("world/camera/image", rr.Image(img_rgb))

        # Overlay: OWL's 2D boxes (rescaled back to native coords).
        owl_dets = owl_by_time.get(time_ns, [])
        if owl_dets:
            owl_bbs = np.array(
                [d["bb2d_xyxy"] for d in owl_dets], dtype=np.float32
            )
            if owl_padded:
                # bb in [image_resize × image_resize] (pad-to-square). Undo:
                #   1) scale from image_resize to the padded-square side
                #   2) subtract pad_left/pad_top → native pixels
                pad = owl_pad_info.get(time_ns, {})
                side = pad.get("side", max(orig_H, orig_W))
                pad_left = pad.get("pad_left", (side - orig_W) // 2)
                pad_top = pad.get("pad_top", (side - orig_H) // 2)
                s = side / owl_image_resize
                owl_bbs[:, 0] = owl_bbs[:, 0] * s - pad_left
                owl_bbs[:, 2] = owl_bbs[:, 2] * s - pad_left
                owl_bbs[:, 1] = owl_bbs[:, 1] * s - pad_top
                owl_bbs[:, 3] = owl_bbs[:, 3] * s - pad_top
            else:
                # Legacy: anisotropic stretch from image_resize → native.
                sx = orig_W / owl_image_resize
                sy = orig_H / owl_image_resize
                owl_bbs[:, 0] *= sx; owl_bbs[:, 2] *= sx
                owl_bbs[:, 1] *= sy; owl_bbs[:, 3] *= sy
            owl_colors = np.array(
                [_color_for_label(d["label"]) for d in owl_dets], dtype=np.uint8
            )
            owl_labels = [f"{d['label']} {d['score']:.2f}" for d in owl_dets]
            rr.log(
                "world/camera/image/owl_bbs",
                rr.Boxes2D(
                    array=owl_bbs,
                    array_format=rr.Box2DFormat.XYXY,
                    colors=owl_colors,
                    labels=owl_labels,
                    radii=2.0,
                ),
            )
        else:
            rr.log("world/camera/image/owl_bbs", rr.Clear(recursive=True))

        # Per-frame 3D boxes + their projections onto the native image.
        frame_boxes = per_frame_boxes.get(time_ns, [])
        if frame_boxes:
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

            strips_uv: list[np.ndarray] = []
            strip_colors: list[tuple[int, int, int]] = []
            for b in frame_boxes:
                R = _quat_to_R(*b["quat_xyzw"])
                corners_w = _box_corners_world(b["center"], b["size"], R)
                uv = _project_box_to_image(
                    corners_w, T_wc_aligned, K_native, img_hw=(orig_H, orig_W)
                )
                if uv is None:
                    continue
                for a, c in _EDGES:
                    strips_uv.append(np.stack([uv[a], uv[c]], axis=0))
                strip_colors.extend([_color_for_label(b["label"])] * 12)
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

        if (i + 1) % 25 == 0 or i == n_keyframes - 1:
            print(f"[ui] logged {i+1}/{n_keyframes} frames")

    print("[ui] done")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--vggt-slam", type=Path, required=True)
    parser.add_argument("--boxer-csv", type=Path, required=True)
    parser.add_argument("--owl-json", type=Path, default=None,
                        help="pipeline.save_owl_bbs JSON of per-frame 2D detections")
    parser.add_argument("--scene-graph", type=Path, default=None)
    parser.add_argument("--max-frames", type=int, default=None)
    parser.add_argument("--save-rrd", type=Path, default=None,
                        help="Save to .rrd instead of spawning; open with `rerun <file>`")
    args = parser.parse_args()

    log_to_rerun(
        vggt_slam_dir=args.vggt_slam,
        boxer_csv=args.boxer_csv,
        owl_json=args.owl_json,
        scene_graph=args.scene_graph,
        max_frames=args.max_frames,
        save_rrd=args.save_rrd,
    )

    if args.save_rrd is not None:
        print(f"[ui] open with: rerun {args.save_rrd}")


if __name__ == "__main__":
    main()
