"""Per-box diagnostics for Boxer 3D detections — three orthogonal metrics.

Given per-frame OBBs (boxer_3dbbs.csv), the VGGT-SLAM scene cloud, and the
OWL 2D detections (owl_2dbbs.json), score each 3D box on three metrics:

  A. dist_to_cloud  (nearest VGGT point to box center)
  B. iou2d          (3D box → 2D projection, IoU with the OWL prompt bbox)
  C. depth_consist  (camera-frame Z of box corners vs. VGGT depth distribution
                     inside the OWL 2D bbox footprint)

All three are purely geometric — no ground-truth labels needed. They each
catch different failure modes:

  * A high means box floats in empty space (common when SL(4) scale or K
    estimation is off enough that BoxerNet put the box outside the scene).
  * Low B means K/pose/3D center aren't self-consistent (the 3D box
    doesn't reproject to where OWL saw the 2D bbox).
  * Large C means box is at wrong depth despite right angular direction.

Combining thresholds lets us filter "keep obvious good boxes, drop clearly
bad ones" before fusion, producing a cleaner scene graph.

Output:
  - `diagnostic_by_box.csv` — one row per per-frame OBB with all three scores.
  - `diagnostic_summary.json` — aggregate distribution stats.
  - optional `boxer_3dbbs_filtered.csv` — the input CSV with bad boxes removed.

CLI:
  python -m pipeline.diagnostic \\
      --vggt-slam test/out_full \\
      --boxer-csv test/boxer_out_full/boxer_3dbbs.csv \\
      --owl-json test/boxer_out_full/owl_2dbbs.json \\
      [--override-k-from-mov test/IMG_6826.MOV] \\
      --out test/boxer_out_full/diagnostic \\
      --filter                   # also write boxer_3dbbs_filtered.csv
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

from pipeline.iphone_k import extract_K_from_mov
from pipeline.run_vggt_slam import load_vggt_slam_output, VGGTSLAMOutput


# --------------------------------------------------------------------------
# Geometry helpers (kept local to this module; pipeline.ui has similar ones)
# --------------------------------------------------------------------------


def _invert_extrinsic_3x4(extr: np.ndarray) -> np.ndarray:
    R = extr[:3, :3]; t = extr[:3, 3]
    T = np.eye(4, dtype=np.float32)
    T[:3, :3] = R.T; T[:3, 3] = -R.T @ t
    return T


def _quat_to_R(qx: float, qy: float, qz: float, qw: float) -> np.ndarray:
    n = math.sqrt(qx*qx + qy*qy + qz*qz + qw*qw)
    qx, qy, qz, qw = qx/n, qy/n, qz/n, qw/n
    return np.array([
        [1 - 2*(qy*qy + qz*qz),     2*(qx*qy - qz*qw),     2*(qx*qz + qy*qw)],
        [    2*(qx*qy + qz*qw), 1 - 2*(qx*qx + qz*qz),     2*(qy*qz - qx*qw)],
        [    2*(qx*qz - qy*qw),     2*(qy*qz + qx*qw), 1 - 2*(qx*qx + qy*qy)],
    ], dtype=np.float32)


def _box_corners_world(center: np.ndarray, size: np.ndarray, R: np.ndarray) -> np.ndarray:
    sx, sy, sz = size / 2.0
    corners_local = np.array([
        [-sx, -sy, -sz], [+sx, -sy, -sz], [+sx, +sy, -sz], [-sx, +sy, -sz],
        [-sx, -sy, +sz], [+sx, -sy, +sz], [+sx, +sy, +sz], [-sx, +sy, +sz],
    ], dtype=np.float32)
    return corners_local @ R.T + center[None, :]


def _iou_axis_aligned(a: np.ndarray, b: np.ndarray) -> float:
    """IoU between two xyxy boxes [x1, y1, x2, y2]."""
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1 = max(ax1, bx1); iy1 = max(ay1, by1)
    ix2 = min(ax2, bx2); iy2 = min(ay2, by2)
    iw = max(0.0, ix2 - ix1); ih = max(0.0, iy2 - iy1)
    inter = iw * ih
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    denom = area_a + area_b - inter
    return float(inter / denom) if denom > 0 else 0.0


# --------------------------------------------------------------------------
# Data loading
# --------------------------------------------------------------------------


@dataclass
class FrameContext:
    """Per-frame precomputed state shared by all three metrics."""
    time_ns: int
    frame_idx: int
    image_hw: tuple[int, int]            # native (H, W)
    T_world_camera: np.ndarray           # [4, 4] in Boxer aligned world
    K_native: np.ndarray                 # [3, 3] at native (H, W) — iPhone K if overridden
    # World points for this frame, in Boxer aligned world, conf-filtered
    world_points: np.ndarray             # [H_vggt, W_vggt, 3]
    world_points_conf: np.ndarray        # [H_vggt, W_vggt]
    vggt_image_hw: tuple[int, int]       # (294, 518) — VGGT's internal shape
    owl_dets: list[dict]                 # list of {bb2d_xyxy, score, label} in owl_resize px
    owl_image_resize: int                # 960 typically
    owl_pad_info: dict                   # {orig_hw, side, pad_top, pad_left, ...}


def _load_boxer_csv(csv_path: Path) -> dict[int, list[dict]]:
    out: dict[int, list[dict]] = {}
    with open(csv_path) as f:
        for r in csv.DictReader(f):
            t = int(r["time_ns"])
            out.setdefault(t, []).append(r)
    return out


def _prepare_frame_context(
    vggt_out: VGGTSLAMOutput,
    frame_idx: int,
    R_align_3: np.ndarray,
    R_align_4: np.ndarray,
    K_native_override: Optional[np.ndarray],
    owl_entry: dict,
    owl_by_time: dict[int, list[dict]],
    owl_pad_info: dict[int, dict],
) -> Optional[FrameContext]:
    img_path = vggt_out.image_names[frame_idx]
    img_bgr = cv2.imread(img_path)
    if img_bgr is None:
        return None
    orig_h, orig_w = img_bgr.shape[:2]

    src_hw = vggt_out.image_size_hw  # (294, 518)
    K_vggt = vggt_out.intrinsic[frame_idx].numpy().astype(np.float32)
    K_native_vggt = K_vggt.copy()
    K_native_vggt[0, 0] *= orig_w / src_hw[1]; K_native_vggt[0, 2] *= orig_w / src_hw[1]
    K_native_vggt[1, 1] *= orig_h / src_hw[0]; K_native_vggt[1, 2] *= orig_h / src_hw[0]
    K_native = K_native_override if K_native_override is not None else K_native_vggt

    extr = vggt_out.extrinsic[frame_idx].numpy().astype(np.float32)
    T_wc = _invert_extrinsic_3x4(extr)
    T_wc_aligned = R_align_4 @ T_wc

    # Apply the same K-consistency world-point rescale as run_boxer would have.
    wp = vggt_out.world_points[frame_idx].numpy().astype(np.float32)  # [H, W, 3]
    if K_native_override is not None:
        fx_ratio = float(K_native_vggt[0, 0] / K_native_override[0, 0])
        fy_ratio = float(K_native_vggt[1, 1] / K_native_override[1, 1])
        # Camera-frame rescale (same math as run_boxer._rescale_world_points_for_new_K).
        R_cw = extr[:3, :3]; t_cw = extr[:3, 3]
        pts = wp.reshape(-1, 3)
        pts_cam = pts @ R_cw.T + t_cw[None]
        pts_cam[:, 0] *= fx_ratio
        pts_cam[:, 1] *= fy_ratio
        pts = (pts_cam - t_cw[None]) @ R_cw
        wp = pts.reshape(wp.shape)
    # Rotate into Boxer-aligned world.
    wp_aligned = wp @ R_align_3.T

    time_ns = int(vggt_out.frame_ids[frame_idx] * 1_000_000)
    return FrameContext(
        time_ns=time_ns,
        frame_idx=frame_idx,
        image_hw=(orig_h, orig_w),
        T_world_camera=T_wc_aligned,
        K_native=K_native,
        world_points=wp_aligned,
        world_points_conf=vggt_out.world_points_conf[frame_idx].numpy().astype(np.float32),
        vggt_image_hw=src_hw,
        owl_dets=owl_by_time.get(time_ns, []),
        owl_image_resize=owl_entry.get("image_resize", 960),
        owl_pad_info=owl_pad_info.get(time_ns, {}),
    )


# --------------------------------------------------------------------------
# Metric A — distance from box center to nearest cloud point
# --------------------------------------------------------------------------


def _metric_a_dist_to_cloud(box_center: np.ndarray, cloud_sample: np.ndarray) -> float:
    d2 = np.sum((cloud_sample - box_center[None, :]) ** 2, axis=1)
    return float(np.sqrt(d2.min()))


# --------------------------------------------------------------------------
# Metric B — project 3D box corners to native image, IoU with OWL bbox
# --------------------------------------------------------------------------


def _project_points(
    pts_world: np.ndarray,       # [N, 3]
    T_wc: np.ndarray,             # [4, 4]
    K: np.ndarray,                # [3, 3]
) -> tuple[np.ndarray, np.ndarray]:
    """Return (uv [N, 2], z [N]). z > 0 means in front of camera."""
    T_cw = np.linalg.inv(T_wc)
    pts_h = np.concatenate([pts_world, np.ones((pts_world.shape[0], 1), dtype=np.float32)], axis=1)
    pts_cam = pts_h @ T_cw.T
    z = pts_cam[:, 2]
    with np.errstate(divide="ignore", invalid="ignore"):
        u = pts_cam[:, 0] / z * K[0, 0] + K[0, 2]
        v = pts_cam[:, 1] / z * K[1, 1] + K[1, 2]
    return np.stack([u, v], axis=1).astype(np.float32), z.astype(np.float32)


def _owl_bbox_to_native_xyxy(entry: dict, ctx: FrameContext) -> np.ndarray:
    """Convert OWL's resized-coord xyxy to native-image xyxy using pad_info."""
    bb = np.array(entry["bb2d_xyxy"], dtype=np.float32)
    pad = ctx.owl_pad_info
    H, W = ctx.image_hw
    if pad:
        side = pad.get("side", max(H, W))
        pad_left = pad.get("pad_left", (side - W) // 2)
        pad_top = pad.get("pad_top", (side - H) // 2)
        s = side / ctx.owl_image_resize
        bb[0] = bb[0] * s - pad_left
        bb[2] = bb[2] * s - pad_left
        bb[1] = bb[1] * s - pad_top
        bb[3] = bb[3] * s - pad_top
    else:
        sx = W / ctx.owl_image_resize; sy = H / ctx.owl_image_resize
        bb[0] *= sx; bb[2] *= sx
        bb[1] *= sy; bb[3] *= sy
    return bb


def _metric_b_iou2d(
    corners_world: np.ndarray,
    ctx: FrameContext,
    owl_match: Optional[dict],
) -> tuple[float, Optional[np.ndarray], Optional[np.ndarray]]:
    """Returns (iou, projected_bbox_xyxy, owl_bbox_xyxy) — latter two are native coords.

    iou = NaN if no valid projection or no matching OWL bbox (same label).
    """
    uv, z = _project_points(corners_world, ctx.T_world_camera, ctx.K_native)
    if not np.all(np.isfinite(uv)) or np.all(z <= 1e-3):
        return float("nan"), None, None
    # Projected bbox as axis-aligned hull of the 8 corners (only those in front).
    mask = (z > 1e-3) & np.isfinite(uv).all(axis=1)
    if mask.sum() < 2:
        return float("nan"), None, None
    uu = uv[mask, 0]; vv = uv[mask, 1]
    proj_bb = np.array([uu.min(), vv.min(), uu.max(), vv.max()], dtype=np.float32)

    if owl_match is None:
        return float("nan"), proj_bb, None
    owl_bb = _owl_bbox_to_native_xyxy(owl_match, ctx)
    return _iou_axis_aligned(proj_bb, owl_bb), proj_bb, owl_bb


def _match_owl_for_box(
    box_label: str,
    box_center_world: np.ndarray,
    ctx: FrameContext,
) -> Optional[dict]:
    """Among OWL dets on this frame with same label, pick the one whose 2D bbox
    center is nearest to the 3D box's projected center. This reconstructs the
    OWL-bbox ↔ BoxerNet-output correspondence that we lost when filtering by
    thresh_3d in run_boxer (original correspondence was index-based)."""
    if not ctx.owl_dets:
        return None
    uv, z = _project_points(box_center_world[None, :], ctx.T_world_camera, ctx.K_native)
    if z[0] <= 1e-3 or not np.all(np.isfinite(uv)):
        return None
    proj_center = uv[0]
    best = None; best_d = float("inf")
    for d in ctx.owl_dets:
        if d["label"] != box_label:
            continue
        bb_native = _owl_bbox_to_native_xyxy(d, ctx)
        owl_c = np.array([
            0.5 * (bb_native[0] + bb_native[2]),
            0.5 * (bb_native[1] + bb_native[3]),
        ], dtype=np.float32)
        dd = float(np.linalg.norm(owl_c - proj_center))
        if dd < best_d:
            best_d = dd; best = d
    return best


# --------------------------------------------------------------------------
# Metric C — SDP depth consistency inside OWL 2D bbox
# --------------------------------------------------------------------------


def _metric_c_depth_consistency(
    corners_world: np.ndarray,    # [8, 3]
    ctx: FrameContext,
    owl_match: Optional[dict],
) -> tuple[float, dict]:
    """Returns (mismatch_meters, details). NaN if insufficient data.

    Collect VGGT world points whose projection lands inside the OWL 2D bbox.
    Compute their camera-frame Z distribution → [q10, q50, q90].
    Compute box's camera-frame Z range from the 8 corners → [z_min, z_max].
    Mismatch = distance between the depth intervals; 0 if they overlap.
    """
    if owl_match is None:
        return float("nan"), {}

    bb_native = _owl_bbox_to_native_xyxy(owl_match, ctx)
    # Project VGGT world points to native image coords.
    H_vggt, W_vggt = ctx.vggt_image_hw
    pts = ctx.world_points.reshape(-1, 3)
    conf = ctx.world_points_conf.reshape(-1)
    # Tight filter: top 30% conf inside this bbox
    c_thresh = np.percentile(conf, 70.0)
    ok = (conf > c_thresh) & np.isfinite(pts).all(axis=1)
    pts = pts[ok]
    uv, z = _project_points(pts, ctx.T_world_camera, ctx.K_native)
    in_front = z > 1e-3
    in_bbox = (
        (uv[:, 0] >= bb_native[0]) & (uv[:, 0] <= bb_native[2]) &
        (uv[:, 1] >= bb_native[1]) & (uv[:, 1] <= bb_native[3])
    )
    keep = in_front & in_bbox & np.isfinite(uv).all(axis=1)
    zs = z[keep]
    if zs.size < 30:
        return float("nan"), {"n_sdp_in_bbox": int(zs.size)}

    z_sdp_q10, z_sdp_q50, z_sdp_q90 = np.percentile(zs, [10, 50, 90]).tolist()

    # 3D box depth range.
    _, z_box = _project_points(corners_world, ctx.T_world_camera, ctx.K_native)
    z_box = z_box[np.isfinite(z_box)]
    if z_box.size == 0:
        return float("nan"), {"reason": "box behind camera"}
    z_box_min = float(z_box.min()); z_box_max = float(z_box.max())

    # Overlap test between [z_sdp_q10, z_sdp_q90] and [z_box_min, z_box_max].
    gap_low = z_sdp_q10 - z_box_max
    gap_high = z_box_min - z_sdp_q90
    gap = max(0.0, gap_low, gap_high)
    return float(gap), {
        "n_sdp_in_bbox": int(zs.size),
        "sdp_q10": float(z_sdp_q10),
        "sdp_q50": float(z_sdp_q50),
        "sdp_q90": float(z_sdp_q90),
        "box_z_min": z_box_min,
        "box_z_max": z_box_max,
    }


# --------------------------------------------------------------------------
# Main driver
# --------------------------------------------------------------------------


def run(
    vggt_slam_dir: Path,
    boxer_csv: Path,
    owl_json: Path,
    out_dir: Path,
    *,
    override_k_from_mov: Optional[Path] = None,
    filter_thresholds: Optional[dict] = None,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)

    vggt_out = load_vggt_slam_output(vggt_slam_dir)
    per_time_boxes = _load_boxer_csv(boxer_csv)
    with open(owl_json) as f:
        owl_data = json.load(f)
    owl_by_time: dict[int, list[dict]] = {
        int(k): v for k, v in owl_data.get("detections_by_time_ns", {}).items()
    }
    owl_pad_info: dict[int, dict] = {
        int(k): v for k, v in owl_data.get("pad_info_by_time_ns", {}).items()
    }

    # Gravity used by run_boxer (from gravity.json next to CSV).
    grav_path = boxer_csv.parent / "gravity.json"
    if grav_path.exists():
        with open(grav_path) as f:
            R_align_3 = np.array(json.load(f)["R_gravity"], dtype=np.float32)
    else:
        raise FileNotFoundError(
            f"No gravity.json next to {boxer_csv}; diagnostics need the same R_align."
        )
    R_align_4 = np.eye(4, dtype=np.float32); R_align_4[:3, :3] = R_align_3

    K_native_override = None
    if override_k_from_mov is not None:
        K_native_override, _, meta = extract_K_from_mov(override_k_from_mov)
        print(f"[diag] K override: fx={K_native_override[0,0]:.1f} HFOV={meta['hfov_deg']:.1f}°")

    # Precompute a cloud sample for metric A.
    pts_all = vggt_out.world_points.reshape(-1, 3).numpy().astype(np.float32)
    conf_all = vggt_out.world_points_conf.reshape(-1).numpy().astype(np.float32)
    # Apply same K rescale as per-frame would, per-frame (for accuracy) — but
    # metric A only needs a rough scene reference, so fuse after per-frame rescale.
    if K_native_override is not None:
        # Rescale per frame and concatenate (mirrors run_boxer._build_global_sdp).
        all_pts = []
        src_hw = vggt_out.image_size_hw
        K_vggt0 = vggt_out.intrinsic[0].numpy().astype(np.float32)
        # iPhone K at VGGT's internal resolution (294×518).
        H_nat, W_nat = map(int, meta["native_hw"])
        s_w = src_hw[1] / W_nat; s_h = src_hw[0] / H_nat
        Ki_at_src = K_native_override.copy()
        Ki_at_src[0, 0] *= s_w; Ki_at_src[0, 2] *= s_w
        Ki_at_src[1, 1] *= s_h; Ki_at_src[1, 2] *= s_h
        fx_ratio = float(K_vggt0[0, 0] / Ki_at_src[0, 0])
        fy_ratio = float(K_vggt0[1, 1] / Ki_at_src[1, 1])
        for i in range(vggt_out.world_points.shape[0]):
            wp = vggt_out.world_points[i].numpy().astype(np.float32)
            extr = vggt_out.extrinsic[i].numpy().astype(np.float32)
            R_cw = extr[:3, :3]; t_cw = extr[:3, 3]
            pts = wp.reshape(-1, 3)
            pts_cam = pts @ R_cw.T + t_cw[None]
            pts_cam[:, 0] *= fx_ratio
            pts_cam[:, 1] *= fy_ratio
            pts = (pts_cam - t_cw[None]) @ R_cw
            conf = vggt_out.world_points_conf[i].numpy().reshape(-1).astype(np.float32)
            thr = np.percentile(conf, 70.0)
            m = (conf > thr) & np.isfinite(pts).all(axis=1)
            all_pts.append(pts[m])
        pts_all = np.concatenate(all_pts, axis=0)
    else:
        thr = np.percentile(conf_all, 70.0)
        m = (conf_all > thr) & np.isfinite(pts_all).all(axis=1)
        pts_all = pts_all[m]
    pts_all = pts_all @ R_align_3.T
    rng = np.random.default_rng(0)
    cloud_sample = pts_all[rng.choice(pts_all.shape[0], size=min(80_000, pts_all.shape[0]), replace=False)]
    print(f"[diag] cloud sample: {cloud_sample.shape[0]} points after K-rescale + rotate")

    # Iterate per frame / per box.
    frame_ids = list(vggt_out.frame_ids)
    kf_for_time = {int(fid * 1_000_000): i for i, fid in enumerate(frame_ids)}

    rows_out: list[dict] = []
    for time_ns, csv_rows in sorted(per_time_boxes.items()):
        if time_ns not in kf_for_time:
            continue
        frame_idx = kf_for_time[time_ns]
        ctx = _prepare_frame_context(
            vggt_out, frame_idx, R_align_3, R_align_4, K_native_override,
            owl_data, owl_by_time, owl_pad_info,
        )
        if ctx is None:
            continue
        for r in csv_rows:
            center = np.array([
                float(r["tx_world_object"]),
                float(r["ty_world_object"]),
                float(r["tz_world_object"]),
            ], dtype=np.float32)
            size = np.array([
                float(r["scale_x"]), float(r["scale_y"]), float(r["scale_z"]),
            ], dtype=np.float32)
            R_obj = _quat_to_R(
                float(r["qx_world_object"]), float(r["qy_world_object"]),
                float(r["qz_world_object"]), float(r["qw_world_object"]),
            )
            corners = _box_corners_world(center, size, R_obj)
            label = r["name"]; prob = float(r["prob"])

            # A
            dist = _metric_a_dist_to_cloud(center, cloud_sample)

            # B + C (both use matched OWL bbox)
            owl_match = _match_owl_for_box(label, center, ctx)
            iou, proj_bb, owl_bb = _metric_b_iou2d(corners, ctx, owl_match)
            depth_gap, depth_info = _metric_c_depth_consistency(corners, ctx, owl_match)

            rows_out.append({
                "time_ns": time_ns,
                "frame_idx": frame_idx,
                "label": label,
                "prob": prob,
                "cx": float(center[0]), "cy": float(center[1]), "cz": float(center[2]),
                "sx": float(size[0]),   "sy": float(size[1]),   "sz": float(size[2]),
                "dist_to_cloud": dist,
                "iou2d": iou,
                "owl_matched": owl_match is not None,
                "depth_gap": depth_gap,
                "n_sdp_in_bbox": depth_info.get("n_sdp_in_bbox"),
                "sdp_q50": depth_info.get("sdp_q50"),
                "box_z_min": depth_info.get("box_z_min"),
                "box_z_max": depth_info.get("box_z_max"),
            })

    # Write per-box CSV.
    diag_csv = out_dir / "diagnostic_by_box.csv"
    with open(diag_csv, "w", newline="") as f:
        fields = list(rows_out[0].keys())
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in rows_out:
            w.writerow(r)
    print(f"[diag] wrote {len(rows_out)} rows → {diag_csv}")

    # Summary.
    summary = _summarize(rows_out)
    summary_json = out_dir / "diagnostic_summary.json"
    with open(summary_json, "w") as f:
        json.dump(summary, f, indent=2)
    print(json.dumps(summary, indent=2))

    # Optional filter.
    if filter_thresholds is not None:
        filtered_csv = out_dir.parent / "boxer_3dbbs_filtered.csv"
        _write_filtered_csv(
            input_csv=boxer_csv,
            diag_rows=rows_out,
            filtered_csv=filtered_csv,
            thresholds=filter_thresholds,
        )


def _summarize(rows: list[dict]) -> dict:
    def _stats(key: str):
        vals = np.array([r[key] for r in rows if r[key] is not None and not (isinstance(r[key], float) and math.isnan(r[key]))])
        if vals.size == 0:
            return None
        return {
            "n": int(vals.size),
            "mean": float(vals.mean()),
            "median": float(np.median(vals)),
            "p90": float(np.percentile(vals, 90)),
            "p99": float(np.percentile(vals, 99)),
            "max": float(vals.max()),
        }
    return {
        "n_boxes": len(rows),
        "n_owl_matched": sum(1 for r in rows if r.get("owl_matched")),
        "dist_to_cloud": _stats("dist_to_cloud"),
        "iou2d": _stats("iou2d"),
        "depth_gap": _stats("depth_gap"),
    }


def _write_filtered_csv(
    input_csv: Path,
    diag_rows: list[dict],
    filtered_csv: Path,
    thresholds: dict,
) -> None:
    # Map (time_ns, center) → diag scores for matching.
    key_to_diag: dict[tuple[int, float, float, float], dict] = {}
    for d in diag_rows:
        key = (d["time_ns"], round(d["cx"], 5), round(d["cy"], 5), round(d["cz"], 5))
        key_to_diag[key] = d

    dmax = thresholds.get("max_dist_to_cloud", float("inf"))
    iou_min = thresholds.get("min_iou2d", -1.0)
    gap_max = thresholds.get("max_depth_gap", float("inf"))

    kept, dropped = 0, 0
    with open(input_csv) as fin, open(filtered_csv, "w", newline="") as fout:
        reader = csv.DictReader(fin)
        writer = csv.DictWriter(fout, fieldnames=reader.fieldnames)
        writer.writeheader()
        for r in reader:
            key = (
                int(r["time_ns"]),
                round(float(r["tx_world_object"]), 5),
                round(float(r["ty_world_object"]), 5),
                round(float(r["tz_world_object"]), 5),
            )
            d = key_to_diag.get(key)
            if d is None:
                writer.writerow(r); kept += 1; continue
            bad = False
            if d["dist_to_cloud"] is not None and d["dist_to_cloud"] > dmax:
                bad = True
            if d["iou2d"] is not None and not math.isnan(d["iou2d"]) and d["iou2d"] < iou_min:
                bad = True
            if d["depth_gap"] is not None and not math.isnan(d["depth_gap"]) and d["depth_gap"] > gap_max:
                bad = True
            if bad:
                dropped += 1
            else:
                writer.writerow(r); kept += 1
    print(f"[diag] filter: kept {kept}, dropped {dropped} → {filtered_csv}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--vggt-slam", type=Path, required=True)
    parser.add_argument("--boxer-csv", type=Path, required=True)
    parser.add_argument("--owl-json", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--override-k-from-mov", type=Path, default=None)
    parser.add_argument("--filter", action="store_true",
                        help="Write boxer_3dbbs_filtered.csv with bad boxes dropped.")
    parser.add_argument("--max-dist-to-cloud", type=float, default=0.8,
                        help="Drop boxes whose center is farther from the cloud than this (m)")
    parser.add_argument("--min-iou2d", type=float, default=0.05,
                        help="Drop boxes whose 2D reprojection has IoU below this with OWL bbox")
    parser.add_argument("--max-depth-gap", type=float, default=0.5,
                        help="Drop boxes whose camera-Z range is this far from SDP depth in the OWL bbox (m)")
    args = parser.parse_args()

    thresholds = None
    if args.filter:
        thresholds = {
            "max_dist_to_cloud": args.max_dist_to_cloud,
            "min_iou2d": args.min_iou2d,
            "max_depth_gap": args.max_depth_gap,
        }

    run(
        vggt_slam_dir=args.vggt_slam,
        boxer_csv=args.boxer_csv,
        owl_json=args.owl_json,
        out_dir=args.out,
        override_k_from_mov=args.override_k_from_mov,
        filter_thresholds=thresholds,
    )


if __name__ == "__main__":
    main()
