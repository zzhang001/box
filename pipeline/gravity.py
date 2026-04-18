"""Gravity estimation from VGGT-SLAM world points via floor-plane RANSAC.

Background:
  VGGT-SLAM's world frame inherits the first camera's OpenCV convention
  (Y-axis ≈ gravity). Boxer expects Z-down gravity. Our original adapter
  hardcoded `Rx(-π/2)` — fine when the iPhone starts exactly upright, but
  off by a few degrees if the user tilts the phone.

  Standard iPhone .MOV files don't carry IMU data (Camera.app doesn't
  embed it). So we recover gravity from geometry: fit the dominant
  horizontal plane to the fused world points (the floor), use its
  normal as gravity direction, and build the rotation that maps it
  onto Boxer's Z-down convention.

Algorithm:
  1. Fuse all per-frame world_points into one cloud, confidence-filter.
  2. Apply an *initial guess* rotation (same as the hardcoded `Rx(-π/2)`)
     so the floor is roughly at the bottom — lets us easily isolate
     floor-candidate points as "lowest k% along the initial Z-axis".
  3. Run RANSAC plane fit (open3d) on those floor candidates.
  4. Check the plane's normal is close enough to vertical (cos-sim with
     initial up). If yes, compute a small correction rotation to make
     the plane's normal exactly `[0, 0, -1]`. Compose with the initial
     guess → final `R_gravity`.
  5. If the plane fit is unreliable (too few inliers, normal too far
     from vertical), fall back to the initial guess.

This is a one-shot estimate per video, applied uniformly to every frame
downstream. For long scans with sustained phone tilt drift, a per-frame
correction from IMU would be stronger — but for typical iPhone room
scans this is usually within a degree.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np

from pipeline._common import R_ALIGN_FALLBACK
from pipeline.run_vggt_slam import VGGTSLAMOutput


# Initial guess: Rx(-π/2) maps VGGT's Y-down world to Boxer's Z-down world.
_R_INITIAL_GUESS = R_ALIGN_FALLBACK


def _collect_confident_world_points(
    vggt_out: VGGTSLAMOutput,
    conf_percentile: float = 50.0,
    max_points: int = 500_000,
) -> np.ndarray:
    """Flatten per-frame world_points to an [N, 3] cloud, conf-filtered."""
    pts = vggt_out.world_points.reshape(-1, 3).numpy().astype(np.float32)
    conf = vggt_out.world_points_conf.reshape(-1).numpy().astype(np.float32)
    thresh = float(np.percentile(conf, conf_percentile))
    mask = (conf > thresh) & np.isfinite(pts).all(axis=1)
    pts = pts[mask]
    if pts.shape[0] > max_points:
        idx = np.random.default_rng(0).choice(pts.shape[0], size=max_points, replace=False)
        pts = pts[idx]
    return pts


def _rotation_aligning_vectors(src: np.ndarray, dst: np.ndarray) -> np.ndarray:
    """3×3 rotation taking unit vector src onto unit vector dst (Rodrigues)."""
    src = src / (np.linalg.norm(src) + 1e-9)
    dst = dst / (np.linalg.norm(dst) + 1e-9)
    c = float(np.dot(src, dst))
    if c > 0.999999:
        return np.eye(3, dtype=np.float32)
    if c < -0.999999:
        # 180° rotation about any axis perpendicular to src.
        axis = np.array([1, 0, 0], dtype=np.float32)
        if abs(src[0]) > 0.9:
            axis = np.array([0, 1, 0], dtype=np.float32)
        axis = axis - np.dot(axis, src) * src
        axis /= np.linalg.norm(axis)
        K = np.array(
            [[0, -axis[2], axis[1]],
             [axis[2], 0, -axis[0]],
             [-axis[1], axis[0], 0]],
            dtype=np.float32,
        )
        return (np.eye(3, dtype=np.float32) + 2 * K @ K).astype(np.float32)
    v = np.cross(src, dst)
    K = np.array(
        [[0, -v[2], v[1]],
         [v[2], 0, -v[0]],
         [-v[1], v[0], 0]],
        dtype=np.float32,
    )
    R = np.eye(3, dtype=np.float32) + K + K @ K * (1.0 / (1.0 + c))
    return R.astype(np.float32)


def estimate_gravity_rotation(
    vggt_out: VGGTSLAMOutput,
    *,
    floor_fraction: float = 0.20,
    ransac_dist: float = 0.02,
    ransac_n: int = 3,
    ransac_iterations: int = 2000,
    min_floor_inliers: int = 500,
    max_tilt_rad: float = np.deg2rad(25.0),
) -> tuple[np.ndarray, dict]:
    """Estimate the rotation that takes VGGT-SLAM's world into Boxer's Z-down world.

    Args:
        vggt_out: loaded VGGT-SLAM output.
        floor_fraction: bottom fraction of points (along the initial-guess Z
            axis) used as floor candidates. 0.3 is forgiving — tilted floors
            or bumpy clouds still work.
        ransac_dist: plane inlier distance threshold (meters).
        ransac_n, ransac_iterations: passed to open3d RANSAC.
        min_floor_inliers: below this, fall back to the initial guess.
        max_tilt_rad: if the fitted plane's normal is more than this many
            radians away from the initial-guess up direction, we don't trust it.

    Returns:
        (R_gravity, info) where R_gravity is a (3, 3) float32 matrix and info
        is a diagnostic dict (inlier count, tilt correction angle, method).
    """
    import open3d as o3d

    pts = _collect_confident_world_points(vggt_out)
    info: dict = {
        "num_candidate_points": int(pts.shape[0]),
        "method": "initial_guess",
    }

    if pts.shape[0] < 10:
        return _R_INITIAL_GUESS.copy(), info

    # Rotate by the initial guess so gravity is -Z. In that frame, the floor
    # sits at the *smallest* Z values (gravity pulls down along -Z). Keep the
    # bottom `floor_fraction` by Z as candidates.
    pts_rot = (pts @ _R_INITIAL_GUESS.T).astype(np.float32)
    z_coords = pts_rot[:, 2]
    cutoff = np.quantile(z_coords, floor_fraction)
    floor_candidates = pts_rot[z_coords <= cutoff]
    info["num_floor_candidates"] = int(floor_candidates.shape[0])

    if floor_candidates.shape[0] < min_floor_inliers:
        return _R_INITIAL_GUESS.copy(), info

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(floor_candidates.astype(np.float64))
    plane_model, inliers = pcd.segment_plane(
        distance_threshold=ransac_dist,
        ransac_n=ransac_n,
        num_iterations=ransac_iterations,
    )
    info["num_plane_inliers"] = int(len(inliers))

    if len(inliers) < min_floor_inliers:
        return _R_INITIAL_GUESS.copy(), info

    # plane_model: [a, b, c, d] with a*x + b*y + c*z + d = 0. Normal is [a,b,c].
    normal = np.array(plane_model[:3], dtype=np.float32)
    normal /= np.linalg.norm(normal) + 1e-9

    # Floor's upward-pointing normal opposes gravity. In our initial-guess
    # frame gravity is -Z, so the upward normal should have POSITIVE Z
    # component. Flip if RANSAC returned the other orientation.
    if normal[2] < 0:
        normal = -normal

    # Angle between this normal and the initial-guess up direction [0,0,1].
    cos_tilt = float(np.clip(normal[2], -1.0, 1.0))
    tilt_rad = float(np.arccos(cos_tilt))
    info["tilt_correction_rad"] = tilt_rad
    info["tilt_correction_deg"] = float(np.rad2deg(tilt_rad))

    if tilt_rad > max_tilt_rad:
        # Untrustworthy — floor probably isn't what we fit.
        info["method"] = "initial_guess_tilt_too_large"
        return _R_INITIAL_GUESS.copy(), info

    # Build correction rotation: map `normal` onto [0, 0, 1] (gravity is -Z).
    R_correction = _rotation_aligning_vectors(normal, np.array([0, 0, 1], dtype=np.float32))
    R_gravity = (R_correction @ _R_INITIAL_GUESS).astype(np.float32)
    info["method"] = "floor_plane_ransac"
    return R_gravity, info


def main() -> None:
    import argparse
    import json

    from pipeline.run_vggt_slam import load_vggt_slam_output

    parser = argparse.ArgumentParser(
        description="Estimate gravity rotation from VGGT-SLAM world points",
    )
    parser.add_argument("--vggt-slam", type=Path, required=True,
                        help="Directory containing vggt_slam_output.pt")
    parser.add_argument("--out-json", type=Path, default=None,
                        help="Where to write the estimated rotation + diagnostics")
    args = parser.parse_args()

    out = load_vggt_slam_output(args.vggt_slam)
    R, info = estimate_gravity_rotation(out)
    print("R_gravity:")
    print(R)
    print("info:", info)

    if args.out_json is not None:
        args.out_json.parent.mkdir(parents=True, exist_ok=True)
        payload = {"R_gravity": R.tolist(), **info}
        with open(args.out_json, "w") as f:
            json.dump(payload, f, indent=2)
        print(f"Wrote {args.out_json}")


if __name__ == "__main__":
    main()
