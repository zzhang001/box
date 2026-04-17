"""Convert VGGT-SLAM outputs to Boxer input format.

VGGT-SLAM preserves VGGT's OpenCV camera convention (x-right, y-down,
z-forward) per-submap and rectifies submaps to a globally consistent
SL(4)-aligned world frame after loop closure. Boxer expects a world frame
with SE(3) poses + a gravity vector.

Key conversions:
  - Extrinsic: camera_from_world [3,4] → T_world_camera [4,4] (invert)
  - Intrinsic: Pinhole K [3,3] → CameraTW struct fields
  - Points: Dense [S,H,W,3] → Semi-dense [N,3] (confidence-filtered subsample)
  - Gravity: In OpenCV world frame, gravity points down = [0, 1, 0]
"""

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

from pipeline.run_vggt_slam import VGGTSLAMOutput


@dataclass
class BoxerInput:
    """Per-frame input data formatted for Boxer."""

    T_world_camera: torch.Tensor     # [S, 4, 4] SE(3) world-from-camera
    rotation_flat: torch.Tensor      # [S, 9] flattened 3x3 rotation (world_from_camera)
    translation: torch.Tensor        # [S, 3] translation (world_from_camera)
    fx: torch.Tensor                 # [S] focal length x
    fy: torch.Tensor                 # [S] focal length y
    cx: torch.Tensor                 # [S] principal point x
    cy: torch.Tensor                 # [S] principal point y
    image_width: int
    image_height: int
    gravity: torch.Tensor            # [3] gravity unit vector in world frame
    semidense_points: torch.Tensor   # [N, 3] world-space 3D points


def invert_extrinsic(extrinsic: torch.Tensor) -> torch.Tensor:
    """
    Invert [S, 3, 4] camera_from_world extrinsics to [S, 4, 4] world_from_camera.

    Given extrinsic E = [R | t] where p_cam = R @ p_world + t,
    the inverse T_world_camera satisfies p_world = R^T @ p_cam - R^T @ t.
    """
    S = extrinsic.shape[0]
    R = extrinsic[:, :3, :3]  # [S, 3, 3]
    t = extrinsic[:, :3, 3:]  # [S, 3, 1]

    R_inv = R.transpose(1, 2)  # [S, 3, 3]
    t_inv = -R_inv @ t         # [S, 3, 1]

    T = torch.eye(4).unsqueeze(0).expand(S, -1, -1).clone()
    T[:, :3, :3] = R_inv
    T[:, :3, 3:] = t_inv
    return T


def extract_semidense_points(
    world_points: torch.Tensor,
    world_points_conf: torch.Tensor,
    max_points: int = 10000,
    conf_threshold: float = 1.5,
) -> torch.Tensor:
    """
    Filter and subsample dense world points to semi-dense representation.

    Args:
        world_points: [S, H, W, 3] dense 3D points from VGGT.
        world_points_conf: [S, H, W] confidence scores.
        max_points: Maximum number of points to keep.
        conf_threshold: Minimum confidence to include a point.

    Returns:
        [N, 3] tensor of world-space 3D points.
    """
    # Flatten across all frames
    points_flat = world_points.reshape(-1, 3)       # [S*H*W, 3]
    conf_flat = world_points_conf.reshape(-1)       # [S*H*W]

    # Filter by confidence
    mask = conf_flat > conf_threshold
    points_filtered = points_flat[mask]

    if points_filtered.shape[0] == 0:
        print("WARNING: No points passed confidence filter, lowering threshold")
        median_conf = conf_flat.median().item()
        mask = conf_flat > median_conf
        points_filtered = points_flat[mask]

    # Filter NaN/Inf
    valid = torch.isfinite(points_filtered).all(dim=1)
    points_filtered = points_filtered[valid]

    # Uniform subsample if too many points
    if points_filtered.shape[0] > max_points:
        indices = torch.linspace(0, points_filtered.shape[0] - 1, max_points).long()
        points_filtered = points_filtered[indices]

    print(f"Semi-dense points: {points_filtered.shape[0]} (from {points_flat.shape[0]} dense)")
    return points_filtered


def convert_vggt_to_boxer(
    vggt_output: VGGTSLAMOutput,
    max_semidense_points: int = 10000,
) -> BoxerInput:
    """
    Convert VGGT-SLAM outputs to Boxer-compatible input format.

    Args:
        vggt_output: Output from VGGT-SLAM inference.
        max_semidense_points: Max semi-dense points for Boxer.

    Returns:
        BoxerInput with all fields populated.
    """
    # Invert extrinsics: camera_from_world → world_from_camera
    T_world_camera = invert_extrinsic(vggt_output.extrinsic)  # [S, 4, 4]

    # Extract rotation (flattened 3x3) and translation for Boxer PoseTW format
    R_world = T_world_camera[:, :3, :3]  # [S, 3, 3]
    t_world = T_world_camera[:, :3, 3]   # [S, 3]
    rotation_flat = R_world.reshape(-1, 9)  # [S, 9]

    # Extract intrinsic parameters
    K = vggt_output.intrinsic  # [S, 3, 3]
    fx = K[:, 0, 0]
    fy = K[:, 1, 1]
    cx = K[:, 0, 2]
    cy = K[:, 1, 2]

    H, W = vggt_output.image_size_hw

    # Gravity: In VGGT's OpenCV world frame, Y-axis points down = gravity direction
    # Boxer expects a unit gravity vector in world frame
    gravity = torch.tensor([0.0, 1.0, 0.0], dtype=torch.float32)

    # Semi-dense points from VGGT world points
    semidense_points = extract_semidense_points(
        vggt_output.world_points,
        vggt_output.world_points_conf,
        max_points=max_semidense_points,
    )

    return BoxerInput(
        T_world_camera=T_world_camera,
        rotation_flat=rotation_flat,
        translation=t_world,
        fx=fx, fy=fy, cx=cx, cy=cy,
        image_width=W,
        image_height=H,
        gravity=gravity,
        semidense_points=semidense_points,
    )


def save_boxer_input(boxer_input: BoxerInput, output_dir: Path) -> None:
    """Save Boxer input to disk."""
    output_dir.mkdir(parents=True, exist_ok=True)
    torch.save({
        "T_world_camera": boxer_input.T_world_camera,
        "rotation_flat": boxer_input.rotation_flat,
        "translation": boxer_input.translation,
        "fx": boxer_input.fx,
        "fy": boxer_input.fy,
        "cx": boxer_input.cx,
        "cy": boxer_input.cy,
        "image_width": boxer_input.image_width,
        "image_height": boxer_input.image_height,
        "gravity": boxer_input.gravity,
        "semidense_points": boxer_input.semidense_points,
    }, output_dir / "boxer_input.pt")
    print(f"Saved Boxer input to {output_dir / 'boxer_input.pt'}")
