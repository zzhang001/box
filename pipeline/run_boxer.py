"""Run Boxer inference on cached VGGT-SLAM output.

Per keyframe:
  1. Load RGB image from disk, resize to BoxerNet's square input size.
  2. Build a Boxer `datum` dict (image + CameraTW + PoseTW + semi-dense points).
  3. Run OWLv2 to get open-vocab 2D boxes from text labels.
  4. Run BoxerNet to lift 2D → 3D oriented bounding boxes in the world frame.
  5. Accumulate per-frame results; optionally fuse offline.

Gravity alignment:
  VGGT-SLAM's world frame inherits VGGT's OpenCV camera convention
  (Y-axis down = gravity). BoxerNet expects Z-down (its
  `gravity_align_T_world_cam` default). We apply a fixed `Rx(-π/2)` so that
  the VGGT Y-axis maps onto the Boxer -Z-axis across all poses and world
  points. After this rotation Boxer's auto T_world_voxel works without
  further overrides.

See the "Inside Boxer: the Three Models" section in README.md for the
full DINOv3/OWLv2/BoxerNet story.
"""

from __future__ import annotations

import os
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
import torch

from pipeline.gravity import estimate_gravity_rotation
from pipeline.iphone_k import extract_K_from_mov
from pipeline.run_vggt_slam import VGGTSLAMOutput, load_vggt_slam_output

_REPO_ROOT = Path(__file__).resolve().parent.parent
_BOXER_DIR = _REPO_ROOT / "extern" / "boxer"
_BOXER_CKPT_DIR = _BOXER_DIR / "ckpts"

# Put extern/boxer on sys.path once. Boxer expects to run from its own source
# tree (imports like `from loaders.base_loader import BaseLoader`) rather than
# being a pip-installed package.
if str(_BOXER_DIR) not in sys.path:
    sys.path.insert(0, str(_BOXER_DIR))


# --------------------------------------------------------------------------
# Output types
# --------------------------------------------------------------------------


@dataclass
class BoxerObject:
    """A single detected 3D object in the world frame."""

    label: str
    center: tuple[float, float, float]      # (x, y, z) meters
    size: tuple[float, float, float]        # (w, h, d) meters in object frame
    yaw: float                              # rotation around gravity axis (rad)
    confidence: float                       # mean(2D score, 3D prob)
    uncertainty: float                      # exp(logvar) from BoxerNet
    frame_idx: int                          # keyframe index it came from
    time_ns: int                            # synthetic timestamp


@dataclass
class BoxerOutput:
    """Per-frame raw results + flattened object list."""

    objects: list[BoxerObject]
    raw_boxes_per_frame: dict[int, torch.Tensor]  # frame_idx → ObbTW tensor
    labels_per_frame: dict[int, list[str]] = field(default_factory=dict)


# --------------------------------------------------------------------------
# Geometry helpers
# --------------------------------------------------------------------------


# Fallback rotation taking VGGT's Y-down world to Boxer's Z-down world when
# gravity estimation is disabled or fails. Rx(-π/2) maps (x,y,z) → (x,-z,y)
# so Y_world_vggt ([0,1,0]) → [0,0,-1]. `run_boxer()` can override this per
# call with an estimate from floor-plane RANSAC (pipeline.gravity).
_R_ALIGN_FALLBACK_3 = np.array(
    [[1, 0, 0],
     [0, 0, 1],
     [0, -1, 0]],
    dtype=np.float32,
)


def _make_R_align_4(R3: np.ndarray) -> np.ndarray:
    R4 = np.eye(4, dtype=np.float32)
    R4[:3, :3] = R3.astype(np.float32)
    return R4


def _invert_extrinsic_3x4(extr: np.ndarray) -> np.ndarray:
    """[3,4] camera_from_world → [4,4] world_from_camera (T_world_camera)."""
    R = extr[:3, :3]
    t = extr[:3, 3]
    T = np.eye(4, dtype=np.float32)
    T[:3, :3] = R.T
    T[:3, 3] = -R.T @ t
    return T


def _rescale_world_points_for_new_K(
    world_points: np.ndarray,   # [H, W, 3] in VGGT world frame
    extrinsic: np.ndarray,      # [3, 4] camera_from_world
    fx_ratio: float,
    fy_ratio: float,
) -> np.ndarray:
    """Rescale VGGT world points to be consistent with a new K.

    VGGT unprojects a pixel (u, v) at depth z via its own K:
        X_cam_vggt = (u - cx) * z / fx_vggt
    If we replace K with K_real (different fx), the correct camera-frame X is
    X_cam_real = X_cam_vggt * (fx_vggt / fx_real) = X_cam_vggt * fx_ratio.
    Same for Y. Z (depth) is unchanged.

    Since we need to stay in world frame (for pose consistency), we:
      1. Transform world points → camera frame via the frame's extrinsic.
      2. Scale X, Y in camera frame.
      3. Transform back to world with the same extrinsic.

    Different frames rotate the per-axis rescale differently in world-frame
    coordinates; that's expected — the correction is per-ray-of-that-camera.
    """
    H, W, _ = world_points.shape
    R_cw = extrinsic[:3, :3]     # camera_from_world rotation
    t_cw = extrinsic[:3, 3]      # camera_from_world translation
    pts_world = world_points.reshape(-1, 3)
    # world → camera
    pts_cam = (pts_world - t_cw[None]) @ R_cw.T     # wait: actually (R_cw @ (p_w - t))... let me be careful
    # Actually extrinsic as [R|t] with p_cam = R p_w + t. So pts_cam = pts_world @ R_cw.T + t_cw.
    pts_cam = pts_world @ R_cw.T + t_cw[None]
    # Rescale X and Y in camera frame
    pts_cam[:, 0] *= fx_ratio
    pts_cam[:, 1] *= fy_ratio
    # camera → world: p_w = R_cw.T @ (p_c - t)
    pts_world_new = (pts_cam - t_cw[None]) @ R_cw
    return pts_world_new.reshape(H, W, 3).astype(np.float32)


def _scale_K(K: np.ndarray, src_hw: tuple[int, int], dst_hw: tuple[int, int]) -> np.ndarray:
    """Scale a pinhole K from src_hw to dst_hw (anisotropic if aspect differs)."""
    src_h, src_w = src_hw
    dst_h, dst_w = dst_hw
    sx = dst_w / src_w
    sy = dst_h / src_h
    K2 = K.copy().astype(np.float32)
    K2[0, 0] *= sx  # fx
    K2[0, 2] *= sx  # cx
    K2[1, 1] *= sy  # fy
    K2[1, 2] *= sy  # cy
    return K2


def _sample_sdp_from_world_points(
    world_points: np.ndarray,      # [H, W, 3] already aligned to Boxer world frame
    conf: np.ndarray,              # [H, W]
    conf_percentile: float = 90.0,
    num_samples: int = 10000,
) -> torch.Tensor:
    """Flatten per-frame world_points, confidence-filter, NaN-pad to fixed count.

    conf_percentile=90 keeps the top 10% of points by VGGT confidence. Tighter
    than the 50% default: BoxerNet treats SDP as a depth prior, so noisy
    monocular-depth outliers from low-confidence pixels (surface boundaries,
    untextured walls, sky) confuse its 3D lifting — especially in submaps
    where SL(4) scale drift already stresses consistency. Fewer but higher-
    quality points push BoxerNet toward the correct depth.
    """
    pts = world_points.reshape(-1, 3)
    c = conf.reshape(-1)
    thresh = float(np.percentile(c, conf_percentile))
    mask = (c > thresh) & np.isfinite(pts).all(axis=1)
    pts = pts[mask]
    if pts.shape[0] > num_samples:
        idx = np.random.choice(pts.shape[0], size=num_samples, replace=False)
        pts = pts[idx]
    sdp = torch.from_numpy(pts.astype(np.float32))
    if sdp.shape[0] < num_samples:
        pad = torch.full((num_samples - sdp.shape[0], 3), float("nan"), dtype=torch.float32)
        sdp = torch.cat([sdp, pad], dim=0)
    return sdp


def _build_global_sdp(
    vggt_out: VGGTSLAMOutput,
    R_align_3: np.ndarray,
    conf_percentile: float = 90.0,
    max_points: int = 400_000,
    fx_ratio: Optional[float] = None,
    fy_ratio: Optional[float] = None,
) -> np.ndarray:
    """Fuse all per-frame world points into ONE global cloud in Boxer world frame.

    If (fx_ratio, fy_ratio) is given, per-frame world points are rescaled
    to be consistent with an overridden K before fusion. See
    `_rescale_world_points_for_new_K` for the math.
    """
    all_pts: list[np.ndarray] = []
    S = vggt_out.world_points.shape[0]
    for i in range(S):
        wp = vggt_out.world_points[i].numpy().astype(np.float32)
        conf = vggt_out.world_points_conf[i].numpy().astype(np.float32)
        if fx_ratio is not None and fy_ratio is not None:
            wp = _rescale_world_points_for_new_K(
                wp, vggt_out.extrinsic[i].numpy().astype(np.float32),
                fx_ratio=fx_ratio, fy_ratio=fy_ratio,
            )
        pts = wp.reshape(-1, 3)
        c = conf.reshape(-1)
        thresh = float(np.percentile(c, conf_percentile))
        mask = (c > thresh) & np.isfinite(pts).all(axis=1)
        all_pts.append(pts[mask])
    pts = np.concatenate(all_pts, axis=0).astype(np.float32)
    rng = np.random.default_rng(0)
    if pts.shape[0] > max_points:
        idx = rng.choice(pts.shape[0], size=max_points, replace=False)
        pts = pts[idx]
    return (pts @ R_align_3.T).astype(np.float32)


def _crop_global_sdp_to_frustum(
    global_sdp: np.ndarray,        # [N, 3] in Boxer world frame
    T_world_camera: np.ndarray,    # [4, 4] in same world
    K: np.ndarray,                 # [3, 3] for BoxerNet input resolution (hw)
    hw: int,
    near: float = 0.1,
    far: float = 10.0,
    num_samples: int = 10000,
) -> torch.Tensor:
    """Project global SDP into the current camera; keep points inside frustum."""
    T_cw = np.linalg.inv(T_world_camera)
    pts_h = np.concatenate([global_sdp, np.ones((global_sdp.shape[0], 1), dtype=np.float32)], axis=1)
    pts_cam = pts_h @ T_cw.T
    z = pts_cam[:, 2]
    with np.errstate(divide="ignore", invalid="ignore"):
        u = (pts_cam[:, 0] / z) * K[0, 0] + K[0, 2]
        v = (pts_cam[:, 1] / z) * K[1, 1] + K[1, 2]
    in_frustum = (z > near) & (z < far) & (u >= 0) & (u < hw) & (v >= 0) & (v < hw)
    visible = global_sdp[in_frustum]
    if visible.shape[0] > num_samples:
        idx = np.random.choice(visible.shape[0], size=num_samples, replace=False)
        visible = visible[idx]
    sdp = torch.from_numpy(visible.astype(np.float32))
    if sdp.shape[0] < num_samples:
        pad = torch.full((num_samples - sdp.shape[0], 3), float("nan"), dtype=torch.float32)
        sdp = torch.cat([sdp, pad], dim=0)
    return sdp


# --------------------------------------------------------------------------
# Per-frame datum construction
# --------------------------------------------------------------------------


def _build_datum(
    vggt_out: VGGTSLAMOutput,
    frame_idx: int,
    boxer_hw: int,
    R_align_3: np.ndarray,
    R_align_4: np.ndarray,
    global_sdp: Optional[np.ndarray] = None,
    K_native_override: Optional[np.ndarray] = None,
    native_hw_override: Optional[tuple[int, int]] = None,
):
    """Build a Boxer datum dict for one VGGT-SLAM keyframe.

    `R_align_3` / `R_align_4` are the gravity-alignment rotations taking
    VGGT-SLAM's world frame into Boxer's Z-down world. Supplied by the
    caller so the same one is used across all frames (estimated once per
    video via floor-plane RANSAC or fallback Rx(-π/2)).

    Image handling: pad to square (aspect-preserving) then isotropic-
    resize to (hw, hw). **Do NOT anisotropically stretch 1920×1080 → hw²**
    — BoxerNet is trained on ~4:3 indoor data and 16:9 iPhone stretch
    creates a K with fx/fy ratio ~1.76 that BoxerNet isn't used to,
    producing badly-localized 3D boxes.

    Returns (datum, img_bgr_cv2) — img_bgr_cv2 is handy for OWL visualization.
    """
    from loaders.base_loader import BaseLoader
    from utils.tw.pose import PoseTW

    # 1. Image: load at native resolution, pad to square, isotropic-resize to
    # boxer_hw × boxer_hw. Black-bar pads keep aspect; K scales isotropically.
    img_path = vggt_out.image_names[frame_idx]
    img_bgr = cv2.imread(img_path)
    if img_bgr is None:
        raise FileNotFoundError(f"Could not read {img_path}")
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    orig_h, orig_w = img_rgb.shape[:2]
    side = max(orig_h, orig_w)
    pad_top = (side - orig_h) // 2
    pad_bottom = side - orig_h - pad_top
    pad_left = (side - orig_w) // 2
    pad_right = side - orig_w - pad_left
    img_square = cv2.copyMakeBorder(
        img_rgb, pad_top, pad_bottom, pad_left, pad_right,
        cv2.BORDER_CONSTANT, value=(0, 0, 0),
    )
    img_resized = cv2.resize(img_square, (boxer_hw, boxer_hw), interpolation=cv2.INTER_AREA)
    img0 = BaseLoader.img_to_tensor(img_resized)  # [1, 3, hw, hw] in [0, 1]

    # 2. Intrinsics: VGGT K is calibrated for its isotropic downscale (294×518).
    # Scale it to the native image first (isotropic scale — 1920/518 ≈ 1080/294),
    # then apply the pad offset to cy, then uniform-scale to (hw, hw).
    src_hw = vggt_out.image_size_hw
    K_vggt = vggt_out.intrinsic[frame_idx].numpy().astype(np.float32)
    K_native_vggt = _scale_K(K_vggt, src_hw=src_hw, dst_hw=(orig_h, orig_w))
    # If the caller supplies a native-resolution K override (e.g. from iPhone
    # EXIF), use it instead of VGGT's estimate. The expected native_hw must
    # match the image we loaded.
    if K_native_override is not None:
        if native_hw_override is not None:
            assert tuple(native_hw_override) == (orig_h, orig_w), (
                f"native_hw_override={native_hw_override} vs loaded image {(orig_h, orig_w)}"
            )
        K_native = K_native_override.astype(np.float32)
    else:
        K_native = K_native_vggt
    # Shift principal point into the padded square.
    K_padded = K_native.copy()
    K_padded[0, 2] += pad_left
    K_padded[1, 2] += pad_top
    # Uniform resize side → boxer_hw.
    s = boxer_hw / side
    K_boxer = K_padded.copy()
    K_boxer[0, 0] *= s; K_boxer[0, 2] *= s
    K_boxer[1, 1] *= s; K_boxer[1, 2] *= s
    cam = BaseLoader.pinhole_from_K(
        w=boxer_hw, h=boxer_hw,
        fx=float(K_boxer[0, 0]),
        fy=float(K_boxer[1, 1]),
        cx=float(K_boxer[0, 2]),
        cy=float(K_boxer[1, 2]),
    )

    # 3. Pose: invert extrinsic → T_world_camera, then apply gravity alignment.
    extr = vggt_out.extrinsic[frame_idx].numpy().astype(np.float32)
    T_wc = _invert_extrinsic_3x4(extr)
    T_wc_aligned = R_align_4 @ T_wc
    R = torch.from_numpy(T_wc_aligned[:3, :3].copy())
    t = torch.from_numpy(T_wc_aligned[:3, 3].copy())
    T_world_rig0 = PoseTW.from_Rt(R, t)

    # 4. Semi-dense points.
    if global_sdp is not None:
        # Mode: global — use the scene-wide fused cloud cropped to this camera's
        # frustum. Every frame sees the same globally-consistent geometry, which
        # sidesteps per-submap SL(4) scale drift.
        sdp_w = _crop_global_sdp_to_frustum(
            global_sdp, T_wc_aligned, K_boxer, hw=boxer_hw, num_samples=10000,
        )
    else:
        # Mode: per-frame — use this frame's VGGT world points after conf filter.
        wp = vggt_out.world_points[frame_idx].numpy().astype(np.float32)   # [H, W, 3]
        # If we overrode K, rescale VGGT's points so they remain consistent with
        # the new pinhole geometry (per-frame, in camera frame — see docstring).
        if K_native_override is not None:
            fx_ratio = float(K_native_vggt[0, 0] / K_native_override[0, 0])
            fy_ratio = float(K_native_vggt[1, 1] / K_native_override[1, 1])
            wp = _rescale_world_points_for_new_K(
                wp, extr, fx_ratio=fx_ratio, fy_ratio=fy_ratio,
            )
        wp_aligned = wp @ R_align_3.T                                       # rotate into Z-down world
        conf = vggt_out.world_points_conf[frame_idx].numpy().astype(np.float32)
        sdp_w = _sample_sdp_from_world_points(wp_aligned, conf, num_samples=10000)

    # 5. Misc.
    time_ns = int(vggt_out.frame_ids[frame_idx] * 1_000_000)  # frame_id is float

    datum = {
        "img0": img0.float(),
        "cam0": cam.float(),
        "T_world_rig0": T_world_rig0.float(),
        "sdp_w": sdp_w,
        "time_ns0": time_ns,
        # BoxerNet.process_camera asserts rotated0.ndim == 1 (per-batch flag).
        "rotated0": torch.tensor([False]),
        "bb2d0": torch.zeros(0, 4, dtype=torch.float32),
    }
    return datum, img_bgr


# --------------------------------------------------------------------------
# Main inference loop
# --------------------------------------------------------------------------


def _resolve_device(device: Optional[str]) -> str:
    if device:
        return device
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def _default_ckpt() -> Path:
    return _BOXER_CKPT_DIR / "boxernet_hw960in4x6d768-wssxpf9p.ckpt"


def run_boxer(
    vggt_slam_output_dir: Path,
    output_dir: Path,
    labels: list[str],
    device: Optional[str] = None,
    thresh_2d: float = 0.25,
    thresh_3d: float = 0.5,
    detector_hw: int = 960,
    num_sdp_samples: int = 10000,
    ckpt_path: Optional[Path] = None,
    max_frames: Optional[int] = None,
    estimate_gravity: bool = True,
    sdp_mode: str = "perframe",
    override_k_from_mov: Optional[Path] = None,
) -> BoxerOutput:
    """Run OWLv2 + BoxerNet on a cached VGGT-SLAM output.

    Args:
        vggt_slam_output_dir: directory containing `vggt_slam_output.pt`.
        output_dir: where to write the Boxer CSV + scene graph.
        labels: list of text prompts for OWLv2 (e.g. ["chair", "table"]).
        device: "cuda" | "mps" | "cpu"; auto-detected if None.
        thresh_2d: OWLv2 min confidence.
        thresh_3d: BoxerNet min prob to keep a 3D box.
        detector_hw: OWLv2 input resize (matches BoxerNet hw).
        num_sdp_samples: semi-dense points fed to BoxerNet per frame.
        ckpt_path: override BoxerNet checkpoint path.
        max_frames: cap keyframes processed (for quick runs).
        estimate_gravity: if True (default), fit the world floor plane via
            RANSAC and use its normal for gravity alignment. Falls back to the
            fixed Rx(-π/2) if the fit is unreliable.
        sdp_mode: "perframe" (default) uses each frame's VGGT world points
            with tight conf filter — simplest, fastest. "global" fuses all
            per-frame world points into one globally-consistent cloud and
            crops it to each frame's view frustum — better at suppressing
            per-submap SL(4) scale drift at a small compute cost.
    """
    from boxernet.boxernet import BoxerNet
    from owl.owl_wrapper import OwlWrapper
    from utils.file_io import ObbCsvWriter2
    from utils.tw.tensor_utils import pad_string, string2tensor

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / "boxer_3dbbs.csv"
    device = _resolve_device(device)
    print(f"[boxer] device = {device}")

    # Load cached VGGT-SLAM output.
    vggt_out = load_vggt_slam_output(Path(vggt_slam_output_dir))
    n_keyframes = vggt_out.extrinsic.shape[0]
    if max_frames is not None:
        n_keyframes = min(n_keyframes, max_frames)
    print(f"[boxer] {n_keyframes} keyframes from {vggt_slam_output_dir}")

    # Estimate gravity-aligning rotation once for the whole video.
    if estimate_gravity:
        R_align_3, grav_info = estimate_gravity_rotation(vggt_out)
        print(f"[boxer] gravity: method={grav_info['method']} "
              f"tilt_correction={grav_info.get('tilt_correction_deg', 0):.2f}° "
              f"inliers={grav_info.get('num_plane_inliers', 0)}")
    else:
        R_align_3 = _R_ALIGN_FALLBACK_3.copy()
        grav_info = {"method": "fixed_Rx_minus_pi_2"}
        print("[boxer] gravity: fixed Rx(-π/2) (estimate_gravity=False)")
    R_align_4 = _make_R_align_4(R_align_3)

    # Persist the alignment so downstream tools (pipeline.ui) can reproduce
    # the exact world frame BoxerNet placed its boxes in. Without this, a
    # viewer that re-estimates gravity from the same VGGT-SLAM output can
    # get a slightly different rotation (RANSAC is stochastic) and
    # projected OBBs will visibly drift.
    import json as _json
    with open(output_dir / "gravity.json", "w") as f:
        _json.dump({"R_gravity": R_align_3.tolist(), **grav_info}, f, indent=2)

    # Optional known-K override (e.g. from iPhone EXIF) — replaces VGGT's
    # estimated K in the Boxer datum, AND rescales per-frame world points
    # so they stay consistent with the new pinhole geometry.
    K_native_override: Optional[np.ndarray] = None
    native_hw_override: Optional[tuple[int, int]] = None
    fx_ratio: Optional[float] = None
    fy_ratio: Optional[float] = None
    if override_k_from_mov is not None:
        K_native_override, native_hw_override, meta = extract_K_from_mov(Path(override_k_from_mov))
        print(f"[boxer] K override from {override_k_from_mov}: "
              f"fx={K_native_override[0,0]:.1f} fy={K_native_override[1,1]:.1f} "
              f"HFOV={meta['hfov_deg']:.1f}° (lens={meta.get('lens_model')})")
        # Compute the ratio VGGT's native-scaled K / iPhone K, used for
        # per-frame point rescaling inside _build_datum / _build_global_sdp.
        # The ratio is the same at VGGT's original res (294×518) and at
        # native (1080×1920) because both K's scale isotropically between
        # them, so we use the override's native K.
        src_hw = vggt_out.image_size_hw
        Ki_at_src = K_native_override.copy()
        Ki_at_src[0, 0] *= src_hw[1] / native_hw_override[1]
        Ki_at_src[0, 2] *= src_hw[1] / native_hw_override[1]
        Ki_at_src[1, 1] *= src_hw[0] / native_hw_override[0]
        Ki_at_src[1, 2] *= src_hw[0] / native_hw_override[0]
        K_vggt0 = vggt_out.intrinsic[0].numpy().astype(np.float32)
        fx_ratio = float(K_vggt0[0, 0] / Ki_at_src[0, 0])
        fy_ratio = float(K_vggt0[1, 1] / Ki_at_src[1, 1])
        print(f"[boxer]   world-point rescale ratios: fx_ratio={fx_ratio:.3f} fy_ratio={fy_ratio:.3f}")

    # Optionally precompute the globally-fused scene cloud (used when
    # sdp_mode="global"). Expensive for ~300k points, but a one-time cost.
    global_sdp: Optional[np.ndarray] = None
    if sdp_mode == "global":
        global_sdp = _build_global_sdp(
            vggt_out, R_align_3, fx_ratio=fx_ratio, fy_ratio=fy_ratio,
        )
        print(f"[boxer] sdp_mode=global: {global_sdp.shape[0]} fused points"
              + (" (K-rescaled)" if fx_ratio is not None else ""))
    elif sdp_mode == "perframe":
        print("[boxer] sdp_mode=perframe (VGGT per-frame points, top 10% conf)"
              + (" (K-rescaled)" if fx_ratio is not None else ""))
    else:
        raise ValueError(f"Unknown sdp_mode={sdp_mode!r}; use 'perframe' or 'global'")

    # Load OWLv2 (text encoder + vision detector). Text prompts are encoded and
    # cached internally.
    print(f"[boxer] loading OWLv2 with {len(labels)} labels: {labels[:8]}{'...' if len(labels) > 8 else ''}")
    owl = OwlWrapper(
        device=device,
        text_prompts=labels,
        min_confidence=thresh_2d,
        precision="float32" if device in ("cpu", "mps") else None,
    )

    # Load BoxerNet (DINOv3 backbone lives inside).
    ckpt = Path(ckpt_path) if ckpt_path else _default_ckpt()
    print(f"[boxer] loading BoxerNet from {ckpt}")
    boxernet = BoxerNet.load_from_checkpoint(str(ckpt), device=device)
    boxer_hw = boxernet.hw
    if isinstance(boxer_hw, (tuple, list)):
        boxer_hw = int(boxer_hw[0])
    print(f"[boxer] BoxerNet expects {boxer_hw}x{boxer_hw} images")

    # CSV writer.
    writer = ObbCsvWriter2(str(csv_path))
    sem_name_to_id = {lab: i for i, lab in enumerate(labels)}
    sem_id_to_name = {v: k for k, v in sem_name_to_id.items()}

    all_objects: list[BoxerObject] = []
    raw_boxes_per_frame: dict[int, torch.Tensor] = {}
    labels_per_frame: dict[int, list[str]] = {}

    t0 = time.time()
    for i in range(n_keyframes):
        datum, _img_bgr = _build_datum(
            vggt_out, i, boxer_hw, R_align_3, R_align_4,
            global_sdp=global_sdp,
            K_native_override=K_native_override,
            native_hw_override=native_hw_override,
        )

        # OWLv2 expects input in [0, 255].
        img_torch_255 = datum["img0"].clone() * 255.0
        bb2d, scores2d, label_ints, _ = owl.forward(
            img_torch_255,
            rotated=False,
            resize_to_HW=(detector_hw, detector_hw),
        )
        if bb2d.shape[0] == 0:
            print(f"[boxer] frame {i}: 0 OWL detections")
            continue
        labels2d = [labels[li] for li in label_ints]
        datum["bb2d"] = bb2d

        # BoxerNet forward.
        if device == "mps":
            outputs = boxernet.forward(datum)
        elif device == "cuda" and torch.cuda.is_bf16_supported():
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                outputs = boxernet.forward(datum)
        else:
            outputs = boxernet.forward(datum)

        obb_pr_w = outputs["obbs_pr_w"].cpu()[0]
        assert len(obb_pr_w) == len(labels2d), (len(obb_pr_w), len(labels2d))

        # Tag sem_id from OWL labels.
        sem_ids = torch.zeros(len(labels2d), dtype=torch.int32)
        for j, lab in enumerate(labels2d):
            if lab not in sem_name_to_id:
                new_id = len(sem_name_to_id)
                sem_name_to_id[lab] = new_id
                sem_id_to_name[new_id] = lab
            sem_ids[j] = sem_name_to_id[lab]
        obb_pr_w.set_sem_id(sem_ids)

        # Filter by 3D prob, combine with 2D score for final confidence.
        prob_3d = obb_pr_w.prob.squeeze(-1)
        keep = prob_3d >= thresh_3d
        obb_pr_w = obb_pr_w[keep].clone()
        labels3d = [labels2d[j] for j in range(len(labels2d)) if keep[j]]
        if len(obb_pr_w) == 0:
            continue
        mean_scores = (scores2d[keep] + prob_3d[keep]) / 2.0
        obb_pr_w.set_prob(mean_scores)

        text_data = torch.stack(
            [string2tensor(pad_string(lab, max_len=128)) for lab in labels3d]
        )
        obb_pr_w.set_text(text_data)

        # Write per-frame OBBs to CSV (Boxer's standard format).
        writer.write(obb_pr_w, datum["time_ns0"], sem_id_to_name=sem_id_to_name)

        # Also collect into our BoxerOutput.
        raw_boxes_per_frame[i] = obb_pr_w.clone()
        labels_per_frame[i] = labels3d

        bb3 = obb_pr_w.bb3_object.numpy()             # [M, 6]
        T_wo = obb_pr_w.T_world_object                # PoseTW-like
        centers = T_wo.t.numpy()                       # [M, 3]
        rot_flat = T_wo.R.numpy().reshape(-1, 9)       # [M, 9]
        # Yaw = atan2(R[1,0], R[0,0]) on the Z-up-world convention
        R_mat = T_wo.R.numpy()                         # [M, 3, 3]
        yaws = np.arctan2(R_mat[:, 1, 0], R_mat[:, 0, 0])
        logvar = outputs.get("obbs_pr_logvar", None)
        if logvar is not None:
            lv = logvar.cpu()[0][keep].squeeze(-1).numpy()
        else:
            lv = np.zeros(len(obb_pr_w))
        for j in range(len(obb_pr_w)):
            w = float(bb3[j, 1] - bb3[j, 0])
            h = float(bb3[j, 3] - bb3[j, 2])
            d = float(bb3[j, 5] - bb3[j, 4])
            all_objects.append(BoxerObject(
                label=labels3d[j],
                center=tuple(centers[j].tolist()),
                size=(w, h, d),
                yaw=float(yaws[j]),
                confidence=float(mean_scores[j]),
                uncertainty=float(np.exp(lv[j])),
                frame_idx=i,
                time_ns=datum["time_ns0"],
            ))

        if (i + 1) % 10 == 0 or i == n_keyframes - 1:
            elapsed = time.time() - t0
            print(f"[boxer] {i+1}/{n_keyframes} frames — {elapsed:.1f}s elapsed, "
                  f"{len(all_objects)} total 3D dets")

    writer.close()
    print(f"[boxer] wrote CSV → {csv_path}")
    print(f"[boxer] total 3D detections: {len(all_objects)}")
    return BoxerOutput(
        objects=all_objects,
        raw_boxes_per_frame=raw_boxes_per_frame,
        labels_per_frame=labels_per_frame,
    )


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Run Boxer on a VGGT-SLAM output directory")
    parser.add_argument("--vggt-slam", type=Path, required=True,
                        help="Directory containing vggt_slam_output.pt")
    parser.add_argument("--output", type=Path, required=True,
                        help="Boxer output directory")
    parser.add_argument("--labels", type=str, default="chair,table,sofa,bed,monitor,keyboard,laptop,book,lamp,plant",
                        help="Comma-separated OWLv2 text prompts")
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--thresh-2d", type=float, default=0.25)
    parser.add_argument("--thresh-3d", type=float, default=0.5)
    parser.add_argument("--max-frames", type=int, default=None,
                        help="Cap number of keyframes (for quick smoke tests)")
    parser.add_argument("--sdp-mode", type=str, default="perframe",
                        choices=["perframe", "global"],
                        help="'perframe' (default) uses each frame's VGGT depth; "
                             "'global' passes the fused scene cloud cropped to "
                             "the frustum — more robust to per-submap scale drift")
    parser.add_argument("--override-k-from-mov", type=Path, default=None,
                        help="Parse iPhone .MOV EXIF to get the real camera K "
                             "and substitute it for VGGT's estimate. Rescales "
                             "per-frame world points to stay consistent with "
                             "the corrected pinhole geometry.")
    parser.add_argument("--scene-graph", type=Path, default=None,
                        help="Also write scene_graph.json here (default: <output>/scene_graph.json)")
    args = parser.parse_args()

    labels = [s.strip() for s in args.labels.split(",") if s.strip()]
    out = run_boxer(
        vggt_slam_output_dir=args.vggt_slam,
        output_dir=args.output,
        labels=labels,
        device=args.device,
        thresh_2d=args.thresh_2d,
        thresh_3d=args.thresh_3d,
        max_frames=args.max_frames,
        sdp_mode=args.sdp_mode,
        override_k_from_mov=args.override_k_from_mov,
    )

    # Also emit scene_graph.json for a self-contained CLI run.
    from pipeline.export import export_scene_graph

    scene_graph_path = args.scene_graph or (args.output / "scene_graph.json")
    n_frames = len(out.raw_boxes_per_frame)
    export_scene_graph(out, scene_graph_path, source_video="", num_frames=n_frames)
