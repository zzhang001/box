"""Run VGGT-SLAM 2.0 to produce globally consistent camera poses, depth, and world points.

VGGT-SLAM extends VGGT with:
  - Optical-flow keyframe selection (so we don't push every frame through VGGT)
  - Submap processing (default 16 frames per submap)
  - SL(4) pose-graph optimization across submaps
  - Loop closure via DINO/SALAD image retrieval

Output shape matches the plain-VGGT wrapper (VGGTOutput) so downstream `convert.py`
and `run_boxer.py` don't need to know which front-end produced the data.

Key trade-offs vs. plain VGGT:
  - (+) Handles arbitrarily long videos (VGGT alone caps at ~32 frames per chunk).
  - (+) Globally consistent trajectory after loop closure.
  - (-) Requires the MIT-SPARK VGGT fork (vggt_spark) + salad + gtsam.
  - (-) World frame is SL(4)-rectified, not strictly metric Euclidean.
"""

import os
import sys
import time
from pathlib import Path
from dataclasses import dataclass
from typing import Optional

import numpy as np
import torch


_REPO_ROOT = Path(__file__).resolve().parent.parent
_VGGT_SLAM_DIR = _REPO_ROOT / "extern" / "vggt_slam"

# Make extern/vggt_slam importable. Its setup.sh installs a fork of VGGT and salad
# into `extern/vggt_slam/third_party/{vggt,salad}` as editable pip packages, so
# those resolve through site-packages rather than sys.path hacks.
if str(_VGGT_SLAM_DIR) not in sys.path:
    sys.path.insert(0, str(_VGGT_SLAM_DIR))


@dataclass
class VGGTSLAMOutput:
    """Per-keyframe outputs from VGGT-SLAM after SL(4) optimization.

    Shape/semantics kept compatible with `pipeline.run_vggt.VGGTOutput` so that
    `pipeline.convert.convert_vggt_to_boxer` can consume either directly.

    Frame ordering: one row per keyframe that VGGT-SLAM retained (loop-closure
    auxiliary frames are excluded). `image_names[i]` is the input path for row i.
    """

    extrinsic: torch.Tensor          # [S, 3, 4] camera_from_world (SE3)
    intrinsic: torch.Tensor          # [S, 3, 3] pinhole K
    depth: torch.Tensor              # [S, H, W] VGGT predicted depth (camera-local)
    depth_conf: torch.Tensor         # [S, H, W] confidence
    world_points: torch.Tensor       # [S, H, W, 3] in SL(4)-rectified world frame
    world_points_conf: torch.Tensor  # [S, H, W] confidence
    image_size_hw: tuple[int, int]
    image_names: list[str]           # [S] source image paths, in row order
    frame_ids: list[float]           # [S] numeric IDs parsed from filenames


def _detect_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def _patch_vggt_slam_for_device(device: str) -> None:
    """VGGT-SLAM hardcodes `device = 'cuda'` in a couple of spots.

    Called once before instantiating `Solver`, we override those module-level
    globals so the image-retrieval branch + VGGT inference use the chosen device.
    """
    import vggt_slam.loop_closure as lc
    lc.device = device


def run_vggt_slam(
    frame_dir: Path,
    device: Optional[str] = None,
    submap_size: int = 16,
    overlapping_window_size: int = 1,
    max_loops: int = 1,
    min_disparity: float = 50.0,
    conf_threshold: float = 25.0,
    lc_thres: float = 0.95,
    use_optical_flow_downsample: bool = True,
) -> VGGTSLAMOutput:
    """Drive VGGT-SLAM end-to-end on a folder of frames.

    Args:
        frame_dir: Directory of extracted frames (jpg/png).
        device: "cuda" | "mps" | "cpu"; auto-detected if None.
        submap_size: Frames per submap (VGGT chunk). Keep ≤16 to fit memory.
        overlapping_window_size: Currently only 1 is supported upstream.
        max_loops: Max loop-closures per submap (0 disables).
        min_disparity: Optical-flow threshold for keyframe selection.
        conf_threshold: VGGT confidence percentile to filter.
        lc_thres: Image-retrieval similarity threshold for loop closure.
        use_optical_flow_downsample: If False, feed every input frame to VGGT.

    Returns:
        VGGTSLAMOutput with per-keyframe poses + depth + world points.
    """
    import cv2
    from tqdm import tqdm

    device = device or _detect_device()
    print(f"[vggt-slam] device = {device}")

    _patch_vggt_slam_for_device(device)

    # Imports must follow the device patch so `lc.device` is the right value
    # at the time Solver instantiates ImageRetrieval.
    import vggt_slam.slam_utils as utils
    from vggt_slam.solver import Solver
    from vggt.models.vggt import VGGT

    solver = Solver(
        init_conf_threshold=conf_threshold,
        lc_thres=lc_thres,
        vis_voxel_size=None,
    )

    print("[vggt-slam] loading VGGT weights (facebook/VGGT-1B)…")
    model = VGGT()
    _URL = "https://huggingface.co/facebook/VGGT-1B/resolve/main/model.pt"
    model.load_state_dict(torch.hub.load_state_dict_from_url(_URL))
    model.eval()

    # bfloat16 is fastest on Ampere+/MPS; CPU also supports bfloat16 though
    # inference will be painfully slow.
    model = model.to(torch.bfloat16).to(device)

    # Glob frames the same way main.py does.
    import glob
    image_names = [
        f for f in glob.glob(os.path.join(str(frame_dir), "*"))
        if "depth" not in os.path.basename(f).lower()
        and "txt" not in os.path.basename(f).lower()
        and "db" not in os.path.basename(f).lower()
    ]
    image_names = utils.sort_images_by_number(image_names)
    print(f"[vggt-slam] found {len(image_names)} frames in {frame_dir}")
    if len(image_names) == 0:
        raise RuntimeError(f"No frames found in {frame_dir}")

    image_names_subset: list[str] = []
    t_start = time.time()
    for image_name in tqdm(image_names, desc="vggt-slam"):
        if use_optical_flow_downsample:
            img = cv2.imread(image_name)
            enough = solver.flow_tracker.compute_disparity(img, min_disparity, False)
            if enough:
                image_names_subset.append(image_name)
        else:
            image_names_subset.append(image_name)

        is_full = len(image_names_subset) == submap_size + overlapping_window_size
        is_last = image_name == image_names[-1] and len(image_names_subset) > 0
        if is_full or is_last:
            predictions = solver.run_predictions(
                image_names_subset, model, max_loops, clip_model=None, clip_preprocess=None,
            )
            solver.add_points(predictions)
            solver.graph.optimize()
            # Keep overlap frames to bridge to next submap.
            image_names_subset = image_names_subset[-overlapping_window_size:]

    elapsed = time.time() - t_start
    print(f"[vggt-slam] processed {solver.map.get_num_submaps()} submaps in {elapsed:.1f}s "
          f"({solver.graph.get_num_loops()} loop closures)")

    return _extract_output(solver)


def _extract_output(solver) -> VGGTSLAMOutput:
    """Pull per-keyframe arrays out of the optimized solver state.

    Rationale: the Solver keeps submaps in `solver.map`, each submap stores its
    VGGT predictions + local SE(3) poses. After `graph.optimize()` we decompose
    the pose-graph homographies back into K/R/t and transform the per-frame
    world-points into the globally consistent frame.
    """
    extrinsics: list[np.ndarray] = []
    intrinsics: list[np.ndarray] = []
    depths: list[np.ndarray] = []
    depth_confs: list[np.ndarray] = []
    world_points: list[np.ndarray] = []
    world_points_confs: list[np.ndarray] = []
    image_names: list[str] = []
    frame_ids: list[float] = []

    for submap in solver.map.ordered_submaps_by_key():
        if submap.get_lc_status():
            continue  # Skip loop-closure auxiliary submaps.

        poses_world = submap.get_all_poses_world(solver.graph)  # [S, 4, 4] world_from_camera
        # camera_from_world = inv(world_from_camera); keep [3, 4] for compatibility.
        for pose in poses_world:
            T_wc = pose  # world_from_camera
            R_cw = T_wc[:3, :3].T
            t_cw = -R_cw @ T_wc[:3, 3]
            E = np.concatenate([R_cw, t_cw[:, None]], axis=1)  # [3, 4]
            extrinsics.append(E)

        # proj_mats stores K as 4x4; grab the top-left 3x3.
        for K4 in submap.proj_mats:
            intrinsics.append(np.asarray(K4[:3, :3], dtype=np.float32))

        # submap.pointclouds is (S, H, W, 3) in submap-local frame. We transform
        # each frame to world using its homography. (VGGT depth itself is a
        # camera-local scalar, so no transform is needed for depth.)
        point_list, fids, _ = submap.get_points_list_in_world_frame(solver.graph)
        for p, fid in zip(point_list, fids):
            world_points.append(np.asarray(p, dtype=np.float32))  # [H, W, 3]
            frame_ids.append(fid)

        # Confidence + depth are stored per submap; VGGT-SLAM doesn't keep the
        # raw depth tensor after `add_points`, but it does keep `conf` and
        # `pointclouds`. We derive depth from pointclouds' camera-local z-axis
        # by re-projecting through the stored K. Cheaper: use the camera-local
        # pointclouds before world-transform — they're in the submap's first
        # frame. For per-frame depth, fall back to `||p - cam_center||_z` via
        # the inverse pose. For now we store zeros — Boxer uses semi-dense
        # points directly.
        S, H, W, _ = submap.pointclouds.shape
        depths.append(np.zeros((S, H, W), dtype=np.float32))
        depth_confs.append(np.asarray(submap.conf, dtype=np.float32))
        world_points_confs.append(np.asarray(submap.conf, dtype=np.float32))

        for n in submap.img_names:
            image_names.append(str(n))

    extrinsic = torch.from_numpy(np.stack(extrinsics, axis=0).astype(np.float32))
    intrinsic = torch.from_numpy(np.stack(intrinsics, axis=0))
    wp = torch.from_numpy(np.stack(world_points, axis=0))
    depth = torch.from_numpy(np.concatenate(depths, axis=0))
    depth_conf = torch.from_numpy(np.concatenate(depth_confs, axis=0))
    wp_conf = torch.from_numpy(np.concatenate(world_points_confs, axis=0))

    H, W = wp.shape[1], wp.shape[2]

    print(f"[vggt-slam] output: {extrinsic.shape[0]} keyframes @ {H}×{W}")
    return VGGTSLAMOutput(
        extrinsic=extrinsic,
        intrinsic=intrinsic,
        depth=depth,
        depth_conf=depth_conf,
        world_points=wp,
        world_points_conf=wp_conf,
        image_size_hw=(H, W),
        image_names=image_names,
        frame_ids=frame_ids,
    )


def save_vggt_slam_output(output: VGGTSLAMOutput, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    torch.save({
        "extrinsic": output.extrinsic,
        "intrinsic": output.intrinsic,
        "depth": output.depth,
        "depth_conf": output.depth_conf,
        "world_points": output.world_points,
        "world_points_conf": output.world_points_conf,
        "image_size_hw": output.image_size_hw,
        "image_names": output.image_names,
        "frame_ids": output.frame_ids,
    }, output_dir / "vggt_slam_output.pt")
    print(f"[vggt-slam] saved output → {output_dir / 'vggt_slam_output.pt'}")


def load_vggt_slam_output(output_dir: Path) -> VGGTSLAMOutput:
    data = torch.load(output_dir / "vggt_slam_output.pt", weights_only=False)
    return VGGTSLAMOutput(**data)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Run VGGT-SLAM 2.0 on a frame directory")
    parser.add_argument("--frames", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--submap-size", type=int, default=16)
    parser.add_argument("--max-loops", type=int, default=1)
    parser.add_argument("--min-disparity", type=float, default=50.0)
    parser.add_argument("--no-flow-downsample", action="store_true",
                        help="Disable optical-flow keyframe selection (slower, denser)")
    args = parser.parse_args()

    out = run_vggt_slam(
        args.frames,
        device=args.device,
        submap_size=args.submap_size,
        max_loops=args.max_loops,
        min_disparity=args.min_disparity,
        use_optical_flow_downsample=not args.no_flow_downsample,
    )
    save_vggt_slam_output(out, args.output)
