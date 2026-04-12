"""Run Boxer inference to produce 3D bounding boxes.

Takes VGGT-derived poses, intrinsics, and semi-dense points plus RGB frames,
runs OWLv2 2D detection, then lifts to 3D with BoxerNet.
"""

import sys
from pathlib import Path
from dataclasses import dataclass

import torch
import numpy as np
from PIL import Image

# Ensure extern/boxer is importable
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "extern" / "boxer"))

from pipeline.convert import BoxerInput


@dataclass
class BoxerObject:
    """A single 3D detected object."""

    label: str
    center: tuple[float, float, float]  # (x, y, z) in world frame
    size: tuple[float, float, float]    # (width, height, depth)
    yaw: float                          # rotation around gravity axis
    confidence: float
    uncertainty: float
    first_seen_frame: int
    last_seen_frame: int


@dataclass
class BoxerOutput:
    """Structured output from Boxer inference."""

    objects: list[BoxerObject]
    raw_boxes_per_frame: dict[int, torch.Tensor]  # frame_idx → [M, 7]


def prepare_boxer_sequence_dir(
    frame_paths: list[Path],
    boxer_input: BoxerInput,
    output_dir: Path,
) -> Path:
    """
    Prepare a sequence directory in the format Boxer's run_boxer.py expects.

    Boxer reads sequences from a directory structure. We create a compatible
    layout with camera metadata and semi-dense points derived from VGGT.
    """
    seq_dir = output_dir / "sequence"
    seq_dir.mkdir(parents=True, exist_ok=True)

    # Save frame list
    frames_file = seq_dir / "frames.txt"
    with open(frames_file, "w") as f:
        for i, path in enumerate(frame_paths):
            f.write(f"{i} {path}\n")

    # Save camera data per frame as numpy
    camera_dir = seq_dir / "camera"
    camera_dir.mkdir(exist_ok=True)

    for i in range(len(frame_paths)):
        np.savez(
            camera_dir / f"frame_{i:05d}.npz",
            rotation=boxer_input.rotation_flat[i].numpy(),       # [9]
            translation=boxer_input.translation[i].numpy(),     # [3]
            fx=boxer_input.fx[i].numpy(),
            fy=boxer_input.fy[i].numpy(),
            cx=boxer_input.cx[i].numpy(),
            cy=boxer_input.cy[i].numpy(),
            width=boxer_input.image_width,
            height=boxer_input.image_height,
        )

    # Save gravity
    np.save(seq_dir / "gravity.npy", boxer_input.gravity.numpy())

    # Save semi-dense points
    np.save(seq_dir / "semidense_points.npy", boxer_input.semidense_points.numpy())

    return seq_dir


def run_boxer(
    frame_paths: list[Path],
    boxer_input: BoxerInput,
    output_dir: Path,
    labels: str = "lvisplus",
    thresh_2d: float = 0.25,
    thresh_3d: float = 0.5,
    track: bool = False,
    fuse: bool = True,
    device: str = "mps",
) -> BoxerOutput:
    """
    Run Boxer 3D object detection.

    This function wraps Boxer's inference pipeline, feeding it VGGT-derived
    camera parameters and semi-dense depth instead of LiDAR/ARKit data.

    Args:
        frame_paths: List of frame image paths.
        boxer_input: Camera and depth data from VGGT (via convert.py).
        output_dir: Where to write Boxer results.
        labels: Object label taxonomy ("lvisplus" or comma-separated labels).
        thresh_2d: 2D detection confidence threshold.
        thresh_3d: 3D box confidence threshold.
        track: Enable online temporal tracking.
        fuse: Enable post-hoc 3D box fusion.
        device: Compute device.

    Returns:
        BoxerOutput with detected 3D objects.
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    # Prepare sequence directory
    seq_dir = prepare_boxer_sequence_dir(frame_paths, boxer_input, output_dir)

    # NOTE: The actual Boxer integration depends on Boxer's exact Python API.
    # Boxer's run_boxer.py is designed for Aria/ScanNet data formats.
    # Below is the integration point — once Boxer's API is understood in detail,
    # this section translates our prepared data into Boxer's datum format.

    print(f"Running Boxer on {len(frame_paths)} frames...")
    print(f"  Labels: {labels}")
    print(f"  Semi-dense points: {boxer_input.semidense_points.shape[0]}")
    print(f"  Device: {device}")

    # TODO: Replace with actual Boxer API calls once integration is tested.
    # The Boxer model expects a `datum` dict per frame with:
    #   - image: [B, 3, H, W] normalized RGB
    #   - bbs2d: [B, M, 4] 2D boxes from OWLv2
    #   - camera: CameraTW object
    #   - pose: PoseTW object (SE3)
    #   - gravity: [3] unit vector
    #   - sdp: [B, N, 3] semi-dense points (optional)
    #
    # Integration sketch:
    #
    #   from boxernet.boxernet import BoxerNet
    #   from boxernet.detector import OWLv2Detector
    #
    #   detector = OWLv2Detector(labels=labels, threshold=thresh_2d)
    #   model = BoxerNet.from_pretrained().to(device)
    #
    #   for i, frame_path in enumerate(frame_paths):
    #       image = load_image(frame_path)  # [1, 3, H, W]
    #       boxes_2d = detector(image)       # [1, M, 4]
    #       datum = build_datum(image, boxes_2d, boxer_input, frame_idx=i)
    #       result = model(datum)
    #       # result["obbs_pr_params"] → [1, M, 7] (center + size + yaw)
    #
    #   if fuse:
    #       fused = fuse_boxes_across_frames(all_results)
    #
    # For now, output placeholder structure:

    print("NOTE: Boxer integration requires extern/boxer to be installed.")
    print("      Run ./setup.sh to install dependencies.")
    print(f"      Sequence data prepared at: {seq_dir}")

    return BoxerOutput(objects=[], raw_boxes_per_frame={})


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Run Boxer inference")
    parser.add_argument("--frames", type=Path, required=True)
    parser.add_argument("--vggt-output", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--labels", type=str, default="lvisplus")
    parser.add_argument("--track", action="store_true")
    parser.add_argument("--device", type=str, default="mps")
    args = parser.parse_args()

    from pipeline.run_vggt import load_vggt_output
    from pipeline.convert import convert_vggt_to_boxer

    vggt_output = load_vggt_output(args.vggt_output)
    boxer_input = convert_vggt_to_boxer(vggt_output)

    frame_paths = sorted(Path(args.frames).glob("*.jpg"))
    if not frame_paths:
        frame_paths = sorted(Path(args.frames).glob("*.png"))

    result = run_boxer(
        frame_paths, boxer_input, args.output,
        labels=args.labels, track=args.track, device=args.device,
    )
    print(f"Detected {len(result.objects)} objects")
