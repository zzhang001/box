"""Run VGGT inference to produce camera poses, depth, and point clouds."""

import sys
from pathlib import Path
from dataclasses import dataclass

import numpy as np
import torch

# Ensure extern/vggt is importable
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "extern" / "vggt"))


@dataclass
class VGGTOutput:
    """Structured output from VGGT inference."""

    extrinsic: torch.Tensor       # [S, 3, 4] camera_from_world
    intrinsic: torch.Tensor       # [S, 3, 3] pinhole K
    depth: torch.Tensor           # [S, H, W] metric depth
    depth_conf: torch.Tensor      # [S, H, W] confidence
    world_points: torch.Tensor    # [S, H, W, 3] dense 3D points
    world_points_conf: torch.Tensor  # [S, H, W] confidence
    image_size_hw: tuple[int, int]


def run_vggt(
    frame_paths: list[Path],
    device: str = "mps",
    model_name: str = "facebook/VGGT-1B",
) -> VGGTOutput:
    """
    Run VGGT on a set of frames.

    Args:
        frame_paths: List of image file paths.
        device: Compute device ("mps", "cuda", "cpu").
        model_name: HuggingFace model identifier.

    Returns:
        VGGTOutput with poses, depth, and point clouds.
    """
    from vggt.models.vggt import VGGT
    from vggt.utils.load_fn import load_and_preprocess_images
    from vggt.utils.pose_enc import pose_encoding_to_extri_intri

    print(f"Loading VGGT model: {model_name}")
    model = VGGT.from_pretrained(model_name).to(device)
    model.eval()

    # Select dtype based on device
    if device == "cuda" and torch.cuda.get_device_capability()[0] >= 8:
        dtype = torch.bfloat16
    elif device == "cuda":
        dtype = torch.float16
    else:
        dtype = torch.float32

    print(f"Loading {len(frame_paths)} frames...")
    image_names = [str(p) for p in frame_paths]
    images = load_and_preprocess_images(image_names).to(device)  # [S, 3, H, W]
    images = images.unsqueeze(0)  # [1, S, 3, H, W]

    image_size_hw = (images.shape[-2], images.shape[-1])

    print("Running VGGT inference...")
    with torch.no_grad():
        if device == "cuda":
            with torch.cuda.amp.autocast(dtype=dtype):
                predictions = model(images)
        else:
            predictions = model(images)

    # Extract and convert pose encoding to extrinsic/intrinsic matrices
    pose_enc = predictions["pose_enc"]  # [1, S, 9]
    extrinsic, intrinsic = pose_encoding_to_extri_intri(pose_enc, image_size_hw)

    output = VGGTOutput(
        extrinsic=extrinsic.squeeze(0).cpu(),           # [S, 3, 4]
        intrinsic=intrinsic.squeeze(0).cpu(),           # [S, 3, 3]
        depth=predictions["depth"].squeeze(0).squeeze(-1).cpu(),  # [S, H, W]
        depth_conf=predictions["depth_conf"].squeeze(0).cpu(),
        world_points=predictions["world_points"].squeeze(0).cpu(),
        world_points_conf=predictions["world_points_conf"].squeeze(0).cpu(),
        image_size_hw=image_size_hw,
    )

    print(f"VGGT complete: {output.extrinsic.shape[0]} frames, depth {output.depth.shape[1:]}px")
    return output


def save_vggt_output(output: VGGTOutput, output_dir: Path) -> None:
    """Save VGGT outputs to disk as .pt files."""
    output_dir.mkdir(parents=True, exist_ok=True)
    torch.save({
        "extrinsic": output.extrinsic,
        "intrinsic": output.intrinsic,
        "depth": output.depth,
        "depth_conf": output.depth_conf,
        "world_points": output.world_points,
        "world_points_conf": output.world_points_conf,
        "image_size_hw": output.image_size_hw,
    }, output_dir / "vggt_output.pt")
    print(f"Saved VGGT output to {output_dir / 'vggt_output.pt'}")


def load_vggt_output(output_dir: Path) -> VGGTOutput:
    """Load previously saved VGGT outputs."""
    data = torch.load(output_dir / "vggt_output.pt", weights_only=False)
    return VGGTOutput(**data)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Run VGGT inference")
    parser.add_argument("--frames", type=Path, required=True, help="Directory of frame images")
    parser.add_argument("--output", type=Path, required=True, help="Output directory")
    parser.add_argument("--device", type=str, default="mps")
    args = parser.parse_args()

    frame_paths = sorted(Path(args.frames).glob("*.jpg"))
    if not frame_paths:
        frame_paths = sorted(Path(args.frames).glob("*.png"))

    result = run_vggt(frame_paths, device=args.device)
    save_vggt_output(result, args.output)
