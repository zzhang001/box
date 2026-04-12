"""End-to-end pipeline: video → 3D scene graph."""

import argparse
import time
from pathlib import Path

from pipeline.config import PipelineConfig
from pipeline.extract_frames import extract_frames
from pipeline.run_vggt import run_vggt, save_vggt_output, load_vggt_output
from pipeline.convert import convert_vggt_to_boxer, save_boxer_input
from pipeline.run_boxer import run_boxer
from pipeline.export import export_scene_graph


def run_pipeline(config: PipelineConfig) -> Path:
    """
    Run the full VGGT → Boxer pipeline.

    Steps:
        1. Extract frames from video (ffmpeg)
        2. Run VGGT inference (poses, depth, points)
        3. Convert VGGT output to Boxer input format
        4. Run Boxer inference (2D detection + 3D lifting)
        5. Export scene graph JSON

    Args:
        config: Pipeline configuration.

    Returns:
        Path to the output scene_graph.json.
    """
    config.output_dir.mkdir(parents=True, exist_ok=True)
    t0 = time.time()

    # Step 1: Extract frames
    print("\n=== Step 1/5: Extracting frames ===")
    frame_paths = extract_frames(
        config.video_path, config.frames_dir,
        fps=config.fps, max_frames=config.max_frames,
    )

    # Step 2: VGGT inference
    print("\n=== Step 2/5: Running VGGT ===")
    vggt_output = run_vggt(frame_paths, device=config.device, model_name=config.vggt_model)
    save_vggt_output(vggt_output, config.vggt_output_dir)

    # Step 3: Format conversion
    print("\n=== Step 3/5: Converting VGGT → Boxer format ===")
    boxer_input = convert_vggt_to_boxer(
        vggt_output, max_semidense_points=config.max_semidense_points,
    )
    save_boxer_input(boxer_input, config.boxer_output_dir)

    # Step 4: Boxer inference
    print("\n=== Step 4/5: Running Boxer ===")
    boxer_output = run_boxer(
        frame_paths, boxer_input, config.boxer_output_dir,
        labels=config.labels,
        thresh_2d=config.thresh_2d,
        thresh_3d=config.thresh_3d,
        track=config.track,
        fuse=config.fuse,
        device=config.device,
    )

    # Step 5: Export
    print("\n=== Step 5/5: Exporting scene graph ===")
    output_path = config.output_dir / "scene_graph.json"
    export_scene_graph(
        boxer_output, output_path,
        source_video=config.video_path.name,
        num_frames=len(frame_paths),
    )

    elapsed = time.time() - t0
    print(f"\n=== Pipeline complete in {elapsed:.1f}s ===")
    print(f"Scene graph: {output_path}")
    return output_path


def main() -> None:
    parser = argparse.ArgumentParser(
        description="VGGT + Boxer: video → 3D scene graph",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python -m pipeline.run --video room.mov --output output/
  python -m pipeline.run --video room.mov --output output/ --fps 2 --labels "chair,table,sofa"
  python -m pipeline.run --video room.mov --output output/ --track --device cuda
        """,
    )
    parser.add_argument("--video", type=Path, required=True, help="Input video file")
    parser.add_argument("--output", type=Path, default=Path("output"), help="Output directory")
    parser.add_argument("--fps", type=float, default=1.0, help="Frames per second to extract")
    parser.add_argument("--max-frames", type=int, default=50, help="Max frames to process")
    parser.add_argument("--labels", type=str, default="lvisplus", help="Object labels or taxonomy")
    parser.add_argument("--track", action="store_true", help="Enable temporal tracking")
    parser.add_argument("--device", type=str, default=None, help="Force device (mps/cuda/cpu)")
    args = parser.parse_args()

    config = PipelineConfig(
        video_path=args.video,
        output_dir=args.output,
        fps=args.fps,
        max_frames=args.max_frames,
        labels=args.labels,
        track=args.track,
    )
    if args.device:
        config.device = args.device

    run_pipeline(config)


if __name__ == "__main__":
    main()
