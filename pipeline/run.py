"""End-to-end pipeline: video → 3D scene graph.

Two front-ends are available for producing camera poses + depth:
  - vggt_slam (default): globally consistent trajectory via submaps + loop closure.
  - vggt: single-chunk VGGT, only suitable for short clips (<32 frames).

Both produce a shape-compatible output object that `convert.py` + `run_boxer.py`
consume unchanged.
"""

import argparse
import time
from pathlib import Path

from pipeline.config import PipelineConfig
from pipeline.extract_frames import extract_frames
from pipeline.convert import convert_vggt_to_boxer, save_boxer_input
from pipeline.run_boxer import run_boxer
from pipeline.export import export_scene_graph


def _run_front_end(config: PipelineConfig, frame_paths: list[Path]):
    """Dispatch to VGGT-SLAM or plain VGGT based on config.front_end."""
    if config.front_end == "vggt_slam":
        from pipeline.run_vggt_slam import run_vggt_slam, save_vggt_slam_output
        out = run_vggt_slam(
            frame_dir=config.frames_dir,
            device=config.device,
            submap_size=config.submap_size,
            max_loops=config.max_loops,
            min_disparity=config.min_disparity,
            conf_threshold=config.conf_threshold,
            lc_thres=config.lc_thres,
        )
        save_vggt_slam_output(out, config.front_end_output_dir)
        return out

    if config.front_end == "vggt":
        from pipeline.run_vggt import run_vggt, save_vggt_output
        out = run_vggt(frame_paths, device=config.device, model_name=config.vggt_model)
        save_vggt_output(out, config.front_end_output_dir)
        return out

    raise ValueError(f"Unknown front_end: {config.front_end!r}")


def run_pipeline(config: PipelineConfig) -> Path:
    """
    Run the full front-end → Boxer pipeline.

    Steps:
        1. Extract frames from video (ffmpeg)
        2. Run the selected front-end (VGGT-SLAM or VGGT)
        3. Convert front-end output to Boxer input format
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

    # Step 2: Front-end inference (VGGT-SLAM or VGGT)
    print(f"\n=== Step 2/5: Running front-end ({config.front_end}) ===")
    front_end_output = _run_front_end(config, frame_paths)

    # Step 3: Format conversion
    print("\n=== Step 3/5: Converting front-end → Boxer format ===")
    boxer_input = convert_vggt_to_boxer(
        front_end_output, max_semidense_points=config.max_semidense_points,
    )
    save_boxer_input(boxer_input, config.boxer_output_dir)

    # Step 4: Boxer inference
    print("\n=== Step 4/5: Running Boxer ===")
    # VGGT-SLAM picks keyframes; the frame list Boxer sees must match them.
    if hasattr(front_end_output, "image_names"):
        boxer_frame_paths = [Path(p) for p in front_end_output.image_names]
    else:
        boxer_frame_paths = frame_paths
    boxer_output = run_boxer(
        boxer_frame_paths, boxer_input, config.boxer_output_dir,
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
        num_frames=len(boxer_frame_paths),
    )

    elapsed = time.time() - t0
    print(f"\n=== Pipeline complete in {elapsed:.1f}s ===")
    print(f"Scene graph: {output_path}")
    return output_path


def main() -> None:
    parser = argparse.ArgumentParser(
        description="VGGT-SLAM (or VGGT) + Boxer: video → 3D scene graph",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Default: VGGT-SLAM front-end, auto device
  python -m pipeline.run --video room.mov --output output/

  # Use plain VGGT for a short clip
  python -m pipeline.run --video short.mov --output output/ --front-end vggt

  # Tune VGGT-SLAM for a longer video
  python -m pipeline.run --video tour.mov --fps 15 --submap-size 16 --max-loops 1
        """,
    )
    parser.add_argument("--video", type=Path, required=True, help="Input video file")
    parser.add_argument("--output", type=Path, default=Path("output"), help="Output directory")
    parser.add_argument("--fps", type=float, default=12.0, help="Frames per second to extract")
    parser.add_argument("--max-frames", type=int, default=2000, help="Max frames to process")
    parser.add_argument("--front-end", type=str, default="vggt_slam",
                        choices=["vggt_slam", "vggt"], help="Pose+depth front-end")
    parser.add_argument("--submap-size", type=int, default=16, help="VGGT-SLAM submap size")
    parser.add_argument("--max-loops", type=int, default=1, help="VGGT-SLAM max loop closures/submap")
    parser.add_argument("--min-disparity", type=float, default=50.0,
                        help="Keyframe selection optical-flow threshold (VGGT-SLAM)")
    parser.add_argument("--labels", type=str, default="lvisplus", help="Object labels or taxonomy")
    parser.add_argument("--track", action="store_true", help="Enable temporal tracking")
    parser.add_argument("--device", type=str, default=None, help="Force device (mps/cuda/cpu)")
    args = parser.parse_args()

    config = PipelineConfig(
        video_path=args.video,
        output_dir=args.output,
        fps=args.fps,
        max_frames=args.max_frames,
        front_end=args.front_end,
        submap_size=args.submap_size,
        max_loops=args.max_loops,
        min_disparity=args.min_disparity,
        labels=args.labels,
        track=args.track,
    )
    if args.device:
        config.device = args.device

    run_pipeline(config)


if __name__ == "__main__":
    main()
