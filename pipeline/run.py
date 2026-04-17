"""End-to-end pipeline: iPhone video → 3D scene graph, via VGGT-SLAM + Boxer."""

import argparse
import time
from pathlib import Path

from pipeline.config import PipelineConfig
from pipeline.extract_frames import extract_frames
from pipeline.run_vggt_slam import run_vggt_slam, save_vggt_slam_output
from pipeline.run_boxer import run_boxer
from pipeline.export import export_scene_graph


def run_pipeline(config: PipelineConfig) -> Path:
    """
    Run the full VGGT-SLAM → Boxer pipeline.

    Steps:
        1. Extract frames from video (ffmpeg)
        2. Run VGGT-SLAM (submapping + SL(4) optimization + loop closure)
        3. Run Boxer (OWLv2 + BoxerNet) on each VGGT-SLAM keyframe
        4. Export scene graph JSON

    Args:
        config: Pipeline configuration.

    Returns:
        Path to the output scene_graph.json.
    """
    config.output_dir.mkdir(parents=True, exist_ok=True)
    t0 = time.time()

    # Step 1: Extract frames
    print("\n=== Step 1/4: Extracting frames ===")
    extract_frames(
        config.video_path, config.frames_dir,
        fps=config.fps, max_frames=config.max_frames,
    )

    # Step 2: VGGT-SLAM inference
    print("\n=== Step 2/4: Running VGGT-SLAM ===")
    vggt_slam_output = run_vggt_slam(
        frame_dir=config.frames_dir,
        device=config.device,
        submap_size=config.submap_size,
        max_loops=config.max_loops,
        min_disparity=config.min_disparity,
        conf_threshold=config.conf_threshold,
        lc_thres=config.lc_thres,
    )
    save_vggt_slam_output(vggt_slam_output, config.vggt_slam_output_dir)

    # Step 3: Boxer inference (OWLv2 + BoxerNet) per keyframe
    print("\n=== Step 3/4: Running Boxer ===")
    labels = [s.strip() for s in config.labels.split(",") if s.strip()]
    boxer_output = run_boxer(
        vggt_slam_output_dir=config.vggt_slam_output_dir,
        output_dir=config.boxer_output_dir,
        labels=labels,
        device=config.device,
        thresh_2d=config.thresh_2d,
        thresh_3d=config.thresh_3d,
    )

    # Step 4: Export scene graph
    print("\n=== Step 4/4: Exporting scene graph ===")
    output_path = config.output_dir / "scene_graph.json"
    export_scene_graph(
        boxer_output, output_path,
        source_video=config.video_path.name,
        num_frames=vggt_slam_output.extrinsic.shape[0],
    )

    elapsed = time.time() - t0
    print(f"\n=== Pipeline complete in {elapsed:.1f}s ===")
    print(f"Scene graph: {output_path}")
    return output_path


def main() -> None:
    parser = argparse.ArgumentParser(
        description="VGGT-SLAM + Boxer: iPhone video → 3D scene graph",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic: auto-detect device, 12 fps extraction, default Boxer labels
  python -m pipeline.run --video test/IMG_6826.MOV --output output/

  # Custom object labels for OWLv2
  python -m pipeline.run --video tour.mov --labels "chair,table,monitor,laptop,lamp"

  # Force CPU on Mac mini
  python -m pipeline.run --video tour.mov --device cpu
        """,
    )
    parser.add_argument("--video", type=Path, required=True, help="Input video file")
    parser.add_argument("--output", type=Path, default=Path("output"), help="Output directory")
    parser.add_argument("--fps", type=float, default=12.0, help="Frames per second to extract")
    parser.add_argument("--max-frames", type=int, default=2000, help="Max frames to process")
    parser.add_argument("--submap-size", type=int, default=16, help="VGGT-SLAM submap size")
    parser.add_argument("--max-loops", type=int, default=1, help="Max loop closures per submap")
    parser.add_argument("--min-disparity", type=float, default=50.0,
                        help="Keyframe selection optical-flow threshold")
    parser.add_argument(
        "--labels", type=str,
        default="chair,table,sofa,bed,monitor,keyboard,laptop,book,lamp,plant",
        help="Comma-separated OWLv2 text prompts",
    )
    parser.add_argument("--thresh-2d", type=float, default=0.25)
    parser.add_argument("--thresh-3d", type=float, default=0.5)
    parser.add_argument("--device", type=str, default=None, help="Force device (cuda/mps/cpu)")
    args = parser.parse_args()

    config = PipelineConfig(
        video_path=args.video,
        output_dir=args.output,
        fps=args.fps,
        max_frames=args.max_frames,
        submap_size=args.submap_size,
        max_loops=args.max_loops,
        min_disparity=args.min_disparity,
        labels=args.labels,
        thresh_2d=args.thresh_2d,
        thresh_3d=args.thresh_3d,
    )
    if args.device:
        config.device = args.device

    run_pipeline(config)


if __name__ == "__main__":
    main()
