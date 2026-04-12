"""Extract frames from video using ffmpeg."""

import subprocess
from pathlib import Path

from tqdm import tqdm


def extract_frames(
    video_path: Path,
    output_dir: Path,
    fps: float = 1.0,
    max_frames: int = 50,
) -> list[Path]:
    """
    Extract frames from a video file using ffmpeg.

    Args:
        video_path: Path to input video (.mov, .mp4, etc.)
        output_dir: Directory to write extracted frames
        fps: Frames per second to extract
        max_frames: Maximum number of frames to extract

    Returns:
        Sorted list of extracted frame paths.
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    cmd = [
        "ffmpeg", "-i", str(video_path),
        "-vf", f"fps={fps}",
        "-frames:v", str(max_frames),
        "-q:v", "2",
        str(output_dir / "frame_%05d.jpg"),
        "-y",
    ]

    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"ffmpeg failed:\n{result.stderr}")

    frames = sorted(output_dir.glob("frame_*.jpg"))
    print(f"Extracted {len(frames)} frames from {video_path.name}")
    return frames


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Extract frames from video")
    parser.add_argument("--video", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--fps", type=float, default=1.0)
    parser.add_argument("--max-frames", type=int, default=50)
    args = parser.parse_args()

    extract_frames(args.video, args.output, args.fps, args.max_frames)
