"""Pipeline configuration."""

from dataclasses import dataclass, field
from pathlib import Path

import torch


def _detect_device() -> str:
    """Auto-detect the best available compute device."""
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


@dataclass
class PipelineConfig:
    """Configuration for the full front-end -> Boxer pipeline.

    The `front_end` field selects how camera poses + depth are produced:
      - "vggt_slam": VGGT-SLAM 2.0 (default). Handles long videos via submapping
        + loop closure. Requires MIT-SPARK's VGGT fork + salad + gtsam.
      - "vggt": Plain VGGT single-chunk inference. Fast for short clips (<32
        frames) but no global consistency.
    """

    # Input
    video_path: Path = Path("input.mov")
    output_dir: Path = Path("output")

    # Frame extraction — 10-15 fps is a sweet spot for iPhone clips: enough
    # temporal overlap for VGGT-SLAM's optical-flow keyframe selector, not so
    # dense that we waste compute on near-duplicate frames.
    fps: float = 12.0
    max_frames: int = 2000

    # Front-end selection
    front_end: str = "vggt_slam"  # "vggt_slam" | "vggt"

    # VGGT / VGGT-SLAM shared settings
    vggt_model: str = "facebook/VGGT-1B"
    vggt_image_size: int = 518

    # VGGT-SLAM specific
    submap_size: int = 16
    max_loops: int = 1
    min_disparity: float = 50.0
    conf_threshold: float = 25.0
    lc_thres: float = 0.95

    # Boxer settings
    labels: str = "lvisplus"
    thresh_2d: float = 0.25
    thresh_3d: float = 0.5
    track: bool = False
    fuse: bool = True
    max_semidense_points: int = 10000

    # Hardware
    device: str = field(default_factory=_detect_device)

    # Derived paths (populated after init)
    frames_dir: Path = field(init=False)
    front_end_output_dir: Path = field(init=False)
    boxer_output_dir: Path = field(init=False)

    def __post_init__(self) -> None:
        self.video_path = Path(self.video_path)
        self.output_dir = Path(self.output_dir)
        self.frames_dir = self.output_dir / "frames"
        # Keep stage outputs under a front-end-specific subdir so switching
        # front-ends doesn't clobber the previous run.
        self.front_end_output_dir = self.output_dir / self.front_end
        self.boxer_output_dir = self.output_dir / "boxer"
