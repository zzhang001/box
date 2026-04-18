"""Pipeline configuration."""

from dataclasses import dataclass, field
from pathlib import Path

from pipeline._common import detect_device as _detect_device


@dataclass
class PipelineConfig:
    """Configuration for the VGGT-SLAM → Boxer pipeline."""

    # Input
    video_path: Path = Path("input.mov")
    output_dir: Path = Path("output")

    # Frame extraction — 10-15 fps is a sweet spot for iPhone clips: enough
    # temporal overlap for VGGT-SLAM's optical-flow keyframe selector, not so
    # dense that we waste compute on near-duplicate frames.
    fps: float = 12.0
    max_frames: int = 2000

    # VGGT-SLAM hyperparameters
    submap_size: int = 16
    max_loops: int = 1
    min_disparity: float = 50.0
    conf_threshold: float = 25.0
    lc_thres: float = 0.95

    # Boxer settings — `labels` is a comma-separated list of OWLv2 text prompts.
    labels: str = "chair,table,sofa,bed,monitor,keyboard,laptop,book,lamp,plant"
    thresh_2d: float = 0.25
    thresh_3d: float = 0.5
    max_semidense_points: int = 10000

    # Hardware
    device: str = field(default_factory=_detect_device)

    # Derived paths (populated after init)
    frames_dir: Path = field(init=False)
    vggt_slam_output_dir: Path = field(init=False)
    boxer_output_dir: Path = field(init=False)

    def __post_init__(self) -> None:
        self.video_path = Path(self.video_path)
        self.output_dir = Path(self.output_dir)
        self.frames_dir = self.output_dir / "frames"
        self.vggt_slam_output_dir = self.output_dir / "vggt_slam"
        self.boxer_output_dir = self.output_dir / "boxer"
