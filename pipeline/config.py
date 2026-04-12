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
    """Configuration for the full VGGT -> Boxer pipeline."""

    # Input
    video_path: Path = Path("input.mov")
    output_dir: Path = Path("output")

    # Frame extraction
    fps: float = 1.0
    max_frames: int = 50

    # VGGT settings
    vggt_model: str = "facebook/VGGT-1B"
    vggt_image_size: int = 518

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
    vggt_output_dir: Path = field(init=False)
    boxer_output_dir: Path = field(init=False)

    def __post_init__(self) -> None:
        self.video_path = Path(self.video_path)
        self.output_dir = Path(self.output_dir)
        self.frames_dir = self.output_dir / "frames"
        self.vggt_output_dir = self.output_dir / "vggt"
        self.boxer_output_dir = self.output_dir / "boxer"
