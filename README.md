# VGGT + Boxer Pipeline

**iPhone video to 3D scene graph — no LiDAR required.**

Turn a casual iPhone video into a set of gravity-aligned, metric 3D bounding boxes for every object in the scene. Designed for robotics scene understanding, spatial AI, and offline 3D mapping.

---

## Why This Exists

[Boxer](https://github.com/facebookresearch/boxer) produces state-of-the-art 3D object detection, but expects camera poses, intrinsics, and (optionally) depth as input. [VGGT](https://github.com/facebookresearch/vggt) (CVPR 2025 Best Paper) predicts all of these from plain RGB frames — no LiDAR, no SfM preprocessing, no COLMAP.

This pipeline connects them: **video in, 3D scene graph out.**

```
iPhone Air (no LiDAR)          Mac (Apple Silicon / CUDA)
─────────────────────          ────────────────────────────
                                ┌─────────────────────────┐
  Record video ──transfer──►    │  1. ffmpeg: extract frames│
                                │                          │
                                │  2. VGGT (MPS/CUDA)     │
                                │     ├─ camera poses      │
                                │     ├─ intrinsics        │
                                │     ├─ depth maps        │
                                │     └─ 3D point cloud    │
                                │                          │
                                │  3. Format Conversion    │
                                │     VGGT → Boxer formats │
                                │                          │
                                │  4. Boxer (MPS/CUDA)     │
                                │     ├─ 2D detection (OWL)│
                                │     ├─ 3D box lifting    │
                                │     └─ tracking/fusion   │
                                │                          │
                                │  5. Export               │
                                │     └─ scene_graph.json  │
                                └─────────────────────────┘
```

---

## Architecture

### Data Flow & Tensor Shapes

```
Frames [S, 3, H, W]  (RGB, float32, [0,1])
         │
         ▼
┌─────────────────────────────────────────────────────┐
│  VGGT  (facebook/VGGT-1B)                           │
│                                                     │
│  Outputs:                                           │
│    pose_enc       [B, S, 9]   (absT_quaR_FoV)      │
│    extrinsic      [B, S, 3, 4] (camera_from_world)  │
│    intrinsic      [B, S, 3, 3] (pinhole K matrix)   │
│    depth          [B, S, H, W, 1] (metric depth)    │
│    world_points   [B, S, H, W, 3] (dense 3D)       │
│    depth_conf     [B, S, H, W]                      │
└────────────┬────────────────────────────────────────┘
             │
             ▼
┌─────────────────────────────────────────────────────┐
│  Format Conversion  (pipeline/convert.py)           │
│                                                     │
│  1. Invert extrinsic → T_world_camera (SE3)         │
│  2. Pinhole K → CameraTW struct                     │
│  3. Dense world_points → semi-dense [B, N, 3]       │
│  4. Extract gravity from pose (world Y-axis → down) │
└────────────┬────────────────────────────────────────┘
             │
             ▼
┌─────────────────────────────────────────────────────┐
│  Boxer  (facebookresearch/boxer)                    │
│                                                     │
│  Inputs per frame:                                  │
│    RGB image      [B, 3, H, W]                      │
│    CameraTW       (fx, fy, cx, cy, w, h)            │
│    PoseTW         [12] (R_3x3_flat + t_3)           │
│    gravity        [3]  (unit vector)                 │
│    semi-dense pts [B, N, 3] (optional, from VGGT)   │
│    2D boxes       [B, M, 4] (from OWLv2 detector)   │
│                                                     │
│  Outputs:                                           │
│    obbs_pr_params [B, M, 7] (center+size+yaw)       │
│    obbs_pr_logvar [B, M, 1] (uncertainty)           │
│    labels         per-box text label                 │
└────────────┬────────────────────────────────────────┘
             │
             ▼
┌─────────────────────────────────────────────────────┐
│  Export  (pipeline/export.py)                        │
│                                                     │
│  scene_graph.json:                                  │
│  {                                                  │
│    "objects": [                                      │
│      {                                              │
│        "label": "chair",                            │
│        "center": [x, y, z],                         │
│        "size": [w, h, d],                           │
│        "yaw": 1.23,                                 │
│        "confidence": 0.92,                          │
│        "uncertainty": 0.05                          │
│      }, ...                                         │
│    ],                                               │
│    "coordinate_system": "gravity_aligned_metric",   │
│    "up_axis": "Z"                                   │
│  }                                                  │
└─────────────────────────────────────────────────────┘
```

### Key Coordinate Conversions

| Step | From | To | Operation |
|------|------|----|-----------|
| VGGT extrinsic → Boxer pose | `camera_from_world` [3,4] | `T_world_camera` [4,4] | Invert: `R.T`, `-R.T @ t` |
| VGGT intrinsic → Boxer camera | Pinhole `K` [3,3] | `CameraTW` struct | Extract `fx,fy,cx,cy` + image size |
| VGGT points → Boxer depth | Dense `[S,H,W,3]` | Semi-dense `[N,3]` | Confidence filter + uniform subsample |
| Gravity extraction | VGGT world frame (OpenCV) | Unit vector `[3]` | `[0, -1, 0]` (Y-down = gravity in OpenCV) |

---

## Project Structure

```
vggt-boxer-pipeline/
├── README.md
├── pyproject.toml
├── setup.sh                    # One-command install
│
├── extern/
│   ├── vggt/                   # git submodule → facebookresearch/vggt
│   └── boxer/                  # git submodule → facebookresearch/boxer
│
├── pipeline/
│   ├── __init__.py
│   ├── config.py               # Pipeline configuration dataclass
│   ├── extract_frames.py       # ffmpeg: video → frames
│   ├── run_vggt.py             # VGGT inference wrapper
│   ├── convert.py              # VGGT output → Boxer input format
│   ├── run_boxer.py            # Boxer inference wrapper
│   ├── export.py               # 3D boxes → scene_graph.json
│   └── run.py                  # End-to-end orchestrator
│
├── scripts/
│   └── run_pipeline.sh         # Shell entrypoint
│
└── examples/
    └── README.md               # How to test with sample data
```

---

## Prerequisites

- **Python 3.12**
- **Mac Apple Silicon** (MPS) or **NVIDIA GPU** (CUDA)
- **ffmpeg** installed (`brew install ffmpeg`)
- ~10 GB disk for model weights (VGGT ~5GB + Boxer ~5GB)
- 16 GB+ unified memory (Mac) or 12 GB+ VRAM (CUDA)

---

## Installation

```bash
# Clone with submodules
git clone --recursive https://github.com/zzhang001/vggt-boxer-pipeline.git
cd vggt-boxer-pipeline

# One-command setup (creates venv, installs everything)
./setup.sh

# Or manual setup:
python3.12 -m venv .venv
source .venv/bin/activate

# Install VGGT
pip install -e extern/vggt

# Install Boxer
pip install -e extern/boxer

# Install this pipeline
pip install -e .
```

---

## Usage

### Full Pipeline (video → scene graph)

```bash
# Basic usage
python -m pipeline.run --video path/to/video.mov --output output/

# With custom settings
python -m pipeline.run \
    --video path/to/video.mov \
    --output output/ \
    --fps 2 \                    # Extract 2 frames/sec (default: 1)
    --max-frames 50 \            # Cap at 50 frames
    --labels "chair,table,sofa" \ # Custom object labels
    --track \                    # Enable temporal tracking
    --device mps                 # Force MPS (auto-detected by default)
```

### Step-by-Step

```bash
# 1. Extract frames
python -m pipeline.extract_frames --video input.mov --output frames/ --fps 2

# 2. Run VGGT
python -m pipeline.run_vggt --frames frames/ --output vggt_output/

# 3. Run Boxer (using VGGT outputs as depth source)
python -m pipeline.run_boxer \
    --frames frames/ \
    --vggt-output vggt_output/ \
    --output boxer_output/ \
    --labels "chair,table,desk,monitor,keyboard"

# 4. Export scene graph
python -m pipeline.export --boxer-output boxer_output/ --output scene_graph.json
```

### Output Format

```json
{
  "metadata": {
    "source_video": "living_room.mov",
    "num_frames": 30,
    "pipeline_version": "0.1.0",
    "timestamp": "2026-04-11T12:00:00Z"
  },
  "coordinate_system": {
    "type": "gravity_aligned_metric",
    "up_axis": "Z",
    "units": "meters"
  },
  "objects": [
    {
      "id": 0,
      "label": "chair",
      "center": [1.23, -0.45, 0.42],
      "size": [0.55, 0.55, 0.88],
      "yaw": 1.57,
      "confidence": 0.92,
      "uncertainty": 0.05,
      "first_seen_frame": 3,
      "last_seen_frame": 28
    }
  ]
}
```

---

## Hardware Tested

| Platform | Device | VGGT | Boxer | Notes |
|----------|--------|------|-------|-------|
| macOS 15 | M3 Max 36GB | MPS | MPS | Primary dev target |
| macOS 15 | M1 Pro 16GB | MPS | MPS | Works, slower on large scenes |
| Linux | RTX 4090 | CUDA | CUDA | Fastest |

---

## How It Compares

| Approach | LiDAR? | Realtime? | Platform | Quality |
|----------|--------|-----------|----------|---------|
| **This pipeline** | No | Offline | Mac/Linux | High (VGGT depth) |
| Boxer3D (iOS app) | Yes | Yes | iPhone Pro only | Highest (true depth) |
| COLMAP + Boxer | No | Offline | Mac/Linux | High, but slow SfM |
| NeRF/3DGS + manual | No | Offline | GPU required | Scene-level only |

---

## Roadmap

- [ ] Core pipeline: video → VGGT → Boxer → JSON
- [ ] Batch processing for long videos (chunked VGGT inference)
- [ ] iPhone metadata extraction (gyroscope → gravity, focal length → intrinsics)
- [ ] Interactive 3D visualization (Open3D / rerun.io)
- [ ] Streaming mode: process frames as they arrive
- [ ] Integration with ROS2 for robot navigation

---

## License

This pipeline code is MIT licensed. Note that the submodules have their own licenses:
- **VGGT**: See [extern/vggt/LICENSE](extern/vggt/LICENSE)
- **Boxer**: CC-BY-NC (see [extern/boxer/LICENSE](extern/boxer/LICENSE))

---

## Acknowledgments

- [VGGT](https://github.com/facebookresearch/vggt) — Jianyuan Wang et al., CVPR 2025 Best Paper
- [Boxer](https://github.com/facebookresearch/boxer) — Daniel DeTone et al., Meta Reality Labs
