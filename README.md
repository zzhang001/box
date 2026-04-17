# VGGT-SLAM + Boxer Pipeline

**iPhone video to 3D scene graph — no LiDAR, no ROS bag, no Cartographer.**

Turn a casual iPhone video into a set of gravity-aligned metric 3D bounding boxes for every object in the scene. Designed for robotics scene understanding, spatial AI, and offline 3D mapping.

---

## Why This Exists

[Boxer](https://github.com/facebookresearch/boxer) produces state-of-the-art 3D object detection, but expects **camera poses, intrinsics, and (optionally) depth** as input. At work, those come from a ROS bag with Cartographer poses. For personal iPhone footage, we need a different front-end.

[VGGT-SLAM 2.0](https://github.com/MIT-SPARK/VGGT-SLAM) (MIT-SPARK) extends [VGGT](https://github.com/facebookresearch/vggt) (CVPR 2025 Best Paper) with optical-flow keyframe selection, submap processing, and SL(4) pose-graph optimization + loop closure. That means it scales past VGGT's ~32-frame chunk limit to full 30-second+ videos with globally consistent trajectories.

This pipeline connects them: **video in → VGGT-SLAM → Boxer → 3D scene graph out.**

```
iPhone (no LiDAR needed)       Mac Mini / Apple Silicon / CUDA box
─────────────────────          ──────────────────────────────────
                                ┌────────────────────────────────┐
  Record video ──transfer──►    │ 1. ffmpeg: extract frames @12fps│
                                │                                 │
                                │ 2. VGGT-SLAM 2.0                │
                                │    ├─ optical-flow keyframes    │
                                │    ├─ VGGT per-submap inference │
                                │    ├─ SL(4) pose-graph opt.     │
                                │    ├─ loop closure (DINO+SALAD) │
                                │    └─ globally consistent poses │
                                │                                 │
                                │ 3. Format conversion            │
                                │    VGGT-SLAM → Boxer formats    │
                                │                                 │
                                │ 4. Boxer                        │
                                │    ├─ 2D detection (OWLv2)      │
                                │    ├─ 3D box lifting (BoxerNet) │
                                │    └─ tracking/fusion           │
                                │                                 │
                                │ 5. Export                       │
                                │    └─ scene_graph.json          │
                                └────────────────────────────────┘
```


---

## Architecture

### Data Flow & Tensor Shapes

```
Frames on disk (frame_%05d.jpg)  — written by ffmpeg at configured fps
         │
         ▼
┌─────────────────────────────────────────────────────┐
│  VGGT-SLAM 2.0  (extern/vggt_slam)                  │
│                                                     │
│  Per frame, decide: keyframe? (optical-flow disparity)
│     │                                               │
│     ▼                                               │
│  Accumulate keyframes into submaps (default 16)     │
│     │                                               │
│     ▼                                               │
│  VGGT (MIT-SPARK fork) per submap → pose_enc,       │
│      depth, depth_conf, world_points                │
│     │                                               │
│     ▼                                               │
│  Image-retrieval (DINO+SALAD) → loop-closure edges  │
│     │                                               │
│     ▼                                               │
│  Pose-graph on SL(4) manifold → globally consistent │
│      homographies; decompose into K, R, t per frame │
│                                                     │
│  Exposed per keyframe S (via _extract_output):      │
│    extrinsic       [S, 3, 4] camera_from_world (SE3)│
│    intrinsic       [S, 3, 3] pinhole K              │
│    world_points    [S, H, W, 3] in SL(4) world frame│
│    world_points_conf [S, H, W]                      │
│    image_names     [S] source paths                 │
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
| VGGT-SLAM pose-graph homographies | SL(4) projective | SE(3) per-frame | `decompose_camera` (done inside `submap.get_all_poses_world`) |
| Front-end extrinsic → Boxer pose | `camera_from_world` [3,4] | `T_world_camera` [4,4] | Invert: `R.T`, `-R.T @ t` |
| Front-end intrinsic → Boxer camera | Pinhole `K` [3,3] | `CameraTW` struct | Extract `fx,fy,cx,cy` + image size |
| Front-end points → Boxer depth | Dense `[S,H,W,3]` | Semi-dense `[N,3]` | Confidence filter + uniform subsample |
| Gravity extraction | VGGT first-submap frame (OpenCV Y-down) | Unit vector `[3]` | `[0, 1, 0]` (MVP; see Known Limitations) |

### Known Limitations (VGGT-SLAM front-end)

- **Non-Euclidean world frame.** SL(4) optimization can include a small scale/shear drift relative to a strict metric world. Boxer box sizes may inherit that drift. Plan: add post-hoc Euclidean rectification via plane/gravity priors.
- **Gravity is assumed**, not estimated. If the iPhone starts tilted, Z-up won't hold exactly. Plan: pull gravity from iPhone IMU (CMMotionManager) or estimate via RANSAC floor plane on the fused point cloud.
- **CUDA hard-wired upstream.** `loop_closure.py` has `device = 'cuda'` at module scope; our wrapper patches it at import time (`_patch_vggt_slam_for_device`). `solver.run_predictions` also calls `torch.cuda.get_device_capability()` which will need upstream attention for MPS paths.

---

## Project Structure

```
box/
├── README.md
├── pyproject.toml
├── setup.sh                    # One-command install (chains VGGT-SLAM's setup.sh)
│
├── extern/
│   ├── vggt_slam/              # submodule → MIT-SPARK/VGGT-SLAM 2.0
│   └── boxer/                  # submodule → facebookresearch/boxer
│
├── pipeline/
│   ├── __init__.py
│   ├── config.py               # PipelineConfig dataclass
│   ├── extract_frames.py       # ffmpeg: video → frames
│   ├── run_vggt_slam.py        # VGGT-SLAM 2.0 inference wrapper
│   ├── convert.py              # VGGT-SLAM output → Boxer input format
│   ├── run_boxer.py            # Boxer inference wrapper
│   ├── export.py               # 3D boxes → scene_graph.json
│   └── run.py                  # End-to-end orchestrator
│
├── test/                       # Local iPhone footage (gitignored — never committed)
│
└── examples/
    └── README.md               # How to test with sample data
```

---

## Prerequisites

- **Python 3.11** (preferred) or 3.12 — VGGT-SLAM's upstream pins 3.11
- **Mac Apple Silicon** or **NVIDIA GPU** (Mac mini works for short clips; CUDA is still fastest)
- **ffmpeg** installed (`brew install ffmpeg`)
- ~15 GB disk for model weights (VGGT ~5GB + DINO/SALAD ~1GB + Boxer ~5GB + PE/SAM3 if installed)
- 16 GB+ unified memory (Mac) or 12 GB+ VRAM (CUDA)

---

## Installation

```bash
# Clone with submodules
git clone --recursive https://github.com/zzhang001/box.git
cd box

# One-command setup (creates venv, chains into extern/vggt_slam/setup.sh)
./setup.sh
```

`setup.sh` delegates to VGGT-SLAM's own `setup.sh`, which clones + editable-installs salad, the MIT-SPARK VGGT fork, perception-encoder, and SAM3 into `extern/vggt_slam/third_party/`. That keeps our install in lock-step with upstream.

---

## Usage

### Full Pipeline (video → scene graph)

```bash
# Basic: 12 fps extraction, auto device
python -m pipeline.run --video test/IMG_6826.MOV --output output/

# Tune for a 30-sec / 1000-frame iPhone clip on Mac mini
python -m pipeline.run \
    --video test/IMG_6826.MOV \
    --output output/ \
    --fps 12 \                     # Extract 12 frames/sec
    --max-frames 2000 \            # Safety cap
    --submap-size 16 \             # VGGT submap chunk
    --max-loops 1 \                # Max loop closures per submap
    --labels "chair,table,sofa" \  # Custom Boxer labels
    --device cpu                   # Mac mini: no CUDA; CPU bfloat16 works
```

### Step-by-Step

```bash
# 1. Extract frames
python -m pipeline.extract_frames --video input.mov --output output/frames/ --fps 12

# 2. Run VGGT-SLAM
python -m pipeline.run_vggt_slam --frames output/frames/ --output output/vggt_slam/

# 3. Run Boxer
python -m pipeline.run_boxer \
    --frames output/frames/ \
    --vggt-output output/vggt_slam/ \
    --output output/boxer/ \
    --labels "chair,table,desk,monitor,keyboard"

# 4. Export scene graph
python -m pipeline.export --boxer-output output/boxer/ --output scene_graph.json
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

| Platform | Device | VGGT-SLAM | Boxer | Notes |
|----------|--------|-----------|-------|-------|
| macOS 15 | Mac mini (Apple Silicon) | CPU bfloat16 | CPU/MPS | Dev target for iPhone clips; expect ~minutes for a 30-sec clip |
| macOS 15 | M3 Max 36GB | MPS / CPU | MPS | Headroom for longer footage |
| Linux | RTX 4090 | CUDA | CUDA | Fastest; matches upstream VGGT-SLAM benchmarks |

---

## How It Compares

| Approach | LiDAR? | Global Consistency | Platform | Quality |
|----------|--------|--------------------|----------|---------|
| **This pipeline (VGGT-SLAM → Boxer)** | No | Yes (SL(4) + loop closure) | Mac/Linux | High |
| ROS bag + Cartographer → Boxer | Depends | Yes | Robot rig | Production; needs bags |
| Boxer3D (iOS app) | Yes | Yes | iPhone Pro only | Highest |
| COLMAP + Boxer | No | Yes | Mac/Linux | High, but slow SfM |

---

## Roadmap

- [x] VGGT-SLAM 2.0 wrapper → Boxer-compatible output
- [ ] End-to-end validation on iPhone test clip (`test/IMG_6826.MOV`)
- [ ] Gravity estimation from iPhone IMU / floor-plane RANSAC
- [ ] Post-hoc Euclidean rectification of SL(4)-warped world
- [ ] Full Boxer wiring (OWLv2 + BoxerNet inference + fusion)
- [ ] Interactive 3D visualization (rerun.io or viser)
- [ ] Streaming mode
- [ ] ROS2 bridge

---

## License

This pipeline code is MIT licensed. Submodules have their own licenses:
- **VGGT-SLAM**: see [extern/vggt_slam/LICENSE](extern/vggt_slam/LICENSE)
- **Boxer**: CC-BY-NC (see [extern/boxer/LICENSE](extern/boxer/LICENSE))

---

## Acknowledgments

- [VGGT-SLAM](https://github.com/MIT-SPARK/VGGT-SLAM) — Dominic Maggio et al., MIT-SPARK Lab (2.0, 2026), built on [VGGT](https://github.com/facebookresearch/vggt) (Jianyuan Wang et al., CVPR 2025 Best Paper)
- [Boxer](https://github.com/facebookresearch/boxer) — Daniel DeTone et al., Meta Reality Labs
