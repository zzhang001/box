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
                                │ 3. Boxer (per keyframe)         │
                                │    ├─ OWLv2 open-vocab 2D det.  │
                                │    ├─ BoxerNet 3D lift          │
                                │    │   (uses DINOv3 backbone)   │
                                │    └─ optional fusion/tracking  │
                                │                                 │
                                │ 4. Export                       │
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
│  pipeline/run_boxer.py (per keyframe)                │
│                                                     │
│  1. Load RGB, resize to 960×960                     │
│  2. Rotate world frame Y-down → Z-down              │
│     (all poses + world points by Rx(-π/2))          │
│  3. Build Boxer datum dict:                         │
│       img0, cam0 (CameraTW), T_world_rig0 (PoseTW), │
│       sdp_w (semi-dense points, conf-filtered)      │
│  4. OWLv2: text prompts → bb2d (x1, x2, y1, y2)     │
│  5. BoxerNet: datum + bb2d → 3D OBBs in world frame │
└────────────┬────────────────────────────────────────┘
             │
             ▼
┌─────────────────────────────────────────────────────┐
│  pipeline/export.py                                  │
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
| VGGT-SLAM pose-graph homographies | SL(4) projective | SE(3) per-frame | `decompose_camera` (inside `submap.get_all_poses_world`) |
| VGGT-SLAM extrinsic → Boxer pose | `camera_from_world` [3,4] | `T_world_camera` [4,4] | Invert: `R.T`, `-R.T @ t` |
| VGGT-SLAM intrinsic → Boxer camera | Pinhole `K` [3,3] at VGGT's 294×518 | `CameraTW` at 960×960 | Scale `fx,cx` by `960/518`, `fy,cy` by `960/294` |
| VGGT-SLAM points → Boxer SDP | Dense `[S,H,W,3]` | Semi-dense `[N,3]` | Confidence-percentile filter + random subsample + NaN pad |
| World-frame alignment | VGGT world (OpenCV Y-down) | Boxer world (Z-down) | `Rx(-π/2)` on all poses + world points |

### Known Limitations (VGGT-SLAM front-end)

- **Non-Euclidean world frame.** SL(4) optimization can include a small scale/shear drift relative to a strict metric world. Boxer box sizes may inherit that drift. Plan: add post-hoc Euclidean rectification via plane/gravity priors.
- **Gravity is assumed**, not estimated. If the iPhone starts tilted, Z-up won't hold exactly. Plan: pull gravity from iPhone IMU (CMMotionManager) or estimate via RANSAC floor plane on the fused point cloud.
- **CUDA hard-wired upstream.** `loop_closure.py` has `device = 'cuda'` at module scope; our wrapper patches it at import time (`_patch_vggt_slam_for_device`). `solver.run_predictions` also calls `torch.cuda.get_device_capability()` which will need upstream attention for MPS paths.

---

## Inside Boxer: the Three Models

Boxer's inference pipeline runs **three separate neural nets** doing three distinct jobs. Understanding what each does, where its weights live, and how they hand data off makes `extern/boxer/run_boxer.py` readable and explains why our adapter (`pipeline/run_boxer.py`) looks the way it does.

| Model | Role | Input | Output | Weights |
|-------|------|-------|--------|---------|
| **OWLv2** | Open-vocabulary 2D detector | RGB image + text prompts (e.g. `chair,table`) | 2D boxes `[M, 4]` + scores + label indices | `extern/boxer/ckpts/owlv2-base-patch16-ensemble.pt` (121 MB) |
| **DINOv3** | Vision foundation model (frozen backbone) | RGB image | Dense per-patch feature map | `extern/boxer/ckpts/dinov3_vits16plus_pretrain_lvd1689m-4057cbaa.pth` (115 MB) |
| **BoxerNet** | Category-agnostic 3D lifter | Image + 2D boxes + pose + K + semi-dense world points | 3D oriented bounding boxes (ObbTW `[M, 165]`) | `extern/boxer/ckpts/boxernet_hw960in4x6d768-wssxpf9p.ckpt` (400 MB) |

**Key insight:** DINOv3 is *not* a detector — it's a pure image encoder (frozen weights from Meta's self-supervised ViT-S/16+). BoxerNet calls it internally and never trains it. OWLv2 is an independent external model; its output is just data flowing into BoxerNet.

### End-to-end flow (per keyframe)

```
                       ┌─────────────────────────────────────────────┐
Text labels ────────┐  │  OWLv2  (open-vocabulary 2D detection)      │
  "chair, table,    │──►│  • CLIP-style text encoder (run once)        │
   sofa, ..."       │  │  • ViT vision encoder                       │
                       │  • Class + box heads                        │
                       │  Output: bb2d [M, 4] in (x1,x2,y1,y2) +     │
                       │          scores [M] + label_ints [M]         │
                       └──────────────────┬──────────────────────────┘
                                          │ bb2d, labels
                                          ▼
RGB frame [1,3,H,W] ─────┬────────────► ┌─────────────────────────────────┐
Pose T_world_cam ────────┤              │            BoxerNet              │
Intrinsics K (CameraTW) ─┤              │                                  │
Semi-dense world pts ────┤              │  1. DINOv3 backbone              │
                         │              │     image → [num_patches, dim]   │
                         │              │     (frozen ViT-S/16+)           │
                         │              │                                  │
                         │              │  2. Project world points into    │
                         │              │     DINOv3 feature grid (via K   │
                         │              │     and pose) → per-patch depth  │
                         │              │                                  │
                         │              │  3. For each bb2d, pool patch    │
                         │              │     features inside box → query  │
                         │              │                                  │
                         │              │  4. Cross-attend query against   │
                         │              │     scene tokens (img patches +  │
                         │              │     depth patches)               │
                         │              │                                  │
                         │              │  5. Regression heads:            │
                         │              │     • center in voxel frame      │
                         │              │     • size (w, h, d)             │
                         │              │     • yaw around gravity axis    │
                         │              │     • log-variance (uncertainty) │
                         │              │                                  │
                         │              │  6. Voxel → world coords using   │
                         │              │     T_world_voxel (gravity-      │
                         │              │     aligned from pose)           │
                         │              │                                  │
                         │              │  Output: obbs_pr_w [1, M, 165]   │
                         │              │    (ObbTW: bb3_object, T_world_  │
                         │              │    object, prob, sem_id, text)   │
                         │              └─────────────────┬────────────────┘
                                                          │
                   ┌──────────────────────────────────────┤
                   │                                      │
                   ▼                                      ▼
        Write CSV per frame                 Optional: offline fusion
        (utils/file_io.ObbCsvWriter2)       (utils/fuse_3d_boxes.py)
                                            or online tracking
                                            (utils/track_3d_boxes.py)
```

### What each model contributes

**OWLv2 — the "what and roughly where in 2D"**
- **Open-vocabulary**: you change the category list per run without retraining. Each text prompt is encoded once at init via a CLIP text encoder and cached; at inference only image features + cached text embeddings are consumed.
- **Output convention gotcha**: `OwlWrapper.forward` returns `bb2d` in `(x1, x2, y1, y2)` order — **not** the standard `(x1, y1, x2, y2)`. BoxerNet expects the same Boxer-internal order. `extern/boxer/run_boxer.py:558` reformats to standard order only when writing out a viewer-friendly CSV.
- **Alternatives**: Boxer also accepts `--gt2d` (ground-truth boxes from dataset annotations) or `--cache2d` (replay detections from a previous CSV). These bypass OWLv2.

**DINOv3 — the "visual features" inside BoxerNet**
- Loaded once inside `BoxerNet.__init__` as `self.dino = DinoV3Wrapper("dinov3_vits16plus")`.
- Vision-only; no text, no queries — just image in, feature map out.
- Self-supervised on 1.7 B unlabeled images (`lvd1689m` in the weight filename).
- Frozen weights. BoxerNet training only updates the heads and attention layers on top of DINOv3 features.
- Why DINOv3 over OWL's internal CLIP ViT? DINOv3 features are denser and geometrically richer; OWL's ViT is tuned for language-grounded classification, not spatial reasoning. Using two different ViTs for two different jobs is intentional.

**BoxerNet — the "lift 2D to metric 3D"**
- Category-agnostic. It doesn't know "chair" from "sofa"; it just gets a 2D box and learns how boxes look in 3D given camera context.
- Two crucial scene inputs fuse the geometry:
  1. **Camera pose + intrinsics** (`T_world_rig0`, `cam0`) → lets BoxerNet place predictions in world coords.
  2. **Semi-dense world points** (`sdp_w`) → a cloud of known 3D points visible in the frame. BoxerNet projects these into the DINOv3 feature grid so each image patch gets a rough depth prior. This is the single biggest signal that makes monocular 3D boxes metrically plausible.
- The output is gravity-aligned: yaw is around the gravity axis; the 6-DoF pose of each box is split into gravity-aligned rotation + 3D center + axis-aligned half-extents in the object frame.
- Outputs aleatoric uncertainty (log-variance) per box, used downstream for fusion weighting.

### Where the three plug into our pipeline

Our VGGT-SLAM output supplies exactly the geometric inputs BoxerNet needs:

| BoxerNet input | Source in VGGT-SLAM output |
|----------------|----------------------------|
| `img0` RGB | `VGGTSLAMOutput.image_names[i]` loaded from disk |
| `cam0` CameraTW | Built from `VGGTSLAMOutput.intrinsic[i]` (pinhole K, scaled to 960×960) |
| `T_world_rig0` PoseTW | Invert `VGGTSLAMOutput.extrinsic[i]`, then `Rx(-π/2)` for Z-down gravity |
| `sdp_w` | `VGGTSLAMOutput.world_points[i]` flattened, rotated, confidence-filtered |
| `bb2d` | Computed on-the-fly by OWLv2 per frame |
| `rotated0` | `False` for iPhone landscape clips |
| `time_ns0` | Derived from frame index |

All three checkpoints land in `extern/boxer/ckpts/` via `extern/boxer/scripts/download_ckpts.sh`. `BoxerNet.load_from_checkpoint` wires BoxerNet and DINOv3 together; OWLv2 is loaded independently by `OwlWrapper`.

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
│       └── ckpts/              # boxernet + dinov3 + owlv2 weights
│
├── pipeline/
│   ├── __init__.py
│   ├── config.py               # PipelineConfig dataclass
│   ├── extract_frames.py       # ffmpeg: video → frames
│   ├── run_vggt_slam.py        # VGGT-SLAM 2.0 inference wrapper
│   ├── run_boxer.py            # Boxer (OWLv2 + BoxerNet) inference wrapper
│   ├── visualize.py            # VGGT-SLAM output → fused PLY
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
- ~15 GB disk for model weights (VGGT ~5 GB, DINO+SALAD ~1 GB, Boxer ~640 MB, optional PE/SAM3 larger)
- 16 GB+ unified memory (Mac) or 12 GB+ VRAM (CUDA)

---

## Installation

```bash
# Clone with submodules
git clone --recursive https://github.com/zzhang001/box.git
cd box

# One-command setup (creates venv, chains into extern/vggt_slam/setup.sh,
# then fetches Boxer checkpoints)
./setup.sh
(cd extern/boxer && bash scripts/download_ckpts.sh)
```

`setup.sh` delegates to VGGT-SLAM's own `setup.sh` for the SLAM stack (salad, MIT-SPARK VGGT fork, gtsam, perception-encoder, SAM3 installed into `extern/vggt_slam/third_party/`). Boxer's checkpoints (BoxerNet, DINOv3, OWLv2) are a separate download.

---

## Usage

### Full Pipeline (video → scene graph)

```bash
# Basic: 12 fps extraction, auto device, default Boxer labels
python -m pipeline.run --video test/IMG_6826.MOV --output output/

# Custom OWLv2 labels for a living-room scan
python -m pipeline.run \
    --video test/IMG_6826.MOV \
    --output output/ \
    --fps 12 \                                    # extract 12 frames/sec
    --submap-size 16 \                            # VGGT submap chunk
    --labels "chair,table,sofa,lamp,plant" \      # OWLv2 prompts
    --device cpu                                  # Mac mini: no CUDA
```

### Step-by-Step

```bash
# 1. Extract frames
python -m pipeline.extract_frames --video input.mov --output output/frames/ --fps 12

# 2. Run VGGT-SLAM
python -m pipeline.run_vggt_slam --frames output/frames/ --output output/vggt_slam/

# 3. Run Boxer (OWLv2 + BoxerNet) on the cached VGGT-SLAM output
python -m pipeline.run_boxer \
    --vggt-slam output/vggt_slam/ \
    --output output/boxer/ \
    --labels "chair,table,desk,monitor,keyboard"

# 4. Export scene graph JSON (optional; pipeline.run does this automatically)
python -m pipeline.export --boxer-output output/boxer/ --output scene_graph.json
```

### Visualizing the fused point cloud

```bash
# Produces output/vggt_slam/fused.ply — openable in MeshLab / CloudCompare / 3dviewer.net
python -m pipeline.visualize --input output/vggt_slam/ --output output/vggt_slam/fused.ply
```

### Output Format

```json
{
  "metadata": {
    "source_video": "living_room.mov",
    "num_frames": 223,
    "pipeline_version": "0.1.0",
    "timestamp": "2026-04-17T08:00:00Z"
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
      "frame_idx": 17,
      "time_ns": 17000000
    }
  ]
}
```

---

## Hardware Tested

| Platform | Device | VGGT-SLAM | Boxer | Notes |
|----------|--------|-----------|-------|-------|
| macOS 15 | Mac mini (Apple Silicon) | CPU fp32 | CPU / MPS | Dev target; full 30-sec clip in ~2 hrs including ~16 VGGT submaps |
| macOS 15 | M3 Max 36 GB | MPS / CPU | MPS | Headroom for longer footage |
| Linux | RTX 4090 | CUDA | CUDA | Fastest; matches upstream benchmarks |

We keep VGGT in **fp32 on CPU** (not bf16) because Apple Silicon NEON lacks hardware bf16 GEMM, and PyTorch CPU bf16 falls back to scalar code that's ~50× slower than fp32 (we verified this: same 5-frame submap took >13 min in bf16 vs 17 s in fp32). CUDA/MPS keep bf16.

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

- [x] VGGT-SLAM 2.0 wrapper on Mac mini (CPU fp32 validated end-to-end; 471 frames → 223 keyframes)
- [x] OWLv2 + BoxerNet wired in, per-keyframe 3D boxes
- [ ] End-to-end validation on `test/IMG_6826.MOV` with visual inspection
- [ ] Offline 3D-box fusion via `extern/boxer/utils/fuse_3d_boxes.py`
- [ ] Gravity estimation from iPhone IMU / floor-plane RANSAC (replace the fixed `Rx(-π/2)`)
- [ ] Post-hoc Euclidean rectification of SL(4)-warped world
- [ ] Fix VGGT-SLAM MPS device-mismatch for Apple Silicon acceleration
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
- [OWLv2](https://arxiv.org/abs/2306.09683) — Minderer et al., Google Research (NeurIPS 2023)
- [DINOv3](https://github.com/facebookresearch/dinov3) — Oquab et al., Meta FAIR
