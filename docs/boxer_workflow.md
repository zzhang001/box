# Boxer Workflow: DINOv3 + OWLv2 + BoxerNet

Inside Boxer's 3D-detection pipeline there are **three separate neural nets** doing three distinct jobs. Understanding what each does, where its weights live, and how they hand data off to each other is key to reading `extern/boxer/run_boxer.py` and to wiring VGGT-SLAM output into it.

## The three models in one sentence each

| Model | Role | Input | Output | Where it lives |
|-------|------|-------|--------|----------------|
| **OWLv2** | Open-vocabulary 2D detector | RGB image + text prompts (e.g. `chair,table`) | 2D boxes `[M, 4]` + scores + label indices | `extern/boxer/owl/owl_wrapper.py`, ckpt `owlv2-base-patch16-ensemble.pt` |
| **DINOv3** | Vision foundation model (frozen backbone) | RGB image | Dense per-patch feature map | `extern/boxer/boxernet/dinov3_wrapper.py`, ckpt `dinov3_vits16plus_pretrain_lvd1689m-4057cbaa.pth` |
| **BoxerNet** | Category-agnostic 3D lifter | Image + 2D boxes + pose + K + semi-dense world points | 3D oriented bounding boxes `[M, 165]` (ObbTW) | `extern/boxer/boxernet/boxernet.py`, ckpt `boxernet_hw960in4x6d768-wssxpf9p.ckpt` |

**Key insight:** DINOv3 is *not* a detector. It's a pure image encoder — frozen weights from Meta's self-supervised ViT. BoxerNet calls it internally to get image features and never trains it. OWLv2 is an independent external model; its output is just data flowing into BoxerNet.

## End-to-end flow (per keyframe)

```
                       ┌─────────────────────────────────────────────┐
Text labels ────────┐  │  OWLv2  (open-vocabulary 2D detection)      │
  "chair,table,     │──►│  • CLIP-style text encoder (run once)       │
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
Gravity vector ──────────┘              │     image → [num_patches, dim]   │
                                          │     (frozen ViT-S/16+)           │
                                          │                                  │
                                          │  2. Project world points into    │
                                          │     DINOv3 feature grid (via K   │
                                          │     and pose) → per-patch depth  │
                                          │                                  │
                                          │  3. For each bb2d, pool patch    │
                                          │     features inside box → query  │
                                          │                                  │
                                          │  4. Cross-attend query against   │
                                          │     scene tokens (img patches +  │
                                          │     depth patches)               │
                                          │                                  │
                                          │  5. Regression heads:            │
                                          │     • center in voxel frame      │
                                          │     • size (w, h, d)             │
                                          │     • yaw around gravity axis    │
                                          │     • log-variance (uncertainty) │
                                          │                                  │
                                          │  6. Voxel → world coords using   │
                                          │     T_world_voxel (gravity-      │
                                          │     aligned from pose)           │
                                          │                                  │
                                          │  Output: obbs_pr_w [1, M, 165]   │
                                          │    (ObbTW: bb3_object, T_world_  │
                                          │    object, prob, sem_id, text)   │
                                          └─────────────────┬────────────────┘
                                                            │
                    ┌───────────────────────────────────────┤
                    │                                       │
                    ▼                                       ▼
        Write CSV per frame                 Optional: offline fusion
        (utils/file_io.ObbCsvWriter2)        (utils/fuse_3d_boxes.py)
                                             or online tracking
                                             (utils/track_3d_boxes.py)
```

## What each model contributes

### OWLv2 — the "what and roughly where in 2D"
- **Open-vocabulary**: you change the category list per run without retraining. Internally, each text prompt is encoded once at init via a CLIP text encoder and cached; at inference only image features + cached text embeddings are consumed.
- **Output convention gotcha**: `OwlWrapper.forward` returns `bb2d` in `(x1, x2, y1, y2)` order — **not** the standard `(x1, y1, x2, y2)`. BoxerNet expects this same Boxer-internal order. `extern/boxer/run_boxer.py:558` reformats to standard order only when writing out a viewer-friendly CSV.
- **Replacement options**: Boxer also accepts `--gt2d` (use ground-truth boxes from dataset annotations) or `--cache2d` (replay detections from a previous CSV). These bypass OWLv2 entirely.

### DINOv3 — the "visual features" inside BoxerNet
- Loaded once inside `BoxerNet.__init__` as `self.dino = DinoV3Wrapper("dinov3_vits16plus")`.
- Vision-only; no text, no queries — just image in, feature map out.
- Self-supervised on 1.7 B unlabeled images (`lvd1689m` in the weight filename).
- Frozen weights. BoxerNet training only updates the heads and attention layers on top of DINOv3 features.
- Why DINOv3 over OWL's internal CLIP ViT? DINOv3 features are denser and geometrically richer; OWL's ViT is tuned for language-grounded classification, not spatial reasoning. Using two different ViTs for two different jobs is intentional.

### BoxerNet — the "lift 2D to metric 3D"
- Category-agnostic. It doesn't know "chair" from "sofa"; it just gets a 2D box and learns how boxes look in 3D given camera context.
- Two crucial scene inputs fuse the geometry:
  1. **Camera pose + intrinsics** (`T_world_rig0`, `cam0`) → lets BoxerNet place predictions in world coords.
  2. **Semi-dense world points** (`sdp_w`) → a cloud of known 3D points visible in the frame. BoxerNet projects these into the DINOv3 feature grid so each image patch gets a rough depth prior. This is the single biggest signal that makes monocular 3D boxes metrically plausible.
- The output is gravity-aligned: yaw is around the gravity axis; the 6-DoF pose of each box is split into gravity-aligned rotation + 3D center + axis-aligned half-extents in the object frame.
- Outputs aleatoric uncertainty (log-variance) per box, used downstream for fusion weighting.

## Where the three plug into our pipeline

Our VGGT-SLAM output supplies exactly the geometric inputs BoxerNet needs:

| BoxerNet input | Source in VGGT-SLAM output |
|----------------|----------------------------|
| `img0` RGB | `VGGTSLAMOutput.image_names[i]` loaded from disk |
| `cam0` CameraTW | Built from `VGGTSLAMOutput.intrinsic[i]` (pinhole K) |
| `T_world_rig0` PoseTW | Invert `VGGTSLAMOutput.extrinsic[i]` (camera_from_world → world_from_camera) |
| `sdp_w` | Flatten `VGGTSLAMOutput.world_points[i]` with `world_points_conf` filter |
| `bb2d` | Computed on-the-fly by OWLv2 per frame |
| `rotated0` | `False` for iPhone landscape clips |
| `time_ns0` | Derived from frame index |

**World-frame gravity adjustment:** VGGT-SLAM's world frame inherits VGGT's OpenCV camera convention (Y-axis down = gravity direction). BoxerNet's `gravity_align_T_world_cam` defaults to Z-down (`[0, 0, -1]`). We pre-rotate all poses and world points by `Rx(-π/2)` so our world becomes Z-down, letting BoxerNet's auto-`T_world_voxel0` computation work without further overrides.

## Checkpoint summary

All three checkpoints are downloaded by `extern/boxer/scripts/download_ckpts.sh` from `huggingface.co/facebook/boxer` into `extern/boxer/ckpts/`:

- `boxernet_hw960in4x6d768-wssxpf9p.ckpt` — 400 MB, BoxerNet weights
- `dinov3_vits16plus_pretrain_lvd1689m-4057cbaa.pth` — 115 MB, DINOv3 backbone
- `owlv2-base-patch16-ensemble.pt` — 121 MB, OWLv2 detector (Meta's repackaging of Google's OWLv2 without HuggingFace transformers dependency)

`BoxerNet.load_from_checkpoint` wires them together: the checkpoint ships with a `cfg` that points BoxerNet at the DINOv3 file; OWLv2 is loaded independently by `OwlWrapper`.
