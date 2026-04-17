# Examples

## Quick Test

Record a short video (10-40 seconds) of a room with your iPhone, transfer it to your Mac, then:

```bash
# Defaults: 12 fps extraction, submap size 16, auto device (MPS on Mac, CUDA on Linux)
python -m pipeline.run --video path/to/my_room.mov --output output/
```

Local iPhone footage lives in `test/` (gitignored). For example:

```bash
python -m pipeline.run --video test/IMG_6826.MOV --output test/out/ --device mps
```

## Sample Workflow

1. **Record**: Walk slowly through a room, keep the phone roughly upright, avoid fast rotations.
2. **Transfer**: AirDrop or USB to your Mac.
3. **Run**: `python -m pipeline.run --video room.mov --output output/`
4. **Inspect**: Open `output/scene_graph.json` for the 3D object list (Boxer stage — WIP).

## Tips for Best Results

- **Keep FPS ≥ 10**: Below that, VGGT-SLAM's optical-flow keyframe selector loses temporal baseline. 12 fps is a good default.
- **Steady movement**: Slow, smooth camera motion gives VGGT better depth estimates.
- **Overlap**: Ensure consecutive keyframes have >60% visual overlap — the keyframe selector drops frames whose optical-flow disparity is below `--min-disparity` (default 50).
- **Lighting**: Well-lit scenes produce better 2D detections.
- **Submap size**: 16 frames per submap balances VGGT memory with pose-graph granularity. Drop to 8-12 if you OOM on MPS.
- **Loop closures**: `--max-loops 1` (default) is enough for room-scale scans. For longer walks with revisits, the image-retrieval backbone (DINOv2 + SALAD) handles it automatically.

## Disk / Weight Downloads

First run downloads:
- VGGT-1B weights (~5 GB) → `~/.cache/torch/hub/checkpoints/`
- DINOv2 ViT-B/14 (~330 MB) → same cache
- `dino_salad.ckpt` (~335 MB, fetched by `setup.sh`)

Subsequent runs reuse the cache.
