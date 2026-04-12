# Examples

## Quick Test

Record a short video (10-30 seconds) of a room with your iPhone, transfer it to your Mac, then:

```bash
python -m pipeline.run --video examples/my_room.mov --output examples/output/ --fps 2
```

## Sample Workflow

1. **Record**: Walk slowly through a room, keep the phone roughly upright, avoid fast rotations.
2. **Transfer**: AirDrop or USB to your Mac.
3. **Run**: `python -m pipeline.run --video room.mov --output output/`
4. **Inspect**: Open `output/scene_graph.json` for the 3D object list.

## Tips for Best Results

- **Steady movement**: Slow, smooth camera motion gives VGGT better depth estimates.
- **Overlap**: Ensure consecutive frames have >60% visual overlap.
- **Lighting**: Well-lit scenes produce better 2D detections.
- **FPS setting**: 1-2 fps is usually enough. More frames = better but slower.
- **Max frames**: 50 frames covers a typical room scan at 1 fps.
