"""Export Boxer 3D detection results to scene graph JSON."""

import json
from datetime import datetime, timezone
from pathlib import Path

from pipeline.run_boxer import BoxerOutput


def export_scene_graph(
    boxer_output: BoxerOutput,
    output_path: Path,
    source_video: str = "",
    num_frames: int = 0,
) -> Path:
    """
    Export Boxer results as a structured scene graph JSON.

    Args:
        boxer_output: Output from Boxer inference.
        output_path: Path to write the JSON file.
        source_video: Name of the source video file.
        num_frames: Number of frames processed.

    Returns:
        Path to the written JSON file.
    """
    scene_graph = {
        "metadata": {
            "source_video": source_video,
            "num_frames": num_frames,
            "pipeline_version": "0.1.0",
            "timestamp": datetime.now(timezone.utc).isoformat(),
        },
        "coordinate_system": {
            "type": "gravity_aligned_metric",
            "up_axis": "Z",
            "units": "meters",
            "convention": "VGGT OpenCV world frame, gravity = +Y",
        },
        "objects": [
            {
                "id": i,
                "label": obj.label,
                "center": list(obj.center),
                "size": list(obj.size),
                "yaw": obj.yaw,
                "confidence": obj.confidence,
                "uncertainty": obj.uncertainty,
                "first_seen_frame": obj.first_seen_frame,
                "last_seen_frame": obj.last_seen_frame,
            }
            for i, obj in enumerate(boxer_output.objects)
        ],
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(scene_graph, f, indent=2)

    print(f"Exported {len(boxer_output.objects)} objects to {output_path}")
    return output_path


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Export scene graph JSON")
    parser.add_argument("--boxer-output", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("scene_graph.json"))
    args = parser.parse_args()

    # Load boxer output (placeholder for now)
    boxer_output = BoxerOutput(objects=[], raw_boxes_per_frame={})
    export_scene_graph(boxer_output, args.output)
