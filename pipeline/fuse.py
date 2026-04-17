"""Offline fuse per-frame Boxer 3D boxes into canonical per-object boxes.

Wraps `extern/boxer/utils/fuse_3d_boxes.fuse_obbs_from_csv`:
  1. Read `boxer_3dbbs.csv` written by pipeline.run_boxer.
  2. Cluster per-frame OBBs by 3D IoU graph + connected components.
  3. Drop clusters with fewer than `min_detections` frames.
  4. Produce one confidence/variance-weighted OBB per cluster.
  5. Write a fused CSV + a scene_graph.json built from the fused OBBs.

See README's "Post-processing: fusion vs. tracking" section for the
motivation and how this differs from online tracking.
"""

from __future__ import annotations

import csv
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

_REPO_ROOT = Path(__file__).resolve().parent.parent
_BOXER_DIR = _REPO_ROOT / "extern" / "boxer"
if str(_BOXER_DIR) not in sys.path:
    sys.path.insert(0, str(_BOXER_DIR))


def fuse_csv(
    input_csv: Path,
    output_csv: Optional[Path] = None,
    scene_graph_path: Optional[Path] = None,
    *,
    iou_threshold: float = 0.3,
    min_detections: int = 4,
    conf_threshold: float = 0.55,
) -> tuple[Path, list[dict]]:
    """Fuse per-frame CSV → per-object CSV + scene graph.

    Returns (fused_csv_path, objects_list).
    """
    from utils.fuse_3d_boxes import fuse_obbs_from_csv

    input_csv = Path(input_csv)
    if output_csv is None:
        output_csv = input_csv.with_name(input_csv.stem + "_fused.csv")
    output_csv = Path(output_csv)

    # Boxer's fuse_obbs_from_csv handles both the IoU-clustering and CSV write.
    fuse_obbs_from_csv(
        str(input_csv),
        str(output_csv),
        iou_threshold=iou_threshold,
        min_detections=min_detections,
        conf_threshold=conf_threshold,
    )

    # Read the fused CSV back so we can build a scene graph out of it.
    objects = _read_fused_csv(output_csv)

    if scene_graph_path is None:
        scene_graph_path = output_csv.with_name("scene_graph_fused.json")
    scene_graph_path = Path(scene_graph_path)

    scene_graph = {
        "metadata": {
            "source_csv": str(input_csv),
            "num_objects": len(objects),
            "fusion_params": {
                "iou_threshold": iou_threshold,
                "min_detections": min_detections,
                "conf_threshold": conf_threshold,
            },
            "pipeline_version": "0.1.0",
            "timestamp": datetime.now(timezone.utc).isoformat(),
        },
        "coordinate_system": {
            "type": "gravity_aligned_metric",
            "up_axis": "Z",
            "units": "meters",
            "convention": "Boxer world frame (Z-down gravity), rotated from "
                          "VGGT-SLAM Y-down by estimated floor-plane normal",
        },
        "objects": objects,
    }
    scene_graph_path.parent.mkdir(parents=True, exist_ok=True)
    with open(scene_graph_path, "w") as f:
        json.dump(scene_graph, f, indent=2)

    print(f"[fuse] wrote {len(objects)} fused objects → {scene_graph_path}")
    return output_csv, objects


def _read_fused_csv(path: Path) -> list[dict]:
    """Parse Boxer's fused CSV into a list of scene-graph object dicts."""
    objects: list[dict] = []
    with open(path) as f:
        reader = csv.DictReader(f)
        for i, row in enumerate(reader):
            objects.append({
                "id": i,
                "label": row.get("name", ""),
                "center": [
                    float(row["tx_world_object"]),
                    float(row["ty_world_object"]),
                    float(row["tz_world_object"]),
                ],
                "size": [
                    float(row["scale_x"]),
                    float(row["scale_y"]),
                    float(row["scale_z"]),
                ],
                "orientation_quat": [
                    float(row["qx_world_object"]),
                    float(row["qy_world_object"]),
                    float(row["qz_world_object"]),
                    float(row["qw_world_object"]),
                ],
                "confidence": float(row.get("prob", 0.0)),
                "num_detections": int(row.get("num_detections", 0) or 0),
                "sem_id": int(row.get("sem_id", 0) or 0),
            })
    return objects


def _print_summary(objects: list[dict]) -> None:
    from collections import Counter
    c = Counter(o["label"] for o in objects)
    print(f"\n[fuse] {len(objects)} fused objects:")
    for label, n in c.most_common():
        print(f"    {label:20s} {n}")


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Fuse per-frame 3D boxes into canonical objects")
    parser.add_argument("--input", type=Path, required=True,
                        help="Path to boxer_3dbbs.csv from pipeline.run_boxer")
    parser.add_argument("--output-csv", type=Path, default=None,
                        help="Fused CSV path (default: <input>_fused.csv)")
    parser.add_argument("--scene-graph", type=Path, default=None,
                        help="scene_graph_fused.json path (default: next to output CSV)")
    parser.add_argument("--iou-threshold", type=float, default=0.3)
    parser.add_argument("--min-detections", type=int, default=4)
    parser.add_argument("--conf-threshold", type=float, default=0.55)
    args = parser.parse_args()

    _, objects = fuse_csv(
        input_csv=args.input,
        output_csv=args.output_csv,
        scene_graph_path=args.scene_graph,
        iou_threshold=args.iou_threshold,
        min_detections=args.min_detections,
        conf_threshold=args.conf_threshold,
    )
    _print_summary(objects)


if __name__ == "__main__":
    main()
