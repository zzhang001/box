"""Run OWLv2 on VGGT-SLAM keyframes and save per-frame 2D detections as JSON.

Output JSON keyed by `time_ns` (matches boxer_3dbbs.csv). Each entry is a
list of `{bb2d_xyxy, score, label}` where `bb2d_xyxy` is in the coordinate
system of the 960×960 image we fed OWL (the same image BoxerNet saw).

Downstream viewers rescale those boxes to the original frame resolution
for display on native (non-stretched) iPhone frames.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
import torch

from pipeline._common import detect_device
from pipeline.run_vggt_slam import load_vggt_slam_output

_REPO_ROOT = Path(__file__).resolve().parent.parent
_BOXER_DIR = _REPO_ROOT / "extern" / "boxer"
if str(_BOXER_DIR) not in sys.path:
    sys.path.insert(0, str(_BOXER_DIR))


def run_owl_on_keyframes(
    vggt_slam_dir: Path,
    output_json: Path,
    labels: list[str],
    *,
    device: Optional[str] = None,
    image_resize: int = 960,
    thresh_2d: float = 0.25,
    max_frames: Optional[int] = None,
) -> None:
    from owl.owl_wrapper import OwlWrapper

    if device is None:
        device = detect_device()
    print(f"[owl] device={device}")

    vggt_out = load_vggt_slam_output(Path(vggt_slam_dir))
    n_keyframes = vggt_out.extrinsic.shape[0]
    if max_frames is not None:
        n_keyframes = min(n_keyframes, max_frames)

    print(f"[owl] loading OWLv2 with {len(labels)} labels")
    owl = OwlWrapper(
        device=device,
        text_prompts=labels,
        min_confidence=thresh_2d,
        precision="float32" if device in ("cpu", "mps") else None,
    )

    results: dict[str, list[dict]] = {}
    # Pad-to-square metadata (same scheme as pipeline.run_boxer) so downstream
    # viewers can map 2D bboxes in [image_resize × image_resize] coords back
    # to native image coords.
    pad_info_per_frame: dict[str, dict] = {}
    t0 = time.time()
    for i in range(n_keyframes):
        img_path = vggt_out.image_names[i]
        img_bgr = cv2.imread(img_path)
        if img_bgr is None:
            continue
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        orig_h, orig_w = img_rgb.shape[:2]
        side = max(orig_h, orig_w)
        pad_top = (side - orig_h) // 2
        pad_bottom = side - orig_h - pad_top
        pad_left = (side - orig_w) // 2
        pad_right = side - orig_w - pad_left
        img_square = cv2.copyMakeBorder(
            img_rgb, pad_top, pad_bottom, pad_left, pad_right,
            cv2.BORDER_CONSTANT, value=(0, 0, 0),
        )
        img_resized = cv2.resize(img_square, (image_resize, image_resize), interpolation=cv2.INTER_AREA)
        img_torch = torch.from_numpy(img_resized).permute(2, 0, 1).float()[None]  # [1,3,H,W] in [0,255]

        bb_boxer, scores, label_ints, _ = owl.forward(img_torch, rotated=False)
        time_ns = int(vggt_out.frame_ids[i] * 1_000_000)
        # bb_boxer is in Boxer's (x1, x2, y1, y2) convention — convert to standard xyxy.
        bb = bb_boxer[:, [0, 2, 1, 3]].cpu().numpy()
        entries = []
        for j in range(bb.shape[0]):
            entries.append({
                "bb2d_xyxy": [float(x) for x in bb[j].tolist()],
                "score": float(scores[j].item()),
                "label": labels[int(label_ints[j].item())],
            })
        results[str(time_ns)] = entries
        pad_info_per_frame[str(time_ns)] = {
            "orig_hw": [orig_h, orig_w],
            "side": side,
            "pad_top": pad_top, "pad_left": pad_left,
            "pad_bottom": pad_bottom, "pad_right": pad_right,
        }

        if (i + 1) % 10 == 0 or i == n_keyframes - 1:
            elapsed = time.time() - t0
            n_dets = sum(len(v) for v in results.values())
            print(f"[owl] {i+1}/{n_keyframes} frames — {elapsed:.1f}s, {n_dets} dets")

    output_json.parent.mkdir(parents=True, exist_ok=True)
    with open(output_json, "w") as f:
        json.dump({
            "image_resize": image_resize,
            "image_pad_to_square": True,
            "labels": labels,
            "thresh_2d": thresh_2d,
            "detections_by_time_ns": results,
            "pad_info_by_time_ns": pad_info_per_frame,
        }, f)
    print(f"[owl] wrote {output_json}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--vggt-slam", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True,
                        help="Output JSON path")
    parser.add_argument("--labels", type=str, required=True,
                        help="Comma-separated text prompts")
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--image-resize", type=int, default=960,
                        help="Match what BoxerNet saw (bb2d coords are in this space)")
    parser.add_argument("--thresh-2d", type=float, default=0.25)
    parser.add_argument("--max-frames", type=int, default=None)
    args = parser.parse_args()

    labels = [s.strip() for s in args.labels.split(",") if s.strip()]
    run_owl_on_keyframes(
        vggt_slam_dir=args.vggt_slam,
        output_json=args.output,
        labels=labels,
        device=args.device,
        image_resize=args.image_resize,
        thresh_2d=args.thresh_2d,
        max_frames=args.max_frames,
    )


if __name__ == "__main__":
    main()
