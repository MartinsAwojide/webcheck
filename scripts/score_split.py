"""Scoring a split on the Jetson Orin Nano: replay one split of one placement through the shipped cascade (gate threshold as given,
right-camera column gating, defect log written as the demo writes it), then check the log against the split's ground
truth with webcheck.inference.log_inspector. Each web position is inspected once, as frame index md_offset_mm / frame
length, so the log's web positions line up with the annotations' web_extent_mm. Positions that do not have both cameras'
frames in --frames are skipped and only defects seen in replayed frames count.

usage: python3 score_split.py --pkg DIR --build-dir DIR --gate-dir DIR --det-dir DIR --engine-cache DIR --placement A
           --frames DIR --manifest manifest.jsonl --coco _annotations.coco.json [--prefix A_] --gate-threshold T
           --lane-width-mm W --log out.jsonl --out report.json
Needs TensorRT and nvcc; runs on the board, not under pytest."""
import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from PIL import Image


def main():
    ap = argparse.ArgumentParser()
    for name in ("pkg", "build-dir", "gate-dir", "det-dir", "engine-cache", "placement", "frames", "manifest", "coco", "log", "out"):
        ap.add_argument(f"--{name}", required=True)
    ap.add_argument("--prefix", default="", help="added to the manifest's file names to find the frames")
    ap.add_argument("--gate-threshold", type=float, required=True)
    ap.add_argument("--lane-width-mm", type=float, required=True)
    args = ap.parse_args()
    sys.path.insert(0, args.pkg)

    from webcheck.inference.cascade import inspect_slice
    from webcheck.inference.defect_log import DefectLogWriter, LineGeometry
    from webcheck.inference.engine import EngineCache
    from webcheck.inference.gate import GateConfig
    from webcheck.inference.gpu_resample import GpuGateInputs
    from webcheck.inference.log_inspector import ground_truth, inspect_log
    from webcheck.inference.overlap import inspect_columns
    from webcheck.inference.trt import DetectorModel, GateModel, build_engine, tensorrt_version

    frames_dir = Path(args.frames)
    rows = [json.loads(line) for line in open(args.manifest) if line.strip()]
    at = {}
    for row in rows:
        at.setdefault(row["md_offset_mm"], {})[row["camera"]] = row["file_name"]
    geometry = LineGeometry(args.lane_width_mm, 123.0, datetime(2026, 1, 1, tzinfo=timezone.utc))
    pitch = geometry.frame_pitch_mm
    positions, skipped = [], 0
    for md in sorted(at):
        names = at[md]
        if set(names) == {"left", "right"} and all((frames_dir / (args.prefix + n)).is_file() for n in names.values()):
            k = round(md / pitch)
            assert abs(k * pitch - md) < 1e-6, f"position {md} mm is not a whole number of frame lengths"
            positions.append((k, names))
        else:
            skipped += 1

    cache = EngineCache(args.engine_cache, tensorrt_version(), build_engine)
    gate = GateModel(cache.get(str(Path(args.gate_dir, "trained.onnx"))), cache.get(str(Path(args.gate_dir, "trained_head.onnx"))))
    det = DetectorModel(cache.get(str(Path(args.det_dir, "trained.onnx"))))
    config = GateConfig(threshold=args.gate_threshold)
    columns = inspect_columns("right", geometry)
    options = {"gate_inputs": GpuGateInputs(args.build_dir),
               "camera_options": {"right": {"columns": columns, "gate_inputs": GpuGateInputs(args.build_dir, columns)}}}
    load = lambda name: np.asarray(Image.open(frames_dir / (args.prefix + name)).convert("RGB"))

    Path(args.log).unlink(missing_ok=True)
    replayed = set()
    with DefectLogWriter(args.log) as log:
        for k, names in positions:
            inspect_slice({c: load(n) for c, n in names.items()}, k, gate, det, config, geometry, log, **options)
            replayed.update(names.values())
    records = [json.loads(line) for line in open(args.log) if line.strip()]
    truth = ground_truth(json.load(open(args.coco)), replayed)
    result = inspect_log(records, truth)
    report = {"placement": args.placement, "manifest": args.manifest, "frames": str(frames_dir), "positions_replayed": len(positions),
              "positions_skipped": skipped, "gate_threshold": args.gate_threshold, "right_camera_columns": list(columns),
              "tensorrt": tensorrt_version(), "summary": result.summary(),
              "missed": [g.__dict__ for g in result.missed],
              "false_alarms": result.false_alarms, "wrong_type": [{"record": r, "truth": g.__dict__} for r, g in result.wrong_type]}
    Path(args.out).write_text(json.dumps(report, indent=1) + "\n")
    s = result.summary()
    print(json.dumps({"placement": args.placement, "manifest": args.manifest, "positions": len(positions), "skipped": skipped,
                      "overall": s["overall"], "not_detectable": s["not_detectable"]}), flush=True)


if __name__ == "__main__":
    main()
