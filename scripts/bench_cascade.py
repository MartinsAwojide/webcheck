"""Benchmark on the Jetson Orin Nano: the cascade's steady state (gate only), worst case (YOLO-Pro on every
inspected tile) and as-shipped throughput, through the benchmark rig (webcheck.bench), with the defect log published to
MQTT as the demo ships it. Slices are replayed back to back, not paced, so the figure is how fast the board inspects.

usage: python3 bench_cascade.py --pkg DIR --build-dir DIR --gate-dir DIR --det-dir DIR --engine-cache DIR
           --station A=DIR --gate-threshold T --lane-width-mm W --max-temp-c C --max-load L --out report.json
           [--slices-per-block 100] [--preload 60] [--repetitions 5] [--warmups 1] [--mqtt localhost] [--modes ...]
The station DIR holds the test frames and manifest.jsonl (the test set, timing only, not scored); the first --preload
positions in web order are decoded into RAM before timing. Needs TensorRT and nvcc; runs on the board, not under pytest."""
import argparse
import json
import platform
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from PIL import Image


def main():
    ap = argparse.ArgumentParser()
    for name in ("pkg", "build-dir", "gate-dir", "det-dir", "engine-cache", "station", "out"):
        ap.add_argument(f"--{name}", required=True)
    ap.add_argument("--gate-threshold", type=float, required=True)
    ap.add_argument("--lane-width-mm", type=float, required=True)
    ap.add_argument("--max-temp-c", type=float, required=True, help="readiness: hottest thermal zone allowed at the start")
    ap.add_argument("--max-load", type=float, required=True, help="readiness: 1-minute load average allowed at the start")
    ap.add_argument("--wait-ready-s", type=float, default=600.0, help="how long to wait for the board to settle")
    ap.add_argument("--slices-per-block", type=int, default=100)
    ap.add_argument("--preload", type=int, default=60)
    ap.add_argument("--repetitions", type=int, default=5)
    ap.add_argument("--warmups", type=int, default=1)
    ap.add_argument("--mqtt", default="localhost", help="broker for the defect log, HOST[:PORT], or none")
    ap.add_argument("--modes", nargs="+", default=["steady_state", "worst_case", "as_shipped"])
    args = ap.parse_args()
    sys.path.insert(0, args.pkg)

    import tensorrt
    from webcheck.bench import (Configuration, check_ready, detect_host, numeric_format, power_mode, read_load,
                                read_temperatures, run_benchmark, wait_until_ready, write_report)
    from webcheck.inference.cascade import inspect_slice
    from webcheck.inference.cascade_bench import mode_config, slice_block
    from webcheck.inference.defect_log import LineGeometry
    from webcheck.inference.engine import EngineCache
    from webcheck.inference.gpu_resample import GpuGateInputs
    from webcheck.inference.hmi import web_order_pairs
    from webcheck.inference.overlap import inspect_columns
    from webcheck.inference.trt import DetectorModel, GateModel, build_engine, tensorrt_version

    placement, _, folder = args.station.partition("=")
    folder = Path(folder)
    rows = [json.loads(line) for line in open(folder / "manifest.jsonl") if line.strip()]
    last = max(r["md_offset_mm"] for r in rows)
    rows = [r for r in rows if r["md_offset_mm"] != last]          # the test set's last position has one camera only
    pairs = web_order_pairs(rows)[:args.preload]
    load = lambda name: np.asarray(Image.open(folder / name).convert("RGB"))
    slices = [{"left": load(p["left"]), "right": load(p["right"])} for p in pairs]   # no PNG decoding in the timing

    model_paths = {"gate_backbone": Path(args.gate_dir, "trained.onnx"), "gate_head": Path(args.gate_dir, "trained_head.onnx"),
                   "detector": Path(args.det_dir, "trained.onnx")}
    cache = EngineCache(args.engine_cache, tensorrt_version(), build_engine)
    gate = GateModel(cache.get(str(model_paths["gate_backbone"])), cache.get(str(model_paths["gate_head"])))
    det = DetectorModel(cache.get(str(model_paths["detector"])))
    geometry = LineGeometry(args.lane_width_mm, 123.0, datetime.now(timezone.utc))
    columns = inspect_columns("right", geometry)
    options = {"gate_inputs": GpuGateInputs(args.build_dir),
               "camera_options": {"right": {"columns": columns, "gate_inputs": GpuGateInputs(args.build_dir, columns)}}}

    class Discard:
        def write(self, record):
            pass

    log, publisher = Discard(), None
    if args.mqtt != "none":
        from webcheck.inference.uns import PublishingLog, UnsPublisher, UnsTopics, broker_address, connect_client
        host, port = broker_address(args.mqtt)
        publisher = UnsPublisher(connect_client(host, port, f"webcheck-bench-{placement}"),
                                 UnsTopics("ENERGY_UP_LLC", "TEXAS_FACILITY"), placement)
        log = PublishingLog(Discard(), publisher)

    limits = {"max_temp_c": args.max_temp_c, "max_load": args.max_load}
    check = lambda: check_ready(read_temperatures(), read_load(), **limits)
    reader = lambda: {"temperatures_c": read_temperatures(), "load_1min": read_load()}
    host = detect_host()
    results = []
    for mode in args.modes:
        config = mode_config(mode, args.gate_threshold)
        counts = {"slices": 0, "tiles": 0, "flagged": 0, "records": 0}

        def inspect(frames, k, config=config, counts=counts):
            outcome = inspect_slice(frames, k, gate, det, config, geometry, Discard(), **options)
            counts["slices"] += 1
            counts["tiles"] += sum(len(r.tiles) for r in outcome.results.values())
            counts["flagged"] += sum(t.flagged for r in outcome.results.values() for t in r.tiles)
            counts["records"] += len(outcome.records)
            return outcome

        configuration = Configuration.build(
            host=host, numeric_format=numeric_format(tensorrt), runtime=f"TensorRT {tensorrt.__version__}",
            models={k: str(v) for k, v in model_paths.items()},
            input_shape={"frame": [1080, 1920], "tile": [640, 640], "tiles_per_frame": 6, "gate_input": [320, 320],
                         "cameras": 2, "right_camera_columns": list(columns)},
            power_mode=power_mode(), versions={"python": platform.python_version(), "tensorrt_engine": tensorrt_version()},
            extras={"mode": mode, "gate_threshold_used": config.threshold, "shipped_gate_threshold": args.gate_threshold,
                    "mqtt": args.mqtt, "placement": placement, "frames": f"{folder} (test set, timing only), "
                    f"first {len(slices)} positions in web order, cycled", "slices_per_block": args.slices_per_block,
                    "readiness_limits": limits, "paced": False})
        print(f"[{mode}] waiting for the board to be ready ({limits})", file=sys.stderr, flush=True)
        wait_until_ready(check, timeout_s=args.wait_ready_s)
        result = run_benchmark(slice_block(slices, inspect, log, args.slices_per_block), name=f"cascade {mode}",
                               configuration=configuration, readiness=check, reader=reader,
                               repetitions=args.repetitions, warmups=args.warmups)
        result.configuration.extras["counts_incl_warmup"] = dict(counts)
        results.append(result)
        print(json.dumps({"mode": mode, "ms_per_slice": result.ms_per_item, "slices_per_s": result.items_per_s,
                          "counts_incl_warmup": counts}), flush=True)
        write_report(results, args.out)
    if publisher is not None:
        print(json.dumps({"mqtt_sent": publisher.sent, "mqtt_failed": publisher.failed}), flush=True)
        publisher.close()


if __name__ == "__main__":
    main()
