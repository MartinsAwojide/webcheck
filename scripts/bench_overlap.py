"""Overlap benchmark on the Jetson Orin Nano: how many slices a second the HMI's loop sustains with the display hand-off
(`on_slice`: picture snapshot, crops' sources, telemetry) run in the driver's loop ("sync") or on its own thread
(`hmi.AsyncSink`, "async"), everything else as the HMI runs it: the display process serving the HMI page, the defect
log written and published to MQTT, the gate's maps off. The line is replayed faster than the board can inspect, so the
queue stays full and the figure is the loop's capacity. Phases alternate (sync, async, sync, ...) to spread drift.

usage: python3 bench_overlap.py --pkg DIR --build-dir DIR --gate-dir DIR --det-dir DIR --engine-cache DIR
           --station A=DIR --gate-threshold T --lane-width-mm W --out report.json
           [--slices 300] [--rounds 3] [--preload 40] [--mqtt localhost]
Needs TensorRT and nvcc; runs on the board, not under pytest."""
import argparse
import json
import sys
import tempfile
import time
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
    ap.add_argument("--slices", type=int, default=300, help="slices per phase")
    ap.add_argument("--rounds", type=int, default=3, help="sync/async pairs")
    ap.add_argument("--preload", type=int, default=40)
    ap.add_argument("--mqtt", default="localhost")
    args = ap.parse_args()
    sys.path.insert(0, args.pkg)

    import tensorrt
    from webcheck.bench import detect_host, numeric_format, power_mode, read_temperatures
    from webcheck.inference.cascade import inspect_slice
    from webcheck.inference.defect_log import DefectLogWriter
    from webcheck.inference.display import Display
    from webcheck.inference.driver import ReplaySource, run_driver
    from webcheck.inference.engine import EngineCache
    from webcheck.inference.gate import GateConfig
    from webcheck.inference.gpu_resample import GpuGateInputs
    from webcheck.inference.hmi import AsyncSink, StageTimer, TimedGeometry, slice_telemetry, web_order_pairs
    from webcheck.inference.overlap import inspect_columns
    from webcheck.inference.scheduler import SchedulerConfig
    from webcheck.inference.trt import DetectorModel, GateModel, build_engine, tensorrt_version
    from webcheck.inference.uns import PublishingLog, UnsPublisher, UnsTopics, broker_address, connect_client

    placement, _, folder = args.station.partition("=")
    folder = Path(folder)
    rows = [json.loads(line) for line in open(folder / "manifest.jsonl") if line.strip()]
    last = max(r["md_offset_mm"] for r in rows)
    rows = [r for r in rows if r["md_offset_mm"] != last]
    pairs = web_order_pairs(rows)[:args.preload]
    load = lambda name: np.asarray(Image.open(folder / name).convert("RGB"))
    slices = [{"left": load(p["left"]), "right": load(p["right"])} for p in pairs]

    cache = EngineCache(args.engine_cache, tensorrt_version(), build_engine)
    gate = GateModel(cache.get(str(Path(args.gate_dir, "trained.onnx"))), cache.get(str(Path(args.gate_dir, "trained_head.onnx"))))
    det = DetectorModel(cache.get(str(Path(args.det_dir, "trained.onnx"))))
    cfg = GateConfig(threshold=args.gate_threshold)
    columns = inspect_columns("right", TimedGeometry(args.lane_width_mm, 123.0, datetime.now(timezone.utc)))
    options = {"gate_inputs": GpuGateInputs(args.build_dir),
               "camera_options": {"right": {"columns": columns, "gate_inputs": GpuGateInputs(args.build_dir, columns)}}}

    class Discard:
        def write(self, record):
            pass

    display = Display("127.0.0.1", 0, hmi=True)
    display.start()
    display.setup((placement,), 123.0, "left")
    display.set_gate_threshold(cfg.threshold)
    publisher = None
    if args.mqtt != "none":
        host, port = broker_address(args.mqtt)
        publisher = UnsPublisher(connect_client(host, port, f"webcheck-bench-overlap-{placement}"),
                                 UnsTopics("ENERGY_UP_LLC", "TEXAS_FACILITY"), placement)

    geometry = TimedGeometry(args.lane_width_mm, 123.0, datetime.now(timezone.utc))
    for i in range(3):                                                    # warm-up, not timed
        inspect_slice(slices[0], i, gate, det, cfg, geometry, Discard(), **options)

    results = []
    with tempfile.TemporaryDirectory() as tmp:
        for phase, mode in enumerate(["sync", "async"] * args.rounds):
            # faster than the board can go: the queue stays full, so the loop runs at its capacity
            geometry = TimedGeometry(args.lane_width_mm, 2000.0, datetime.now(timezone.utc), roll_frames=len(slices))
            timer = StageTimer()
            tg, td = timer.wrap("gate", gate), timer.wrap("detector", det)
            timings, shown = {}, {"n": 0, "records": 0}

            def inspect(frames, k, geometry=geometry, timer=timer, tg=tg, td=td, timings=timings):
                timer.take()
                outcome = inspect_slice(frames, k, tg, td, cfg, geometry, Discard(), **options, anomaly_maps=display.heat)
                timings[k] = timer.take()
                return outcome

            def on_slice(k, frames, outcome, geometry=geometry, timings=timings, shown=shown):
                shown["n"] += 1
                shown["records"] += len(outcome.records)
                mqtt = None if publisher is None else (publisher.sent, publisher.failed)
                display.sink(k, frames, outcome, telemetry=slice_telemetry(k, geometry, k + 1, 0.0, timings.pop(k, {}),
                                                                           None, outcome, shown["records"], mqtt))

            sink = AsyncSink(on_slice) if mode == "async" else on_slice
            config = SchedulerConfig(1e6, 4 * sum(f.nbytes for f in slices[0].values()), "block")
            temp_start = max(read_temperatures().values(), default=None)
            with DefectLogWriter(Path(tmp) / f"log_{phase}.jsonl") as file_log:
                log = file_log if publisher is None else PublishingLog(file_log, publisher)
                began = time.perf_counter()
                report = run_driver(ReplaySource(slices, geometry, args.slices), inspect, log, config, on_slice=sink)
                if mode == "async":
                    sink.close()
                wall = time.perf_counter() - began
            row = {"phase": phase, "mode": mode, "slices": report.written, "shown": shown["n"], "wall_s": round(wall, 2),
                   "slices_per_s": round(report.written / wall, 3), "ms_per_slice": round(1000 * wall / report.written, 1),
                   "inspect_ms_mean": round(1000 * float(np.mean(report.service_s)), 1), "alarms": len(report.alarms),
                   "temp_c_start": temp_start, "temp_c_end": max(read_temperatures().values(), default=None)}
            results.append(row)
            print(json.dumps(row), flush=True)
    display.close()
    if publisher is not None:
        publisher.close()
    by_mode = {m: [r["slices_per_s"] for r in results if r["mode"] == m] for m in ("sync", "async")}
    summary = {m: {"median_slices_per_s": float(np.median(v)), "runs": v} for m, v in by_mode.items()}
    summary["gain_pct"] = round(100 * (summary["async"]["median_slices_per_s"] / summary["sync"]["median_slices_per_s"] - 1), 1)
    out = {"what": "HMI loop capacity, display hand-off in the loop (sync) vs on its own thread (async)",
           "rung": "R4", "measured": True, "host": str(detect_host()), "power_mode": power_mode(),
           "numeric_format": numeric_format(tensorrt), "tensorrt": tensorrt_version(), "placement": placement,
           "gate_threshold": cfg.threshold, "mqtt": args.mqtt, "preload": len(slices), "results": results, "summary": summary,
           "date": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")}
    json.dump(out, open(args.out, "w"), indent=1)
    print(json.dumps(summary), flush=True)


if __name__ == "__main__":
    main()
