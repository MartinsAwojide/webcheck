"""Demo driver on the Jetson Orin Nano: replays real two-camera slices at a chosen line speed through the scheduler, the
gate, YOLO-Pro, the camera-overlap de-duplication and the defect log (webcheck.inference.driver).

usage: python3 demo_driver.py --pkg DIR --build-dir DIR --gate-dir DIR --det-dir DIR --frames-dir DIR --pairs slice_pairs.json
           --engine-cache DIR --out out.json --slices 120 --ceiling-s S --max-queue-bytes N --gate-threshold T --lane-width-mm W
           SPEED [SPEED ...]
With --serve-port, the annotated picture and the defect panel are served as a page on that port (open
http://<board>:<port> in Chrome on the TV). --raw-camera NAME=/dev/videoN (repeatable) adds a live camera to the page as a raw
picture-in-picture, never inspected; the slices still come from the stored frames. SPEED is the belt speed in m/min; it sets both the arrival schedule and the logged timestamps. Every run writes a defect
log next to --out (<out stem>_<speed>.jsonl). Needs TensorRT and nvcc; runs on the board, not under pytest.

--hmi --station A=DIR [--station B=DIR] serves the HMI page. Each DIR holds one placement's test frames and its
manifest.jsonl (the test set, for viewing only). The page opens on a splash screen where the operator picks the station,
the line speed (default --start-speed, 123 m/min) and the first camera; only then is the roll loaded and the line started.
The roll plays in web order and loops; the page has a speed slider (1-370 m/min), pause, step and a camera switch.
--frames-dir, --pairs, --preload and SPEED are not used. Positions missing from a DIR end its roll at the first gap.
Timings taken while someone moves the controls are not benchmarks.
--gate NAME=DIR:THRESHOLD (repeatable, HMI only) offers named gates on the splash screen in place of --gate-dir and
--gate-threshold; the first is the default, and the chosen gate's engines are loaded once the line starts.

--mqtt HOST[:PORT] also publishes each logged record to UNS topics on that broker
(<enterprise>/<site>/Printing_Lamination/Line_1/Placement_<A|B>/defect, webcheck.inference.uns); the log file is still
written first. Without --hmi it needs --placement. Needs paho-mqtt."""
import argparse
import json
import random
import socket
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from PIL import Image


def main():
    ap = argparse.ArgumentParser()
    for name in ("pkg", "build-dir", "engine-cache", "out"):
        ap.add_argument(f"--{name}", required=True)
    ap.add_argument("--det-dir", help="the detector's ONNX folder (without --detector)")
    ap.add_argument("--detector", action="append", default=[], metavar="NAME=DIR",
                    help="a YOLO-Pro size the operator can pick on the splash screen, repeatable; the first is the default (HMI)")
    ap.add_argument("--gate-dir", help="the gate's ONNX folder (without --gate)")
    ap.add_argument("--gate", action="append", default=[], metavar="NAME=DIR:THRESHOLD",
                    help="a gate the operator can pick on the splash screen, repeatable; the first is the default (HMI)")
    ap.add_argument("--frames-dir", help="stored frames (replay without --hmi)")
    ap.add_argument("--pairs", help="slice_pairs.json (replay without --hmi)")
    ap.add_argument("--slices", type=int, required=True)
    ap.add_argument("--ceiling-s", type=float, required=True, help="worst-case latency ceiling, seconds")
    ap.add_argument("--max-queue-bytes", type=int, required=True)
    ap.add_argument("--gate-threshold", type=float, help="anomaly-score threshold of the --gate-dir gate")
    ap.add_argument("--lane-width-mm", type=float, required=True, help="SKU lane width")
    ap.add_argument("--on-overload", choices=("block", "fault"), default="block")
    ap.add_argument("--read-workers", type=int, default=3,
                    help="HMI: threads decoding the test frames ahead of the line (one decodes about 8 slices/s on the board)")
    ap.add_argument("--preload", type=int, default=60, help="distinct slices held in RAM and cycled")
    ap.add_argument("--no-gating", action="store_true", help="inspect every tile of the right camera too")
    ap.add_argument("--backend", choices=("tensorrt", "onnxruntime"),
                    help="the model runtime (HMI): TensorRT where it is installed, ONNX Runtime otherwise")
    ap.add_argument("--gpu", choices=("auto", "direct", "webgpu", "none"), default="auto",
                    help="ONNX Runtime's GPU route: direct (CUDA or DirectML), webgpu (Vulkan, Metal), none (CPU); "
                         "auto takes direct, then WebGPU, then the CPU")
    ap.add_argument("--source", choices=("replay",), default="replay", help="where the slices come from (the USB cameras come after the camera spike)")
    ap.add_argument("--raw-camera", action="append", default=[], metavar="NAME=DEVICE",
                    help="show a live USB camera as a raw view on the page, not inspected (repeatable, e.g. left=/dev/video0 right=/dev/video2)")
    ap.add_argument("--raw-size", default="1280x720", help="raw camera picture size, WIDTHxHEIGHT (MJPEG)")
    ap.add_argument("--raw-fps", type=int, default=None, help="raw camera frame rate (default: a rate the camera offers)")
    ap.add_argument("--serve-port", type=int, help="serve the annotated page on this port (0 picks a free one)")
    ap.add_argument("--host", default="0.0.0.0", help="address the page is served on")
    ap.add_argument("--hmi", action="store_true", help="serve the HMI page; the operator picks the station on its splash screen")
    ap.add_argument("--station", action="append", default=[], metavar="NAME=DIR",
                    help="a placement's test frames and manifest.jsonl for the HMI (repeatable, e.g. A=../test_frames/A_1080)")
    ap.add_argument("--start-speed", type=float, default=123.0, help="the splash screen's default line speed, m/min (HMI)")
    ap.add_argument("--start-view", choices=("left", "right"), default="left", help="the splash screen's default camera (HMI)")
    ap.add_argument("--mqtt", metavar="HOST[:PORT]", help="also publish the defect log to UNS topics on this MQTT broker")
    ap.add_argument("--uns-enterprise", default="ENERGY_UP_LLC", help="UNS enterprise level (the converter company)")
    ap.add_argument("--uns-site", default="TEXAS_FACILITY", help="UNS site level (no spaces: the defect map shows _ as a space)")
    ap.add_argument("--placement", choices=("A", "B"), help="the placement the replayed slices come from (with --mqtt, without --hmi)")
    ap.add_argument("speeds", type=float, nargs="*")
    args = ap.parse_args()
    if args.hmi and (not args.station or args.serve_port is None):
        ap.error("--hmi needs at least one --station and --serve-port")
    if not args.hmi and (args.pairs is None or args.frames_dir is None or not args.speeds):
        ap.error("--pairs, --frames-dir and at least one SPEED are needed without --hmi")
    if args.mqtt and not args.hmi and args.placement is None:
        ap.error("--mqtt without --hmi needs --placement")
    if args.mqtt and args.hmi and any(s.partition("=")[0] not in ("A", "B") for s in args.station):
        ap.error("--mqtt needs every --station to be named A or B (its UNS work cell)")
    if args.detector and not args.hmi:
        ap.error("--detector is the splash screen's choice; without --hmi use --det-dir")
    if bool(args.detector) == bool(args.det_dir):
        ap.error("use --detector (HMI) or --det-dir, one of them")
    if args.gate and not args.hmi:
        ap.error("--gate is the splash screen's choice; without --hmi use --gate-dir and --gate-threshold")
    if args.gate and (args.gate_dir is not None or args.gate_threshold is not None):
        ap.error("use --gate, or --gate-dir with --gate-threshold, not both")
    if not args.gate and (args.gate_dir is None or args.gate_threshold is None):
        ap.error("--gate-dir and --gate-threshold are needed without --gate")
    sys.path.insert(0, args.pkg)
    if args.hmi:
        return run_hmi(args)
    publisher = None if args.mqtt is None else uns_publisher(args, args.placement)

    from webcheck.inference.camera import camera_specs
    from webcheck.inference.cascade import inspect_slice
    from webcheck.inference.defect_log import DefectLogWriter, LineGeometry
    from webcheck.inference.display import Display
    from webcheck.inference.driver import ReplaySource, run_driver
    from webcheck.inference.engine import EngineCache
    from webcheck.inference.gate import GateConfig
    from webcheck.inference.gpu_resample import GpuGateInputs
    from webcheck.inference.overlap import inspect_columns
    from webcheck.inference.scheduler import SchedulerConfig
    from webcheck.inference.trt import DetectorModel, GateModel, build_engine, tensorrt_version

    cache = EngineCache(args.engine_cache, tensorrt_version(), build_engine)
    gate = GateModel(cache.get(str(Path(args.gate_dir, "trained.onnx"))), cache.get(str(Path(args.gate_dir, "trained_head.onnx"))))
    det = DetectorModel(cache.get(str(Path(args.det_dir, "trained.onnx"))))
    cfg = GateConfig(threshold=args.gate_threshold)
    lane_width_mm = args.lane_width_mm
    probe = LineGeometry(lane_width_mm, args.speeds[0], datetime.now(timezone.utc))
    gpu = GpuGateInputs(args.build_dir)
    options = {"gate_inputs": gpu}
    if not args.no_gating:
        columns = inspect_columns("right", probe)
        options["camera_options"] = {"right": {"columns": columns, "gate_inputs": GpuGateInputs(args.build_dir, columns)}}

    pairs = [p for ps in json.load(open(args.pairs)).values() for p in ps]
    random.Random(1).shuffle(pairs)
    load = lambda n: np.asarray(Image.open(Path(args.frames_dir, n)).convert("RGB"))
    slices = [{"left": load(p["left"]), "right": load(p["right"])} for p in pairs[:args.preload]]   # no PNG decoding in the timing

    class Discard:
        def write(self, record):
            pass

    display = None
    if args.raw_camera and args.serve_port is None:
        ap.error("--raw-camera needs --serve-port (the raw view is part of the page)")
    if args.serve_port is not None:
        cameras = None
        if args.raw_camera:
            width, height = (int(v) for v in args.raw_size.lower().split("x"))
            cameras = camera_specs(args.raw_camera, width, height, args.raw_fps)
        display = Display(args.host, args.serve_port, cameras=cameras)
        display.start()
        print(f"page: http://{socket.gethostname()}.local:{display.port}  (or the board's IP)", file=sys.stderr, flush=True)

    results = []
    for speed in args.speeds:
        def make_inspect(geometry):
            def inspect(frames, k):
                return inspect_slice(frames, k, gate, det, cfg, geometry, Discard(), **options)   # the driver logs result.records
            return inspect

        warm = make_inspect(LineGeometry(lane_width_mm, speed, datetime.now(timezone.utc)))
        for i in range(3):                                                  # warm-up, not logged
            warm(slices[i], i)
        geometry = LineGeometry(lane_width_mm, speed, datetime.now(timezone.utc))      # frame 0 is captured now
        inspect = make_inspect(geometry)
        out = Path(args.out)
        log_path = out.with_name(f"{out.stem}_{speed:g}.jsonl")
        log_path.unlink(missing_ok=True)
        config = SchedulerConfig(args.ceiling_s, args.max_queue_bytes, args.on_overload)
        def status(t, m, speed=speed):
            line = f"[{speed:g} m/min] t={t:6.1f}s {m['state']:<10} queue={m['queue_depth']:>3} oldest={m['oldest_age_s']:.2f}s written={m['written']}"
            print(line, file=sys.stderr, flush=True)
            if display is not None:
                display.status(f"{speed:g} m/min   {m['state']}   queue {m['queue_depth']}   slices logged {m['written']}")

        with DefectLogWriter(log_path) as log:
            report = run_driver(ReplaySource(slices, geometry, args.slices), inspect, publishing(log, publisher), config,
                                on_status=status, on_slice=display.sink if display is not None else None)
        res = {"speed_m_per_min": speed, "offered_slices_per_s": speed / 60.0 * 1000.0 / geometry.frame_pitch_mm, "gating": not args.no_gating,
               "ceiling_s": args.ceiling_s, "log_lines": sum(1 for _ in open(log_path)), **mqtt_counts(publisher),
               **report.summary(args.ceiling_s)}
        print(json.dumps(res), flush=True)
        results.append(res)
    json.dump({"tensorrt": tensorrt_version(), "results": results}, open(args.out, "w"), indent=1)
    if display is not None:
        display.close()
    close_publisher(publisher)


def uns_publisher(args, placement):
    """A UNS publisher for one placement on the --mqtt broker; it connects in the background."""
    from webcheck.inference.uns import UnsPublisher, UnsTopics, broker_address, connect_client

    topics = UnsTopics(args.uns_enterprise, args.uns_site)
    host, port = broker_address(args.mqtt)
    client = connect_client(host, port, f"webcheck-{socket.gethostname()}-{placement}")
    print(f"mqtt: {host}:{port}, topics {topics.defect(placement)}[/...]", file=sys.stderr, flush=True)
    return UnsPublisher(client, topics, placement)


def publishing(log, publisher):
    from webcheck.inference.uns import PublishingLog

    return log if publisher is None else PublishingLog(log, publisher)


def mqtt_counts(publisher):
    return {} if publisher is None else {"mqtt_sent": publisher.sent, "mqtt_failed": publisher.failed}


def close_publisher(publisher):
    if publisher is not None:
        publisher.close()


def run_hmi(args):
    """The HMI demo: the roll in web order, driven from the page."""
    import threading
    from webcheck.inference.camera import camera_specs
    from webcheck.inference.cascade import inspect_slice
    from webcheck.inference.defect_log import DefectLogWriter
    from webcheck.inference.display import Display
    from webcheck.inference.driver import run_driver
    from webcheck.inference.engine import EngineCache
    from webcheck.inference.gate import GateConfig
    from importlib.util import find_spec
    from webcheck.inference.hmi import (AsyncSink, ControlledSource, GateChoice, LineControl, ReadAhead, StageTimer, TimedGeometry,
                                        parse_detectors, parse_gates, runtime_panel, slice_telemetry, web_order_pairs)
    from webcheck.inference.devices import detect_options, preferred_option
    from webcheck.inference.ort import GPU_ROUTES, backend_info, load_models, select_backend, webgpu_device
    from webcheck.inference.overlap import inspect_columns
    from webcheck.inference.scheduler import SchedulerConfig
    trt_ok, ort_ok = find_spec("tensorrt") is not None, find_spec("onnxruntime") is not None
    try:
        select_backend(args.backend, trt_ok, ort_ok)                    # a --backend that cannot run here stops with why
    except ValueError as exc:
        raise SystemExit(f"--backend: {exc}")
    if args.gpu not in GPU_ROUTES:
        raise SystemExit(f"--gpu must be one of {', '.join(GPU_ROUTES)}")
    providers, webgpu_ready = (), False
    if ort_ok:                                                          # what ONNX Runtime can reach here: nothing is timed
        import onnxruntime
        providers = tuple(onnxruntime.get_available_providers())
        webgpu_ready = webgpu_device(onnxruntime) is not None
    device_options = detect_options(trt_ok, ort_ok, providers, webgpu_ready)
    first = preferred_option(device_options, args.backend, gpu=args.gpu)
    print("devices: " + ", ".join(o.label for o in device_options) + f"; first {first.label}", file=sys.stderr, flush=True)

    stations = {}
    for item in args.station:
        name, _, folder = item.partition("=")
        if not name or not Path(folder, "manifest.jsonl").is_file() or name in stations:
            raise SystemExit(f"--station {item!r}: needs NAME=DIR, a DIR with manifest.jsonl, and a name used once")
        stations[name] = Path(folder)
    try:
        gates = parse_gates(args.gate)
    except ValueError as exc:
        raise SystemExit(f"--gate: {exc}")
    for name, choice in gates.items():
        if not all(Path(choice.folder, f).is_file() for f in ("trained.onnx", "trained_head.onnx")):
            raise SystemExit(f"--gate {name}: {choice.folder} needs trained.onnx and trained_head.onnx")
    try:
        detectors = parse_detectors(args.detector)
    except ValueError as exc:
        raise SystemExit(f"--detector: {exc}")
    for name, folder in detectors.items():
        if not Path(folder, "trained.onnx").is_file():
            raise SystemExit(f"--detector {name}: {folder} needs trained.onnx")

    def load_roll(folder):
        rows = [json.loads(line) for line in open(folder / "manifest.jsonl") if line.strip()]
        cameras_at = {}
        for row in rows:
            cameras_at.setdefault(row["md_offset_mm"], set()).add(row["camera"])
        last = max(cameras_at)
        if len(cameras_at[last]) == 1:                                  # the test set's last position has one camera (A35: 553 images)
            rows = [row for row in rows if row["md_offset_mm"] != last]
            print(f"roll: dropped the last position ({last:g} mm), which has only the {sorted(cameras_at[last])[0]} frame",
                  file=sys.stderr, flush=True)
        pairs = []
        for pair in web_order_pairs(rows):
            if not all((folder / pair[c]).is_file() for c in ("left", "right")):
                break
            pairs.append(pair)
        if not pairs:
            raise SystemExit(f"no complete position at the start of the roll in {folder}")
        return pairs

    cameras = None
    if args.raw_camera:
        width, height = (int(v) for v in args.raw_size.lower().split("x"))
        cameras = camera_specs(args.raw_camera, width, height, args.raw_fps)
    display = Display(args.host, args.serve_port, cameras=cameras, hmi=True,
                      gate_threshold=None if gates else args.gate_threshold)
    display.start()
    display.setup(tuple(stations), first.speed, args.start_view, gates=tuple(gates), gate=next(iter(gates), None),
                  detectors=tuple(detectors), detector=next(iter(detectors), None),
                  devices=tuple(o.as_choice() for o in device_options), device=first.key)
    print(f"HMI: http://{socket.gethostname()}.local:{display.port}  (or the board's IP); waiting for the splash screen's start",
          file=sys.stderr, flush=True)
    start = None
    while start is None:                                                # the page refuses everything else until the line starts
        command = display.next_command(0.5)
        if command is not None and "start" in command:
            start = command["start"]
    picked = next(o for o in device_options if o.key == start["device"])
    kind, gpu = picked.backend, picked.gpu
    print(f"backend: {kind}" + (f", gpu {gpu}" if kind == "onnxruntime" else ""), file=sys.stderr, flush=True)
    options = {}
    if kind == "tensorrt":                                              # the GPU resize kernel needs CUDA; ONNX Runtime uses the CPU resize
        from webcheck.inference.gpu_resample import GpuGateInputs
        from webcheck.inference.trt import DetectorModel, GateModel, build_engine, tensorrt_version
        cache = EngineCache(args.engine_cache, tensorrt_version(), build_engine)
        options["gate_inputs"] = GpuGateInputs(args.build_dir)
    if not args.no_gating:
        columns = inspect_columns("right", TimedGeometry(args.lane_width_mm, start["speed"], datetime.now(timezone.utc)))
        options["camera_options"] = {"right": {"columns": columns}}
        if kind == "tensorrt":
            options["camera_options"]["right"]["gate_inputs"] = GpuGateInputs(args.build_dir, columns)
    chosen = gates[start["gate"]] if gates else GateChoice(args.gate_dir, args.gate_threshold)
    det_dir = detectors[start["detector"]] if detectors else args.det_dir
    from webcheck.device import describe_device
    backend = None
    if kind == "tensorrt":
        det = DetectorModel(cache.get(str(Path(det_dir, "trained.onnx"))))
        gate = GateModel(cache.get(str(Path(chosen.folder, "trained.onnx"))), cache.get(str(Path(chosen.folder, "trained_head.onnx"))))
        try:                                                            # what the models run on, read from the backend
            import tensorrt as trt
            from webcheck.bench import numeric_format
            backend = ("TensorRT", tensorrt_version(), "GPU", numeric_format(trt))
        except Exception as exc:
            print(f"runtime not known: {exc}", file=sys.stderr, flush=True)
    else:
        import onnxruntime
        try:
            gate, det, provider = load_models(chosen.folder, det_dir, gpu=gpu)
        except ValueError as exc:
            raise SystemExit(f"--gpu: {exc}")
        print(f"models on {provider}", file=sys.stderr, flush=True)
        backend = backend_info(onnxruntime, provider)
    cfg = GateConfig(threshold=chosen.threshold)
    if gates:
        display.set_gate_threshold(cfg.threshold)
    display.runtime(runtime_panel(describe_device(), backend))
    folder, speed = stations[start["placement"]], start["speed"]
    from webcheck.inference.cascade import STATION_CLASSES
    options.update(classes=STATION_CLASSES.get(start["placement"]))     # a class the station cannot have is set aside
    pairs = load_roll(folder)
    print(f"roll: placement {start['placement']}, {len(pairs)} positions from {pairs[0]['md_mm']:g} mm (test set, viewing only), "
          f"{speed:g} m/min, {start['view']} camera first, gate {start.get('gate', chosen.folder)} at {cfg.threshold:g}",
          file=sys.stderr, flush=True)

    load = lambda name: np.asarray(Image.open(folder / name).convert("RGB"))
    load_pair = lambda i: {"left": load(pairs[i]["left"]), "right": load(pairs[i]["right"])}

    class Discard:
        def write(self, record):
            pass

    geometry = TimedGeometry(args.lane_width_mm, speed, datetime.now(timezone.utc))
    warm = load_pair(0)
    for i in range(3):                                                  # warm-up on the first position, not logged
        inspect_slice(warm, i, gate, det, cfg, geometry, Discard(), **options)
    geometry = TimedGeometry(args.lane_width_mm, speed, datetime.now(timezone.utc), roll_frames=len(pairs))   # the run starts now; the roll loops

    timer = StageTimer()
    timed_gate, timed_det = timer.wrap("gate", gate), timer.wrap("detector", det)    # per tile; each call waits for the GPU
    timings, latest = {}, {}

    def inspect(frames, k):
        timer.take()
        outcome = inspect_slice(frames, k, timed_gate, timed_det, cfg, geometry, Discard(), **options,
                                anomaly_maps=display.heat)                  # the gate's maps only while the page shows them
        timings[k] = timer.take()
        return outcome

    control = LineControl(speed, view=start["view"])
    reader = ReadAhead(load_pair, len(pairs), ahead=8, workers=args.read_workers)   # PNG decoding, side by side
    done = threading.Event()

    def follow_commands():
        last = None
        while not done.is_set():
            command = display.next_command(0.2)
            if command is not None:
                try:
                    control.apply(command)
                except ValueError as exc:                               # a step while running: nothing to step
                    print(f"control refused: {exc}", file=sys.stderr, flush=True)
            state = control.state()
            if state != last:
                display.control_state(state)
                last = state

    follower = threading.Thread(target=follow_commands, name="hmi-commands", daemon=True)
    follower.start()

    logged = {"records": 0}

    def on_slice(k, frames, outcome):
        logged["records"] += len(outcome.records)
        mqtt = None if publisher is None else (publisher.sent, publisher.failed)
        display.sink(k, frames, outcome, telemetry=slice_telemetry(k, geometry, source.taken, source.now(), timings.pop(k, {}),
                                                                   latest.get("status"), outcome, logged["records"], mqtt))

    def status(t, m):
        latest["status"] = m
        print(f"[HMI {control.speed:g} m/min] t={t:6.1f}s {m['state']:<10} queue={m['queue_depth']:>3} written={m['written']}",
              file=sys.stderr, flush=True)
        display.status(f"slices {m['written']}   {m['state']}   queue {m['queue_depth']}")

    out = Path(args.out)
    log_path = out.with_name(f"{out.stem}_hmi.jsonl")
    log_path.unlink(missing_ok=True)
    source = ControlledSource(reader, geometry, control, args.slices)
    config = SchedulerConfig(args.ceiling_s, args.max_queue_bytes, args.on_overload)
    publisher = None if args.mqtt is None else uns_publisher(args, start["placement"])
    sink = AsyncSink(on_slice)                                          # slice k goes to the screen while k+1 is inspected
    try:
        with DefectLogWriter(log_path) as log:
            report = run_driver(source, inspect, publishing(log, publisher), config, on_status=status, on_slice=sink,
                                max_sleep_s=0.05)
    finally:
        done.set()
        follower.join(2.0)
        reader.close()
        try:
            sink.close()
        except Exception as exc:
            print(f"display hand-off: {exc}", file=sys.stderr, flush=True)
        display.close()
        close_publisher(publisher)
    res = {"mode": "hmi (operator-driven, not a benchmark)", "placement": start["placement"], "positions": len(pairs),
           "gate": start.get("gate"), "gate_dir": chosen.folder, "gate_threshold": cfg.threshold,
           "detector": start.get("detector"), "det_dir": det_dir,
           "log_lines": sum(1 for _ in open(log_path)), **mqtt_counts(publisher),
           **report.summary(args.ceiling_s)}
    print(json.dumps(res), flush=True)
    json.dump({"backend": backend[0] + " " + backend[1] if backend else kind, "results": [res]}, open(args.out, "w"), indent=1)


if __name__ == "__main__":
    main()
