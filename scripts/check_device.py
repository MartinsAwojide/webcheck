"""Check this machine against the Jetson, for anyone trying the demo on hardware the project has not
run on (a Mac, a Windows PC, another Linux GPU). It needs the demo folder from the README (the Studio models in
onnx_gate6 and onnx_det5) and the test frames from scripts/download_test_frames.py.

    uv run python scripts/check_device.py --dir ~/webcheck_demo [--frames ~/test_frames] [--gpu auto|direct|webgpu|none]
                                          [--ladder 4,6,8,10,12] [--out check_report.json]

1. Parity: the 8 reference test frames (48 tiles) through the gate and YOLO-Pro on the GPU route asked for, compared
   with the board's TensorRT results in scripts/reference/board_parity.json, with the time per tile.
2. Speed ladder (skip with --ladder ""): the real HMI started with the launcher, held at each speed, read from its log;
   the best speed is the highest that kept up.
3. One JSON report (--out) to send back, and a summary on screen."""
import argparse
import dataclasses
import json
import statistics
import subprocess
import sys
import time
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from PIL import Image

from webcheck.device import describe_device
from webcheck.inference import selfcheck
from webcheck.inference.detector import decode_detections
from webcheck.inference.frame import tile_views
from webcheck.inference.gate import gate_score
from webcheck.inference.ort import backend_info, load_models
from webcheck.inference.preprocess import detector_input, gate_input

REFERENCE = Path(__file__).resolve().parent / "reference" / "board_parity.json"
DEVICE_KEY = {"CUDAExecutionProvider": "direct", "DmlExecutionProvider": "direct", "WebGpuExecutionProvider": "webgpu",
              "CPUExecutionProvider": "cpu"}


def run_frames(gate, det, frames_dir: Path):
    """Rows {frame, tile, score, dets} for the reference frames, and the gate and detector times per tile (seconds)."""
    rows, t_gate, t_det = [], [], []
    for name in selfcheck.REFERENCE_FRAMES:
        frame = np.asarray(Image.open(frames_dir / name).convert("RGB"))
        for r, c, tile in tile_views(frame):
            tile = np.ascontiguousarray(tile)
            g = gate_input(tile)
            t0 = time.perf_counter()
            anomaly_map = gate(g)
            t_gate.append(time.perf_counter() - t0)
            d = detector_input(tile)
            t0 = time.perf_counter()
            raw = det(d)
            t_det.append(time.perf_counter() - t0)
            dets = [{"cls": x.label, "score": round(float(x.score), 4),
                     "box": [round(float(v), 4) for v in (x.x1, x.y1, x.x2, x.y2)]} for x in decode_detections(raw)]
            rows.append({"frame": name, "tile": [r, c], "score": float(gate_score(anomaly_map)), "dets": dets})
    return rows, t_gate, t_det


def _post(payload: dict, port: int = 8080) -> None:
    req = urllib.request.Request(f"http://127.0.0.1:{port}/control", json.dumps(payload).encode(),
                                 {"Content-Type": "application/json"})
    urllib.request.urlopen(req, timeout=10).read()


def _status(log: Path):
    return [s for s in (selfcheck.parse_status(line) for line in log.read_text(errors="replace").splitlines()) if s]


def _wait_until(log: Path, t: float, limit_s: float = 900):
    end = time.monotonic() + limit_s
    while time.monotonic() < end:
        s = _status(log)
        if s and s[-1][0] >= t:
            return
        time.sleep(2)
    raise TimeoutError(f"the HMI's log did not reach t={t:.0f} s")


def run_ladder(demo: Path, gpu: str, device_key: str, speeds, settle_s: float = 20, hold_s: float = 60):
    """Hold the real HMI at each speed and measure what it writes; the launcher starts and stops it by its PIDs."""
    launch = [sys.executable, "-m", "webcheck.launcher", "--dir", str(demo)]
    subprocess.run(launch + ["start", "--backend", "onnxruntime", "--gpu", gpu, "--wait-s", "300"], check=True)
    log, samples = demo / "hmi_demo.log", []
    try:
        _post({"start": {"placement": "A", "speed": speeds[0], "view": "left", "gate": "best_recall",
                         "detector": "small", "device": device_key}})
        _wait_until(log, 1)
        for speed in speeds:
            _post({"speed": speed})
            now = _status(log)[-1][0]
            _wait_until(log, now + settle_s)
            start = _status(log)[-1]
            _wait_until(log, start[0] + hold_s)
            window = [s for s in _status(log) if s[0] >= start[0]]
            end = window[-1]
            samples.append({"speed": speed, "rate": (end[3] - start[3]) / (end[0] - start[0]),
                            "need": selfcheck.slices_needed(speed), "max_queue": max(s[2] for s in window),
                            "states": sorted({s[1] for s in window})})
            print(f"  {speed:g} m/min: {samples[-1]['rate']:.3f} of {samples[-1]['need']:.3f} slices/s, "
                  f"queue up to {samples[-1]['max_queue']}", flush=True)
    finally:
        subprocess.run(launch + ["stop"])
    return samples


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--dir", required=True, help="the demo folder (onnx_gate6, onnx_det5, demo_driver.py, ...)")
    ap.add_argument("--frames", help="the folder holding A_1080 and B_1080 (default: ../test_frames from --dir)")
    ap.add_argument("--gpu", default="auto", choices=("auto", "direct", "webgpu", "none"))
    ap.add_argument("--ladder", default="4,6,8,10,12", help="speeds in m/min to hold the HMI at; empty to skip")
    ap.add_argument("--hold-s", type=float, default=60, help="seconds measured at each speed")
    ap.add_argument("--out", default="check_report.json")
    args = ap.parse_args()
    demo = Path(args.dir).expanduser().resolve()
    frames = Path(args.frames).expanduser() if args.frames else demo.parent / "test_frames"

    import onnxruntime
    gate, det, provider = load_models(demo / "onnx_gate6", demo / "onnx_det5", gpu=args.gpu)
    device = describe_device()
    print(f"device: {device.name} ({device.system}); route: {provider}", flush=True)
    rows, t_gate, t_det = run_frames(gate, det, frames)
    reference = json.loads(REFERENCE.read_text())
    parity = selfcheck.compare(reference["rows"], rows)
    timing = {"gate_ms_median": 1000 * statistics.median(t_gate), "detector_ms_median": 1000 * statistics.median(t_det)}
    print(f"parity with the board ({reference['backend']}): {'PASS' if parity['pass'] else 'FAIL'}; "
          f"gate scores within {100 * parity['max_rel_diff']:.2f} %, detections {parity['detections']['matched']} "
          f"of {parity['detections']['reference']} matched", flush=True)
    print(f"per tile: gate {timing['gate_ms_median']:.1f} ms, detector {timing['detector_ms_median']:.1f} ms (median)")

    samples, best = [], None
    speeds = [float(s) for s in args.ladder.split(",") if s.strip()]
    if speeds:
        print("speed ladder on the real HMI:", flush=True)
        samples = run_ladder(demo, args.gpu, DEVICE_KEY.get(provider, "cpu"), speeds, hold_s=args.hold_s)
        best = selfcheck.ladder_result(samples)
        print(f"best speed that kept up: {best:g} m/min" if best else "none of the speeds kept up")

    report = {"date": datetime.now(timezone.utc).isoformat(timespec="seconds"), "device": dataclasses.asdict(device),
              "route": provider, "runtime": list(backend_info(onnxruntime, provider)), "python": sys.version.split()[0],
              "parity": parity, "timing": timing, "ladder": samples, "best_speed_m_per_min": best, "rows": rows}
    Path(args.out).write_text(json.dumps(report, indent=1))
    print(f"report written to {args.out}: please send it back")
    return 0 if parity["pass"] else 1


if __name__ == "__main__":
    sys.exit(main())
