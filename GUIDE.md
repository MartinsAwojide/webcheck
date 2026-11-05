# webcheck: the full guide

The [README](README.md) has the short version: what the demo is and how to start it. This guide has every step,
setting and page in full.

Print-defect detection for flexible-packaging web, running on an NVIDIA Jetson Orin Nano. A prototype: the models are
trained on synthetic images of a printed and laminated web, rendered with BlenderProc, and the demo replays synthetic
test frames as if they came off a line.

Each frame is cut into tiles. A **FOMO-AD** anomaly gate (Edge Impulse) scores every tile; only the tiles it flags go to
a **YOLO-Pro** detector (Edge Impulse), which names the defect: streaks, dot skip, mottling, tunneling, bubbles or
silvering. Both models run as TensorRT engines on the board's GPU (float32, TF32 allowed). Every detection is written to
a defect log and published over MQTT on a Unified Namespace topic tree.

The demo has two pages, both served by the board:

- **Inspection HMI** (port 8080): the web as the camera sees it with the detections boxed, the latest defect, per-class
  counts, model timings, and controls for line speed (1 to 450 m/min), pause, step, camera, gate and detector.
- **Defect map** (port 8090): the MQTT log streaming live, every detection folded onto one print repeat of the web as a
  per-class heatmap, counts by class and severity, and the defect rate in defects per million tiles (DPMO) with the
  sigma level. History is kept in SQLite on the board (since start, 30 min, 12 h shift, 24 h day).

Measured on the Jetson Orin Nano Super (MAXN_SUPER, float32 with TF32 allowed, best-F1 gate, YOLO-Pro small, both pages,
the defect map and MQTT running): the demo keeps up with the line up to 450 m/min and tops out at about 15 slices a
second, about 490 m/min. Those figures are with the browser on another machine (a PC or a TV). With a browser on the
Jetson itself showing both pages (measured with a headless Firefox ESR on the board, which draws on the CPU), it keeps
up at 330 m/min but not at 450 m/min (10.8 of the 13.9 slices a second it needs; from that, an estimated ceiling of about
350 m/min).

## What is here

| Path | What it is |
|---|---|
| `src/webcheck/` | The Python package: the inference wrapper (tiling, gate, detector, TensorRT glue, GPU resize kernel), the cascade and scheduler, the defect log, MQTT publishing, the HMI and defect map pages with their fonts and assets |
| `scripts/demo_driver.py` | The HMI: replays the test frames through the cascade and serves the page |
| `scripts/defect_map.py` | The defect map server (MQTT subscriber, SQLite history, page) |
| `src/webcheck/launcher.py`, `scripts/webcheck_demo.py`, `Makefile` | The demo launcher for Linux, macOS and Windows: start, stop and status on the board, or drive the board from a PC over ssh (`webcheck-demo`); the Makefile is a shortcut for it on Linux and macOS |
| `scripts/run_demo.sh` | Run the HMI from this repository's layout instead |
| `scripts/download_test_frames.py` | Download the test frames from Hugging Face |
| `scripts/bench_cascade.py`, `scripts/bench_overlap.py`, `scripts/score_split.py` | Benchmarks and the scorer that checks the defect log against ground truth |
| `scripts/check_device.py`, `scripts/reference/` | Checks a machine the project has not run on against the Jetson's results |
| `scripts/webgpu_op_check.py`, `ci/webgpu-linux.yml` | Checks that every operator of the models runs on WebGPU; the CI workflow that runs it on Linux |
| `pyproject.toml`, `uv.lock` | The Python environment (uv) |

## What you need

- An NVIDIA **Jetson Orin Nano** with JetPack 6 (tested on L4T R36.4.7) and the TensorRT 10.3 and CUDA toolkit
  (`nvcc`) that come with it. TensorRT is not installable with pip; the demo uses the system's.
- **Mosquitto** MQTT broker on the board: `sudo apt install mosquitto` (it runs as a service on port 1883).
- Python 3.10 with numpy, Pillow and paho-mqtt. On the board: `sudo apt install python3-pil python3-numpy` and
  `pip3 install paho-mqtt`. On a PC: `uv sync --extra demo`.
- About 2 GB of disk for the test frames and models, and a browser on the same network (a TV works).
- Optional: one or two USB cameras for the raw camera view (`/dev/video0`, `/dev/video2`). Without them the page shows the
  web itself.

## Set up the board

1. **Get the code** on the board and lay out a demo folder:

   ```bash
   git clone <this repository> webcheck && cd webcheck
   mkdir -p ~/webcheck_demo/pkg
   cp -r src/webcheck ~/webcheck_demo/pkg/
   cp scripts/demo_driver.py scripts/defect_map.py scripts/webcheck_demo.py ~/webcheck_demo/
   ```

2. **Download the test frames** (synthetic, CC-BY-4.0, from
   [huggingface.co/datasets/martinsawojide/webcheck-test-frames](https://huggingface.co/datasets/martinsawojide/webcheck-test-frames)).
   Every frame is checked against the dataset's checksums; a rerun only fetches what is missing.

   ```bash
   python3 scripts/download_test_frames.py --dest ~/test_frames
   ```

   The launcher reads `../test_frames` from the demo folder by default; `start --frames DIR` uses another place.

3. **Export the models from Edge Impulse Studio.** The models live in two public Edge Impulse projects; you need a free
   Edge Impulse account to clone them.

   | Folder in the demo | Project | Impulse | Model |
   |---|---|---|---|
   | `onnx_gate6/` | [webcheck_poc_ad](https://studio.edgeimpulse.com/public/1147680/latest) | 6 | FOMO-AD gate, best recall (the default) |
   | `onnx_gate7/` | [webcheck_poc_ad](https://studio.edgeimpulse.com/public/1147680/latest) | 7 | FOMO-AD gate, best F1 |
   | `onnx_det5/` | [webcheck_poc_od](https://studio.edgeimpulse.com/public/1136708/latest) | 5 | YOLO-Pro small (the default) |
   | `onnx_det_medium/` | [webcheck_poc_od](https://studio.edgeimpulse.com/public/1136708/latest) | 6 | YOLO-Pro medium (optional) |

   For each row: open the public project, choose **Clone this project**, and in your clone open **Deployment** for that
   impulse. Pick **TensorRT library**, the **float32** model, and **Build**. Unzip the download and copy its ONNX files
   into the folder in the first column of the demo folder: `trained.onnx` (and, for the gates, `trained_head.onnx`).
   The board builds TensorRT engines from them on the first start and keeps them in `engine_cache/`, so the first start
   takes several minutes.

4. **Start the demo** from the demo folder:

   ```bash
   cd ~/webcheck_demo
   python3 webcheck_demo.py start          # add --clear-db to empty the defect map's history
   python3 webcheck_demo.py start --open   # on the board itself, with a monitor: opens the HMI in its browser
   python3 webcheck_demo.py status
   python3 webcheck_demo.py stop
   ```

   It prints the two addresses: open `http://<board>:8080` for the HMI and `http://<board>:8090` for the defect map,
   on the board itself or on any machine on its network. The launcher needs only the board's python3, no uv.
   From a clone of this repository on the board, `make demo`, `make stop` and `make status` do the same
   (`DEMO_DIR` is the demo folder, default `~/webcheck_demo`).
   On the HMI's first screen pick the station (A, after printing; B, after lamination), the line speed, the gate and the
   detector, then start the line. The GPU resize kernel is compiled with `nvcc` on the first start and cached.

### From a PC (Linux, macOS or Windows)

From a clone of this repository with [uv](https://docs.astral.sh/uv/), the launcher copies the code and the demo
scripts to the board over ssh and starts the demo there. Name the board with `BOARD=user@host` or put `user@host` on
the first line of `~/.config/webcheck/board`; the demo folder on the board is `REMOTE_DIR` (relative to the remote
home directory, default `webcheck_demo`).

| System | Start | Stop / status / log |
|---|---|---|
| Linux, macOS | `make board` (or `uv run webcheck-demo board start`) | `make board-stop`, `make board-status`, `make board-log` |
| Windows (PowerShell) | `$env:BOARD="user@host"; uv run webcheck-demo board start` | `uv run webcheck-demo board stop` / `status` / `log` |

Windows needs nothing beyond uv and the OpenSSH client that ships with Windows 10 and 11. The Windows and macOS paths
have not been run on those systems yet.

## Without a Jetson (Linux, macOS or Windows)

The same Studio exports also run through [ONNX Runtime](https://onnxruntime.ai/) on a PC or a Mac, with the same pages.
Lay out a demo folder as in steps 1 to 3 above, then:

```bash
uv sync --extra demo --extra onnx
uv run webcheck-demo start --dir ~/webcheck_demo --backend onnxruntime --open
```

The splash screen lists the device options this machine has (TensorRT, ONNX Runtime on a direct GPU, on WebGPU, on
the CPU) and picks the best one; picking another sets the line speed to a starting speed measured for that option
beforehand (nothing is timed on your machine). From the command line, `--gpu` picks the first choice:

| `--gpu` | What it uses | Install |
|---|---|---|
| `direct` | the vendor's provider: CUDA on an NVIDIA GPU, DirectML on Windows | `onnxruntime-gpu` or `onnxruntime-directml` in place of `onnxruntime` |
| `webgpu` | the WebGPU provider: Vulkan on Linux, Metal on a Mac with Apple silicon, D3D12 on Windows | `uv sync --extra webgpu` (Python 3.11+; on Linux the Vulkan loader, `libvulkan.so.1`) |
| `none` | the CPU | nothing more |
| `auto` (default) | direct if its provider is installed, then WebGPU, then the CPU | |

Per tile on a laptop (Intel Core i7-1355U with an NVIDIA RTX 2050, R1, float32, measured): about 1.5 s on the CPU and
about 0.16 s through WebGPU (Vulkan), against about 0.02 s on the Jetson with TensorRT. On the CPU the demo managed
about 18 s per slice, roughly 2 m/min of web, so a laptop shows the line at a crawl rather than at line speed. On a
Mac the WebGPU route has not been run yet. The HMI's header names the runtime, the GPU route, the device and the rung,
so a laptop run (R1) is never read as the board's figures. Checked against the board on 8 test frames, on the CPU and
through WebGPU: gate scores within 0.22 %, the same detections.

### Checking a machine the project has not run on

If you run the demo on hardware not listed above (a Mac, a Windows PC, another GPU), this checks it against the Jetson
and measures the speed it keeps up with. With the demo folder and the test frames in place:

```bash
uv sync --extra demo --extra onnx --extra webgpu
uv run python scripts/check_device.py --dir ~/webcheck_demo --gpu webgpu      # or --gpu direct / none
```

It runs 8 published test frames through the models and compares every gate score and detection with the board's
results (`scripts/reference/board_parity.json`), times each tile, then holds the real HMI at 4 to 12 m/min to see
which speed keeps up (about 7 minutes; `--ladder ""` skips it). Please send back the report it writes,
`check_report.json`: it holds the device, the route, the comparison, the timings and the ladder, and nothing else.
`scripts/webgpu_op_check.py` checks that every operator of the models runs on WebGPU. `ci/webgpu-linux.yml` runs it
on Linux through a software Vulkan driver; copy it to `.github/workflows/` to run it on GitHub Actions.

## Settings

Most settings are command-line options: `--help` on `webcheck-demo` and on each script lists them. A few come from the
environment or from one config file. Nothing needs a password or key, so there is no `.env` file.

| Setting | Used by | What it does | Default |
|---|---|---|---|
| `BOARD` | `webcheck-demo board ...` | `user@host` of the Jetson | the first line of `~/.config/webcheck/board` |
| `~/.config/webcheck/board` | the same | its first line, `user@host`, is used when `BOARD` is not set | none |
| `REMOTE_DIR` | the same | the demo folder on the board, relative to its home folder | `webcheck_demo` |
| `NO_COLOR` | `webcheck-demo` | any value turns colour off in the terminal output | not set |
| `UI_TTY`, `UI_COLOR`, `UI_UTF8` | `webcheck-demo` | `0` or `1` forces the progress line, colour or box glyphs off or on | read from the terminal |
| `CAMERA` | `scripts/run_demo.sh` | the webcam shown on the page: a `/dev/videoN` device, or `none` | `/dev/video0` when it exists |
| `PORT` | the same | the page's port | `8080` |
| `MQTT` | the same | where the defect log is published: `HOST`, `HOST:PORT`, or `none` | `localhost` |
| `WEBCHECK_PYTHON` | the same | the Python that runs the demo | `python3` |

## Using the HMI

- **Splash screen:** pick the station (A, after printing; B, after lamination), the device (the options this machine
  has, each with its own starting speed), the line speed, the first camera, the anomaly gate (best recall flags more,
  best F1 raises fewer false alarms) and the detector, then start. The line runs once per start; restart the demo to
  start again.
- **Controls:** speed slider (1 to 450 m/min), pause, step and camera switch. The test roll loops at its end.
- **Falling behind:** a banner over the picture says how many slices and seconds the screen is behind the belt.
  Nothing is skipped: every slice is still inspected and logged, late.
- **Counter:** slices inspected, metres of web, the defects per class since the page was opened, and the gate's and
  detector's time with their tile counts. Each station counts only the classes it can have (A: dot skip, mottling,
  streaks; B: all six); a detection of another class is set aside and counted apart.
- **Latest defect and history:** a crop of the latest defect with its class, confidence, position, size and the gate
  score of its tile; the history strip keeps the last ten.
- **Toggles:** History, Camera (an external webcam if one is attached, otherwise the web itself), Shade (the tile
  columns a camera does not inspect), Heatmap (the gate's anomaly map) and Present (hides the controls; `P` toggles).
- **Device:** the button in the header shows the accelerator, the runtime and GPU route, the device, the number
  format and the rung.

| Address | What it shows |
|---|---|
| `/` | the HMI |
| `/?square` | the same page on a 1080x1080 canvas, for square clips |
| `/web/left`, `/web/right` | that camera's frame full screen with nothing drawn on it |
| `/health` | a small JSON status |

## What it writes

The defect log (`driver_hmi_hmi.jsonl` in the demo folder) has one JSON object per defect:

```json
{"defect_type": "streaks", "confidence": 0.81, "timestamp": "2026-10-29T09:00:00.000Z",
 "location": {"md_mm": 4320.5, "cd_mm": 300.5}, "size": {"value": 1200.0, "unit": "mm2"}}
```

- `location.md_mm` is along the web (machine direction), `cd_mm` across it. There is no encoder, so the position is
  the frame's index times the frame length, plus the camera's offset across the web.
- `size` is the box area in square millimetres; `timestamp` is when that part of the web passed the camera at the
  chosen line speed, in UTC.

The summary (`driver_hmi.json`) records the run's slice count, latency statistics and how long the line spent behind.

## MQTT topics

Each station is a work cell in an ISA-95 style tree (the company and site are fictional):

```
ENERGY_UP_LLC/TEXAS_FACILITY/Printing_Lamination/Line_1/Placement_A/defect        the whole record, as in the log file
ENERGY_UP_LLC/TEXAS_FACILITY/Printing_Lamination/Line_1/Placement_A/defect/class  timestamp and defect type (also /confidence, /location, /size)
ENERGY_UP_LLC/TEXAS_FACILITY/Printing_Lamination/Line_1/Placement_A/frame         one message per inspected frame: its tiles, the tiles a defect came from, its records
ENERGY_UP_LLC/TEXAS_FACILITY/Printing_Lamination/Line_1/Placement_B/...           the same for the lamination station
```

## How the code is organised

Everything the demo runs is in `src/webcheck/`. Only `inference/trt.py` and `inference/gpu_resample.py` need an NVIDIA
GPU; the rest is plain Python and NumPy.

| Module | Role |
|---|---|
| `inference/frame.py` | cuts a 1920x1080 frame into six 640x640 tiles (two overlapping rows, three columns) |
| `inference/preprocess.py` | tile to model input: the gate sees a 320x320 Lanczos resize, the detector the tile as is |
| `inference/resample.py`, `resample.cu`, `gpu_resample.py` | the gate's resize on the GPU, byte for byte equal to Pillow's Lanczos |
| `inference/gate.py`, `inference/detector.py` | the gate's score and flag; YOLO-Pro's output decoded, duplicate boxes removed |
| `inference/trt.py`, `inference/engine.py` | TensorRT engines built from the ONNX files, cached and run (the Jetson) |
| `inference/ort.py`, `inference/devices.py` | ONNX Runtime on a direct GPU, WebGPU or the CPU (no Jetson); the device options and their starting speeds |
| `inference/pipeline.py`, `inference/overlap.py`, `inference/cascade.py` | one frame; the overlap between the two cameras; one slice, with the station's classes |
| `inference/defect_log.py`, `inference/uns.py` | the record format and writer; MQTT publishing on UNS topics |
| `inference/scheduler.py`, `inference/driver.py` | the queue and state machine (one slice at a time, in order, never dropped) and its runner |
| `inference/hmi.py`, `inference/display.py`, `inference/overlay.py`, `inference/stream.py` | the operator's line and controls; the page drawn and served in a separate process |
| `inference/camera.py` | the optional webcam view (`gst-launch-1.0`) |
| `inference/defect_map.py`, `inference/severity.py` | the defect map's store and page; severity bands from the training data |
| `inference/selfcheck.py`, `inference/log_inspector.py`, `inference/cascade_bench.py`, `bench.py` | checking a machine, scoring a defect log, benchmarking |
| `launcher.py`, `device.py` | the cross-platform launcher; the machine and rung shown on the HMI |
| `datagen/scene.py` | the frame geometry (millimetres per pixel, camera offsets) shared with the rendered scenes |

## Limits

- **The frames are synthetic** renders of a fictional product, and the demo plays the held-out test set.
- **Gating trades recall for speed.** The right camera inspects only its last tile column, because the left camera
  already covers the other two (`--no-gating` turns it off; the Shade toggle shows the skipped tiles).
- **The webcam is not inspected.** It shows a real camera on the page; the models inspect the stored frames.
- **Off the Jetson the demo is slow:** see the starting speeds on the splash screen.

## Troubleshooting

| Symptom | Check |
|---|---|
| `nvcc` not found | install the CUDA toolkit (`cuda-toolkit-12-6` on JetPack 6); the demo also looks in `/usr/local/cuda/bin` |
| `ModuleNotFoundError: tensorrt` on the Jetson | install `python3-libnvinfer` and run with the system `python3`, not a virtual environment |
| the page does not load | on a first start, wait for the `HMI ready` line (engines are still building); otherwise the port is busy or a firewall blocks it |
| `port 8080 is in use` | another program holds the port; stop it, or start with `--hmi-port` and `--map-port` |
| `no WebGPU device` | install the `webgpu` extra; on Linux install the Vulkan loader (`libvulkan1`) |
| no camera view | only external webcams are shown on Linux; check `ls /dev/video*` |
| no test frames | run `scripts/download_test_frames.py` and point `--frames` at its folder |

## Licences

Code: MIT (`LICENSE`). The web artwork behind the defect map's template and the screenshots in `assets/`: CC-BY-4.0.
The fonts (IBM Plex, Inter Display): SIL Open Font License 1.1. See `REUSE.toml` and `LICENSES/`. The test frames are
CC-BY-4.0 on Hugging Face. All brands, logos and codes in the artwork are fictional.
