#!/bin/bash
# Starts the demo HMI from this repository's layout (see GUIDE.md).
#
# usage: ./scripts/run_demo.sh [SLICES|max] [extra demo_driver.py flags...]
#   SLICES  slices inspected before the run ends (default 1000000); max: no limit, the line runs until stopped
#   extra flags go to the driver as they are, e.g. --no-gating, --host 127.0.0.1, --start-speed 60
# environment:
#   CAMERA  webcam shown on the page, never inspected: a /dev/videoN device, or none
#           (default /dev/video0 when it exists)
#   PORT    the page's port (default 8080)
#   MQTT    broker the defect log is also published to, HOST or HOST:PORT (default localhost: Mosquitto on the board),
#           or none
# gates offered on the splash screen: best recall (models/gate, the default) and best F1 (models/gate_f1) when that folder holds its models
# detectors offered: YOLO-Pro small (models/detector, the default), and medium (models/detector_medium) when it is there
set -euo pipefail
cd "$(dirname "$0")/.."

usage="usage: $0 [SLICES|max] [extra demo_driver.py flags...]"
SLICES=1000000
if [ $# -gt 0 ] && [[ "$1" != --* ]]; then
  SLICES="$1"
  shift
fi
if [ "$SLICES" = "max" ]; then
  SLICES=9223372036854775807
elif ! [[ "$SLICES" =~ ^[1-9][0-9]*$ ]]; then
  echo "$usage  (SLICES must be a positive whole number or max, got '$SLICES')" >&2
  exit 2
fi

stations=()
for name in A B; do
  if [ -f "frames/$name/manifest.jsonl" ]; then
    stations+=(--station "$name=frames/$name")
  fi
done
if [ ${#stations[@]} -eq 0 ]; then
  echo "no test frames found: expected frames/A/manifest.jsonl or frames/B/manifest.jsonl under $(pwd)" >&2
  exit 1
fi

camera=()
case "${CAMERA:-}" in
  none) ;;
  "")
    if [ -c /dev/video0 ]; then
      camera=(--raw-camera left=/dev/video0)
    fi
    ;;
  *)
    if [ ! -c "$CAMERA" ]; then
      echo "CAMERA=$CAMERA is not a camera device (use /dev/videoN, or CAMERA=none for no webcam)" >&2
      exit 2
    fi
    camera=(--raw-camera "left=$CAMERA")
    ;;
esac

gates=(--gate best_recall=models/gate:20.3522)                      # the first gate is the splash screen's default
if [ -f models/gate_f1/trained.onnx ] && [ -f models/gate_f1/trained_head.onnx ]; then
  gates+=(--gate best_f1=models/gate_f1:25.3)
fi

detectors=(--detector small=models/detector)
if [ -f models/detector_medium/trained.onnx ]; then
  detectors+=(--detector medium=models/detector_medium)
fi

mqtt=()
if [ "${MQTT:-localhost}" != "none" ]; then
  mqtt=(--mqtt "${MQTT:-localhost}")
fi

mkdir -p cache/engines cache/build results

exec "${WEBCHECK_PYTHON:-python3}" scripts/demo_driver.py --hmi \
  --pkg src \
  "${gates[@]}" "${detectors[@]}" \
  "${stations[@]}" \
  --engine-cache cache/engines --build-dir cache/build \
  --out results/driver_hmi.json \
  --slices "$SLICES" \
  --lane-width-mm 550.08 \
  --ceiling-s 0.4 --max-queue-bytes 2684354560 \
  --serve-port "${PORT:-8080}" \
  ${camera[@]+"${camera[@]}"} \
  ${mqtt[@]+"${mqtt[@]}"} \
  "$@"
