"""The demo display: renders and serves the page in a child process, away from the driver.

`Display.sink` is the driver's `on_slice`: it copies what the page needs (half-size frames, boxes, records) and queues
it, so the driver's single thread never waits on the screen. The child renders the JPEG and runs the web server
(stream.py). Pictures go through a one-slot queue where the newest replaces the oldest, so a busy child skips slices;
records and status go through an ordered queue and are never skipped. The child is started with 'spawn', not 'fork',
so it does not inherit the driver's CUDA and TensorRT state. With `hmi=True` the child serves the HMI page,
draws the chosen camera at full size, checks each control posted by the page and queues it back for the driver
(`next_command`); the driver shows the line's state with `control_state`. With a `gate_threshold` the HMI also offers the
gate's anomaly heatmap, off by default: the child keeps the switch and shares it with the driver (`heat`), which asks the
pipeline for the maps only while it is on, so the heatmap costs the inspection nothing when off."""
import math
import multiprocessing
import multiprocessing.connection
import os
import queue
import signal
import sys
import threading
import time
from typing import Optional, Sequence

from webcheck.inference.camera import CameraReader, CameraSpec
from webcheck.inference.hmi import check_command
from webcheck.inference.overlap import CAMERAS
from webcheck.inference.overlay import crop_defect, crop_jpeg, encode_jpeg, render_annotated, render_camera, snapshot
from webcheck.inference.stream import CropStore, DisplayServer, EventLog, Latest, state_key

_START_TIMEOUT_S = 30.0
_STOP_TIMEOUT_S = 5.0
_WATCH_POLL_S = 0.2          # how often the child tells the driver which plain streams have viewers
_CROPS_KEPT = 64             # HMI: the newest defect crops served at /crop/<n>


def _render(snap, shade: bool = True, heat_threshold: Optional[float] = None):
    """The HMI's one full-size camera if the snapshot kept one (gating shade and heatmap only when the page asks), else
    the two-camera picture."""
    if snap.full:
        return render_camera(snap, next(iter(snap.full)), shade=shade, heat_threshold=heat_threshold)
    return render_annotated(snap)


def _serve(pictures, events, ready, host, port, quality, cameras=(), commands=None, watched=None, heat=None, gate_threshold=None):
    """The child process: serve the page and turn queued snapshots into pictures and queued events into the log.
    Each camera is streamed raw at /raw/<name> by a reader in this process; a camera that is lost is announced on the
    event stream and never touches the annotated picture. With `watched` (HMI), each test camera's plain frame is served
    at /web/<camera>, and `watched` tells the driver which of them have a viewer, so only those are copied and encoded.
    `heat` (HMI with a `gate_threshold`) tells the driver whether the page wants the anomaly heatmap."""
    frames, log = Latest(), EventLog(sticky=state_key)         # a page opened late still gets the line and camera states
    readers = {spec.name: CameraReader(spec, Latest(), on_state=lambda name, state: log.append({"type": "camera", "name": name, "state": state}))
               for spec in cameras}
    line = {"placements": (), "gates": (), "detectors": (), "devices": (), "started": False, "shade": False, "heat": False}   # once the driver has sent its setup, the start comes first
    line_lock = threading.Lock()
    gate = {"threshold": gate_threshold}     # the running gate's threshold; the driver sends it once the line has started
    if gate_threshold is not None:
        log.append({"type": "gate", "threshold": float(gate_threshold)})   # the page's heat legend and score caption

    def on_control(command):
        if isinstance(command, dict) and ("shade" in command or "heat" in command):   # the screen's overlays: kept here, not sent on
            checked = check_command(command)
            (key, on), = checked.items()
            if key == "heat":
                if heat is None or gate["threshold"] is None:
                    raise ValueError("this display has no gate threshold to draw the heatmap with")
                heat.value = 1 if on else 0
            line[key] = on
            log.append({"type": key, "on": on})
            return checked
        with line_lock:                        # a bad control is refused here and never reaches the driver
            if line["placements"] and not line["started"]:
                if not isinstance(command, dict) or "start" not in command:
                    raise ValueError("start the line from the splash screen first")
                checked = check_command(command, line["placements"], line["gates"], line["detectors"], line["devices"])
                line["started"] = True
                station = {"type": "station", "placement": checked["start"]["placement"]}     # the page titles itself from it
                if "gate" in checked["start"]:
                    station["gate"] = checked["start"]["gate"]
                if "detector" in checked["start"]:
                    station["detector"] = checked["start"]["detector"]
                log.append(station)
            else:
                checked = check_command(command)
            commands.put(checked)
        return checked

    hmi = commands is not None
    plain = {camera: Latest() for camera in CAMERAS} if watched is not None else {}
    crops = CropStore(_CROPS_KEPT) if hmi else None
    server = DisplayServer(frames, log, host=host, port=port, raw=readers, hmi=hmi, on_control=on_control if hmi else None,
                           plain=plain, crops=crops)

    def stop_cameras():
        for reader in readers.values():
            reader.stop()

    def on_terminate(*_):                      # Display.close() ends a child that will not stop by itself: take the cameras down with it
        stop_cameras()
        sys.exit(0)

    def follow_parent(parent):                 # a driver killed without close() must not leave the page and its port behind
        multiprocessing.connection.wait([parent.sentinel])
        stop_cameras()
        os._exit(0)

    signal.signal(signal.SIGTERM, on_terminate)
    parent = multiprocessing.parent_process()
    if parent is not None:
        threading.Thread(target=follow_parent, args=(parent,), daemon=True).start()
    server.start()
    for reader in readers.values():
        reader.start()
    ready.put(server.port)

    def follow_events():
        while True:
            event = events.get()
            if event is None:
                return
            if event.get("type") == "setup":
                with line_lock:
                    line["placements"] = tuple(event["placements"])
                    line["gates"] = tuple(event.get("gates", ()))
                    line["detectors"] = tuple(event.get("detectors", ()))
                    line["devices"] = tuple(d["key"] for d in event.get("devices", ()))
            elif event.get("type") == "gate":
                gate["threshold"] = event["threshold"]
            for index, pixels, box, label in event.pop("_crops", ()):     # encoded here, off the driver's time
                try:
                    event["records"][index]["crop"] = "/crop/%d" % crops.put(crop_jpeg(pixels, box, label))
                except ValueError:
                    pass
            log.append(event)

    def follow_viewers():
        while True:
            for index, camera in enumerate(CAMERAS):
                watched[index] = 1 if server.watching(camera) else 0
            time.sleep(_WATCH_POLL_S)

    follower = threading.Thread(target=follow_events, daemon=True)
    follower.start()
    if watched is not None:
        threading.Thread(target=follow_viewers, daemon=True).start()
    while True:
        snap = pictures.get()
        if snap is None:
            break
        frames.put(encode_jpeg(_render(snap, line["shade"], gate["threshold"] if line["heat"] else None), quality))
        for camera, frame in snap.plain.items():
            plain[camera].put(encode_jpeg(frame, quality))
    follower.join(_STOP_TIMEOUT_S)
    server.stop()
    stop_cameras()


def _attach_sources(event: dict, frames, outcome) -> None:
    """HMI: each record of the slice event gets its gate score, and a window around it is cut from its camera's frame
    for the display process to draw and encode (the log and MQTT never see either)."""
    sources = getattr(outcome, "sources", ())
    if len(sources) != len(event["records"]):
        return
    crops = []
    for index, (record, source) in enumerate(zip(event["records"], sources)):
        if source.gate_score is not None:
            record["gate_score"] = round(float(source.gate_score), 2)
        if source.camera in frames:
            try:
                crops.append((index, *crop_defect(frames[source.camera], source.box), record["defect_type"]))
            except ValueError:
                pass
    if crops:
        event["_crops"] = crops


class Display:
    def __init__(self, host: str = "0.0.0.0", port: int = 8080, quality: int = 80, cameras: Optional[Sequence[CameraSpec]] = None,
                 hmi: bool = False, gate_threshold: Optional[float] = None):
        if isinstance(port, bool) or not isinstance(port, int) or not 0 <= port <= 65535:
            raise ValueError(f"port must be an integer from 0 to 65535, got {port!r}")
        if isinstance(quality, bool) or not isinstance(quality, int) or not 1 <= quality <= 100:
            raise ValueError(f"quality must be an integer from 1 to 100, got {quality!r}")
        if cameras is not None and (not isinstance(cameras, (list, tuple)) or not all(isinstance(c, CameraSpec) for c in cameras)
                                    or len({c.name for c in cameras}) != len(cameras)):
            raise ValueError("cameras must be a list of CameraSpec with distinct names")
        if not isinstance(hmi, bool):
            raise ValueError(f"hmi must be True or False, got {hmi!r}")
        if gate_threshold is not None and (isinstance(gate_threshold, bool) or not isinstance(gate_threshold, (int, float))
                                           or not math.isfinite(gate_threshold) or gate_threshold <= 0):
            raise ValueError(f"gate_threshold must be a positive finite number, got {gate_threshold!r}")
        self.gate_threshold = gate_threshold
        self._cameras, self.hmi, self.view = tuple(cameras or ()), hmi, "left"
        self._commands = self._watched = self._heat = None
        self.host, self._requested_port, self._quality = host, port, quality
        self._process = self._pictures = self._events = None
        self._port: Optional[int] = None
        self._closed = False

    @property
    def port(self) -> int:
        return self._port if self._port is not None else self._requested_port

    @property
    def running(self) -> bool:
        return self._process is not None and not self._closed and self._process.is_alive()

    def start(self) -> None:
        if self._process is not None:
            raise ValueError("the display is already started")
        ctx = multiprocessing.get_context("spawn")
        self._pictures, self._events, ready = ctx.Queue(maxsize=1), ctx.Queue(), ctx.Queue()
        self._commands = ctx.Queue() if self.hmi else None
        self._watched = ctx.Array("b", len(CAMERAS)) if self.hmi else None
        self._heat = ctx.Value("b", 0) if self.hmi else None
        self._process = ctx.Process(target=_serve, args=(self._pictures, self._events, ready, self.host, self._requested_port,
                                                         self._quality, self._cameras, self._commands, self._watched,
                                                         self._heat, self.gate_threshold), daemon=True)
        self._process.start()
        try:
            self._port = ready.get(timeout=_START_TIMEOUT_S)
        except queue.Empty:
            self._process.terminate()
            raise RuntimeError("the display process did not start")

    @property
    def heat(self) -> bool:
        """Whether the page has the anomaly heatmap on, so the driver should ask for the maps
        (`process_frame(anomaly_maps=...)`); always False without the HMI or a gate threshold."""
        return self._heat is not None and bool(self._heat.value)

    def sink(self, frame_index: int, frames, outcome, telemetry: Optional[dict] = None) -> bool:
        """The driver's `on_slice`. Returns False when the display has been closed. Never waits for the screen.
        `telemetry` (hmi.slice_telemetry) rides on the slice's event."""
        if self._process is None:
            raise ValueError("the display is not started")
        if telemetry is not None and not isinstance(telemetry, dict):
            raise ValueError(f"telemetry must be a dict, got {telemetry!r}")
        if self._closed:
            return False
        watched = tuple(c for i, c in enumerate(CAMERAS) if self._watched is not None and self._watched[i] and c in frames)
        snap = snapshot(frame_index, frames, outcome, full=self.view if self.hmi else None, plain=watched)
        event = {"type": "slice", "slice": frame_index, "records": snap.records}
        if self.hmi:
            _attach_sources(event, frames, outcome)
        if telemetry is not None:
            event["telemetry"] = dict(telemetry)
        self._events.put(event)
        try:
            self._pictures.put_nowait(snap)
        except queue.Full:
            try:
                self._pictures.get_nowait()          # the child is busy: the older picture is skipped
            except queue.Empty:
                pass
            try:
                self._pictures.put_nowait(snap)
            except queue.Full:
                pass
        return True

    def status(self, text: str) -> None:
        if self._process is None:
            raise ValueError("the display is not started")
        if not self._closed:
            self._events.put({"type": "status", "text": str(text)})

    def next_command(self, timeout: float) -> Optional[dict]:
        """The next control posted on the HMI page (already checked), or None if none came within `timeout` seconds.
        A view command also switches the camera the display draws."""
        if not self.hmi:
            raise ValueError("only an HMI display takes commands")
        if self._process is None:
            raise ValueError("the display is not started")
        try:
            command = self._commands.get(timeout=timeout)
        except queue.Empty:
            return None
        if "view" in command:
            self.view = command["view"]
        elif "start" in command:
            self.view = command["start"]["view"]
        return command

    def setup(self, placements, speed_m_per_min: float, view: str, gates: tuple = (), gate: Optional[str] = None,
              detectors: tuple = (), detector: Optional[str] = None, devices: tuple = (), device: Optional[str] = None) -> None:
        """Offer the splash screen's choices: the placements with test frames on the board, the starting speed and
        camera, and, if any, the named anomaly gates with the one picked first, the detectors, and the device options
        ({key, label, speed} each, `devices.DeviceOption.as_choice`). From then on the page must start the line, once,
        before any other control is taken."""
        if self._process is None:
            raise ValueError("the display is not started")
        if not isinstance(placements, tuple) or not placements or len(set(placements)) != len(placements) \
                or not all(isinstance(p, str) and p for p in placements):
            raise ValueError(f"placements must be a tuple of distinct names, got {placements!r}")
        if not isinstance(gates, tuple) or len(set(gates)) != len(gates) or not all(isinstance(g, str) and g for g in gates) \
                or (gates and gate not in gates) or (not gates and gate is not None):
            raise ValueError(f"gates must be a tuple of distinct names with gate one of them, got {gates!r} and {gate!r}")
        if not isinstance(detectors, tuple) or len(set(detectors)) != len(detectors) \
                or not all(isinstance(d, str) and d for d in detectors) \
                or (detectors and detector not in detectors) or (not detectors and detector is not None):
            raise ValueError(f"detectors must be a tuple of distinct names with detector one of them, got {detectors!r} and {detector!r}")
        keys = [d.get("key") if isinstance(d, dict) else None for d in devices] if isinstance(devices, tuple) else None
        if keys is None or not all(isinstance(d, dict) and set(d) == {"key", "label", "speed"} and isinstance(d["key"], str)
                                   and d["key"] and isinstance(d["label"], str) for d in devices) \
                or len(set(keys)) != len(keys) or (devices and device not in keys) or (not devices and device is not None):
            raise ValueError(f"devices must be a tuple of {{key, label, speed}} with device one of the keys, got {devices!r} and {device!r}")
        for d in devices:
            check_command({"speed": d["speed"]})
        speed, view = check_command({"speed": speed_m_per_min})["speed"], check_command({"view": view})["view"]
        event = {"type": "setup", "placements": list(placements), "speed_m_per_min": speed, "view": view}
        if gates:
            event.update(gates=list(gates), gate=gate)
        if detectors:
            event.update(detectors=list(detectors), detector=detector)
        if devices:
            event.update(devices=[dict(d) for d in devices], device=device)
        if not self._closed:
            self._events.put(event)

    def set_gate_threshold(self, threshold: float) -> None:
        """The running gate's threshold, which the heatmap is drawn against (sent once the line has started)."""
        if self._process is None or not self.hmi:
            raise ValueError("the gate threshold goes to a started HMI display")
        if isinstance(threshold, bool) or not isinstance(threshold, (int, float)) or not math.isfinite(threshold) or threshold <= 0:
            raise ValueError(f"the gate threshold must be a positive finite number, got {threshold!r}")
        self.gate_threshold = float(threshold)
        if not self._closed:
            self._events.put({"type": "gate", "threshold": float(threshold)})

    def runtime(self, info: dict) -> None:
        """What the models run on (hmi.runtime_info), shown in the HMI's header; a late page learns it too."""
        if self._process is None or not self.hmi:
            raise ValueError("the runtime goes to a started HMI display")
        if not isinstance(info, dict):
            raise ValueError(f"the runtime must be a dict, got {info!r}")
        if not self._closed:
            self._events.put(dict(info, type="runtime"))

    def control_state(self, state: dict) -> None:
        """Show the line's state (hmi.LineControl.state()) on the HMI page."""
        if self._process is None:
            raise ValueError("the display is not started")
        if not self._closed:
            self._events.put({"type": "control", "state": dict(state)})

    def close(self) -> None:
        if self._process is None or self._closed:
            return
        self._closed = True
        self._events.put(None)
        try:
            self._pictures.put(None, timeout=2.0)
        except queue.Full:
            pass
        self._process.join(_STOP_TIMEOUT_S)
        if self._process.is_alive():
            self._process.terminate()
            self._process.join(_STOP_TIMEOUT_S)
