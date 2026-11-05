"""Raw camera view for the demo page: a USB camera streamed to the page and never inspected.

A camera is a `gst-launch-1.0` child process that writes MJPEG to its standard output. `split_jpegs` cuts that byte
stream into whole pictures without decoding them, and a `CameraReader` keeps the newest picture in a mailbox for the
page's `/raw/<name>` stream. A camera that stops or unplugs is reported as lost and its process is started again; the
inspection and the annotated picture never wait for it. Standard library only (GStreamer is run as a subprocess)."""
import re
import subprocess
import threading
from dataclasses import dataclass
from typing import Callable, Iterable, Iterator, List, Optional, Sequence, Tuple

from webcheck.inference.stream import Latest

DEFAULT_MAX_FRAME_BYTES = 8_000_000
_SOI, _EOI = b"\xff\xd8", b"\xff\xd9"
_DEVICE = re.compile(r"/dev/video[0-9]+\Z")
_NAME = re.compile(r"[A-Za-z0-9_-]+\Z")
_READ_BYTES = 65536
_STOP_TIMEOUT_S = 5.0


def _is_int(value) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def _check_name(name) -> str:
    if not isinstance(name, str) or not _NAME.match(name):
        raise ValueError(f"a camera name is letters, digits, '-' and '_', got {name!r}")
    return name


def split_jpegs(chunks: Iterable[bytes], max_frame_bytes: int = DEFAULT_MAX_FRAME_BYTES) -> Iterator[bytes]:
    """Cut a byte stream into whole JPEG pictures (start marker to end marker). Bytes before a start marker are
    dropped, an unfinished last picture is not handed out, and a picture larger than `max_frame_bytes` is dropped
    and the stream resumes at the next start marker. Pictures with an embedded thumbnail are not supported; camera
    MJPEG has none."""
    if not _is_int(max_frame_bytes) or max_frame_bytes < 1:
        raise ValueError(f"max_frame_bytes must be a positive integer, got {max_frame_bytes!r}")
    return _split(chunks, max_frame_bytes)


def _split(chunks, max_frame_bytes):
    buf = bytearray()
    for chunk in chunks:
        if not isinstance(chunk, (bytes, bytearray)):
            raise ValueError(f"chunks must be bytes, got {type(chunk).__name__}")
        buf += chunk
        while True:
            start = buf.find(_SOI)
            if start < 0:
                del buf[: len(buf) - 1 if buf.endswith(b"\xff") else len(buf)]     # keep a lone 0xFF: the marker may be split
                break
            del buf[:start]
            end = buf.find(_EOI, 2)
            if end < 0:
                if len(buf) > max_frame_bytes:
                    del buf[:2]                                     # too long without an end: drop it and look for the next start
                    continue
                break
            if end + 2 > max_frame_bytes:
                del buf[:2]
                continue
            frame = bytes(buf[: end + 2])
            del buf[: end + 2]
            yield frame


def gst_command(device: str, width: int = 1280, height: int = 720, fps: Optional[int] = None) -> List[str]:
    """The `gst-launch-1.0` command that writes the camera's MJPEG to its standard output. With `fps` None the caps
    name no frame rate and the camera runs at a rate it offers for that size."""
    if not isinstance(device, str) or not _DEVICE.match(device):
        raise ValueError(f"device must be a /dev/videoN node, got {device!r}")
    limits = (("width", width, 1, 7680), ("height", height, 1, 4320)) + ((("fps", fps, 1, 60),) if fps is not None else ())
    for label, value, low, high in limits:
        if not _is_int(value) or not low <= value <= high:
            raise ValueError(f"{label} must be an integer from {low} to {high}, got {value!r}")
    caps = f"image/jpeg,width={width},height={height}" + (f",framerate={fps}/1" if fps is not None else "")
    return ["gst-launch-1.0", "-q", "v4l2src", f"device={device}", "!", caps, "!", "fdsink", "fd=1"]


@dataclass(frozen=True)
class CameraSpec:
    """A named camera and the command that streams it. Plain data, so it can be sent to the display process."""
    name: str
    command: Tuple[str, ...]

    def __post_init__(self):
        _check_name(self.name)
        if not isinstance(self.command, tuple) or not self.command or not all(isinstance(c, str) for c in self.command):
            raise ValueError("command must be a non-empty tuple of strings")


def camera_specs(items: Sequence[str], width: int = 1280, height: int = 720, fps: Optional[int] = None) -> List[CameraSpec]:
    """Turn `name=/dev/videoN` items (the command-line form) into specs."""
    if not isinstance(items, (list, tuple)) or not items:
        raise ValueError("items must be a non-empty list of name=/dev/videoN")
    specs, seen = [], set()
    for item in items:
        if not isinstance(item, str) or "=" not in item:
            raise ValueError(f"expected name=/dev/videoN, got {item!r}")
        name, device = item.split("=", 1)
        _check_name(name)
        if name in seen:
            raise ValueError(f"camera name {name!r} is used twice")
        seen.add(name)
        specs.append(CameraSpec(name, tuple(gst_command(device, width, height, fps))))
    return specs


class CameraReader:
    """Runs a camera's command, puts each whole picture in `frames` (a mailbox that keeps only the newest) and restarts
    the command `retry_s` seconds after it stops. `state` is 'starting', 'live', 'lost' or 'stopped'; `on_state(name,
    state)` is called from the reader's thread when it changes ('live' and 'lost' only)."""

    def __init__(self, spec: CameraSpec, frames: Latest, retry_s: float = 2.0, on_state: Optional[Callable[[str, str], None]] = None):
        if not isinstance(spec, CameraSpec):
            raise ValueError(f"spec must be a CameraSpec, got {type(spec).__name__}")
        if not isinstance(frames, Latest):
            raise ValueError(f"frames must be a Latest mailbox, got {type(frames).__name__}")
        if isinstance(retry_s, bool) or not isinstance(retry_s, (int, float)) or not retry_s > 0:
            raise ValueError(f"retry_s must be a positive number, got {retry_s!r}")
        if on_state is not None and not callable(on_state):
            raise ValueError("on_state must be callable")
        self.spec, self.frames, self._retry_s, self._on_state = spec, frames, float(retry_s), on_state
        self.state, self.restarts, self.pid = "starting", 0, None
        self._thread, self._proc, self._stop, self._lock, self._started = None, None, threading.Event(), threading.Lock(), False

    def start(self) -> None:
        if self._started:
            raise ValueError("the reader was already started")
        self._started = True
        self._thread = threading.Thread(target=self._run, name=f"camera-{self.spec.name}", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        with self._lock:
            proc = self._proc
        if proc is not None:
            _end(proc)
        if self._thread is not None:
            self._thread.join(_STOP_TIMEOUT_S)
        self.state, self.pid = "stopped", None

    def _set(self, state: str) -> None:
        if state == self.state:
            return
        self.state = state
        if self._on_state is not None:
            self._on_state(self.spec.name, state)

    def _run(self) -> None:
        while not self._stop.is_set():
            try:
                proc = subprocess.Popen(self.spec.command, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, bufsize=0)
            except OSError:
                proc = None
            if proc is not None:
                with self._lock:
                    self._proc = proc
                self.pid = proc.pid
                if self._stop.is_set():
                    _end(proc)
                try:
                    for jpeg in split_jpegs(iter(lambda: proc.stdout.read(_READ_BYTES), b"")):
                        self.frames.put(jpeg)
                        self._set("live")
                except (OSError, ValueError):
                    pass
                finally:
                    _end(proc)
                    with self._lock:
                        self._proc = None
            self.pid = None
            if self._stop.is_set():
                break
            self._set("lost")
            self.restarts += 1
            self._stop.wait(self._retry_s)


def _end(proc: subprocess.Popen) -> None:
    """Stop a child process and reap it."""
    if proc.poll() is None:
        proc.terminate()
        try:
            proc.wait(_STOP_TIMEOUT_S)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait()
    if proc.stdout is not None:
        proc.stdout.close()
    proc.wait()
