"""The demo launcher for Linux, macOS and Windows: starts, stops and reports the defect map and the
inspection HMI in a demo folder, or drives a Jetson's demo folder from a PC over ssh. Standard library only, so it runs
with the board's plain python3 as well as `uv run webcheck-demo` on a PC.

    webcheck-demo [--dir DIR] start [--clear-db] [SLICES|max] [--open] [--frames DIR]
    webcheck-demo [--dir DIR] stop | status
    webcheck-demo board start [--clear-db] [SLICES|max] | stop | status | log

Each process is started detached with its PID in a file and is stopped by that PID, never by a name."""
import argparse
import io
import os
import platform
import re
import shlex
import signal
import socket
import subprocess
import sys
import tarfile
import time
import webbrowser
from pathlib import Path

HMI_LOG, HMI_PID = "hmi_demo.log", "hmi_demo.pid"
MAP_LOG, MAP_PID = "defect_map.log", "defect_map.pid"
DB = "defect_map.sqlite"
MAX_SLICES = 2 ** 63 - 1
SPLASH = "waiting for the splash"
REMOTE_DIR = "webcheck_demo"
BOARD_SCRIPTS = ("demo_driver.py", "defect_map.py", "webcheck_demo.py")


def parse_slices(text: str) -> int:
    """Slices inspected before the run ends: a positive whole number, or max for no limit."""
    if text == "max":
        return MAX_SLICES
    if not re.fullmatch(r"[1-9][0-9]*", text or ""):
        raise ValueError(f"SLICES must be a positive whole number or max, got {text!r}")
    return int(text)


def _slices_text(text: str) -> str:
    parse_slices(text)
    return text


def external_cameras(sysfs="/sys") -> list:
    """The attached cameras on Linux: V4L2 capture nodes (index 0) whose USB port Linux does not mark built in
    (`removable` = fixed), so a laptop's own webcam is left out; a camera not on USB counts as attached."""
    found = []
    for node in Path(sysfs, "class", "video4linux").glob("video*"):
        try:
            if node.joinpath("index").read_text().strip() != "0":
                continue
            usb = node.joinpath("device").resolve().parent
            removable = usb.joinpath("removable").read_text().strip() if usb.joinpath("removable").is_file() else None
        except OSError:
            continue
        if removable != "fixed" and node.name[5:].isdigit():
            found.append(int(node.name[5:]))
    return [f"/dev/video{n}" for n in sorted(found)]


def hmi_command(folder, python: str, slices: int, frames, port: int = 8080, system: str = None, backend: str = None,
                gpu: str = None, cameras=None) -> list:
    """demo_driver.py with the arguments start_hmi.sh gave it; the cameras (on Linux the external ones found, at most
    two: left, right) are named only on Linux; `backend`
    (tensorrt or onnxruntime) and `gpu` (ONNX Runtime's route) are passed on when given, else the driver picks."""
    system = system or platform.system()
    frames = Path(frames)
    detectors = ["--detector", "small=onnx_det5"]
    if (Path(folder) / "onnx_det_medium" / "trained.onnx").is_file():
        detectors += ["--detector", "medium=onnx_det_medium"]
    if system == "Linux" and cameras is None:
        cameras = external_cameras()
    cameras = [arg for side, dev in zip(("left", "right"), cameras or []) for arg in ("--raw-camera", f"{side}={dev}")] \
        if system == "Linux" else []
    return [python, "demo_driver.py", "--pkg", "pkg", "--build-dir", "build_cu",
            "--gate", "best_recall=onnx_gate6:20.3522", "--gate", "best_f1=onnx_gate7:25.3", *detectors,
            "--station", f"A={frames / 'A_1080'}", "--station", f"B={frames / 'B_1080'}",
            "--engine-cache", "engine_cache", "--out", "driver_hmi.json", "--slices", str(slices), "--ceiling-s", "0.4",
            "--max-queue-bytes", "2684354560", "--lane-width-mm", "550.08", "--serve-port", str(port),
            *cameras, "--hmi", "--mqtt", "localhost", *(["--backend", backend] if backend else []),
            *(["--gpu", gpu] if gpu else [])]


def map_command(python: str, port: int = 8090) -> list:
    return [python, "defect_map.py", "--pkg", "pkg", "--db", DB, "--port", str(port)]


def detach_options(system: str = None) -> dict:
    """Popen options that let a process outlive the launcher and its terminal."""
    if (system or platform.system()) == "Windows":
        return {"creationflags": 0x00000008 | 0x00000200}           # DETACHED_PROCESS | CREATE_NEW_PROCESS_GROUP
    return {"start_new_session": True}


def local_address() -> str:
    """This machine's address on its network, from the route the system would pick (no packet is sent)."""
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
            s.connect(("192.0.2.1", 9))
            return s.getsockname()[0]
    except OSError:
        return "127.0.0.1"


def port_in_use(port: int, host: str = "127.0.0.1") -> bool:
    with socket.socket() as s:
        s.settimeout(0.5)
        return s.connect_ex((host, port)) == 0


def pid_alive(pid: int) -> bool:
    if platform.system() == "Windows":                              # os.kill(pid, 0) would end the process there
        import ctypes
        handle = ctypes.windll.kernel32.OpenProcess(0x1000, False, pid)     # PROCESS_QUERY_LIMITED_INFORMATION
        if not handle:
            return False
        code = ctypes.c_ulong()
        ctypes.windll.kernel32.GetExitCodeProcess(handle, ctypes.byref(code))
        ctypes.windll.kernel32.CloseHandle(handle)
        return code.value == 259                                    # STILL_ACTIVE
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    try:                                                            # a child of this process that already ended
        done, _ = os.waitpid(pid, os.WNOHANG)
        return done == 0
    except ChildProcessError:
        return True


def ssh_command(board: str, remote: str, tty: bool = False) -> list:
    return ["ssh", *(["-t"] if tty else []), board, remote]


class Screen:
    """The launcher's look: banner, step lines, a progress line while the HMI starts and the address box. Colour only
    on a terminal (TERM not dumb, NO_COLOR unset), the progress line only on a terminal, glyphs only when the stream
    is UTF-8; UI_TTY, UI_COLOR and UI_UTF8 (0/1) override."""

    WORDMARK = r"""              _          _               _
__      _____| |__   ___| |__   ___  ___| | __
\ \ /\ / / _ \ '_ \ / __| '_ \ / _ \/ __| |/ /
 \ V  V /  __/ |_) | (__| | | |  __/ (__|   <
  \_/\_/ \___|_.__/ \___|_| |_|\___|\___|_|\_\
"""

    def __init__(self, tty=False, color=False, utf8=False, out=None, err=None):
        self.tty, self.color, self.utf8 = tty, color, utf8
        self.out = out if out is not None else sys.stdout
        self.err = err if err is not None else sys.stderr
        self._frame = 0
        if utf8:
            self.g = dict(step="▸", ok="✔", warn="▲", fail="✖", full="█", empty="░", roll="▓▒░",
                          tl="╭", tr="╮", bl="╰", br="╯", h="─", v="│", spin="⠋⠙⠹⠸⠼⠴⠦⠧⠇⠏")
        else:
            self.g = dict(step=">", ok="ok", warn="!", fail="x", full="#", empty="-", roll="==",
                          tl="+", tr="+", bl="+", br="+", h="-", v="|", spin="|/-\\")

    @classmethod
    def detect(cls, out=None, err=None, env=None):
        env = os.environ if env is None else env
        out = out if out is not None else sys.stdout
        flag = lambda name: None if env.get(name) in (None, "") else env.get(name) == "1"
        tty = flag("UI_TTY")
        if tty is None:
            tty = bool(getattr(out, "isatty", lambda: False)())
        color = flag("UI_COLOR")
        if color is None:
            color = tty and not env.get("NO_COLOR") and env.get("TERM") != "dumb"
        utf8 = flag("UI_UTF8")
        if utf8 is None:
            utf8 = (getattr(out, "encoding", None) or "").lower().replace("-", "") == "utf8"
        if color and os.name == "nt":
            os.system("")                                           # switches the Windows console to ANSI sequences
        return cls(tty=tty, color=bool(color), utf8=utf8, out=out, err=err)

    def _c(self, code: str) -> str:
        return f"\033[{code}m" if self.color else ""

    def _say(self, stream, text: str):
        stream.write(text + "\n")
        stream.flush()

    def banner(self):
        acc, dim, bold, reset = self._c("1;38;5;44"), self._c("2"), self._c("1"), self._c("0")
        g = self.g
        self.out.write(acc + self.WORDMARK + reset)
        self._say(self.out, f"{dim}{g['roll']} print {g['step']} laminate {g['step']} inspect {g['roll']}{reset}")
        self._say(self.out, f"{bold}webcheck{reset}  print-defect inspection on the web: FOMO-AD gate, then YOLO-Pro detector\n")

    def step(self, text: str):
        self._say(self.out, f"  {self._c('38;5;44')}{self.g['step']}{self._c('0')} {text}")

    def ok(self, text: str):
        self._say(self.out, f"  {self._c('38;5;42')}{self.g['ok']}{self._c('0')} {text}")

    def warn(self, text: str):
        self._say(self.err, f"  {self._c('38;5;214')}{self.g['warn']}{self._c('0')} {text}")

    def fail(self, text: str):
        self._say(self.err, f"  {self._c('38;5;203')}{self.g['fail']} {text}{self._c('0')}")

    def bar(self, done: int, total: int, width: int = 24) -> str:
        pct = 100 if total <= 0 else min(max(done, 0), total) * 100 // total
        fill = pct * width // 100
        return (f"[{self._c('38;5;44')}{self.g['full'] * fill}{self._c('2')}{self.g['empty'] * (width - fill)}"
                f"{self._c('0')}] {pct:3d}%")

    def wait(self, label: str, done: int, total: int):
        if not self.tty:
            return
        spin = self.g["spin"][self._frame % len(self.g["spin"])]
        self._frame += 1
        self.out.write(f"\r  {self._c('38;5;44')}{spin}{self._c('0')} {self.bar(done, total, 20)} {label[:44]:<44}")
        self.out.flush()

    def wait_end(self):
        if self.tty:
            self._say(self.out, "")

    def addresses(self, ip: str, hmi_port: int, map_port: int):
        g, w, acc, reset = self.g, 46, self._c("38;5;44"), self._c("0")
        rows = [f"HMI         http://{ip}:{hmi_port}", f"Defect map  http://{ip}:{map_port}", "",
                "On the TV: Chrome, ?square for 1080x1080"]
        lines = [f"\n  {acc}{g['tl']}{g['h'] * w}{g['tr']}{reset}"]
        lines += [f"  {acc}{g['v']}{reset} {row:<{w - 2}} {acc}{g['v']}{reset}" for row in rows]
        lines += [f"  {acc}{g['bl']}{g['h'] * w}{g['br']}{reset}\n"]
        self._say(self.out, "\n".join(lines))


def _read_pid(pidfile: Path):
    try:
        return int(pidfile.read_text().strip())
    except (OSError, ValueError):
        return None


def stop_pid(pidfile: Path, screen: Screen):
    """Stop the process named in `pidfile` by its PID and remove the file; a stale file is just removed."""
    if not pidfile.exists():
        return None
    pid = _read_pid(pidfile)
    stopped = None
    if pid is not None and pid_alive(pid):
        try:
            os.kill(pid, signal.SIGTERM)
        except OSError:
            pass
        for _ in range(80):
            if not pid_alive(pid):
                break
            time.sleep(0.25)
        screen.ok(f"stopped {pid} ({pidfile.name})")
        stopped = pid
    pidfile.unlink(missing_ok=True)
    return stopped


def _spawn(argv: list, folder: Path, log: str):
    env = dict(os.environ, PYTHONUNBUFFERED="1")
    with open(folder / log, "wb") as out:
        return subprocess.Popen(argv, cwd=folder, stdout=out, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL,
                                env=env, **detach_options())


def _tail(path: Path, n: int = 5) -> str:
    try:
        return "\n".join(path.read_text(errors="replace").splitlines()[-n:])
    except OSError:
        return ""


def cmd_start(folder: Path, args, screen: Screen) -> int:
    screen.banner()
    stop_pid(folder / HMI_PID, screen)
    stop_pid(folder / MAP_PID, screen)
    for port in (args.hmi_port, args.map_port):
        if port_in_use(port):
            screen.fail(f"port {port} is in use by a program not started from this folder; stop that one first")
            return 1
    if args.clear_db:
        for name in (DB, DB + "-wal", DB + "-shm"):
            (folder / name).unlink(missing_ok=True)
        screen.ok("defect map database cleared")
    frames = Path(args.frames) if args.frames else Path("..") / "test_frames"
    python = sys.executable
    screen.step(f"defect map on port {args.map_port} (log {MAP_LOG})")
    mapper = _spawn(map_command(python, args.map_port), folder, MAP_LOG)
    (folder / MAP_PID).write_text(str(mapper.pid))
    screen.step(f"HMI on port {args.hmi_port} (log {HMI_LOG})")
    hmi = _spawn(hmi_command(folder, python, parse_slices(args.slices), frames, args.hmi_port, backend=args.backend,
                             gpu=args.gpu),
                 folder, HMI_LOG)
    (folder / HMI_PID).write_text(str(hmi.pid))
    ready, began = False, time.monotonic()
    while True:
        waited = time.monotonic() - began
        log = folder / HMI_LOG
        if SPLASH in log.read_text(errors="replace"):
            ready = True
            break
        if hmi.poll() is not None:
            if SPLASH in log.read_text(errors="replace"):               # it wrote the line, then ended
                ready = True
                break
            screen.wait_end()
            screen.fail(f"the HMI stopped while starting; the end of {HMI_LOG}:")
            screen.err.write(_tail(log) + "\n")
            screen.err.flush()
            (folder / HMI_PID).unlink(missing_ok=True)
            return 1
        if waited >= args.wait_s:
            break
        screen.wait(f"starting the HMI ({int(waited)} s)", int(waited), args.wait_s)
        time.sleep(0.25)
    screen.wait_end()
    if ready:
        screen.ok("HMI ready, waiting at the splash screen (the first start builds the engines after Start)")
    else:
        screen.warn(f"no splash screen after {args.wait_s} s; see {HMI_LOG}")
    screen.addresses(local_address(), args.hmi_port, args.map_port)
    if args.open:
        webbrowser.open(f"http://localhost:{args.hmi_port}")
    screen.step("stop with: python3 webcheck_demo.py stop (or make stop from the repository)")
    return 0


def cmd_stop(folder: Path, screen: Screen) -> int:
    stop_pid(folder / HMI_PID, screen)
    stop_pid(folder / MAP_PID, screen)
    return 0


def cmd_status(folder: Path, screen: Screen) -> int:
    for name in (HMI_PID, MAP_PID):
        pid = _read_pid(folder / name)
        if pid is not None and pid_alive(pid):
            screen.ok(f"{name}: running {pid}")
        else:
            screen.step(f"{name}: not running")
    last = _tail(folder / HMI_LOG, 1)
    if last:
        screen._say(screen.out, last)
    return 0


def _package_tar(src: Path) -> bytes:
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w") as tar:
        tar.add(src, arcname=src.name, filter=lambda info: None if "__pycache__" in info.name else info)
    return buf.getvalue()


def _board_name(screen: Screen):
    board = os.environ.get("BOARD", "").strip()
    config = Path.home() / ".config" / "webcheck" / "board"
    if not board and config.is_file():
        lines = config.read_text().splitlines()
        board = lines[0].strip() if lines else ""
    if not board:
        screen.fail("name the board: set BOARD=user@host, or put user@host in ~/.config/webcheck/board")
    return board or None


def cmd_board(args, screen: Screen) -> int:
    board = _board_name(screen)
    if board is None:
        return 2
    folder = os.environ.get("REMOTE_DIR") or REMOTE_DIR
    q = shlex.quote(folder)
    if args.action == "start":
        repo = Path(__file__).resolve().parents[2]
        sends = [(f"mkdir -p {q}/pkg && tar -C {q}/pkg -xf -", _package_tar(Path(__file__).resolve().parent))]
        sends += [(f"cat > {q}/{name}" + (f" && chmod +x {q}/{name}" if name == "webcheck_demo.py" else ""),
                   (repo / "scripts" / name).read_bytes()) for name in BOARD_SCRIPTS]
        for remote, data in sends:
            done = subprocess.run(ssh_command(board, remote), input=data)
            if done.returncode != 0:
                screen.fail(f"copying to {board} failed: {remote}")
                return done.returncode
        clear = "--clear-db " if args.clear_db else ""
        remote = f"cd {q} && python3 webcheck_demo.py start {clear}{args.slices}"
        return subprocess.run(ssh_command(board, remote, tty=sys.stdout.isatty())).returncode
    if args.action == "log":
        return subprocess.run(ssh_command(board, f"tail -f {q}/{HMI_LOG}")).returncode
    return subprocess.run(ssh_command(board, f"cd {q} && python3 webcheck_demo.py {args.action}")).returncode


def _parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(prog="webcheck-demo", description=__doc__.split("\n\n")[0])
    ap.add_argument("--dir", default=".", help="the demo folder (default: the current folder)")
    sub = ap.add_subparsers(dest="cmd", required=True)
    start = sub.add_parser("start", help="start the defect map and the HMI")
    start.add_argument("--dir", default=argparse.SUPPRESS, help=argparse.SUPPRESS)
    start.add_argument("--clear-db", action="store_true", help="delete the defect map's database first")
    start.add_argument("slices", nargs="?", default="1000000", type=_slices_text, help="SLICES or max (default 1000000)")
    start.add_argument("--open", action="store_true", help="open the HMI in this machine's browser")
    start.add_argument("--frames", help="the folder holding A_1080 and B_1080 (default ../test_frames)")
    start.add_argument("--hmi-port", type=int, default=8080)
    start.add_argument("--map-port", type=int, default=8090)
    start.add_argument("--wait-s", type=int, default=180, help="how long to wait for the HMI's splash screen")
    start.add_argument("--backend", choices=("tensorrt", "onnxruntime"),
                       help="the model runtime (default: TensorRT where installed, else ONNX Runtime)")
    start.add_argument("--gpu", choices=("auto", "direct", "webgpu", "none"),
                       help="ONNX Runtime's GPU route: direct (CUDA, DirectML), webgpu (Vulkan, Metal), none (CPU)")
    for name in ("stop", "status"):
        p = sub.add_parser(name, help=f"{name} the demo in the folder")
        p.add_argument("--dir", default=argparse.SUPPRESS, help=argparse.SUPPRESS)
    board = sub.add_parser("board", help="drive the demo on a Jetson over ssh (BOARD=user@host, REMOTE_DIR)")
    actions = board.add_subparsers(dest="action", required=True)
    bstart = actions.add_parser("start", help="send the code to the board and start the demo there")
    bstart.add_argument("--clear-db", action="store_true")
    bstart.add_argument("slices", nargs="?", default="1000000", type=_slices_text)
    for name in ("stop", "status", "log"):
        actions.add_parser(name)
    return ap


def main(argv=None) -> int:
    args = _parser().parse_args(argv)
    screen = Screen.detect()
    if args.cmd == "board":
        return cmd_board(args, screen)
    folder = Path(args.dir).resolve()
    if args.cmd == "start":
        return cmd_start(folder, args, screen)
    if args.cmd == "stop":
        return cmd_stop(folder, screen)
    return cmd_status(folder, screen)


if __name__ == "__main__":
    sys.exit(main())
