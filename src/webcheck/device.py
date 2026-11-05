"""What machine the demo runs on and its rung, on Linux, macOS and Windows, for the HMI's header.
Standard library only, and it never raises: a machine it cannot read is still named. Unlike `bench.detect_host`, which
refuses every board but the Orin Nano because the benchmark rig measures only that one, any board is named here; one
outside the project's rungs carries none. The macOS and Windows sources are from references checked 2026-11-05 and
have not been run on those systems."""
import platform
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Optional

ORIN_NANO = "Jetson Orin Nano"
WINDOWS_CPU_KEY = r"HARDWARE\DESCRIPTION\System\CentralProcessor\0"


@dataclass(frozen=True)
class Device:
    name: str
    rung: Optional[str]          # R1 PC or laptop, R2 container, R4 Jetson Orin Nano; None for any other board
    system: str


def _read(path: Path) -> Optional[str]:
    try:
        return path.read_text(errors="replace")
    except (OSError, ValueError):
        return None


def _board_name(model: str) -> str:
    return model.replace(" Engineering Reference Developer Kit", "").replace(" Developer Kit", "")


def _run(argv) -> str:
    return subprocess.run(argv, capture_output=True, text=True, timeout=5, check=True).stdout


def _registry(key: str, value: str) -> str:
    import winreg
    with winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE, key) as handle:
        return str(winreg.QueryValueEx(handle, value)[0])


def _ask(fn: Callable, *args) -> str:
    try:
        return (fn(*args) or "").strip()
    except Exception:
        return ""


def describe_device(root="/", system: str = None, run: Callable = _run, registry: Callable = _registry) -> Device:
    """The machine under `root` (the real one by default) on `system` (this one by default)."""
    system = system or platform.system()
    root = Path(root)
    fallback = lambda: _ask(platform.processor) or _ask(platform.machine) or "unknown device"
    if system == "Linux":
        model = (_read(root / "proc/device-tree/model") or "").strip("\0\n ")
        if model:
            return Device(_board_name(model), "R4" if ORIN_NANO in model else None, system)
        cpu = next((line.split(":", 1)[1].strip() for line in (_read(root / "proc/cpuinfo") or "").splitlines()
                    if line.startswith("model name") and ":" in line), "")
        return Device(cpu or fallback(), "R2" if (root / ".dockerenv").exists() else "R1", system)
    if system == "Darwin":
        return Device(_ask(run, ["sysctl", "-n", "machdep.cpu.brand_string"]) or fallback(), "R1", system)
    if system == "Windows":
        return Device(_ask(registry, WINDOWS_CPU_KEY, "ProcessorNameString") or fallback(), "R1", system)
    return Device(fallback(), "R1", system)
