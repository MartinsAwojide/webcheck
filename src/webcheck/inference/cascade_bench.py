"""The cascade's benchmark modes. The gate threshold sets the mode: steady state runs the gate alone
(no tile can be flagged, so YOLO-Pro never runs), worst case flags every inspected tile (YOLO-Pro runs on all of them),
as shipped uses the demo's own threshold. A block inspects a fixed number of slices for `webcheck.bench.run_benchmark`."""
from typing import Callable, Sequence

from webcheck.inference.gate import GateConfig

MODES = ("steady_state", "worst_case", "as_shipped")
_NEVER, _ALWAYS = 1e30, 1e-30             # thresholds no anomaly map reaches / every map reaches (GateConfig wants > 0)


def mode_config(mode: str, threshold: float) -> GateConfig:
    if mode not in MODES:
        raise ValueError(f"mode must be one of {MODES}, got {mode!r}")
    return GateConfig(threshold={"steady_state": _NEVER, "worst_case": _ALWAYS, "as_shipped": threshold}[mode])


def slice_block(slices: Sequence, inspect: Callable, log, size: int) -> Callable[[], int]:
    """A block that inspects `size` slices, cycling through `slices` and numbering them on along the web from one block
    to the next, writes their records to `log`, and returns `size`."""
    if isinstance(size, bool) or not isinstance(size, int) or size < 1:
        raise ValueError(f"a block size must be a positive whole number of slices, got {size!r}")
    if not slices:
        raise ValueError("a block needs at least one slice")
    position = [0]

    def block() -> int:
        for _ in range(size):
            k = position[0]
            outcome = inspect(slices[k % len(slices)], k)
            for record in outcome.records:
                log.write(record)
            position[0] += 1
        return size

    return block
