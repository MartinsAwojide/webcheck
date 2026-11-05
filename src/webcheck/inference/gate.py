"""The FOMO-AD gate: the tile's score is the max of the 40x40 anomaly map; a tile is flagged when the score
reaches the threshold. The threshold is a policy choice, so it has no default."""
import math
from dataclasses import dataclass

import numpy as np

MAP_SIZE = 40


@dataclass(frozen=True)
class GateConfig:
    threshold: float

    def __post_init__(self):
        if not (isinstance(self.threshold, (int, float)) and math.isfinite(self.threshold) and self.threshold > 0):
            raise ValueError(f"gate threshold must be a positive finite number, got {self.threshold!r}")


def gate_score(anomaly_map) -> float:
    m = np.asarray(anomaly_map)
    if m.shape == (1, MAP_SIZE, MAP_SIZE):
        m = m[0]
    if m.shape != (MAP_SIZE, MAP_SIZE):
        raise ValueError(f"anomaly map must be 1x{MAP_SIZE}x{MAP_SIZE} or {MAP_SIZE}x{MAP_SIZE}, got {m.shape}")
    if np.isnan(m).any():
        raise ValueError("anomaly map contains NaN")
    return float(m.max())


def is_anomalous(score: float, config: GateConfig) -> bool:
    return bool(score >= config.threshold)
