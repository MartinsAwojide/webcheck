"""Severity from a defect's area: per class, below the train split's 50th percentile of defect box areas is minor, from
it major, from its 90th percentile critical (static/severity_bands.json). The
bands are quantiles of the synthetic training data, not a quality standard; the defect log carries no severity."""
import json
import math
from functools import lru_cache
from pathlib import Path
from typing import Dict

BANDS_FILE = Path(__file__).parent / "static" / "severity_bands.json"
SEVERITIES = ("minor", "major", "critical")


@lru_cache(maxsize=1)
def load_bands() -> Dict:
    return json.loads(BANDS_FILE.read_text())


def severity(defect_type: str, area_mm2: float) -> str:
    band = load_bands()["classes"].get(defect_type) if isinstance(defect_type, str) else None
    if band is None:
        raise ValueError(f"no severity bands for class {defect_type!r}")
    if isinstance(area_mm2, bool) or not isinstance(area_mm2, (int, float)) or not math.isfinite(area_mm2) or area_mm2 < 0:
        raise ValueError(f"area must be a finite number of mm² from 0, got {area_mm2!r}")
    if area_mm2 >= band["p90_mm2"]:
        return "critical"
    return "major" if area_mm2 >= band["p50_mm2"] else "minor"
