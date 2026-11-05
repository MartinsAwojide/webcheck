"""The log inspector: a defect log checked against the ground truth of the frames that were replayed. Ground
truth is one entry per defect (`instance_id`), its extent the union of the annotations' `web_extent_mm` in the log's own web
coordinates. A record matches a defect of its type when its centre lies in the defect's extent grown by the cameras'
de-duplication margin, most confident record first, nearest defect first; on a defect of another type it is a wrong type,
on none a false alarm. Types the detector does not have are counted apart."""
import math
import statistics
from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Optional, Sequence, Set, Tuple

from webcheck.inference.detector import CLASSES
from webcheck.inference.overlap import DEDUP_GROW_MM


@dataclass(frozen=True)
class GroundTruth:
    instance_id: str
    label: str
    md0: float
    md1: float
    cd0: float
    cd1: float

    @property
    def centre(self) -> Tuple[float, float]:
        return (self.md0 + self.md1) / 2, (self.cd0 + self.cd1) / 2

    @property
    def area_mm2(self) -> float:
        return (self.md1 - self.md0) * (self.cd1 - self.cd0)

    def holds(self, md: float, cd: float, grow: float) -> bool:
        return self.md0 - grow <= md <= self.md1 + grow and self.cd0 - grow <= cd <= self.cd1 + grow


def ground_truth(coco: Dict, files: Set[str]) -> List[GroundTruth]:
    """One GroundTruth per defect with at least one annotation in a frame named in `files`."""
    names = {c["id"]: c["name"] for c in coco["categories"]}
    images = {i["id"]: i["file_name"] for i in coco["images"]}
    extents: Dict[str, List] = {}
    seen: Set[str] = set()
    for a in coco["annotations"]:
        if "instance_id" not in a or "web_extent_mm" not in a:
            raise ValueError(f"annotation {a.get('id')} needs its instance_id and web_extent_mm")
        cd0, md0, cd1, md1 = (float(v) for v in a["web_extent_mm"])
        key, label = a["instance_id"], names[a["category_id"]]
        if key in extents:
            e = extents[key]
            extents[key] = [label, min(e[1], md0), max(e[2], md1), min(e[3], cd0), max(e[4], cd1)]
        else:
            extents[key] = [label, md0, md1, cd0, cd1]
        if images[a["image_id"]] in files:
            seen.add(key)
    truth = [GroundTruth(key, *extents[key]) for key in seen]
    return sorted(truth, key=lambda g: (g.md0, g.cd0, g.instance_id))


@dataclass(frozen=True)
class Match:
    record: Dict
    instance_id: str
    label: str
    location_error_mm: float
    size_ratio: float


@dataclass
class Inspection:
    matches: List[Match] = field(default_factory=list)
    missed: List[GroundTruth] = field(default_factory=list)
    false_alarms: List[Dict] = field(default_factory=list)
    wrong_type: List[Tuple[Dict, GroundTruth]] = field(default_factory=list)
    not_detectable: List[GroundTruth] = field(default_factory=list)
    records: int = 0

    def summary(self) -> Dict:
        per_type = {}
        for label in CLASSES:
            matched = sum(1 for m in self.matches if m.label == label)
            defects = matched + sum(1 for g in self.missed if g.label == label)
            alarms = sum(1 for r in self.false_alarms if r["defect_type"] == label)
            wrong = sum(1 for r, _ in self.wrong_type if r["defect_type"] == label)
            records = matched + alarms + wrong
            per_type[label] = {"defects": defects, "matched": matched, "missed": defects - matched, "false_alarms": alarms,
                               "wrong_type": wrong, "recall": matched / defects if defects else None,
                               "precision": matched / records if records else None}
        defects = len(self.matches) + len(self.missed)
        not_detectable: Dict[str, int] = {}
        for g in self.not_detectable:
            not_detectable[g.label] = not_detectable.get(g.label, 0) + 1
        confusion: Dict[str, int] = {}
        for r, g in self.wrong_type:
            name = f"{r['defect_type']} on {g.label}"
            confusion[name] = confusion.get(name, 0) + 1
        return {"per_type": per_type,
                "overall": {"defects": defects, "matched": len(self.matches), "records": self.records,
                            "recall": len(self.matches) / defects if defects else None,
                            "precision": len(self.matches) / self.records if self.records else None},
                "not_detectable": not_detectable, "wrong_type": confusion,
                "location_error_mm": _spread([m.location_error_mm for m in self.matches]),
                "size_ratio": _spread([m.size_ratio for m in self.matches])}


def _spread(values: Sequence[float]) -> Optional[Dict[str, float]]:
    if not values:
        return None
    return {"median": statistics.median(values), "mean": statistics.fmean(values), "min": min(values), "max": max(values)}


def _distance(g: GroundTruth, md: float, cd: float) -> float:
    gm, gc = g.centre
    return math.hypot(md - gm, cd - gc)


def inspect_log(records: Iterable[Dict], truth: Sequence[GroundTruth], grow_mm: float = DEDUP_GROW_MM) -> Inspection:
    """Match log records (as written to the JSON Lines log) to the ground truth."""
    records = list(records)
    for r in records:
        if r["defect_type"] not in CLASSES:
            raise ValueError(f"a log record's defect_type must be one of {CLASSES}, got {r['defect_type']!r}")
    result = Inspection(records=len(records))
    detectable = [g for g in truth if g.label in CLASSES]
    result.not_detectable = [g for g in truth if g.label not in CLASSES]
    open_truth = list(detectable)
    for r in sorted(records, key=lambda r: -r["confidence"]):
        md, cd = r["location"]["md_mm"], r["location"]["cd_mm"]
        same = [g for g in open_truth if g.label == r["defect_type"] and g.holds(md, cd, grow_mm)]
        if same:
            g = min(same, key=lambda g: _distance(g, md, cd))
            open_truth.remove(g)
            result.matches.append(Match(r, g.instance_id, g.label, _distance(g, md, cd), r["size"]["value"] / g.area_mm2))
            continue
        other = [g for g in truth if g.label != r["defect_type"] and g.holds(md, cd, grow_mm)]
        if other:
            result.wrong_type.append((r, min(other, key=lambda g: _distance(g, md, cd))))
        else:
            result.false_alarms.append(r)
    result.missed = [g for g in detectable if g in open_truth]
    return result
