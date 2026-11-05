"""The defect map: the live defect log drawn over the printed web, like player positions on a pitch.

`scripts/defect_map.py` runs on the board beside the HMI. It subscribes to the whole-record UNS topic of every placement
(`uns.UnsTopics`, `.../+/defect`; the four field topics would count each record five times), keeps every detection in a
SQLite database (`DefectDB`), and serves a page with the UNS log streaming on the left and, on the right, one unit of
the web (one print repeat, both lanes side by side) with the detections of a time window folded onto it as a heatmap
(one layer per class in its colour, the class with the largest typical spot at the bottom and the smallest on top), and their counts by class and area band
under it, and the window's defect rate: DPMO with the sigma level in braces, an opportunity being one inspected tile
of the station and a defective tile one a logged record came from (the driver's `.../+/frame` messages). The
TV's browser does the
drawing; the board only keeps points. Stations are kept apart (two different webs), and the page
counts detections: the demo roll loops, so a printed defect is detected again on every pass. The artwork is placed as
the renders place it."""
import json
import math
import sqlite3
import threading
import time
from collections import deque
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from statistics import NormalDist
from typing import Dict, List, Mapping, Optional

from webcheck.inference.detector import CLASSES
from webcheck.inference.hmi import INSPECTION_TITLES
from webcheck.inference.overlay import CLASS_COLORS
from webcheck.inference.stream import _FONT_DIR, _REG_MARK, FONT_FILES, _font_faces
from webcheck.inference.severity import load_bands, severity
from webcheck.inference.uns import PLACEMENTS, UnsTopics

MAP_TEMPLATE = Path(__file__).parent / "static" / "map_template.png"     # one print repeat of the artwork, both lanes
HEAT_MIN_RADIUS_MM = 8.0       # a blob is never smaller than this, so a tiny defect still shows
HEAT_SPREAD = 1.5              # a blob is drawn to this many times its radius; it fades to nothing at the edge
HEAT_TINT = 0.6                # the light end of a class's ramp: its colour mixed this far toward white
HEAT_SHADE = 0.35              # the dark end: its colour mixed this far toward black


def heat_radius_mm(area_mm2) -> float:
    """A detection's heat blob radius: the radius of a disc of its area, never under HEAT_MIN_RADIUS_MM."""
    area = area_mm2 if _finite(area_mm2) and area_mm2 > 0 else 0.0
    return max(HEAT_MIN_RADIUS_MM, math.sqrt(area / math.pi))


def class_heat_ramps() -> Dict[str, List[tuple]]:
    """Each class's heat ramp in detector order: [light tint, its colour, darker shade], few detections to many."""
    def mix(rgb, to, t):
        return tuple(round(c + (to - c) * t) for c in rgb)
    return {c: [mix(CLASS_COLORS[c], 255, HEAT_TINT), CLASS_COLORS[c], mix(CLASS_COLORS[c], 0, HEAT_SHADE)] for c in CLASSES}


def heat_order() -> List[str]:
    """The heat layers bottom to top: by each class's median defect area in the training split, largest first, so the
    small spots are drawn last and stay visible."""
    bands = load_bands()["classes"]
    return sorted(CLASSES, key=lambda c: (-bands[c]["p50_mm2"], CLASSES.index(c)))


def defect_filter(topics: UnsTopics) -> str:
    """The subscription: the whole record of every placement of the line."""
    return f"{topics.enterprise}/{topics.site}/{topics.area}/{topics.line}/+/defect"


def frame_filter(topics: UnsTopics) -> str:
    """The second subscription: one message per inspected slice of every placement, for the defect rate."""
    return f"{topics.enterprise}/{topics.site}/{topics.area}/{topics.line}/+/frame"


def _whole(value) -> bool:
    return isinstance(value, int) and not isinstance(value, bool) and value >= 0


def dpmo(tiles: int, defective: int) -> Optional[float]:
    """Defective tiles per million tiles; None when no tile was counted."""
    if not _whole(tiles) or not _whole(defective) or defective > tiles:
        raise ValueError(f"tiles and defective must be whole numbers from 0 with defective <= tiles, got {tiles!r}, {defective!r}")
    return None if tiles == 0 else defective / tiles * 1_000_000.0


def sigma_level(rate: Optional[float]) -> Optional[float]:
    """The short-term sigma level of a DPMO: the normal quantile of the yield plus the 1.5 sigma shift. None when there
    is no rate, no defective tile or every tile defective (the quantile is infinite)."""
    if rate is None:
        return None
    if not _finite(rate) or not 0 <= rate <= 1_000_000:
        raise ValueError(f"rate must be a DPMO from 0 to 1,000,000, got {rate!r}")
    if rate in (0, 1_000_000):
        return None
    return NormalDist().inv_cdf(1.0 - rate / 1_000_000.0) + 1.5


def rate_label(tiles: int, defective: int) -> str:
    """'12,345 DPMO (3.75σ)': DPMO first, the sigma level in braces."""
    rate = dpmo(tiles, defective)
    if rate is None:
        return "no tiles counted yet"
    sigma = sigma_level(rate)
    return "%s DPMO (%s)" % (f"{round(rate):,}", "σ —" if sigma is None else f"{sigma:.2f}σ")


def parse_frame(topic: str, payload, topics: UnsTopics) -> Optional[Dict]:
    """A frame message from its UNS topic, or None for another topic or a malformed message."""
    placement = next((p for p in PLACEMENTS if topic == topics.frame(p)), None)
    if placement is None:
        return None
    try:
        message = json.loads(payload)
        frame = {"placement": placement, "frame_index": message["frame_index"], "tiles": message["tiles"],
                 "defective_tiles": message["defective_tiles"]}
    except (ValueError, TypeError, KeyError):
        return None
    if not all(_whole(frame[k]) for k in ("frame_index", "tiles", "defective_tiles")) or frame["defective_tiles"] > frame["tiles"]:
        return None
    return frame


def _finite(value) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def parse_message(topic: str, payload, topics: UnsTopics) -> Optional[Dict]:
    """A defect record from its UNS topic as a map point, or None for another topic or a malformed record."""
    placement = next((p for p in PLACEMENTS if topic == topics.defect(p)), None)
    if placement is None:
        return None
    try:
        record = json.loads(payload)
        location, size = record["location"], record["size"]
        point = {"placement": placement, "defect_type": record["defect_type"], "confidence": record["confidence"],
                 "md_mm": location["md_mm"], "cd_mm": location["cd_mm"], "size_mm2": size["value"],
                 "timestamp": record["timestamp"], "topic": topic}
    except (ValueError, TypeError, KeyError):
        return None
    if point["defect_type"] not in CLASSES or not isinstance(point["timestamp"], str):
        return None
    if not all(_finite(point[k]) for k in ("confidence", "md_mm", "cd_mm", "size_mm2")) or not 0 <= point["confidence"] <= 1:
        return None
    return point


def on_connect(topics: UnsTopics, health: dict):
    """paho (VERSION2) on_connect: subscribe on every connect, so a broker restart does not stop the map."""
    def handler(client, userdata, flags, reason_code, properties):
        client.subscribe(defect_filter(topics), qos=0)
        client.subscribe(frame_filter(topics), qos=0)
        health["connected"] = True
    return handler


def on_disconnect(health: dict):
    def handler(client, userdata, flags, reason_code, properties):
        health["connected"] = False
    return handler


def on_message(store: "DefectDB", topics: UnsTopics, health: dict):
    def handler(client, userdata, message):
        point = parse_message(message.topic, message.payload, topics)
        if point is None:
            frame = parse_frame(message.topic, message.payload, topics)
            if frame is None:
                health["ignored"] = health.get("ignored", 0) + 1
            else:
                store.add_frame(frame["placement"], tiles=frame["tiles"], defective_tiles=frame["defective_tiles"])
            return
        store.add(point)
        health["received"] = health.get("received", 0) + 1
    return handler


class DefectDB:
    """Every detection in SQLite on the board, so the map can show the last 30 minutes, a 12 h shift or a day. Each row
    keeps the record (time, placement, class, confidence, centroid = the box centre along and across the web, area), the
    topic levels, where it falls on one unit of the web (MD modulo the print pitch), its severity and when it arrived.
    Windows are counted exactly in SQL; the heat gets at most `sample` of the newest points. Reset only restarts the
    'since start' view; rows older than `retention_h` are pruned. Each inspected slice's tile counts are kept in their own
    table (time, station, tiles, defective tiles) for the window's defect rate."""

    _COLUMNS = ("id", "received_at", "timestamp", "enterprise", "site", "area", "line", "placement", "defect_type",
                "confidence", "md_mm", "cd_mm", "unit_md_mm", "area_mm2", "severity")

    def __init__(self, path, layout: Mapping, retention_h: float = 48.0, sample: int = 3000, clock=time.time):
        self._db = sqlite3.connect(str(path), check_same_thread=False)
        self._lock, self._clock = threading.Lock(), clock
        self._pitch = layout["sheet_length_mm"] + layout["sheet_gap_mm"]
        self._retention_s, self._sample, self._adds = retention_h * 3600.0, sample, 0
        with self._lock:
            self._db.execute("PRAGMA journal_mode=WAL")
            self._db.execute("CREATE TABLE IF NOT EXISTS defects (id INTEGER PRIMARY KEY AUTOINCREMENT, received_at REAL, "
                             "timestamp TEXT, enterprise TEXT, site TEXT, area TEXT, line TEXT, placement TEXT, defect_type TEXT, "
                             "confidence REAL, md_mm REAL, cd_mm REAL, unit_md_mm REAL, area_mm2 REAL, severity TEXT)")
            self._db.execute("CREATE INDEX IF NOT EXISTS by_time ON defects (placement, received_at)")
            self._db.execute("CREATE TABLE IF NOT EXISTS slices (id INTEGER PRIMARY KEY AUTOINCREMENT, received_at REAL, "
                             "placement TEXT, tiles INTEGER, defective_tiles INTEGER)")
            self._db.execute("CREATE INDEX IF NOT EXISTS slices_by_time ON slices (placement, received_at)")
            self._db.commit()
        self.baseline = clock()

    def add(self, point: Mapping) -> int:
        levels = point["topic"].split("/")
        row = (self._clock(), point["timestamp"], levels[0], levels[1], levels[2], levels[3], point["placement"],
               point["defect_type"], point["confidence"], point["md_mm"], point["cd_mm"], point["md_mm"] % self._pitch,
               point["size_mm2"], severity(point["defect_type"], point["size_mm2"]))
        with self._lock:
            cur = self._db.execute("INSERT INTO defects (%s) VALUES (%s)" % (", ".join(self._COLUMNS[1:]), ", ".join("?" * 14)), row)
            self._db.commit()
            self._adds += 1
        if self._adds % 500 == 0:
            self.prune()
        return cur.lastrowid

    def add_frame(self, placement: str, tiles: int, defective_tiles: int) -> None:
        if placement not in PLACEMENTS:
            raise ValueError(f"placement must be one of {PLACEMENTS}, got {placement!r}")
        dpmo(tiles, defective_tiles)                                    # checks the counts
        with self._lock:
            self._db.execute("INSERT INTO slices (received_at, placement, tiles, defective_tiles) VALUES (?, ?, ?, ?)",
                             (self._clock(), placement, tiles, defective_tiles))
            self._db.commit()
            self._adds += 1
        if self._adds % 500 == 0:
            self.prune()

    def prune(self) -> None:
        with self._lock:
            self._db.execute("DELETE FROM defects WHERE received_at < ?", (self._clock() - self._retention_s,))
            self._db.execute("DELETE FROM slices WHERE received_at < ?", (self._clock() - self._retention_s,))
            self._db.commit()

    def since(self, after: int, limit: int = 200) -> List[Dict]:
        with self._lock:
            rows = self._db.execute("SELECT %s FROM defects WHERE id > ? ORDER BY id LIMIT ?" % ", ".join(self._COLUMNS),
                                    (after, limit)).fetchall()
        return [dict(zip(self._COLUMNS, r)) for r in rows]

    def latest(self, n: int) -> List[Dict]:
        """The newest `n` detections, oldest first: what a fresh page's log opens on."""
        with self._lock:
            rows = self._db.execute("SELECT %s FROM defects ORDER BY id DESC LIMIT ?" % ", ".join(self._COLUMNS),
                                    (max(0, int(n)),)).fetchall()
        return [dict(zip(self._COLUMNS, r)) for r in reversed(rows)]

    def window(self, placement: str, minutes: Optional[float]) -> Dict:
        if placement not in PLACEMENTS:
            raise ValueError(f"placement must be one of {PLACEMENTS}, got {placement!r}")
        if minutes is not None and (isinstance(minutes, bool) or not isinstance(minutes, (int, float)) or not minutes > 0):
            raise ValueError(f"minutes must be a positive number or None (since start), got {minutes!r}")
        start = self.baseline if minutes is None else self._clock() - minutes * 60.0
        op = ">=" if minutes is None else ">"                   # since start includes the start; a window excludes its edge
        with self._lock:
            groups = self._db.execute("SELECT defect_type, severity, COUNT(*) FROM defects WHERE placement = ? AND received_at %s ? " % op + 
                                      "GROUP BY defect_type, severity", (placement, start)).fetchall()
            points = self._db.execute("SELECT defect_type, cd_mm, unit_md_mm, area_mm2, severity FROM defects WHERE placement = ? AND "
                                      "received_at %s ? ORDER BY id DESC LIMIT ?" % op, (placement, start, self._sample)).fetchall()
            tiles, defective = self._db.execute("SELECT COALESCE(SUM(tiles), 0), COALESCE(SUM(defective_tiles), 0) FROM slices "
                                                "WHERE placement = ? AND received_at %s ?" % op, (placement, start)).fetchone()
        counts: Dict[str, Dict[str, int]] = {}
        for defect_type, sev, n in groups:
            counts.setdefault(defect_type, {})[sev] = n
        total = sum(n for _, _, n in groups)
        rate = dpmo(tiles, defective)
        return {"counts": counts, "total": total, "sampled": total > len(points), "from": start,
                "tiles": tiles, "defective_tiles": defective, "dpmo": rate, "sigma": sigma_level(rate),
                "rate": rate_label(tiles, defective),
                "points": [dict(zip(("defect_type", "cd_mm", "unit_md_mm", "area_mm2", "severity"), p)) for p in points]}

    def reset(self) -> None:
        self.baseline = self._clock()

    def __len__(self) -> int:
        with self._lock:
            return self._db.execute("SELECT COUNT(*) FROM defects").fetchone()[0]


def load_layout() -> Dict:
    """The template's geometry in mm (map_template.json)."""
    return json.loads(MAP_TEMPLATE.with_suffix(".json").read_text())


def render_map_page(layout: Mapping) -> str:
    colors = json.dumps({label: "rgb(%d,%d,%d)" % CLASS_COLORS[label] for label in CLASSES})
    ramps = json.dumps({label: [list(c) for c in ramp] for label, ramp in class_heat_ramps().items()})
    return """<!doctype html>
<html lang="en"><head><meta charset="utf-8"><title>Defect map</title>
<style>
:root{--canvas:#0E0F12;--plate:#16171B;--plate-raised:#1D1F24;--hairline:#2A2C33;--hairline-strong:#3D4049;--paper:#F4F5F7;--ink-2:#B4B8C0;--ink-3:#858A94;--on-paper:#0E0F12}
__FONTS__
html,body{margin:0;background:var(--canvas);color:var(--paper);font-family:'IBM Plex Sans',system-ui,sans-serif;font-variant-numeric:tabular-nums;overflow:hidden}
#stage{position:relative;width:1920px;height:1080px;overflow:hidden;transform-origin:0 0}
#header{position:absolute;left:0;right:0;top:0;height:80px;box-sizing:border-box;padding:0 32px;display:flex;align-items:center;border-bottom:1px solid var(--hairline)}
h1{font-family:'Inter Display','IBM Plex Sans',sans-serif;font-size:44px;font-weight:600;letter-spacing:-1.1px;margin:0}#sub{font-family:'Inter Display','IBM Plex Sans',sans-serif;font-size:44px;font-weight:300;letter-spacing:-1.1px;color:var(--ink-2);margin-left:14px}
#wordmark{display:flex;align-items:center;gap:10px;font-size:28px;font-weight:500;letter-spacing:-0.4px;margin-right:28px}
.reg{width:22px;height:22px;flex:0 0 auto}
#rate{margin-left:auto;font-size:24px;font-weight:500;color:var(--paper)}
#chip{margin-left:24px;padding:8px 16px;border:1px solid var(--hairline-strong);border-radius:9999px;font-size:18px;font-weight:600;letter-spacing:1.6px;text-transform:uppercase;color:var(--ink-3)}
#chip.on{color:var(--paper)}
.eyebrow{font-size:18px;font-weight:600;letter-spacing:1.6px;text-transform:uppercase;color:var(--ink-3)}
#log{position:absolute;left:32px;top:96px;width:600px;bottom:32px;box-sizing:border-box;padding:20px 24px;background:var(--plate);overflow:hidden;
clip-path:polygon(0 0,calc(100% - 14px) 0,100% 14px,100% 100%,0 100%)}
#logrows{margin-top:12px;border-top:1px solid var(--hairline)}
.row{padding:10px 0;border-bottom:1px solid var(--hairline)}
.lvl{font-family:'IBM Plex Mono',monospace;font-size:18px;color:var(--ink-3);display:flex;justify-content:space-between}
.def{display:flex;align-items:center;gap:10px;margin-top:4px}
.def .n{font-size:26px;font-weight:600}.def .conf{margin-left:auto;font-size:24px;color:var(--paper)}
.sev{font-size:16px;font-weight:600;letter-spacing:1px;text-transform:uppercase;padding:2px 10px;border-radius:9999px;border:1px solid var(--hairline-strong);color:var(--ink-2)}
.sev.major{color:var(--paper);border-color:var(--ink-2)}.sev.critical{color:var(--on-paper);background:var(--paper);border-color:var(--paper)}
.pos{font-family:'IBM Plex Mono',monospace;font-size:18px;color:var(--ink-2);margin-top:4px;white-space:nowrap}
.sw{width:12px;height:12px;border-radius:3px;flex:0 0 auto}
#tools{position:absolute;left:664px;right:32px;top:96px;height:48px;display:flex;align-items:center;gap:8px}
#tools button,#classes button{height:44px;padding:0 16px;font-family:inherit;font-size:20px;font-weight:500;border-radius:9999px;border:1px solid var(--hairline-strong);background:transparent;color:var(--paper);display:flex;align-items:center;gap:8px}
#tools button.on,#classes button.on{background:var(--paper);color:var(--on-paper);border-color:var(--paper)}
.sep{width:1px;height:32px;background:var(--hairline-strong);margin:0 8px}
#reset{margin-left:auto}
#classes{position:absolute;left:664px;right:32px;top:152px;display:flex;gap:8px}
#classes button{height:40px;font-size:18px}
#map{position:absolute;left:664px;top:204px}
#chart{position:absolute;left:664px;right:32px;top:652px;bottom:84px}
.bar{display:grid;grid-template-columns:150px 1fr 70px;align-items:center;gap:12px;height:38px;margin-top:6px}
.bar .name{font-size:20px;color:var(--ink-2);text-align:right}.bar .total{font-size:22px;font-weight:500}
.track{display:flex;gap:2px;height:28px}
.seg{height:100%;border-radius:2px;display:flex;align-items:center;justify-content:center;font-size:16px;font-weight:600;color:var(--on-paper);overflow:hidden;white-space:nowrap}
#foot{position:absolute;left:664px;right:32px;bottom:16px;font-size:18px;color:var(--ink-3);line-height:1.35}
#foot b{color:var(--paper);font-weight:500}
</style></head><body><div id="stage">
<div id="header"><span id="wordmark">__REG__webcheck</span><h1>Defect map</h1><span id="sub">· Station A</span><span id="rate" title="Defect rate: tiles with at least one detection per million tiles inspected; σ is the short-term sigma level (with the 1.5 shift)"></span><span id="chip">MQTT</span></div>
<div id="log"><div class="eyebrow">UNS defect log · live · times UTC+0</div><div id="logrows"></div></div>
<div id="tools"><button data-st="A" class="on">Station A</button><button data-st="B">Station B</button><span class="sep"></span>
<button data-min="" class="on">Since start</button><button data-min="30">30 min</button><button data-min="720">12 h shift</button><button data-min="1440">24 h day</button>
<span class="sep"></span><button id="dots" class="on">Dots</button><button id="reset">Reset</button></div>
<div id="classes"></div>
<canvas id="map" width="1224" height="440"></canvas>
<div id="chart"><div class="eyebrow">Detections by class · bar split by area band (synthetic data quantiles): minor · major · critical</div><div id="bars"></div></div>
<div id="foot">One unit of the web: one print repeat, both lanes side by side, across the web left to right, along the web from the far end of the
sheet (top). <b id="total">0</b> detections in this view<span id="sampled"></span>; the roll loops, so a printed defect is counted on every pass.</div>
</div>
<script>
var W=1920,H=1080,stage=document.getElementById("stage");
function fit(){var s=Math.min(innerWidth/W,innerHeight/H);stage.style.transform="translate("+(innerWidth-W*s)/2+"px,"+(innerHeight-H*s)/2+"px) scale("+s+")"}
window.addEventListener("resize",fit);fit();
var L=__LAYOUT__,COLORS=__COLORS__,CLASSES=__CLASSES__,HEAT_ORDER=__HEATORDER__,TITLES=__TITLES__,RAMPS=__RAMPS__,MIN_R_MM=__MINR__,SPREAD=__SPREAD__,LOG_ROWS=30,TOP=34,TPL_ALPHA=0.35,SEVS=["minor","major","critical"],SEV_ALPHA={minor:0.45,major:0.72,critical:1};
var BREADTH=L.n_lanes*L.lane_width_mm+(L.n_lanes-1)*L.lane_gap_mm,PITCH=L.sheet_length_mm+L.sheet_gap_mm;
var cv=document.getElementById("map"),ctx=cv.getContext("2d"),s=cv.width/BREADTH;cv.height=Math.round(PITCH*s)+TOP+8;
var layers={},luts={};CLASSES.forEach(function(c){var k=document.createElement("canvas");k.width=cv.width;k.height=cv.height;layers[c]=k;luts[c]=lut(RAMPS[c])});
var tpl=new Image(),station="A",minutes="",cls=null,dots=true,win=null,last=0;
tpl.onload=function(){draw()};tpl.src="/template.png";
function lut(st){var o=[];for(var i=0;i<256;i++){var t=i/255*(st.length-1),k=Math.min(st.length-2,Math.floor(t)),f=t-k;for(var ch=0;ch<3;ch++){o.push(Math.round(st[k][ch]+(st[k+1][ch]-st[k][ch])*f))}}return o}
function blobR(p){return Math.max(MIN_R_MM,Math.sqrt(Math.max(0,p.area_mm2||0)/Math.PI))}
function nameOf(c){return c.replace(/_/g," ")}
function unitXY(p){return [p.cd_mm*s,TOP+(PITCH-p.unit_md_mm)*s]}
function drawUnit(){ctx.fillStyle="#0E0F12";ctx.fillRect(0,0,cv.width,cv.height);ctx.fillStyle="#16171B";ctx.fillRect(0,TOP,BREADTH*s,PITCH*s);
for(var lane=0;lane<L.n_lanes;lane++){var x=lane*(L.lane_width_mm+L.lane_gap_mm)*s;
ctx.fillStyle="#858A94";ctx.font="600 20px 'IBM Plex Sans'";ctx.fillText("LANE "+(lane+1),x,TOP-10);
if(tpl.complete&&tpl.naturalWidth){ctx.globalAlpha=TPL_ALPHA;ctx.drawImage(tpl,x,TOP+L.sheet_gap_mm*s,L.lane_width_mm*s,L.sheet_length_mm*s);ctx.globalAlpha=1}}}
function clipToLanes(){ctx.save();ctx.beginPath();for(var lane=0;lane<L.n_lanes;lane++){ctx.rect(lane*(L.lane_width_mm+L.lane_gap_mm)*s,TOP,L.lane_width_mm*s,PITCH*s)}ctx.clip()}
function drawHeat(list){HEAT_ORDER.forEach(function(c){if(cls!==null&&c!==cls){return}var pts=list.filter(function(p){return p.defect_type===c});if(!pts.length){return}
var k=layers[c],g=k.getContext("2d"),lu=luts[c];g.clearRect(0,0,k.width,k.height);
pts.forEach(function(p){var q=unitXY(p),r=blobR(p)*SPREAD*s,gr=g.createRadialGradient(q[0],q[1],0,q[0],q[1],r);gr.addColorStop(0,"rgba(0,0,0,0.2)");gr.addColorStop(1,"rgba(0,0,0,0)");
g.fillStyle=gr;g.fillRect(q[0]-r,q[1]-r,2*r,2*r)});
var img=g.getImageData(0,0,k.width,k.height),d=img.data;
for(var i=0;i<d.length;i+=4){var a=d[i+3];if(a){var j=a*3;d[i]=lu[j];d[i+1]=lu[j+1];d[i+2]=lu[j+2];d[i+3]=Math.min(235,90+a)}}
g.putImageData(img,0,0);ctx.drawImage(k,0,0)})}
function drawDots(list){list.slice().sort(function(a,b){return b.area_mm2-a.area_mm2}).forEach(function(p){var q=unitXY(p);ctx.beginPath();ctx.arc(q[0],q[1],4,0,2*Math.PI);ctx.fillStyle=COLORS[p.defect_type]||"#F4F5F7";ctx.fill();
ctx.lineWidth=1.5;ctx.strokeStyle="#0E0F12";ctx.stroke()})}
function chart(w){var bars=document.getElementById("bars");while(bars.firstChild){bars.removeChild(bars.firstChild)}var rows=Object.keys(w.counts).map(function(c){var n=0;["minor","major","critical"].forEach(function(v){n+=w.counts[c][v]||0});return [c,n]}).sort(function(a,b){return b[1]-a[1]});var top=rows.length?rows[0][1]:1;rows.forEach(function(r){var bar=document.createElement("div"),nm=document.createElement("div"),tr=document.createElement("div"),tot=document.createElement("div");bar.className="bar";nm.className="name";nm.textContent=nameOf(r[0]);tr.className="track";tot.className="total";tot.textContent=r[1];["minor","major","critical"].forEach(function(v){var n=w.counts[r[0]][v]||0;if(!n){return}var sg=document.createElement("div");sg.className="seg";sg.style.width=(100*n/top)+"%";sg.style.background=COLORS[r[0]].replace("rgb(","rgba(").replace(")",","+SEV_ALPHA[v]+")");sg.style.color=v==="critical"?"#0E0F12":"#F4F5F7";sg.textContent=n>=Math.max(2,top/12)?n+" "+v:"";sg.title=n+" "+v;tr.appendChild(sg)});bar.appendChild(nm);bar.appendChild(tr);bar.appendChild(tot);bars.appendChild(bar)})}
function classes(w){var cl=document.getElementById("classes");while(cl.firstChild){cl.removeChild(cl.firstChild)}var all=document.createElement("button");all.textContent="All classes";all.className=cls===null?"on":"";all.onclick=function(){cls=null;draw()};cl.appendChild(all);
CLASSES.forEach(function(c){if(!w.counts[c]){return}var b=document.createElement("button"),sw=document.createElement("span"),t=document.createElement("span");sw.className="sw";sw.style.background=COLORS[c];t.textContent=nameOf(c);b.appendChild(sw);b.appendChild(t);b.className=c===cls?"on":"";b.onclick=function(){cls=c;draw()};cl.appendChild(b)})}
function draw(){if(!win){return}var list=win.points.filter(function(p){return cls===null||p.defect_type===cls});drawUnit();clipToLanes();drawHeat(list);if(dots){drawDots(list)}ctx.restore();chart(win);classes(win);
document.getElementById("total").textContent=win.total;document.getElementById("sampled").textContent=win.sampled?" (heat from the newest "+win.points.length+")":"";
document.getElementById("sub").textContent="· Station "+station+" · "+TITLES[station];document.getElementById("rate").textContent="Defect rate "+win.rate}
function refresh(){fetch("/window?placement="+station+"&minutes="+minutes).then(function(r){return r.json()}).then(function(w){win=w;draw()}).catch(function(){})}
var rowsEl=document.getElementById("logrows");
function logRow(p){var r=document.createElement("div"),lv=document.createElement("div"),a=document.createElement("span"),tm=document.createElement("span"),df=document.createElement("div"),sw=document.createElement("span"),n=document.createElement("span"),sv=document.createElement("span"),cf=document.createElement("span"),ps=document.createElement("div");r.className="row";lv.className="lvl";a.textContent=p.enterprise+" / "+nameOf(p.site)+" / "+p.placement;tm.textContent=(p.timestamp.slice(0,10)===new Date().toISOString().slice(0,10)?"":p.timestamp.slice(5,10)+" ")+p.timestamp.slice(11,19);lv.appendChild(a);lv.appendChild(tm);df.className="def";sw.className="sw";sw.style.background=COLORS[p.defect_type]||"#F4F5F7";n.className="n";n.style.color=COLORS[p.defect_type]||"#F4F5F7";n.textContent=nameOf(p.defect_type);sv.className="sev "+p.severity;sv.textContent=p.severity;cf.className="conf";cf.textContent=Math.round(p.confidence*100)+"%";df.appendChild(sw);df.appendChild(n);df.appendChild(sv);df.appendChild(cf);ps.className="pos";ps.textContent=p.md_mm.toFixed(0)+" mm along · "+p.cd_mm.toFixed(0)+" mm across · "+p.area_mm2.toFixed(0)+" mm²";r.appendChild(lv);r.appendChild(df);r.appendChild(ps);rowsEl.insertBefore(r,rowsEl.firstChild);while(rowsEl.children.length>LOG_ROWS){rowsEl.removeChild(rowsEl.lastChild)}}
function slideIn(n){if(!window.matchMedia||window.matchMedia("(prefers-reduced-motion: reduce)").matches){return 0}var h=0;for(var i=0;i<n&&i<rowsEl.children.length;i++){h+=rowsEl.children[i].offsetHeight||0}if(!(h>0)){return 0}rowsEl.style.transition="none";rowsEl.style.transform="translateY(-"+h+"px)";void rowsEl.offsetHeight;rowsEl.style.transition="transform .6s cubic-bezier(.2,.8,.2,1)";rowsEl.style.transform="translateY(0)";return h}
function addRows(points){var fresh=points.slice(-LOG_ROWS);fresh.forEach(logRow);return slideIn(fresh.length)}
function poll(){fetch("/points?after="+last).then(function(r){return r.json()}).then(function(d){if(d.points.length){addRows(d.points);last=d.last}})
.catch(function(){}).then(function(){setTimeout(poll,1000)})}
function health(){fetch("/health").then(function(r){return r.json()}).then(function(h){var chip=document.getElementById("chip");
chip.textContent=h.connected?"MQTT live · "+h.received+" received":"MQTT waiting for the broker";chip.className=h.connected?"on":""}).catch(function(){})}
[].forEach.call(document.querySelectorAll("#tools button[data-st]"),function(b){b.onclick=function(){station=b.getAttribute("data-st");cls=null;[].forEach.call(document.querySelectorAll("#tools button[data-st]"),function(x){x.className=x===b?"on":""});refresh()}});
[].forEach.call(document.querySelectorAll("#tools button[data-min]"),function(b){b.onclick=function(){minutes=b.getAttribute("data-min");[].forEach.call(document.querySelectorAll("#tools button[data-min]"),function(x){x.className=x===b?"on":""});refresh()}});
document.getElementById("dots").onclick=function(){dots=!dots;this.className=dots?"on":"";draw()};
document.getElementById("reset").onclick=function(){fetch("/reset",{method:"POST"}).then(function(){refresh()})};
fetch("/points?tail="+LOG_ROWS).then(function(r){return r.json()}).catch(function(){return {points:[],last:0}}).then(function(d){d.points.forEach(logRow);last=d.last;poll()});
refresh();health();setInterval(refresh,2000);setInterval(health,2000);
</script></body></html>""".replace("__FONTS__", _font_faces()).replace("__LAYOUT__", json.dumps(dict(layout))) \
        .replace("__COLORS__", colors).replace("__CLASSES__", json.dumps(list(CLASSES))).replace("__HEATORDER__", json.dumps(heat_order())) \
        .replace("__TITLES__", json.dumps(INSPECTION_TITLES)).replace("__REG__", _REG_MARK).replace("__RAMPS__", ramps) \
        .replace("__MINR__", repr(HEAT_MIN_RADIUS_MM)).replace("__SPREAD__", repr(HEAT_SPREAD))


class MapServer:
    """Serves the map page, its template, its fonts, the log after a number, a time window, a reset and the health."""

    def __init__(self, store: "DefectDB", layout: Mapping, health: dict, host: str = "0.0.0.0", port: int = 8090):
        if isinstance(port, bool) or not isinstance(port, int) or not 0 <= port <= 65535:
            raise ValueError(f"port must be an integer from 0 to 65535, got {port!r}")
        self.store, self.layout, self.health, self.host, self._port = store, layout, health, host, port
        self._server = self._thread = None

    @property
    def port(self) -> int:
        return self._server.server_address[1] if self._server else self._port

    def start(self) -> None:
        store, health = self.store, self.health
        page = render_map_page(self.layout).encode()
        files = {"/template.png": ("image/png", MAP_TEMPLATE.read_bytes())}
        files.update({"/fonts/" + name: ("font/woff2", (_FONT_DIR / name).read_bytes()) for name in FONT_FILES})

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *args):
                pass

            def _reply(self, status, kind, body, cache=False):
                self.send_response(status)
                self.send_header("Content-Type", kind)
                self.send_header("Content-Length", str(len(body)))
                self.send_header("Cache-Control", "max-age=86400" if cache else "no-cache")
                self.end_headers()
                self.wfile.write(body)

            def do_GET(self):
                path, _, query = self.path.partition("?")
                if path == "/":
                    self._reply(200, "text/html; charset=utf-8", page)
                elif path in files:
                    self._reply(200, files[path][0], files[path][1], cache=True)
                elif path == "/points":
                    tail = query.partition("tail=")[2].partition("&")[0]
                    after = query.partition("after=")[2].partition("&")[0]
                    after = int(after) if after.isdigit() else 0
                    points = store.latest(min(int(tail), 200)) if tail.isdigit() else store.since(after)
                    last = points[-1]["id"] if points else after
                    self._reply(200, "application/json", json.dumps({"points": points, "last": last}).encode())
                elif path == "/window":
                    args = dict(part.partition("=")[::2] for part in query.split("&") if part)
                    minutes = args.get("minutes", "")
                    try:
                        window = store.window(args.get("placement", ""), float(minutes) if minutes else None)
                    except ValueError as exc:
                        self._reply(400, "application/json", json.dumps({"error": str(exc)}).encode())
                        return
                    self._reply(200, "application/json", json.dumps(window).encode())
                elif path == "/health":
                    body = {"connected": bool(health.get("connected", False)), "received": health.get("received", 0),
                            "ignored": health.get("ignored", 0), "rows": len(store)}
                    self._reply(200, "application/json", json.dumps(body).encode())
                elif path == "/reset":
                    self._reply(405, "text/plain", b"POST to reset")
                else:
                    self._reply(404, "text/plain", b"not found")

            def do_POST(self):
                if self.path.partition("?")[0] != "/reset":
                    self._reply(404, "text/plain", b"not found")
                    return
                store.reset()
                self._reply(200, "application/json", b'{"ok": true}')

        self._server = ThreadingHTTPServer((self.host, self._port), Handler)
        self._server.daemon_threads = True
        self._thread = threading.Thread(target=self._server.serve_forever, kwargs={"poll_interval": 0.1}, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        if self._server is not None:
            self._server.shutdown()
            self._server.server_close()
            self._thread.join(5.0)
