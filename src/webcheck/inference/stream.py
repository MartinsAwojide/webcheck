"""The demo stream: the annotated picture and the defect panel as a web page on the local network.

One page, standard library only. `/annotated` is an MJPEG stream (multipart/x-mixed-replace, which Chrome shows in an
<img>); it is latest-wins, so a slow viewer skips slices (the defect log never does). `/events` is a server-sent event
stream of the defect panel's records and status: an ordered, bounded log that a reconnecting browser resumes with
Last-Event-ID, so the panel does not skip records. The HMI page (`hmi=True`) shows one camera at full size
beside its instruments, with a control bar; its controls are posted as JSON to `/control`, and its fonts are served from
`/fonts/`."""
import json
import re
from pathlib import Path
import select
import socket
import threading
from collections import OrderedDict, deque
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any, Callable, List, Mapping, Optional, Tuple

from webcheck.inference.hmi import INSPECTION_TITLES, inspection_title
from webcheck.inference.overlay import (CANVAS_HEIGHT, CANVAS_WIDTH, CLASS_COLORS, HEAT_AT_COLOR, HEAT_BELOW_COLOR, HEAT_FLOOR,
                                        HEAT_TOP, HEAT_TOP_COLOR)

BOUNDARY = "frame"
_CAMERA_NAME = re.compile(r"[A-Za-z0-9_-]+\Z")
_PIP_WIDTH = 640                 # the raw view's box on the panel, 16:9 (360 high)
_PIP_GAP = 8
_HMI_PIP_WIDTH = 480             # the raw view on the HMI page, in the picture's bottom-right corner
_HMI_BAR_HEIGHT = 88
_HMI_AREA = (32, 96, 1264, 711)      # left of the instrument column, above the strip (the picture itself fills the stage)
_HMI_GAP = 16
_HMI_COLUMN_WIDTH = 1920 - 2 * _HMI_AREA[0] - _HMI_AREA[2] - _HMI_GAP   # the instrument column, 576 px
_HMI_HISTORY = 10
FONT_FILES = ("IBMPlexSans-Light.woff2", "IBMPlexSans-Regular.woff2", "IBMPlexSans-Medium.woff2", "IBMPlexSans-SemiBold.woff2",
              "IBMPlexMono-Regular.woff2", "InterDisplay-Light.woff2", "InterDisplay-Regular.woff2",
              "InterDisplay-SemiBold.woff2")  # self-hosted, OFL-1.1: the TV may only see the board's network
_FONT_FAMILIES = {"IBMPlexSans": "IBM Plex Sans", "IBMPlexMono": "IBM Plex Mono", "InterDisplay": "Inter Display"}
_FONT_WEIGHTS = {"Light": 300, "Regular": 400, "Medium": 500, "SemiBold": 600}
_FONT_DIR = Path(__file__).parent / "static" / "fonts"
_MAX_CONTROL_BYTES = 1024        # a control is a tiny JSON object; anything bigger is refused unread
_POLL_S = 0.5            # how often a stream handler looks at the stop flag
_KEEPALIVE_S = 15.0


def mjpeg_part(jpeg: bytes) -> bytes:
    if not isinstance(jpeg, bytes):
        raise ValueError(f"a picture must be bytes, got {type(jpeg).__name__}")
    return b"--" + BOUNDARY.encode() + b"\r\nContent-Type: image/jpeg\r\nContent-Length: " + str(len(jpeg)).encode() \
        + b"\r\n\r\n" + jpeg + b"\r\n"


def sse_message(data: Any, event: Optional[str] = None, id: Optional[int] = None) -> bytes:
    """One server-sent event; the JSON data is on a single line."""
    if event is not None and (not isinstance(event, str) or not event or any(c.isspace() for c in event)):
        raise ValueError(f"event must be a single word, got {event!r}")
    head = (f"id: {id}\n" if id is not None else "") + (f"event: {event}\n" if event is not None else "")
    return (head + "data: " + json.dumps(data) + "\n\n").encode()


class Latest:
    """A mailbox that keeps only the newest value; readers ask for anything newer than the version they have."""

    def __init__(self):
        self._cond, self._version, self._value = threading.Condition(), 0, None

    @property
    def version(self) -> int:
        return self._version

    def put(self, value) -> int:
        with self._cond:
            self._version += 1
            self._value = value
            self._cond.notify_all()
            return self._version

    def get(self, after: int, timeout: float) -> Optional[Tuple[int, Any]]:
        with self._cond:
            self._cond.wait_for(lambda: self._version > after, timeout)
            return (self._version, self._value) if self._version > after else None


def state_key(event) -> Optional[Tuple[str, ...]]:
    """What a state event describes (the line, or one camera), so the log can keep the newest of each; None otherwise."""
    if not isinstance(event, dict):
        return None
    if event.get("type") in ("control", "setup", "shade", "heat", "station", "gate", "runtime"):
        return (event["type"],)
    if event.get("type") == "camera" and "name" in event:
        return ("camera", event["name"])
    return None


class EventLog:
    """An ordered log of the newest `maxlen` events with increasing ids that are never reused. With `sticky` (a key
    function such as `state_key`), the newest event of each key is kept even after it leaves the window, so a page
    opened late still learns the current state; it comes back in id order before the window."""

    def __init__(self, maxlen: int = 200, sticky: Optional[Callable[[Any], Any]] = None):
        if isinstance(maxlen, bool) or not isinstance(maxlen, int) or maxlen < 1:
            raise ValueError(f"maxlen must be a positive integer, got {maxlen!r}")
        if sticky is not None and not callable(sticky):
            raise ValueError("sticky must be a key function")
        self._cond, self._events, self._seq = threading.Condition(), deque(maxlen=maxlen), 0
        self._key, self._kept = sticky, {}

    def append(self, event) -> int:
        with self._cond:
            self._seq += 1
            self._events.append((self._seq, event))
            key = self._key(event) if self._key else None
            if key is not None:
                self._kept[key] = (self._seq, event)
            self._cond.notify_all()
            return self._seq

    def _after(self, seq: int) -> List[Tuple[int, Any]]:
        if not self._events:
            return []
        oldest = self._events[0][0]
        kept = sorted((item for item in self._kept.values() if seq < item[0] < oldest), key=lambda item: item[0])
        return kept + [(s, e) for s, e in self._events if s > seq]

    def since(self, seq: int) -> List[Tuple[int, Any]]:
        with self._cond:
            return self._after(seq)

    def wait(self, seq: int, timeout: float) -> List[Tuple[int, Any]]:
        with self._cond:
            self._cond.wait_for(lambda: self._seq > seq, timeout)
            return self._after(seq)


class CropStore:
    """The newest `keep` defect crops (JPEG bytes), each under a number that is never reused."""

    def __init__(self, keep: int = 64):
        if isinstance(keep, bool) or not isinstance(keep, int) or keep < 1:
            raise ValueError(f"keep must be a positive integer, got {keep!r}")
        self._keep, self._lock, self._items, self._next = keep, threading.Lock(), OrderedDict(), 0

    def put(self, jpeg: bytes) -> int:
        with self._lock:
            self._next += 1
            self._items[self._next] = jpeg
            while len(self._items) > self._keep:
                self._items.popitem(last=False)
            return self._next

    def get(self, key: int) -> Optional[bytes]:
        with self._lock:
            return self._items.get(key)


def _check_camera_names(names) -> Tuple[str, ...]:
    if not isinstance(names, tuple) or not all(isinstance(n, str) and _CAMERA_NAME.match(n) for n in names) or len(set(names)) != len(names):
        raise ValueError(f"camera names must be a tuple of distinct words (letters, digits, '-', '_'), got {names!r}")
    return names


def _pip_parts(names: Tuple[str, ...]) -> Tuple[str, str, str, int]:
    """CSS, HTML and script for the raw camera view at the bottom of the panel, and how many log rows fit above it."""
    if not names:
        return "", "", "", 14
    width = (_PIP_WIDTH - _PIP_GAP * (len(names) - 1)) // len(names)
    height = width * 9 // 16
    css = ("#stage{position:relative}#pip{position:absolute;right:32px;bottom:24px;width:%dpx}#pipbox{display:flex;gap:%dpx;width:%dpx}"
           ".cam{position:relative;background:#000;overflow:hidden}.cam img{display:block;width:100%%;height:100%%}"
           ".lost{position:absolute;left:0;right:0;top:0;bottom:0;display:flex;align-items:center;justify-content:center;font-size:22px;"
           "background:rgba(13,17,23,.6)}.lost:empty{display:none}#pipnote{font-size:18px;color:#9fb0c3;margin-top:6px}" % (_PIP_WIDTH, _PIP_GAP, _PIP_WIDTH))
    cams = "".join('<div class="cam" style="width:%dpx;height:%dpx"><img src="/raw/%s" alt="raw camera %s"><div class="lost" id="lost_%s">waiting for the camera</div></div>'
                   % (width, height, n, n, n) for n in names)
    html = '<div id="pip"><div id="pipbox">%s</div><div id="pipnote">raw camera, not inspected</div></div>' % cams
    js = 'else if(m.type==="camera"){var lost=document.getElementById("lost_"+m.name);if(lost){lost.textContent=m.state==="live"?"":"camera "+m.state}}'
    rows = 14 - (len(names) == 1) * 6 - (len(names) > 1) * 3          # the rows the picture-in-picture would cover are not listed
    return css, html, js, rows


def render_page(raw: Tuple[str, ...] = (), hmi: bool = False) -> str:
    """The page, with the picture size and the defect colours filled in. `raw` names the cameras shown in the
    picture-in-picture at the bottom of the panel (none: the page has no raw view). `hmi=True` gives the HMI page."""
    if not isinstance(hmi, bool):
        raise ValueError(f"hmi must be True or False, got {hmi!r}")
    if hmi:
        return _render_hmi(_check_camera_names(raw))
    pip_css, pip_html, pip_js, rows = _pip_parts(_check_camera_names(raw))
    colors = json.dumps({label: "rgb(%d,%d,%d)" % c for label, c in CLASS_COLORS.items()})
    return """<!doctype html>
<html lang="en"><head><meta charset="utf-8"><title>Print-defect inspection demo</title>
<style>
html,body{margin:0;background:#0d1117;color:#e6edf3;font-family:system-ui,sans-serif;overflow:hidden}
#stage{width:1920px;height:1080px;display:grid;grid-template-columns:__W__px 960px}
#picture{width:__W__px;height:__H__px;display:block;background:#000}
#panel{box-sizing:border-box;width:960px;height:1080px;padding:28px 32px;display:flex;flex-direction:column}
h1{font-size:34px;margin:0 0 6px}
#status{font-size:22px;color:#9fb0c3;margin-bottom:18px;min-height:30px}
#defects{list-style:none;margin:0;padding:0;overflow:hidden}
#defects li{display:grid;grid-template-columns:18px 1fr;column-gap:14px;padding:12px 0;border-top:1px solid #243041}
.sw{width:18px;height:18px;border-radius:4px;margin-top:8px}
.t{font-size:28px}.d{font-size:20px;color:#9fb0c3}__PIPCSS__
</style></head><body><div id="stage"><img id="picture" src="/annotated" alt="annotated web">
<div id="panel"><h1>Print-defect inspection</h1><div id="status">waiting for the line</div><ul id="defects"></ul></div>__PIPHTML__</div>
<script>
var colors=__COLORS__,list=document.getElementById("defects"),statusEl=document.getElementById("status"),MAX=__ROWS__;
function row(r){var li=document.createElement("li"),sw=document.createElement("div"),box=document.createElement("div"),
t=document.createElement("div"),d=document.createElement("div");sw.className="sw";sw.style.background=colors[r.defect_type]||"#fff";
t.className="t";t.textContent=r.defect_type+"  "+Math.round(r.confidence*100)+"%";d.className="d";
d.textContent="MD "+r.location.md_mm.toFixed(0)+" mm, CD "+r.location.cd_mm.toFixed(0)+" mm, "+r.size.value.toFixed(0)+" "+r.size.unit+"  "+r.timestamp.slice(11,23);
box.appendChild(t);box.appendChild(d);li.appendChild(sw);li.appendChild(box);return li}
var es=new EventSource("/events");
es.onmessage=function(e){var m=JSON.parse(e.data);
if(m.type==="slice"){m.records.forEach(function(r){list.insertBefore(row(r),list.firstChild)});while(list.children.length>MAX){list.removeChild(list.lastChild)}}
else if(m.type==="status"){statusEl.textContent=m.text}__PIPJS__};
</script></body></html>""".replace("__W__", str(CANVAS_WIDTH)).replace("__H__", str(CANVAS_HEIGHT)).replace("__COLORS__", colors) \
        .replace("__PIPCSS__", pip_css).replace("__PIPHTML__", pip_html).replace("__PIPJS__", pip_js).replace("__ROWS__", str(rows))


def _hmi_pip(names: Tuple[str, ...]) -> Tuple[str, str, str, str]:
    """CSS, HTML, script definitions and the event branch for the HMI's one raw box: it shows the raw camera of the view
    on screen (the page starts on the left), names that camera, and keeps every camera's state while another is shown.
    A side with no live camera shows its inspected web with nothing drawn (/plain/<camera>), in step with the slices.
    The box sits at the bottom of the instrument column."""
    css = ("#pip{position:relative;flex:0 0 auto;width:%dpx}#rawbox{position:relative;width:%dpx;height:%dpx;background:var(--command);overflow:hidden}"
           "#raw{display:block;width:100%%;height:100%%}#rawtag{position:absolute;left:0;top:0;padding:4px 12px;"
           "font-size:18px;font-weight:600;letter-spacing:1.6px;text-transform:uppercase;color:var(--paper);background:var(--plate)}"
           "#rawlost{position:absolute;left:0;right:0;top:0;bottom:0;display:flex;align-items:center;justify-content:center;font-size:18px;"
           "color:var(--ink-2);background:rgba(14,15,18,.7)}#rawlost:empty{display:none}#pipnote{position:absolute;left:0;bottom:0;font-size:18px;color:var(--ink-2);"
           "padding:2px 12px;background:var(--plate)}"
           % (_HMI_COLUMN_WIDTH, _HMI_COLUMN_WIDTH, _HMI_COLUMN_WIDTH * 9 // 16))
    src, note = ("/raw/left", "raw camera") if "left" in names else ("/plain/left", "web view")
    html = ('<div id="pip"><div id="rawbox"><img id="raw" src="%s" alt="camera view"><div id="rawtag">left</div><div id="rawlost"></div>'
            '<div id="pipnote">%s</div></div></div>' % (src, note))
    defs = ('var RAW=%s,camState={},rawView="left",rawImg=document.getElementById("raw"),rawTag=document.getElementById("rawtag"),'
            'rawLost=document.getElementById("rawlost"),pipNote=document.getElementById("pipnote");\n'
            'function showRaw(v){rawView=v;rawTag.textContent=v;var live=RAW.indexOf(v)>=0&&camState[v]==="live",src=live?"/raw/"+v:"/plain/"+v;'
            'if(rawImg.getAttribute("src")!==src){rawImg.src=src}pipNote.textContent=live?"raw camera":"web view";rawLost.textContent=""}\n'
            % json.dumps(list(names), separators=(",", ":")))
    defs += ('var rawBtn=document.getElementById("rawbtn");stage.classList.add("raw-on");\n'
             'rawBtn.onclick=function(){var on=stage.classList.toggle("raw-on");rawBtn.className=on?"icon on":"icon"};\n')
    js = 'else if(m.type==="camera"){camState[m.name]=m.state;if(m.name===rawView){showRaw(rawView)}}'
    return css, html, defs, js


_RAW_BUTTON = ('<button id="rawbtn" class="icon on" aria-label="Show or hide the raw camera" title="Raw camera">'
               '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linejoin="round">'
               '<path d="M3 7h12v10H3zM15 10l6-3v10l-6-3"/></svg><span class="lbl">Camera</span></button>')
_REG_MARK = ('<svg class="reg" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.6">'
             '<circle cx="12" cy="12" r="6"/><path d="M12 1v22M1 12h22"/></svg>')


def _font_faces() -> str:
    faces = []
    for name in FONT_FILES:
        family, _, weight = name[:-len(".woff2")].partition("-")
        faces.append("@font-face{font-family:'%s';font-weight:%d;font-display:block;src:url(/fonts/%s) format('woff2')}"
                     % (_FONT_FAMILIES[family], _FONT_WEIGHTS[weight], name))
    return "\n".join(faces)


def _hex(rgb) -> str:
    return "#%02X%02X%02X" % tuple(rgb)


def _render_hmi(names: Tuple[str, ...]) -> str:
    """The HMI page (DESIGN.md, a press-room instrument panel): the chosen camera's picture in the left two-thirds, the
    counter and the latest defect in a column on its right, the defect history below and the control bar along the
    bottom. Chrome is black, white and grey; the only colours are the defect classes and the heatmap ramp. `P` or the
    present button hides the bar for filming; `?square` gives the 1080x1080 canvas. The line's state comes back as
    'control' events, the running gate's threshold as a 'gate' event."""
    pip_css, pip_html, pip_defs, pip_js = _hmi_pip(names)
    colors = json.dumps({label: "rgb(%d,%d,%d)" % c for label, c in CLASS_COLORS.items()})
    left, top, width, height = _HMI_AREA
    column = left + width + _HMI_GAP
    heat_floor, heat_t = 100.0 * HEAT_FLOOR / HEAT_TOP, 100.0 / HEAT_TOP
    heat = "linear-gradient(90deg,transparent %.1f%%,%s %.1f%%,%s %.1f%%,%s 100%%)" % (
        heat_floor, _hex(HEAT_BELOW_COLOR), heat_t - 0.1, _hex(HEAT_AT_COLOR), heat_t, _hex(HEAT_TOP_COLOR))
    return """<!doctype html>
<html lang="en"><head><meta charset="utf-8"><title>Defect inspection HMI</title>
<style>
:root{--canvas:#0E0F12;--plate:#16171B;--plate-raised:#1D1F24;--command:#08090B;--hairline:#2A2C33;--hairline-strong:#3D4049;--paper:#F4F5F7;--ink-2:#B4B8C0;--ink-3:#858A94;--on-paper:#0E0F12}
__FONTS__
html,body{margin:0;background:var(--canvas);color:var(--paper);font-family:'IBM Plex Sans',system-ui,sans-serif;font-variant-numeric:tabular-nums;overflow:hidden}
#stage{position:relative;width:1920px;height:1080px;overflow:hidden;transform-origin:0 0;background:var(--canvas)}
#stage.square{width:1080px;height:1080px}
.eyebrow{font-size:18px;font-weight:600;letter-spacing:1.6px;text-transform:uppercase;color:var(--ink-3)}
.reg{width:22px;height:22px;flex:0 0 auto}
#header{position:absolute;left:0;right:0;top:0;height:80px;box-sizing:border-box;padding:0 __L__px;display:flex;align-items:center;background:rgba(14,15,18,.85);border-bottom:1px solid var(--hairline)}
#title{font-family:'Inter Display','IBM Plex Sans',sans-serif;font-size:44px;font-weight:600;letter-spacing:-1.1px;line-height:1;margin:0;white-space:nowrap}
#model{font-family:'Inter Display','IBM Plex Sans',sans-serif;font-size:44px;font-weight:300;letter-spacing:-1.1px;line-height:1;color:var(--ink-2);margin-left:14px;white-space:nowrap}
#brand{margin-left:auto;display:flex;align-items:center;gap:24px}
#wordmark{display:flex;align-items:center;gap:10px;font-size:28px;font-weight:500;letter-spacing:-0.4px;margin-right:28px}
#live{display:flex;align-items:center;gap:10px;padding:8px 16px;border:1px solid var(--hairline-strong);border-radius:9999px;color:var(--ink-3)}
#live .eyebrow{color:inherit}#live.on{color:var(--paper)}#live.on .reg{animation:pulse 1s ease-in-out infinite}
@keyframes pulse{50%{opacity:.4}}
#picture{position:absolute;left:0px;top:0px;width:1920px;height:1080px;display:block;background:var(--command)}
#overload{position:absolute;left:__L__px;top:__T__px;width:__W__px;box-sizing:border-box;display:none;padding:12px 20px;font-size:22px;font-weight:600;letter-spacing:1.6px;text-transform:uppercase;background:var(--paper);color:var(--on-paper)}
#panel{position:absolute;left:__C__px;top:__T__px;width:__CW__px;height:__CH__px;display:flex;flex-direction:column;gap:__G__px}
.plate{position:relative;box-sizing:border-box;padding:24px;background:rgba(22,23,27,.85);clip-path:polygon(0 0,calc(100% - 14px) 0,100% 14px,100% 100%,0 100%)}
#counter{flex:0 0 auto}
#countrow{display:flex;align-items:baseline;gap:14px;margin-top:8px}
#count{font-family:'Inter Display','IBM Plex Sans',sans-serif;font-size:112px;font-weight:400;line-height:.9;letter-spacing:-3.4px}
.unit{font-size:22px;color:var(--ink-2)}
#metres{font-size:22px;font-weight:500;letter-spacing:-0.2px;margin-top:10px;white-space:nowrap;overflow:hidden;text-overflow:ellipsis}
#aside{font-size:18px;color:var(--ink-3);margin-top:6px}#aside:empty{display:none}
#legend{display:grid;grid-template-columns:repeat(3,1fr);gap:6px 12px;margin-top:14px}
#legend div{display:flex;align-items:center;gap:8px;font-size:18px;color:var(--ink-2);white-space:nowrap}
#legend b{font-size:22px;font-weight:500;color:var(--paper)}
.sw{width:14px;height:14px;border-radius:4px;flex:0 0 auto}
#telemetry{width:100%;margin-top:14px;border-collapse:collapse;border-top:1px solid var(--hairline);font-family:'IBM Plex Mono',monospace;font-size:18px;line-height:1.4;color:var(--ink-2);white-space:nowrap}
#telemetry th,#telemetry td{padding:6px 18px 0 0;text-align:left;font-weight:400}#telemetry th{color:var(--ink-3)}
#latest{flex:0 0 auto}
#latesthead{display:flex;justify-content:space-between}
#latestbody{display:flex;gap:16px;margin-top:16px}
#latestcrop{box-sizing:border-box;width:136px;height:136px;flex:0 0 auto;border:3px solid var(--hairline-strong);background:var(--plate-raised);object-fit:cover;visibility:hidden}
#latesttext{min-width:0}
#latestname{font-family:'Inter Display','IBM Plex Sans',sans-serif;font-size:40px;font-weight:600;letter-spacing:-0.8px;line-height:1;color:var(--ink-3)}
#latestconf{font-family:'Inter Display','IBM Plex Sans',sans-serif;font-size:56px;font-weight:400;letter-spacing:-1.4px;line-height:1;margin-top:6px}
#latestdata{font-size:18px;line-height:1.4;color:var(--ink-2);margin-top:8px;white-space:nowrap}
#scoretrack{position:relative;height:10px;margin-top:16px;background:var(--hairline)}
#scorefill{position:absolute;left:0;top:0;bottom:0;width:0}
#scoretick{position:absolute;top:-4px;bottom:-4px;width:2px;margin-left:-1px;background:var(--paper);display:none}
#scorelabel{font-size:18px;color:var(--ink-3);margin-top:6px}
#latest.flash #latestname{animation:flash .3s}
@keyframes flash{50%{opacity:.6}}
#strip{position:absolute;left:__L__px;right:__SR__px;top:__ST__px;height:152px;display:none;align-items:center;gap:24px;padding:16px 24px;background:rgba(22,23,27,.85)}
#striphead{width:180px;flex:0 0 auto}
#striphead .sub{font-size:18px;color:var(--ink-3);margin-top:4px}
#history{display:flex;align-items:flex-start;gap:12px;flex:0 0 auto}
.slot{box-sizing:border-box;width:90px;height:88px;border-radius:4px;border:1px dashed var(--hairline-strong)}
.slot.full{position:relative;overflow:hidden;border:2px solid;display:flex;flex-direction:column;align-items:center;justify-content:flex-end;background:var(--plate-raised);animation:fadein .2s}
.slot.full img{position:absolute;left:0;top:0;width:100%;height:100%;object-fit:cover}
.slot.full .n{position:relative;width:100%;text-align:center;font-size:18px;color:var(--paper);background:rgba(14,15,18,.75)}
.slotwrap .tm{font-size:18px;color:var(--ink-3);text-align:center;margin-top:4px}
@keyframes fadein{from{opacity:0}}
#heatlegend{position:absolute;left:__L__px;bottom:__HB__px;box-sizing:border-box;width:392px;padding:12px 16px;display:none;background:rgba(22,23,27,.85)}
.history-on #strip{display:flex}.history-on #heatlegend{bottom:__HBS__px}
#heatbar{height:10px;margin:10px 0 6px;border:1px solid var(--hairline-strong);background:__HEAT__}
#heatbar{position:relative}
#heatneedle{position:absolute;top:-5px;left:0;width:4px;height:20px;margin-left:-2px;background:var(--paper);display:none;transition:left .15s}
#heatread{font-size:18px;color:var(--ink-2);margin-top:4px;white-space:nowrap}
#heatlabels{position:relative;height:24px;font-size:18px;color:var(--ink-3)}
#heatlabels span{position:absolute;top:0;white-space:nowrap}
#bar{position:absolute;left:0;right:0;bottom:0;height:__BAR__px;box-sizing:border-box;padding:0 __L__px;display:flex;align-items:center;gap:16px;background:rgba(8,9,11,.85);border-top:1px solid var(--hairline)}
#bar button{height:52px;padding:0 24px;font-family:inherit;font-size:22px;font-weight:500;border-radius:9999px;border:1px solid var(--hairline-strong);background:transparent;color:var(--paper)}
#bar button.on{background:var(--paper);color:var(--on-paper);border-color:var(--paper)}
#pause{background:var(--paper);color:var(--on-paper);border-color:var(--paper)}
#bar button:disabled{opacity:.35}
#bar button.icon{min-width:52px;padding:0 14px;border-radius:4px;display:flex;align-items:center;justify-content:center}#bar button.icon svg{display:block;width:28px;height:28px}
#speed{-webkit-appearance:none;appearance:none;width:360px;height:24px;background:transparent}
#speed::-webkit-slider-runnable-track{height:4px;background:var(--hairline-strong)}
#speed::-webkit-slider-thumb{-webkit-appearance:none;width:24px;height:24px;margin-top:-10px;border-radius:50%;background:var(--paper)}
#speed:disabled{opacity:.35}
#speedval{font-size:28px;font-weight:500;min-width:150px}
#meta{margin-left:24px;display:flex;flex-direction:column;gap:4px}#meta div{display:flex;align-items:center}
#status{font-size:18px;color:var(--ink-3);white-space:nowrap}
#devbtn{margin-left:20px;height:40px;padding:0 18px;font-family:inherit;font-size:18px;border-radius:9999px;border:1px solid var(--hairline-strong);background:transparent;color:var(--ink-2);cursor:pointer}
#devbtn[aria-expanded="true"]{background:var(--paper);color:var(--on-paper);border-color:var(--paper)}
#devpanel{display:none;position:absolute;top:96px;left:24px;z-index:20;padding:18px 24px;border-radius:12px;border:1px solid var(--hairline-strong);background:var(--canvas);color:var(--paper)}
#devpanel.open{display:block}#devpanel dl{display:grid;grid-template-columns:auto auto;gap:6px 24px;margin:0;font-size:18px}
#devpanel dt{color:var(--ink-3)}#devpanel dd{margin:0}
#logchip{margin-left:20px;padding:6px 14px;border:1px solid var(--hairline-strong);border-radius:9999px;font-size:18px;color:var(--ink-2);white-space:nowrap}
#logchip:empty{display:none}#logchip.tick{animation:flash .6s}
#icons{margin-left:auto;display:flex;gap:10px}
#bar button.icon .lbl{font-size:18px;margin-left:8px}
#edge{position:absolute;left:0;right:0;bottom:0;height:24px;display:none;z-index:5}
.present #bar{display:none}.present #edge{display:block}.present #strip{top:__PST__px}.present #panel{height:__PCH__px}
.present #heatlegend{bottom:__L__px}.present.history-on #heatlegend{bottom:__HBP__px}
#stage:not(.raw-on) #pip{display:none}
.raw-on .plate{padding:16px 20px}.raw-on #countrow{margin-top:4px}.raw-on #count{font-size:64px;letter-spacing:-2px}.raw-on #metres{font-size:22px;margin-top:6px;white-space:nowrap}
.raw-on #legend{margin-top:10px}.raw-on #telemetry{margin-top:10px;line-height:1.3}
.raw-on #latestbody{margin-top:10px}.raw-on #latestcrop{width:96px;height:96px}.raw-on #latestname{font-size:26px}
.raw-on #latestconf{font-size:34px}.raw-on #latestdata{margin-top:4px}.raw-on #scoretrack{margin-top:10px}
.square #title,.square #model{font-size:32px}.square #wordmark{display:none}
.square #picture{left:__L__px;top:__T__px;width:1016px;height:571px}
.square #overload{width:1016px}
.square #panel{left:__L__px;top:683px;width:1016px;height:365px;flex-direction:row}
.square #latest{order:1;flex:0 0 600px}.square #counter{order:2;flex:1 1 auto}.square #count{font-size:88px}.square #telemetry{display:none}
.square #metres{font-size:22px;white-space:nowrap}
.square #strip{display:none}
.square #bar{display:none}
.square #pip,.square #edge{display:none}
#splash{position:absolute;left:0;top:0;right:0;bottom:0;display:none;align-items:center;justify-content:center;background:rgba(14,15,18,.96);z-index:10}
#splash .card{box-sizing:border-box;width:760px;padding:48px;background:var(--plate-raised);clip-path:polygon(0 0,calc(100% - 14px) 0,100% 14px,100% 100%,0 100%)}
#splash h2{font-size:44px;font-weight:600;letter-spacing:-1.1px;line-height:1;margin:0 0 10px}#splash .sub{color:var(--ink-2);font-size:22px;margin-bottom:8px}
#splash .field{margin:24px 0}#splash .k{display:block;margin-bottom:10px}
#splash .opts{display:flex;gap:12px}
#splash .opts button{height:52px;padding:0 28px;font-family:inherit;font-size:22px;font-weight:500;border-radius:9999px;border:1px solid var(--hairline-strong);background:transparent;color:var(--paper)}
#splash .opts button.on{background:var(--paper);color:var(--on-paper);border-color:var(--paper)}
#spspeed{font-family:inherit;font-size:28px;width:160px;padding:8px 12px;border-radius:8px;border:1px solid var(--hairline-strong);background:var(--canvas);color:var(--paper)}
#startbtn{display:block;width:100%;height:64px;margin-top:16px;font-family:inherit;font-size:26px;font-weight:600;border:0;border-radius:9999px;background:var(--paper);color:var(--on-paper)}
#spmsg{margin-top:14px;color:var(--ink-2);font-size:18px;min-height:24px}__PIPCSS__
</style></head><body><div id="stage">
<img id="picture" src="/annotated" alt="annotated web">
<div id="header"><span id="wordmark">__REG__webcheck</span><h1 id="title">Defect inspection</h1><span id="model">· Model Cascade</span><div id="meta"><div><span id="status">waiting for the line</span><span id="logchip"></span><button id="devbtn" aria-expanded="false" aria-controls="devpanel">Device</button></div></div>
<div id="brand"><span id="live">__REG__<span class="eyebrow" id="livetext">STANDBY</span></span></div></div>
<div id="devpanel" role="region" aria-label="Device"><dl id="devrows"><dt>Runtime</dt><dd>not known yet</dd></dl></div>
<div id="overload"></div>
<div id="panel"><div id="counter" class="plate"><div class="eyebrow">Inspected</div>
<div id="countrow"><span id="count">0</span><span class="unit">slices</span></div><div id="metres">0.0 m of web</div>
<div id="legend"></div><div id="aside"></div><table id="telemetry"><tr><th>Model</th><th>Latency</th><th>Tiles</th></tr><tr><td>FOMO-AD</td><td id="tgms">–</td><td id="tgtiles">–</td></tr><tr><td id="tdname">YOLO-Pro (small)</td><td id="tdms">–</td><td id="tdtiles">–</td></tr></table></div>
<div id="latest" class="plate"><div id="latesthead"><span class="eyebrow">Latest defect</span><span id="latesttime" class="eyebrow"></span></div>
<div id="latestbody"><img id="latestcrop" alt="the latest defect, cropped from its camera"><div id="latesttext"><div id="latestname">none yet</div><div id="latestconf"></div><div id="latestdata"></div></div></div>
<div id="scoretrack"><div id="scorefill"></div><div id="scoretick"></div></div><div id="scorelabel">confidence</div></div>__PIPHTML__</div>
<div id="strip" class="plate"><div id="striphead"><div class="eyebrow">Defect history</div><div class="sub">newest first</div></div>
<div id="history">__SLOTS__</div></div>
<div id="heatlegend"><div class="eyebrow">Gate heatmap</div><div id="heatbar"><span id="heatneedle"></span></div><div id="heatlabels"><span style="left:0">low</span><span id="heatt" style="left:__HT__%;transform:translateX(-50%)">threshold</span><span style="right:0">high</span></div><div id="heatread">waiting for a slice</div></div>
<div id="bar"><button id="view">Camera: left</button><button id="pause">Pause</button><button id="step" disabled>Step</button>
<input id="speed" type="range" min="1" max="450" step="1" disabled><span id="speedval">– m/min</span>
<span id="icons"><button id="logbtn" class="icon" aria-label="Show or hide the inspection log" title="Defect history">
<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round"><path d="M8 6h13M8 12h13M8 18h13M3 6h.01M3 12h.01M3 18h.01"/></svg><span class="lbl">History</span></button>__RAWBTN__<button id="shadebtn" class="icon" aria-label="Show or hide the columns this camera does not inspect" title="Uninspected columns">
<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><rect x="3" y="4" width="18" height="16"/><path d="M9 4v16M15 4v16"/><path d="M4 8l4-3M4 13l4-4M4 18l4-4" stroke-width="1.5"/></svg><span class="lbl">Shade</span></button><button id="heatbtn" class="icon" aria-label="Show or hide the gate's anomaly heatmap" title="Anomaly heatmap (yellow below the gate threshold, red from it)">
<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><rect x="3" y="3" width="18" height="18"/><rect x="9" y="9" width="6" height="6" fill="currentColor"/><path d="M3 9h18M3 15h18M9 3v18M15 3v18" stroke-width="1"/></svg><span class="lbl">Heatmap</span></button><button id="presentbtn" class="icon" aria-label="Present: hide the controls for filming" title="Present (P)">
<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M3 8V3h5M16 3h5v5M21 16v5h-5M8 21H3v-5"/></svg><span class="lbl">Present</span></button></span></div>
<div id="edge"></div>
<div id="splash"><div class="card"><h2 id="sptitle">__UNKNOWN__</h2><div class="sub">Choose the station and how the line starts</div>
<div class="field"><span class="k eyebrow">Station (placement)</span><div class="opts" id="spplace"></div></div>
<div class="field"><label class="k eyebrow" for="spspeed">Line speed, m/min (1 to 450)</label><input id="spspeed" type="number" min="1" max="450" step="1"></div>
<div class="field"><span class="k eyebrow">First camera view</span><div class="opts" id="spview"><button data-v="left">Left</button><button data-v="right">Right</button></div></div>
<div class="field" id="spdevfield" style="display:none"><span class="k eyebrow">Device</span><div class="opts" id="spdev"></div></div>
<div class="field" id="spgatefield" style="display:none"><span class="k eyebrow">Anomaly gate</span><div class="opts" id="spgate"></div></div>
<div class="field" id="spdetfield" style="display:none"><span class="k eyebrow">Detector (YOLO-Pro)</span><div class="opts" id="spdet"></div></div>
<button id="startbtn">Start the line</button><div id="spmsg"></div></div></div></div>
<script>
var W=1920,H=1080,stage=document.getElementById("stage");
if(location.search.indexOf("square")>=0){stage.classList.add("square");W=1080}
function fit(){var s=Math.min(innerWidth/W,innerHeight/H);stage.style.transform="translate("+(innerWidth-W*s)/2+"px,"+(innerHeight-H*s)/2+"px) scale("+s+")"}
window.addEventListener("resize",fit);fit();
var HISTORY=__HISTORY__,colors=__COLORS__,statusEl=document.getElementById("status");
var viewEl=document.getElementById("view"),pauseEl=document.getElementById("pause"),stepEl=document.getElementById("step"),
speedEl=document.getElementById("speed"),speedVal=document.getElementById("speedval"),line={paused:false,view:"left"},dragging=false;
var liveEl=document.getElementById("live"),liveText=document.getElementById("livetext");
var splash=document.getElementById("splash"),spPlace=document.getElementById("spplace"),spSpeed=document.getElementById("spspeed"),
spView=document.getElementById("spview"),spMsg=document.getElementById("spmsg"),started=false,choice={placement:null,view:"left"};
var titles=__TITLES__,titleEl=document.getElementById("title"),spTitle=document.getElementById("sptitle");
function titleOf(p){return titles[p]||"__UNKNOWN__"}
function station(p){titleEl.textContent=titleOf(p);document.title=titleOf(p)+" HMI"}
function nameOf(c){return c.replace(/_/g," ")}
function send(c){return fetch("/control",{method:"POST",headers:{"Content-Type":"application/json"},body:JSON.stringify(c)})
.then(function(r){return r.json()}).catch(function(){return {ok:false,error:"the board did not answer"}})}
function show(s){started=true;splash.style.display="none";line=s;viewEl.textContent="Camera: "+s.view;pauseEl.textContent=s.paused?"Resume":"Pause";
stepEl.disabled=!s.paused;speedEl.disabled=false;if(!dragging){speedEl.value=Math.round(s.speed_m_per_min)}
speedVal.textContent=Math.round(s.speed_m_per_min)+" m/min";liveText.textContent=s.paused?"PAUSED":"LIVE";liveEl.className=s.paused?"":"on";
if(typeof showRaw==="function"){showRaw(s.view)}}
var logBtn=document.getElementById("logbtn");
logBtn.onclick=function(){var on=stage.classList.toggle("history-on");logBtn.className=on?"icon on":"icon"};
var shadeBtn=document.getElementById("shadebtn"),shadeOn=false;
shadeBtn.onclick=function(){send({shade:!shadeOn})};
var heatBtn=document.getElementById("heatbtn"),heatOn=false,heatLegend=document.getElementById("heatlegend");
heatBtn.onclick=function(){send({heat:!heatOn})};
var present=false,presentBtn=document.getElementById("presentbtn");
function setPresent(on){present=on;if(on){stage.classList.add("present")}else{stage.classList.remove("present")}presentBtn.className=on?"icon on":"icon"}
presentBtn.onclick=function(){setPresent(true)};
document.getElementById("edge").onclick=function(){setPresent(false)};
document.addEventListener("keydown",function(e){if(e.target===spSpeed){return}if(e.key==="p"||e.key==="P"){setPresent(!present)}});
var countEl=document.getElementById("count"),metresEl=document.getElementById("metres"),tgMs=document.getElementById("tgms"),
tgTiles=document.getElementById("tgtiles"),tdMs=document.getElementById("tdms"),tdTiles=document.getElementById("tdtiles"),overEl=document.getElementById("overload");
var heatT=document.getElementById("heatt"),heatNeedle=document.getElementById("heatneedle"),heatRead=document.getElementById("heatread"),gateThreshold=null;
function gateT(t){gateThreshold=t;heatT.textContent="threshold "+t.toFixed(1)}
var tdName=document.getElementById("tdname"),logChip=document.getElementById("logchip"),lastLogged=null,devRows=document.getElementById("devrows"),devPanel=document.getElementById("devpanel"),devBtn=document.getElementById("devbtn");
devBtn.onclick=function(){var open=!devPanel.classList.contains("open");devPanel.classList.toggle("open",open);devBtn.setAttribute("aria-expanded",open?"true":"false")};
function runtime(m){while(devRows.firstChild){devRows.removeChild(devRows.firstChild)}[["Accelerator",m.accelerator],["Runtime",m.runtime],["Device",m.device],["Number format",m.precision],["Rung",m.rung||"no rung"]].forEach(function(r){var dt=document.createElement("dt"),dd=document.createElement("dd");dt.textContent=r[0];dd.textContent=r[1];devRows.appendChild(dt);devRows.appendChild(dd)});devBtn.textContent="Device · "+m.accelerator}
function logTag(t){if(t.log_records===undefined){return}var s=t.log_records+" records logged";if(t.mqtt_sent!==undefined){s+=" · MQTT "+t.mqtt_sent+" sent"+(t.mqtt_failed?", "+t.mqtt_failed+" failed":"")}if(lastLogged!==null&&t.log_records!==lastLogged){logChip.classList.remove("tick");void logChip.offsetWidth;logChip.classList.add("tick")}lastLogged=t.log_records;logChip.textContent=s}
var asideEl=document.getElementById("aside"),asideN=0;
function telemetry(t){logTag(t);if(t.set_aside){asideN+=t.set_aside;asideEl.textContent=asideN+" set aside: classes this station cannot have"}if(t.slices!==undefined){countEl.textContent=t.slices}metresEl.textContent=t.web_m.toFixed(1)+" m inspected · at "+(t.roll_md_mm/1000).toFixed(1)+(t.roll_m!==undefined?" of "+t.roll_m.toFixed(1)+" m test roll":" m of the test roll");if(t.gate_max!==undefined&&gateThreshold){heatNeedle.style.display="block";heatNeedle.style.left=Math.min(100,100*t.gate_max/(gateThreshold*__HEATTOP__))+"%";heatRead.textContent="this slice: highest "+t.gate_max.toFixed(1)+" · "+t.flagged_tiles+" tiles flagged"}if(t.gate_ms!==undefined){tgMs.textContent=t.gate_ms.toFixed(1)+" ms"}if(t.flagged_tiles!==undefined){tgTiles.textContent=t.flagged_tiles+" flagged"}if(t.detector_ms!==undefined){tdMs.textContent=t.detector_ms.toFixed(1)+" ms"}if(t.bbox_tiles!==undefined){tdTiles.textContent=t.bbox_tiles+" with boxes"}
if(t.state==="OVERLOADED"){overEl.textContent="▲ Overloaded — "+t.lag_slices+" slices / "+t.lag_s.toFixed(1)+" s behind the belt";overEl.style.display="block"}
else{overEl.style.display="none"}}
var counts={},legendEl=document.getElementById("legend");
function legend(){while(legendEl.firstChild){legendEl.removeChild(legendEl.firstChild)}Object.keys(colors).forEach(function(c){if(!counts[c]){return}var d=document.createElement("div"),sw=document.createElement("span"),n=document.createElement("b"),l=document.createElement("span");sw.className="sw";sw.style.background=colors[c];n.textContent=counts[c];l.textContent=nameOf(c);d.appendChild(sw);d.appendChild(n);d.appendChild(l);legendEl.appendChild(d)})}
var latestEl=document.getElementById("latest"),latestName=document.getElementById("latestname"),latestConf=document.getElementById("latestconf"),
latestData=document.getElementById("latestdata"),latestTime=document.getElementById("latesttime"),scoreFill=document.getElementById("scorefill"),
latestCrop=document.getElementById("latestcrop"),scoreTick=document.getElementById("scoretick"),scoreLabel=document.getElementById("scorelabel");
latestCrop.onerror=function(){latestCrop.style.visibility="hidden"};
function latest(r){var c=colors[r.defect_type]||"var(--paper)";latestName.textContent=nameOf(r.defect_type);latestName.style.color=c;latestConf.textContent=Math.round(r.confidence*100)+"%";latestData.textContent=r.location.md_mm.toFixed(0)+" mm along · "+r.location.cd_mm.toFixed(0)+" mm across · "+r.size.value.toFixed(0)+" "+r.size.unit.replace("mm2","mm²");latestTime.textContent=r.timestamp.slice(11,19)+" (UTC+0)";if(r.crop){latestCrop.src=r.crop;latestCrop.style.borderColor=c;latestCrop.style.visibility="visible"}else{latestCrop.style.visibility="hidden"}if(r.gate_score!==undefined&&gateThreshold){var top=gateThreshold*__HEATTOP__;scoreFill.style.width=Math.min(100,100*r.gate_score/top)+"%";scoreTick.style.left=(100/__HEATTOP__)+"%";scoreTick.style.display="block";scoreLabel.textContent="gate score "+r.gate_score.toFixed(1)+" · T "+gateThreshold.toFixed(1)}else{scoreFill.style.width=Math.round(r.confidence*100)+"%";scoreTick.style.display="none";scoreLabel.textContent="confidence"}scoreFill.style.background=c;latestEl.classList.remove("flash");void latestEl.offsetWidth;latestEl.classList.add("flash")}
var historyEl=document.getElementById("history");
function history(r){var w=document.createElement("div"),s=document.createElement("div"),n=document.createElement("div"),tm=document.createElement("div");w.className="slotwrap";s.className="slot full";s.style.borderColor=colors[r.defect_type]||"var(--paper)";if(r.crop){var im=document.createElement("img");im.src=r.crop;im.alt="";im.onerror=function(){im.style.display="none"};s.appendChild(im)}n.className="n";n.textContent=nameOf(r.defect_type)+" "+Math.round(r.confidence*100)+"%";tm.className="tm";tm.textContent=r.timestamp.slice(11,19);s.appendChild(n);w.appendChild(s);w.appendChild(tm);historyEl.insertBefore(w,historyEl.firstChild);while(historyEl.children.length>HISTORY){historyEl.removeChild(historyEl.lastChild)}}
function pick(box,v){[].forEach.call(box.children,function(b){b.className=b.getAttribute("data-v")===v?"on":""})}
function setup(m){if(started){return}
while(spPlace.firstChild){spPlace.removeChild(spPlace.firstChild)}
m.placements.forEach(function(p){var b=document.createElement("button");b.setAttribute("data-v",p);b.textContent=p;
b.onclick=function(){choice.placement=p;pick(spPlace,p);spTitle.textContent=titleOf(p)};spPlace.appendChild(b)});
choice.placement=m.placements[0];pick(spPlace,choice.placement);spTitle.textContent=titleOf(choice.placement);spSpeed.value=Math.round(m.speed_m_per_min);
choice.view=m.view;pick(spView,m.view);
var spGate=document.getElementById("spgate");while(spGate.firstChild){spGate.removeChild(spGate.firstChild)}
choice.gate=null;if(m.gates&&m.gates.length){m.gates.forEach(function(g){var b=document.createElement("button");b.setAttribute("data-v",g);
b.textContent=nameOf(g).replace(/f1/i,"F1");b.onclick=function(){choice.gate=g;pick(spGate,g)};spGate.appendChild(b)});
choice.gate=m.gate;pick(spGate,m.gate);document.getElementById("spgatefield").style.display=""}
var spDev=document.getElementById("spdev");while(spDev.firstChild){spDev.removeChild(spDev.firstChild)}
choice.device=null;if(m.devices&&m.devices.length){m.devices.forEach(function(d){var b=document.createElement("button");b.setAttribute("data-v",d.key);
b.textContent=d.label;b.onclick=function(){choice.device=d.key;pick(spDev,d.key);spSpeed.value=Math.round(d.speed)};spDev.appendChild(b)});
choice.device=m.device;pick(spDev,m.device);m.devices.forEach(function(d){if(d.key===m.device){spSpeed.value=Math.round(d.speed)}});document.getElementById("spdevfield").style.display=""}
var spDet=document.getElementById("spdet");while(spDet.firstChild){spDet.removeChild(spDet.firstChild)}
choice.detector=null;if(m.detectors&&m.detectors.length){m.detectors.forEach(function(d){var b=document.createElement("button");b.setAttribute("data-v",d);
b.textContent=d;b.onclick=function(){choice.detector=d;pick(spDet,d)};spDet.appendChild(b)});
choice.detector=m.detector;pick(spDet,m.detector);document.getElementById("spdetfield").style.display=""}
splash.style.display="flex"}
[].forEach.call(spView.children,function(b){b.onclick=function(){choice.view=b.getAttribute("data-v");pick(spView,choice.view)}});
document.getElementById("startbtn").onclick=function(){var v=Number(spSpeed.value);
if(!(v>=1&&v<=450)){spMsg.textContent="the speed must be 1 to 450 m/min";return}
spMsg.textContent="starting the line";
send({start:{placement:choice.placement,speed:v,view:choice.view,gate:choice.gate||undefined,detector:choice.detector||undefined,device:choice.device||undefined}}).then(function(a){if(!a.ok){spMsg.textContent=a.error}})};
viewEl.onclick=function(){send({view:line.view==="left"?"right":"left"})};
pauseEl.onclick=function(){send({pause:!line.paused})};
stepEl.onclick=function(){send({step:true})};
speedEl.oninput=function(){dragging=true;speedVal.textContent=speedEl.value+" m/min"};
speedEl.onchange=function(){dragging=false;send({speed:Number(speedEl.value)})};
__PIPDEFS__var es=new EventSource("/events");
es.onmessage=function(e){var m=JSON.parse(e.data);
if(m.type==="slice"){m.records.forEach(function(r){counts[r.defect_type]=(counts[r.defect_type]||0)+1;history(r)});
if(m.records.length){latest(m.records[m.records.length-1]);legend()}
if(m.telemetry){telemetry(m.telemetry)}}
else if(m.type==="status"){statusEl.textContent=m.text}
else if(m.type==="control"){show(m.state)}
else if(m.type==="setup"){setup(m)}
else if(m.type==="station"){station(m.placement);if(m.detector){tdName.textContent="YOLO-Pro ("+m.detector+")"}}
else if(m.type==="gate"){gateT(m.threshold)}
else if(m.type==="runtime"){runtime(m)}
else if(m.type==="shade"){shadeOn=m.on;shadeBtn.className=m.on?"icon on":"icon"}
else if(m.type==="heat"){heatOn=m.on;heatBtn.className=m.on?"icon on":"icon";heatLegend.style.display=m.on?"block":"none"}__PIPJS__};
</script></body></html>""".replace("__FONTS__", _font_faces()).replace("__COLORS__", colors) \
        .replace("__TITLES__", json.dumps(INSPECTION_TITLES)).replace("__UNKNOWN__", inspection_title(None)) \
        .replace("__HISTORY__", str(_HMI_HISTORY)).replace("__SLOTS__", '<div class="slot"></div>' * _HMI_HISTORY) \
        .replace("__HEATTOP__", "%g" % HEAT_TOP).replace("__HEAT__", heat).replace("__HT__", "%.1f" % heat_t).replace("__BAR__", str(_HMI_BAR_HEIGHT)).replace("__REG__", _REG_MARK) \
        .replace("__L__", str(left)).replace("__T__", str(top)).replace("__W__", str(width)).replace("__H__", str(height)) \
        .replace("__C__", str(column)).replace("__CW__", str(1920 - left - column)).replace("__G__", str(_HMI_GAP)) \
        .replace("__ST__", str(top + height + _HMI_GAP)).replace("__SR__", str(1920 - left - width)) \
        .replace("__CH__", str(1080 - _HMI_BAR_HEIGHT - _HMI_GAP - top)).replace("__PCH__", str(1080 - left - top)) \
        .replace("__HBS__", str(1080 - top - height)).replace("__HBP__", str(left + 152 + _HMI_GAP)) \
        .replace("__HB__", str(_HMI_BAR_HEIGHT + _HMI_GAP)).replace("__PST__", str(1080 - left - 152)) \
        .replace("__PIPCSS__", pip_css).replace("__PIPHTML__", pip_html).replace("__PIPDEFS__", pip_defs).replace("__PIPJS__", pip_js) \
        .replace("__RAWBTN__", _RAW_BUTTON)


PAGE_HTML = render_page()


def web_page(camera: str) -> str:
    """A full-screen page of one camera's plain stream: the test frame with nothing drawn on it, for a camera to film."""
    _check_camera_names((camera,))
    return ('<!doctype html><html lang="en"><head><meta charset="utf-8"><title>web, %s camera</title><style>'
            'html,body{margin:0;background:#000;overflow:hidden;cursor:none}'
            '#web{display:block;width:100vw;height:100vh;object-fit:contain;background:#000}</style></head>'
            '<body><img id="web" src="/plain/%s" alt="%s camera, plain"></body></html>') % (camera, camera, camera)


class DisplayServer:
    """Serves the page, the picture stream and the event stream; `port=0` picks a free port (see `.port`)."""

    def __init__(self, frames: Latest, events: EventLog, host: str = "0.0.0.0", port: int = 8080, raw: Optional[Mapping[str, Any]] = None,
                 hmi: bool = False, on_control: Optional[Callable[[dict], Any]] = None, plain: Optional[Mapping[str, Latest]] = None,
                 crops: Optional[CropStore] = None):
        if isinstance(port, bool) or not isinstance(port, int) or not 0 <= port <= 65535:
            raise ValueError(f"port must be an integer from 0 to 65535, got {port!r}")
        if not isinstance(hmi, bool):
            raise ValueError(f"hmi must be True or False, got {hmi!r}")
        if on_control is not None and not callable(on_control):
            raise ValueError("on_control must be callable")
        if crops is not None and not isinstance(crops, CropStore):
            raise ValueError(f"crops must be a CropStore, got {type(crops).__name__}")
        self.hmi, self.on_control, self.crops = hmi, on_control, crops
        if raw is not None and not isinstance(raw, Mapping):
            raise ValueError(f"raw must map camera names to feeds, got {type(raw).__name__}")
        raw = dict(raw or {})
        _check_camera_names(tuple(raw))
        for name, feed in raw.items():
            if not isinstance(getattr(feed, "frames", None), Latest) or not hasattr(feed, "state"):
                raise ValueError(f"the feed of camera {name!r} needs a `frames` mailbox and a `state`")
        self.raw = raw
        if plain is not None and not isinstance(plain, Mapping):
            raise ValueError(f"plain must map camera names to mailboxes, got {type(plain).__name__}")
        plain = dict(plain or {})
        _check_camera_names(tuple(plain))
        if not all(isinstance(box, Latest) for box in plain.values()):
            raise ValueError("each plain stream needs a Latest mailbox")
        self.plain, self._watching, self._watch_lock = plain, {name: 0 for name in plain}, threading.Lock()
        self.frames, self.events, self.host, self._port = frames, events, host, port
        self._server, self._thread, self._stopping = None, None, threading.Event()

    @property
    def port(self) -> int:
        return self._server.server_address[1] if self._server else self._port

    def watching(self, camera: str) -> int:
        """How many viewers have the camera's plain stream open."""
        with self._watch_lock:
            return self._watching.get(camera, 0)

    def _watch(self, camera: str, change: int) -> None:
        with self._watch_lock:
            self._watching[camera] += change

    def start(self) -> None:
        frames, events, stopping, raw, on_control = self.frames, self.events, self._stopping, self.raw, self.on_control
        plain, watch, crops = self.plain, self._watch, self.crops
        page = render_page(raw=tuple(raw), hmi=self.hmi).encode()
        fonts = {name: (_FONT_DIR / name).read_bytes() for name in FONT_FILES} if self.hmi else {}
        web_pages = {name: web_page(name).encode() for name in plain}

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *args):
                pass

            def _reply(self, status, content_type, body):
                self.send_response(status)
                self.send_header("Content-Type", content_type)
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

            def _stream_headers(self, content_type):
                self.send_response(200)
                self.send_header("Content-Type", content_type)
                self.send_header("Cache-Control", "no-cache, no-store")
                self.end_headers()

            def do_GET(self):
                path = self.path.split("?")[0]
                try:
                    if path == "/":
                        self._reply(200, "text/html; charset=utf-8", page)
                    elif path == "/health":
                        body = {"ok": True, "pictures": frames.version}
                        if raw:
                            body["raw"] = {name: {"pictures": feed.frames.version, "state": feed.state} for name, feed in raw.items()}
                        if plain:
                            body["plain"] = {name: box.version for name, box in plain.items()}
                        self._reply(200, "application/json", json.dumps(body).encode())
                    elif path == "/annotated":
                        self._pictures(frames)
                    elif path.startswith("/raw/") and path[len("/raw/"):] in raw:
                        self._pictures(raw[path[len("/raw/"):]].frames)
                    elif path.startswith("/web/") and path[len("/web/"):] in web_pages:
                        self._reply(200, "text/html; charset=utf-8", web_pages[path[len("/web/"):]])
                    elif path.startswith("/plain/") and path[len("/plain/"):] in plain:
                        name = path[len("/plain/"):]
                        watch(name, 1)
                        try:
                            self._pictures(plain[name])
                        finally:
                            watch(name, -1)
                    elif path.startswith("/fonts/") and path[len("/fonts/"):] in fonts:
                        self.send_response(200)
                        self.send_header("Content-Type", "font/woff2")
                        self.send_header("Content-Length", str(len(fonts[path[len("/fonts/"):]])))
                        self.send_header("Cache-Control", "max-age=86400")
                        self.end_headers()
                        self.wfile.write(fonts[path[len("/fonts/"):]])
                    elif path.startswith("/crop/") and crops is not None:
                        self._crop(path[len("/crop/"):])
                    elif path == "/events":
                        self._events()
                    elif path == "/control" and on_control is not None:
                        self._reply(405, "text/plain", b"POST a control")
                    else:
                        self._reply(404, "text/plain", b"not found")
                except (BrokenPipeError, ConnectionResetError, ConnectionAbortedError):
                    pass

            def do_POST(self):
                if self.path.split("?")[0] != "/control" or on_control is None:
                    self._reply(404, "text/plain", b"not found")
                    return
                try:
                    length = int(self.headers.get("Content-Length", ""))
                except ValueError:
                    length = -1
                if length < 0:
                    self._answer(400, {"ok": False, "error": "a control needs a Content-Length"})
                    return
                if length > _MAX_CONTROL_BYTES:
                    self.close_connection = True
                    self._answer(413, {"ok": False, "error": f"a control is at most {_MAX_CONTROL_BYTES} bytes"})
                    return
                try:
                    command = json.loads(self.rfile.read(length).decode("utf-8"))
                    if not isinstance(command, dict):
                        raise ValueError("a control is a JSON object")
                    state = on_control(command)
                except (ValueError, UnicodeDecodeError) as exc:
                    self._answer(400, {"ok": False, "error": str(exc)})
                    return
                self._answer(200, {"ok": True, "state": state})

            def _crop(self, key):
                jpeg = crops.get(int(key)) if key.isdigit() else None
                if jpeg is None:
                    self._reply(404, "text/plain", b"not found")
                    return
                self.send_response(200)
                self.send_header("Content-Type", "image/jpeg")
                self.send_header("Content-Length", str(len(jpeg)))
                self.send_header("Cache-Control", "max-age=3600")
                self.end_headers()
                self.wfile.write(jpeg)

            def _answer(self, status, body):
                self._reply(status, "application/json", json.dumps(body).encode())

            def _gone(self) -> bool:
                """Whether the viewer has closed the connection (it sends nothing more, so a readable socket means EOF)."""
                try:
                    readable, _, _ = select.select([self.connection], [], [], 0)
                    return bool(readable) and self.connection.recv(1, socket.MSG_PEEK) == b""
                except OSError:
                    return True

            def _pictures(self, mailbox):
                self._stream_headers(f"multipart/x-mixed-replace; boundary={BOUNDARY}")
                seen = 0
                while not stopping.is_set() and not self._gone():
                    item = mailbox.get(seen, _POLL_S)
                    if item is not None:
                        seen, jpeg = item
                        self.wfile.write(mjpeg_part(jpeg))
                        self.wfile.flush()

            def _events(self):
                try:
                    last = max(0, int(self.headers.get("Last-Event-ID", "0")))
                except ValueError:
                    last = 0
                self._stream_headers("text/event-stream")
                quiet = 0.0
                while not stopping.is_set():
                    items = events.wait(last, _POLL_S)
                    for seq, event in items:
                        self.wfile.write(sse_message(event, id=seq))
                        last = seq
                    quiet = 0.0 if items else quiet + _POLL_S
                    if quiet >= _KEEPALIVE_S:
                        self.wfile.write(b": keep-alive\n\n")
                        quiet = 0.0
                    self.wfile.flush()

        server = ThreadingHTTPServer((self.host, self._port), Handler)
        server.daemon_threads = True
        self._server = server
        self._thread = threading.Thread(target=server.serve_forever, kwargs={"poll_interval": 0.1}, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stopping.set()
        if self._server is not None:
            self._server.shutdown()
            self._server.server_close()
            self._thread.join(5.0)
