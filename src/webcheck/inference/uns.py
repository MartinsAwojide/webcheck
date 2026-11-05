"""The defect log on Unified Namespace (UNS) MQTT topics. Topics follow ISA-95,
`<Enterprise>/<Site>/<Area>/<Line>/Placement_<A|B>/defect`, each placement its own work cell. The `defect` topic carries
the whole record, the same JSON as its log-file line; `defect/class`, `/confidence`, `/location` and `/size` each carry
the timestamp and that one field. `frame` carries one message per inspected slice with its tile count, the tiles a logged
record came from and its record count, so a reader can count tiles and defective tiles (the defect map's DPMO). The JSON Lines file stays the record of truth: `PublishingLog` writes the file first and
publishes second, and publishing never raises into the inspection. The client is paho-mqtt (the client behind Arduino App
Lab's MQTT brick), at QoS 0 with a bounded queue."""
import json
from datetime import datetime, timezone
from typing import List, Tuple

from webcheck.inference.defect_log import DefectRecord

PLACEMENTS = ("A", "B")
FIELD_TOPICS = {"class": "defect_type", "confidence": "confidence", "location": "location", "size": "size"}
DEFAULT_PORT = 1883


def _level(value, name: str) -> str:
    if not isinstance(value, str) or not value or any(c in value for c in "/+# \x00"):
        raise ValueError(f"{name} must be a non-empty MQTT topic level without '/', '+', '#', spaces or NUL, got {value!r}")
    return value


def _placement(value) -> str:
    if not isinstance(value, str) or value not in PLACEMENTS:
        raise ValueError(f"placement must be one of {PLACEMENTS}, got {value!r}")
    return value


class UnsTopics:
    """The topic tree of one line: enterprise and site are the converter company, area and line default to the printing and lamination line."""

    def __init__(self, enterprise: str, site: str, area: str = "Printing_Lamination", line: str = "Line_1"):
        self.enterprise, self.site = _level(enterprise, "enterprise"), _level(site, "site")
        self.area, self.line = _level(area, "area"), _level(line, "line")

    def defect(self, placement: str) -> str:
        return f"{self.enterprise}/{self.site}/{self.area}/{self.line}/Placement_{_placement(placement)}/defect"

    def field(self, placement: str, name: str) -> str:
        if name not in FIELD_TOPICS:
            raise ValueError(f"field topic must be one of {tuple(FIELD_TOPICS)}, got {name!r}")
        return f"{self.defect(placement)}/{name}"

    def frame(self, placement: str) -> str:
        return f"{self.enterprise}/{self.site}/{self.area}/{self.line}/Placement_{_placement(placement)}/frame"


def _count(value, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{name} must be a whole number from 0, got {value!r}")
    return value


def tile_counts(outcome) -> Tuple[int, int]:
    """(tiles, defective tiles) of one inspected slice: every gate tile of every camera, and each tile a logged record
    came from, counted once."""
    tiles = sum(len(result.tiles) for result in outcome.results.values())
    return tiles, len({(source.camera, tuple(source.tile)) for source in outcome.sources})


def frame_payload(frame_index: int, tiles: int, defective_tiles: int, defects: int, timestamp: datetime) -> str:
    """JSON text of one inspected slice: its index, its tiles, the tiles a record came from, how many records it logged,
    and when (UTC, milliseconds)."""
    if not isinstance(timestamp, datetime) or timestamp.tzinfo is None:
        raise ValueError(f"timestamp must be a timezone-aware datetime, got {timestamp!r}")
    if _count(defective_tiles, "defective_tiles") > _count(tiles, "tiles"):
        raise ValueError(f"defective_tiles must not exceed tiles, got {defective_tiles!r} of {tiles!r}")
    utc = timestamp.astimezone(timezone.utc)
    stamp = utc.strftime("%Y-%m-%dT%H:%M:%S.") + f"{utc.microsecond // 1000:03d}Z"
    return json.dumps({"frame_index": _count(frame_index, "frame_index"), "tiles": tiles, "defective_tiles": defective_tiles,
                       "defects": _count(defects, "defects"), "timestamp": stamp})


def uns_messages(record: DefectRecord, topics: UnsTopics, placement: str) -> List[Tuple[str, str]]:
    """(topic, JSON text) for one record: the whole record first, then one message per field topic."""
    if not isinstance(record, DefectRecord):
        raise ValueError(f"only DefectRecord can be published, got {type(record).__name__}")
    whole = record.to_dict()
    messages = [(topics.defect(placement), json.dumps(whole, allow_nan=False))]
    for name, key in FIELD_TOPICS.items():
        part = {"timestamp": whole["timestamp"], key: whole[key]}
        messages.append((topics.field(placement, name), json.dumps(part, allow_nan=False)))
    return messages


class UnsPublisher:
    """Publishes the records of one placement. A refused or failed publish is counted in `failed`, never raised."""

    def __init__(self, client, topics: UnsTopics, placement: str):
        self._client, self._topics, self._placement = client, topics, _placement(placement)
        self.sent = self.failed = 0

    def publish(self, record: DefectRecord) -> None:
        for topic, payload in uns_messages(record, self._topics, self._placement):
            self._send(topic, payload)

    def publish_frame(self, frame_index: int, tiles: int, defective_tiles: int, defects: int) -> None:
        """One inspected slice with its tile and record counts, stamped now."""
        self._send(self._topics.frame(self._placement),
                   frame_payload(frame_index, tiles, defective_tiles, defects, datetime.now(timezone.utc)))

    def _send(self, topic: str, payload: str) -> None:
        try:
            ok = self._client.publish(topic, payload, qos=0, retain=False).rc == 0
        except Exception:
            ok = False
        if ok:
            self.sent += 1
        else:
            self.failed += 1

    def close(self) -> None:
        """Disconnect (while the network loop still runs, so the disconnect is sent), then stop the loop."""
        try:
            self._client.disconnect()
        except Exception:
            pass
        self._client.loop_stop()


class PublishingLog:
    """A defect log that writes each record to `log` (the file) and then publishes it."""

    def __init__(self, log, publisher: UnsPublisher):
        self._log, self._publisher = log, publisher

    def write(self, record: DefectRecord) -> None:
        self._log.write(record)
        self._publisher.publish(record)

    def end_slice(self, frame_index: int, outcome) -> None:
        tiles, defective = tile_counts(outcome)
        self._publisher.publish_frame(frame_index, tiles, defective, len(outcome.records))

    def close(self) -> None:
        self._log.close()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()


def broker_address(text) -> Tuple[str, int]:
    """'HOST' or 'HOST:PORT' to (host, port); the port defaults to 1883."""
    if not isinstance(text, str):
        raise ValueError(f"broker must be HOST or HOST:PORT, got {text!r}")
    host, _, port = text.rpartition(":") if ":" in text else (text, "", str(DEFAULT_PORT))
    if not host or not port.isdigit() or not 0 < int(port) < 65536:
        raise ValueError(f"broker must be HOST or HOST:PORT with a port from 1 to 65535, got {text!r}")
    return host, int(port)


def connect_client(host: str, port: int, client_id: str, max_queued: int = 1000):
    """A paho-mqtt 2.x client connecting in the background (a missing broker never blocks the caller), reconnecting on its
    own, with at most `max_queued` messages held while the broker is away. Needs paho-mqtt; checked on the board."""
    if isinstance(max_queued, bool) or not isinstance(max_queued, int) or max_queued < 1:
        raise ValueError(f"max_queued must be a positive integer, got {max_queued!r}")
    import paho.mqtt.client as mqtt
    from paho.mqtt.enums import CallbackAPIVersion

    client = mqtt.Client(callback_api_version=CallbackAPIVersion.VERSION2, client_id=client_id)
    client.max_queued_messages_set(max_queued)
    client.reconnect_delay_set(min_delay=1, max_delay=60)
    client.connect_async(host, port)
    client.loop_start()
    return client
