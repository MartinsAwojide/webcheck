"""The defect map on the board: subscribes to the defect log on the UNS MQTT topics and serves the map page
(webcheck.inference.defect_map) for the TV. Runs beside the HMI; needs paho-mqtt and the broker the HMI publishes to.

usage: python3 defect_map.py --pkg DIR [--mqtt HOST[:PORT]] [--port 8090] [--host 0.0.0.0] [--db defect_map.sqlite]
                             [--uns-enterprise ENERGY_UP_LLC] [--uns-site TEXAS_FACILITY]"""
import argparse
import signal
import socket
import sys
import threading


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pkg", required=True, help="the folder that contains the webcheck package")
    ap.add_argument("--mqtt", default="localhost", help="the broker the defect log is published to, HOST[:PORT]")
    ap.add_argument("--port", type=int, default=8090, help="the map page's port")
    ap.add_argument("--host", default="0.0.0.0", help="address the page is served on")
    ap.add_argument("--uns-enterprise", default="ENERGY_UP_LLC", help="UNS enterprise level (the converter company)")
    ap.add_argument("--uns-site", default="TEXAS_FACILITY", help="UNS site level (no spaces: the defect map shows _ as a space)")
    ap.add_argument("--db", default="defect_map.sqlite", help="the SQLite file the detections are kept in (48 h)")
    args = ap.parse_args()
    sys.path.insert(0, args.pkg)
    import paho.mqtt.client as mqtt
    from paho.mqtt.enums import CallbackAPIVersion

    from webcheck.inference.defect_map import DefectDB, MapServer, load_layout, on_connect, on_disconnect, on_message
    from webcheck.inference.uns import UnsTopics, broker_address

    topics = UnsTopics(args.uns_enterprise, args.uns_site)
    layout = load_layout()
    store, health = DefectDB(args.db, layout), {"connected": False, "received": 0, "ignored": 0}
    server = MapServer(store, layout, health, host=args.host, port=args.port)
    server.start()
    client = mqtt.Client(callback_api_version=CallbackAPIVersion.VERSION2, client_id=f"webcheck-map-{socket.gethostname()}")
    client.on_connect, client.on_disconnect = on_connect(topics, health), on_disconnect(health)
    client.on_message = on_message(store, topics, health)
    client.reconnect_delay_set(min_delay=1, max_delay=60)
    host, port = broker_address(args.mqtt)
    client.connect_async(host, port)
    client.loop_start()
    print(f"defect map: http://{socket.gethostname()}.local:{server.port}  (or the board's IP); MQTT {host}:{port}",
          file=sys.stderr, flush=True)
    done = threading.Event()
    signal.signal(signal.SIGTERM, lambda *_: done.set())
    signal.signal(signal.SIGINT, lambda *_: done.set())
    done.wait()
    client.disconnect()
    client.loop_stop()
    server.stop()


if __name__ == "__main__":
    main()
