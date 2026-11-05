"""Download the demo's test frames from the public Hugging Face dataset
https://huggingface.co/datasets/martinsawojide/webcheck-test-frames (CC-BY-4.0) into the folders the demo reads:
placement A into <dest>/A_1080 and B into <dest>/B_1080, each with its manifest.jsonl. Every PNG is checked against the
dataset's SHA256SUMS and a file that does not match is not kept; a rerun skips the frames already there and correct.
Standard library only; no token, the dataset is public.

usage: python3 scripts/download_test_frames.py [--dest test_frames] [--placements A B] [--workers 8]
"""
import argparse
import hashlib
import http.client
import json
import os
import sys
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Callable, Dict

REPO = "martinsawojide/webcheck-test-frames"
FOLDERS = {"A": "A_1080", "B": "B_1080"}          # the published folder -> the folder the demo reads


def tree_url(folder: str) -> str:
    return f"https://huggingface.co/api/datasets/{REPO}/tree/main/{folder}"


def file_url(path: str) -> str:
    return f"https://huggingface.co/datasets/{REPO}/resolve/main/{path}"


def fetch(url: str, attempts: int = 5) -> bytes:
    """The bytes at `url`; a failed or cut-short download is tried again, waiting a little longer each time."""
    for attempt in range(attempts):
        try:
            with urllib.request.urlopen(url, timeout=60) as response:
                return response.read()
        except (OSError, http.client.HTTPException):
            if attempt == attempts - 1:
                raise
            time.sleep(2 ** attempt)
    raise AssertionError("unreachable")


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def download(placement: str, dest, get: Callable[[str], bytes] = fetch, workers: int = 8) -> Dict[str, int]:
    """Fetch one placement's frames and manifest into <dest>/<FOLDERS[placement]>; returns how many files were
    downloaded and how many were kept from an earlier run."""
    if placement not in FOLDERS:
        raise ValueError(f"placement must be one of {sorted(FOLDERS)}, got {placement!r}")
    names = [Path(e["path"]).name for e in json.loads(get(tree_url(placement))) if e.get("type") == "file"]
    pngs = sorted(n for n in names if n.endswith(".png"))
    sums = {}
    for line in get(file_url(f"{placement}/SHA256SUMS")).decode().splitlines():
        if line.strip():
            digest, name = line.split(maxsplit=1)
            sums[name.strip().lstrip("*")] = digest
    missing = [n for n in pngs if n not in sums]
    if missing:
        raise ValueError(f"no checksum in SHA256SUMS for {', '.join(missing)}")
    out = Path(dest) / FOLDERS[placement]
    out.mkdir(parents=True, exist_ok=True)
    counts = {"downloaded": 0, "kept": 0}

    def one(name: str) -> str:
        target = out / name
        if name in sums and target.is_file() and _sha256(target.read_bytes()) == sums[name]:
            return "kept"
        data = get(file_url(f"{placement}/{name}"))
        if name in sums and _sha256(data) != sums[name]:
            raise ValueError(f"{placement}/{name} does not match its SHA256SUMS entry")
        partial = target.with_name(target.name + ".part")
        partial.write_bytes(data)
        os.replace(partial, target)
        return "downloaded"

    with ThreadPoolExecutor(max(1, workers)) as pool:
        for result in pool.map(one, pngs + ["manifest.jsonl"]):
            counts[result] += 1
    return counts


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--dest", default="test_frames", help="folder that will hold A_1080 and B_1080 (default test_frames)")
    ap.add_argument("--placements", nargs="+", default=sorted(FOLDERS), choices=sorted(FOLDERS))
    ap.add_argument("--workers", type=int, default=8)
    args = ap.parse_args()
    for placement in args.placements:
        counts = download(placement, args.dest, workers=args.workers)
        print(f"{placement}: {counts['downloaded']} downloaded, {counts['kept']} already there -> "
              f"{Path(args.dest) / FOLDERS[placement]}", file=sys.stderr)


if __name__ == "__main__":
    main()
