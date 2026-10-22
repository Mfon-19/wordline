"""Download the detector model and the evaluation clips listed in assets.json.

Clips come from Wikimedia Commons; authors and licences are in clips/SOURCES.md.

Usage: python scripts/fetch_assets.py
"""
import json
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
COMMONS = "https://commons.wikimedia.org/wiki/Special:FilePath/"


def fetch(url, dst):
    if dst.exists() and dst.stat().st_size > 0:
        print(f"have  {dst.relative_to(ROOT)}")
        return
    dst.parent.mkdir(parents=True, exist_ok=True)
    req = urllib.request.Request(url, headers={"User-Agent": "worldline/0.1"})
    for attempt in range(5):  # Wikimedia rate-limits bursts; wait and retry
        try:
            with urllib.request.urlopen(req, timeout=120) as r:
                dst.write_bytes(r.read())
            break
        except urllib.error.HTTPError as e:
            if e.code != 429 or attempt == 4:
                raise
            time.sleep(int(e.headers.get("Retry-After", 0)) or 15 * (attempt + 1))
    time.sleep(2)
    print(f"saved {dst.relative_to(ROOT)} ({dst.stat().st_size / 1e6:.1f} MB)")


def main():
    assets = json.loads((ROOT / "assets.json").read_text())
    fetch(assets["model"]["url"], ROOT / assets["model"]["path"])
    md = ROOT / assets["megadetector"]["path"]
    if not md.exists():
        print(f"missing {md.relative_to(ROOT)}: run {assets['megadetector']['made_by']} to make it")
    for name, title in assets["clips"].items():
        fetch(COMMONS + urllib.parse.quote(title), ROOT / "clips" / f"{name}{Path(title).suffix}")


if __name__ == "__main__":
    main()
