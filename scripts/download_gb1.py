"""Download the unmodified, official eLife Supplementary file 1."""

import hashlib
import sys
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEST = ROOT / "data" / "raw" / "elife-16965-supp1-v4.xlsx"
URL = "https://cdn.elifesciences.org/articles/16965/elife-16965-supp1-v4.xlsx"


def main() -> None:
    DEST.parent.mkdir(parents=True, exist_ok=True)
    if not DEST.exists():
        print(f"Downloading published GB1 measurements from {URL}")
        try:
            urllib.request.urlretrieve(URL, DEST)
        except Exception:
            DEST.unlink(missing_ok=True)
            raise
    with DEST.open("rb") as handle:
        signature = handle.read(4)
    if signature != b"PK\x03\x04":
        raise RuntimeError("Source download is not a valid XLSX/ZIP header")
    print(f"Source: {DEST}")
    print(f"Size: {DEST.stat().st_size:,} bytes")
    print(f"SHA256: {hashlib.sha256(DEST.read_bytes()).hexdigest()}")


if __name__ == "__main__":
    sys.exit(main())

