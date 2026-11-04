"""Download/resume and independently verify Meta Llama 3.1 8B GGUF weights."""

import argparse
import json
import urllib.request
from pathlib import Path

from proteinevo.local_gguf_client import (
    DEFAULT_MODEL_PATH, EXPECTED_SHA256, EXPECTED_SIZE, MODEL_ID,
    MODEL_URL, file_sha256,
)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_MODEL_PATH)
    parser.add_argument("--verify-only", action="store_true")
    args = parser.parse_args()
    path = args.output.resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    if not args.verify_only and (not path.exists() or path.stat().st_size != EXPECTED_SIZE):
        current = path.stat().st_size if path.exists() else 0
        if current > EXPECTED_SIZE:
            parser.error("Model file is larger than expected; no overwrite permitted")
        request = urllib.request.Request(
            MODEL_URL, headers={"Range": f"bytes={current}-"} if current else {}
        )
        try:
            with urllib.request.urlopen(request, timeout=120) as source:
                if current and source.status != 206:
                    parser.error("Server did not honor resume Range; preserve existing bytes")
                with path.open("ab" if current else "wb") as destination:
                    while True:
                        block = source.read(8 * 1024 * 1024)
                        if not block:
                            break
                        destination.write(block)
        except OSError as exc:
            parser.error(f"Download interrupted; run again to resume: {exc}")
    if not path.exists() or path.stat().st_size != EXPECTED_SIZE:
        parser.error("Model missing or incomplete")
    print("Validating SHA-256 of 4.9GB GGUF...", flush=True)
    checksum = file_sha256(path)
    if checksum != EXPECTED_SHA256:
        parser.error("Checksum does not match the original GGUF file; preserve for inspection")
    info = {
        "model": MODEL_ID,
        "source": MODEL_URL,
        "quantization": "Q4_K_M",
        "size_bytes": EXPECTED_SIZE,
        "sha256": checksum,
        "license": "Meta Llama 3.1 Community License",
    }
    manifest = path.parent / "model_info.json"
    manifest.write_text(json.dumps(info, indent=2) + "\n", encoding="utf-8")
    print(f"Model verified: {path}\nMetadata: {manifest}")


if __name__ == "__main__":
    main()
