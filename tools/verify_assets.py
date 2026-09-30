#!/usr/bin/env python3
"""Verify committed test assets without third-party Python dependencies."""
from pathlib import Path
import hashlib
import json


def main() -> None:
    root = Path(__file__).resolve().parents[1] / "tests" / "assets"
    manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    for entry in manifest["files"]:
        path = (root / entry["path"]).resolve()
        if not path.is_relative_to(root.resolve()):
            raise ValueError(f"Unsafe manifest path: {entry['path']}")
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        if digest != entry["sha256"]:
            raise ValueError(f"Asset checksum mismatch: {entry['path']}")
    print(f"Verified {len(manifest['files'])} assets")


if __name__ == "__main__":
    main()
