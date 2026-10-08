#!/usr/bin/env python3
"""Restore exact ignored candidate CSVs from the published lossless archives."""
import gzip
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
FOLDER = Path("docs/zone_study_20261008")


def restore(root=ROOT):
    root = Path(root).resolve()
    manifest = json.loads((root / FOLDER / "candidate_archives.json").read_text())
    expected = {str(FOLDER / f"{symbol}_candidates.csv") for symbol in ("BOOM600", "CRASH600")}
    entries = manifest["entries"]
    if len(entries) != 2 or {e["source_path"] for e in entries} != expected:
        raise ValueError("Archive population differs from the two fixed candidate tables")
    result = []
    for item in entries:
        target, archive = root / item["source_path"], root / item["compressed_path"]
        target.resolve().relative_to(root)
        archive.resolve().relative_to(root)
        if archive != target.with_suffix(".csv.gz") or not 0 <= item["source_bytes"] <= 64 * 1024**2:
            raise ValueError("Unexpected archive name or declared size")
        compressed = archive.read_bytes()
        if len(compressed) != item["compressed_bytes"] or hashlib.sha256(compressed).hexdigest() != item["compressed_sha256"]:
            raise ValueError("Compressed candidate pin changed")
        with gzip.open(archive, "rb") as source:
            payload = source.read(item["source_bytes"] + 1)
        if len(payload) != item["source_bytes"] or hashlib.sha256(payload).hexdigest() != item["source_sha256"]:
            raise ValueError("Restored candidate bytes differ from the exact original")
        if target.exists():
            if hashlib.sha256(target.read_bytes()).hexdigest() != item["source_sha256"]:
                raise ValueError("Refusing to overwrite changed local candidate CSV")
            status = "already_exact"
        else:
            with target.open("xb") as output:
                output.write(payload)
            status = "restored_exact"
        result.append({"path": item["source_path"], "status": status})
    return result


if __name__ == "__main__":
    print(json.dumps(restore()))
