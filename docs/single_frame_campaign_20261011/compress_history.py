"""Optional macOS storage maintenance; scientific bytes and paths stay identical.

Only immutable SHA256-named CSV snapshots are eligible. This uses transparent
filesystem compression, not a new CSV encoding or a changed research dataset.
It is unnecessary on Ubuntu and is never imported by the scientific runner.
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path
import re
import subprocess
import sys
import tempfile
import time

from research_campaign.state import atomic_json, digest, guard, utc


def compress_one(source: Path):
    before = source.stat()
    if before.st_blocks * 512 < before.st_size * .90:
        return None
    original = digest(source)
    if original != source.stem:
        raise ValueError(f"Snapshot name no longer matches its immutable bytes: {source}")
    descriptor, name = tempfile.mkstemp(prefix=".lossless-compression-", suffix=".tmp", dir=source.parent)
    os.close(descriptor)
    destination = Path(name)
    try:
        subprocess.run(["/usr/bin/ditto", "--hfsCompression", "--noclone", str(source), str(destination)],
                       check=True, capture_output=True)
        after = destination.stat()
        if digest(destination) != original or digest(source) != original:
            raise ValueError("Compression changed bytes or the source changed concurrently")
        applied = after.st_blocks < before.st_blocks
        if applied:
            os.replace(destination, source)
        if digest(source) != original:
            raise ValueError("Published snapshot no longer matches its original hash")
        return {"path": str(source), "sha256": original, "logical_size": before.st_size,
                "allocated_before": before.st_blocks * 512,
                "allocated_after": (after if applied else before).st_blocks * 512,
                "applied": applied, "logical_bytes_verified_unchanged": True}
    finally:
        destination.unlink(missing_ok=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--history-dir", type=Path, required=True)
    parser.add_argument("--receipt-dir", type=Path, required=True)
    args = parser.parse_args()
    guard()
    if sys.platform != "darwin":
        raise RuntimeError("Optional transparent compression requires macOS; scientific commands remain portable")
    history = args.history_dir.resolve()
    paths = sorted(p for p in history.glob("*/snapshots/*.csv")
                   if re.fullmatch(r"[0-9a-f]{64}", p.stem) and not p.is_symlink())
    rows = []
    receipt = args.receipt_dir / f"filesystem-compression-{time.time_ns()}.json"
    for path in paths:
        row = compress_one(path)
        if row is not None:
            rows.append(row)
            # Preserve evidence even if storage maintenance is interrupted.
            atomic_json(receipt, {"at": utc(), "status": "RUNNING", "files": rows,
                                 "scientific_bytes_changed": False, "safety": guard()})
    result = {"at": utc(), "status": "COMPLETED", "files": rows,
              "eligible_files": len(paths), "processed_files": len(rows),
              "reclaimed_bytes": sum(r["allocated_before"] - r["allocated_after"] for r in rows),
              "scientific_bytes_changed": False, "safety": guard()}
    atomic_json(receipt, result)
    print({k: v for k, v in result.items() if k != "files"} | {"receipt": str(receipt)})


if __name__ == "__main__":
    main()
