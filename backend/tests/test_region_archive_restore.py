import gzip
import hashlib
import json

import pytest

from scripts.restore_zone_candidates import FOLDER, restore


def setup_archives(tmp_path):
    folder = tmp_path / FOLDER
    folder.mkdir(parents=True)
    entries = []
    for symbol in ("BOOM600", "CRASH600"):
        payload = f"symbol,value\n{symbol},1\n".encode()
        source = FOLDER / f"{symbol}_candidates.csv"
        compressed = gzip.compress(payload, mtime=0)
        archive = source.with_suffix(".csv.gz")
        (tmp_path / archive).write_bytes(compressed)
        entries.append(dict(source_path=str(source), source_sha256=hashlib.sha256(payload).hexdigest(),
                            source_bytes=len(payload), compressed_path=str(archive),
                            compressed_sha256=hashlib.sha256(compressed).hexdigest(), compressed_bytes=len(compressed)))
    (folder / "candidate_archives.json").write_text(json.dumps({"entries": entries}))
    return folder, entries


def test_exact_offline_restore_is_idempotent(tmp_path):
    _, entries = setup_archives(tmp_path)
    assert all(r["status"] == "restored_exact" for r in restore(tmp_path))
    assert all(r["status"] == "already_exact" for r in restore(tmp_path))
    for e in entries:
        assert hashlib.sha256((tmp_path / e["source_path"]).read_bytes()).hexdigest() == e["source_sha256"]


def test_changed_local_file_is_never_overwritten(tmp_path):
    _, entries = setup_archives(tmp_path)
    path = tmp_path / entries[0]["source_path"]
    path.write_text("user data")
    with pytest.raises(ValueError, match="overwrite"): restore(tmp_path)
    assert path.read_text() == "user data"


def test_bad_archive_is_rejected_before_restore(tmp_path):
    _, entries = setup_archives(tmp_path)
    (tmp_path / entries[0]["compressed_path"]).write_bytes(b"bad gzip")
    with pytest.raises(ValueError, match="pin changed"): restore(tmp_path)
    assert not (tmp_path / entries[0]["source_path"]).exists()


def test_manifest_cannot_redirect_output_outside_fixed_candidate_paths(tmp_path):
    folder, entries = setup_archives(tmp_path)
    entries[0]["source_path"] = "../outside.csv"
    (folder / "candidate_archives.json").write_text(json.dumps({"entries": entries}))
    with pytest.raises(ValueError, match="population"): restore(tmp_path)
