import gzip
import json

import pandas as pd
import pytest

from scripts import run_zone_excursions as study


def candidates():
    rows = []
    for i, reason in enumerate(("issued_fixed_region", "h4_h1_context_disagrees", "context_unavailable")):
        for variant in study.VARIANTS:
            rows.append(dict(signal_time=pd.Timestamp("2026-01-01T00:00Z") + pd.Timedelta(minutes=30 * i),
                variant=variant, reason=reason, atr=1., side=1, eligible=i == 0 and variant != "CRT_RETEST"))
    return pd.DataFrame(rows)


def test_clock_membership_uses_past_availability_and_preserves_native_disagreement():
    frame = study.available_anchors(candidates())
    assert len(frame) == 2
    assert frame.eligible_CONTEXT_GEOMETRIC.tolist() == [True, False]
    assert frame.eligible_CRT_RETEST.tolist() == [False, False]
    assert ((frame.anchor_time - frame.issue_time).dt.total_seconds() == 61).all()


def test_region_cannot_be_compared_as_subset_of_a_missing_control():
    data = candidates()
    data.loc[data.variant.eq(study.CONTROL), "eligible"] = False
    with pytest.raises(ValueError, match="superset"): study.available_anchors(data)


def test_exact_actual_entry_anchor_is_preserved_for_filled_unknown_paths():
    source = pd.DataFrame({"signal_time": ["2026-01-01T00:00Z"], "entry_time": ["2026-01-01T00:02:17Z"],
                           "atr": [2.], "side": [-1], "censored": [True], "net_R": [None]})
    selected = study.native_entries(source)
    assert len(selected) == 1 and selected.iloc[0].anchor_time.second == 17
    assert selected.iloc[0].side == -1


def test_archives_preserve_exact_label_bytes_and_refuse_overwriting(tmp_path, monkeypatch):
    monkeypatch.setattr(study, "ROOT", tmp_path)
    path = tmp_path / "labels.csv.gz"
    frame = pd.DataFrame({"y": [-1, 0, 1], "unknown": [True, False, False]})
    artifact = study.archive_frame(path, frame)
    assert gzip.decompress(path.read_bytes()) == frame.to_csv(index=False).encode()
    assert artifact["uncompressed_bytes"] == len(frame.to_csv(index=False).encode())
    with pytest.raises(FileExistsError): study.archive_frame(path, frame)


def test_incomplete_multiplicity_family_cannot_silently_shrink():
    with pytest.raises(ValueError, match="144"): study.attach_holm([])


def test_protocol_exposes_two_fixed_endpoints_without_changing_economics():
    p = study.spec()
    assert p["primary_purge_minutes"] == 32 and p["secondary_purge_minutes"] == 46
    assert len(p["definitions"]) == 12
    assert p["region_parameters_and_profit_ledgers_changed"] is False and p["QUALIFIED"] is False
