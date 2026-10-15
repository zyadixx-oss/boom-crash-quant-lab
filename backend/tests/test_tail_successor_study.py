"""Synthetic and metadata-only premeasurement guards for the successor study."""
import pandas as pd
import pytest

from scripts import run_tail_successor as study


def metadata():
    return {"input_sha256": {}, "sources": {}, "detectors": {}, "observed_rows": {}, "row_cutoffs": {}}


@pytest.fixture
def patched(monkeypatch):
    monkeypatch.setattr(study, "science_hashes", lambda: {"synthetic": "abc"})
    monkeypatch.setattr(study, "verified_lineage", metadata)


@pytest.mark.parametrize("flag", study.FLAGS)
@pytest.mark.parametrize("value", ["true", "1", ""])
def test_safety_rejects_before_output_or_metadata(monkeypatch, tmp_path, flag, value):
    monkeypatch.setenv(flag, value)
    monkeypatch.setattr(study, "verified_lineage", lambda: pytest.fail("metadata accessed before guard"))
    with pytest.raises(ValueError, match="flags"):
        study.declare(tmp_path / "study")
    assert not (tmp_path / "study").exists()


def test_declaration_and_frozen_are_metadata_only(patched, monkeypatch, tmp_path):
    monkeypatch.setattr(study.previous, "load_tick_day", lambda *_: pytest.fail("historical quote decoded"))
    output = tmp_path / "study"
    value = study.declare(output)
    assert study.frozen(output) == value
    assert value["historical_quotes_decoded_for_this_response"] is False
    assert value["successor_response_evaluated"] is False
    assert value["detector_refit"] is False
    assert not (output / "results.json").exists()


def test_exclusive_declaration_cannot_overwrite_partial_failure(patched, tmp_path):
    (tmp_path / "failure.log").write_text("original failure\n")
    with pytest.raises(ValueError, match="empty"):
        study.declare(tmp_path)
    assert (tmp_path / "failure.log").read_text() == "original failure\n"


def test_frozen_rejects_changed_science_before_quotes(patched, monkeypatch, tmp_path):
    study.declare(tmp_path)
    monkeypatch.setattr(study, "science_hashes", lambda: {"synthetic": "changed"})
    monkeypatch.setattr(study.previous, "load_tick_day", lambda *_: pytest.fail("historical quote decoded"))
    with pytest.raises(ValueError, match="Frozen"):
        study.evaluate(tmp_path)


def test_frozen_rejects_changed_lineage(patched, monkeypatch, tmp_path):
    study.declare(tmp_path)
    changed = metadata(); changed["observed_rows"] = {"BOOM600": 999}
    monkeypatch.setattr(study, "verified_lineage", lambda: changed)
    with pytest.raises(ValueError, match="Frozen"):
        study.frozen(tmp_path)


def test_frozen_rejects_declaration_byte_change(patched, tmp_path):
    study.declare(tmp_path)
    with (tmp_path / "declaration.json").open("a") as stream:
        stream.write(" ")
    with pytest.raises(ValueError, match="bytes"):
        study.frozen(tmp_path)


def test_existing_outcome_refused_before_reading(monkeypatch, tmp_path):
    (tmp_path / "results.json").write_text("kept")
    monkeypatch.setattr(study, "frozen", lambda *_: pytest.fail("inputs reopened"))
    with pytest.raises(ValueError, match="overwrite"):
        study.evaluate(tmp_path)
    assert (tmp_path / "results.json").read_text() == "kept"


@pytest.mark.parametrize("value", ["/tmp/x", "../x", "a/../x", "./x", "a//x", "", 1])
def test_repository_path_rejects_aliases(value):
    with pytest.raises(ValueError):
        study.path_for(value)


def test_repository_path_rejects_symlink_component(monkeypatch, tmp_path):
    monkeypatch.setattr(study, "ROOT", tmp_path)
    (tmp_path / "real").mkdir(); (tmp_path / "real/x").write_text("data")
    (tmp_path / "alias").symlink_to(tmp_path / "real", target_is_directory=True)
    with pytest.raises(ValueError, match="symlink"):
        study.path_for("alias/x")


def test_partition_end_uses_first_excluded_quote_and_final_midnight(monkeypatch):
    index = pd.date_range("2026-01-01 23:59:50", periods=20, freq="s", tz="UTC")
    ticks = pd.DataFrame({"quote": 100.}, index=index)
    cuts = {"40": 8, "50": 10, "60": 12, "70": 14, "100": 20}
    old = {"median_abs_log_return": .001}
    lineage = {"row_cutoffs": {"BOOM600": cuts}, "detectors": {"BOOM600": old}}
    calls = []
    def collect(frame, detector, first, endpoint):
        calls.append((first, endpoint, detector.side, detector.median_abs_log_return)); return []
    monkeypatch.setattr(study, "collect_successor_days", collect)
    monkeypatch.setattr(study, "summarize_successor", lambda *_, **__: {})
    value = study.evaluate_symbol(ticks, "BOOM600", lineage)
    assert calls[0][:2] == (int(index[8].timestamp()), int(index[10].timestamp()))
    assert calls[-1][1] == int(pd.Timestamp("2026-01-03", tz="UTC").timestamp())
    assert value["detector"] is old and value["detector_fit_attempts"] == 0
    assert value["strategy_eligible"] is False and value["profit_factor"] is None
    assert set(value["segments"]) == {"wf1", "wf2", "wf3", "dev_validation", "final30"}


def test_configuration_has_one_response_and_no_optimization_grid():
    assert study.CONFIG["response"] == "side*log(P_t_plus_2/P_t_plus_1)"
    assert study.CONFIG["bootstrap_repeats"] == 9999
    assert study.CONFIG["bootstrap_seed"] == 20261007
    assert study.CONFIG["reference_includes_events"] is True
    assert study.CONFIG["detector_refit"] is False
    assert study.CONFIG["history_status"] == "previously_exposed_not_fresh_price_OOS"
