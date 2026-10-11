"""Product evidence bridge checks; all fixtures are explicitly synthetic."""
import csv
import json

from research_campaign.reporting import metric_view, publish_snapshot
from research_campaign.state import SAFETY, atomic_json, read_json


def test_ledger_dates_survive_the_string_only_dashboard_contract():
    result = {"id": "synthetic-fixture", "metrics": {"initial_equity": 100},
              "equity": [{"epoch": 1728000000, "equity": 100}, {"epoch": 1728000060, "equity": 90}],
              "trades": [{"entry_epoch": 1728000000, "exit_epoch": 1728000060,
                          "direction": -1, "net_pnl": -10}], "monthly": []}
    output = metric_view(result, details=True)
    assert output["equity"][0]["time"] == "2024-10-04T00:00:00+00:00"
    assert output["drawdown"][-1]["value"] == .10
    assert output["trades"][0]["side"] == "SELL"
    assert output["trades"][0]["exit_time"] == "2024-10-04T00:01:00+00:00"


def test_snapshot_and_export_disclose_selection_without_borrowing_final(tmp_path):
    symbol = "BOOM500"
    d = {"split": {"start": 1728000000, "end": 1728060000},
         "symbol_catalogue": {symbol: {"api_symbol": symbol, "display_name": "Boom 500"}},
         "datasets": {symbol: {"requested_rows": 999, "study_quality": {
             "internal_missing_minutes": 1, "internal_invalid_rows": 0}}}}
    metrics = {"completed_trades": 120, "profit_factor": 1.4, "net_profit": 2, "max_drawdown": .02}
    base = {"symbol": symbol, "timeframe": "M1", "family": "crt", "status": "TESTED",
            "spec": {"threshold": .8}, "tested_at": "2026-10-11T00:00:00Z", "artifact": {"path": "synthetic"},
            "training_metrics": metrics, "validation_metrics": metrics, "validation_score": 1}
    winner = {**base, "strategy_id": "selected"}
    loser = {**base, "strategy_id": "not-selected", "validation_score": .5}
    cycle = {"number": 1, "status": "COMPLETED", "started_at": "2024-10-04T00:00:00+00:00", "results": [winner, loser]}
    state = {"winners": {symbol + "/M1/crt": winner}, "started_epoch": 1728000000,
             "completed_cycles": 1, "interval_seconds": 900, "target_cycles": 32,
             "final_accessed": False, "updated_at": "2026-10-11T00:00:00Z", "cycles": [cycle],
             "coverage": {}, "current_cycle": None, "safety": SAFETY}
    atomic_json(tmp_path / "declaration.json", d)
    atomic_json(tmp_path / "cycles/01.json", cycle)
    publish_snapshot(tmp_path, tmp_path / "public", state)
    output = read_json(tmp_path / "snapshot.json")
    assert output["strategies"][0]["oos"] is None
    assert output["strategies"][0]["status"] == "NOT_TESTED"
    assert output["symbols"][0]["best_strategy_id"] is None
    assert any("1 دقيقة مفقودة" in n for n in output["source"]["data_notes"])
    with (tmp_path / "public/search.csv").open() as stream:
        exported = list(csv.DictReader(stream))
    assert exported[0]["selection_disposition"] == "SELECTED_VALIDATION_CANDIDATE"
    assert exported[1]["reasons"] == "NOT_BEST_VALIDATION_SCORE_WITHIN_FAMILY"
    assert json.loads(exported[1]["settings_json"]) == base["spec"]
