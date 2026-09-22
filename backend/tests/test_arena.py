import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient
from types import SimpleNamespace

from app.config import Settings
from app.main import app
from app.services.arena.agents import AGENT_SPECS, generate_candidate_payloads
from app.services.arena.evaluator import prepare_arena_frame
from app.services.arena.leaderboard import calculate_composite
from app.services.arena.qualification import qualification_gate
from app.services.backtest import chronological_split
from app.services.deriv_client import DerivClient
from app.services.metrics import spike_detection_metrics, trading_metrics
from app.services.spike import label_spikes
from app.services.safety import reject_order_execution
from app.symbol_profiles import get_profile, list_profiles


def sample_candles(n=500, seed=91):
    rng = np.random.default_rng(seed)
    close = 100 + np.cumsum(rng.normal(0, 0.08, n))
    # deterministic large moves create a few real spike-like observations
    for i in range(100, n, 120):
        close[i : min(i + 3, n)] += np.linspace(0, 1.5, min(3, n - i))
    return [
        {
            "epoch": int(i * 60),
            "open": float(close[i] + rng.normal(0, 0.02)),
            "high": float(close[i] + 0.12),
            "low": float(close[i] - 0.12),
            "close": float(close[i]),
            "tick_count": int(rng.integers(10, 30)),
        }
        for i in range(n)
    ]


def test_strategy_creation_has_all_six_agents():
    rows = generate_candidate_payloads("CRASH500", variants_per_agent=1)
    assert len(rows) == 6
    assert {r["agent_name"] for r in rows} == set(AGENT_SPECS)
    assert len({r["id"] for r in rows}) == 6
    assert all(r["status"] == "created" for r in rows)


def test_symbol_profiles_are_independent_and_support_all_ten():
    profiles = list_profiles()
    assert len(profiles) == 10
    boom = get_profile("BOOM500")
    crash = get_profile("CRASH500")
    assert boom.symbol == "BOOM500" and crash.symbol == "CRASH500"
    assert boom.timeframe_weights != crash.timeframe_weights
    assert boom.support_resistance_lookback == 20
    assert crash.spike_window == 5


def test_backtest_metrics_cover_required_fields():
    metrics = trading_metrics([2.0, -1.0, 3.0, -0.5, 1.0])
    required = {
        "total_trades", "winning_trades", "losing_trades", "win_rate", "net_profit",
        "gross_profit", "gross_loss", "profit_factor", "expectancy", "max_drawdown",
        "average_drawdown", "average_win", "average_loss", "risk_reward",
        "consecutive_losses", "sharpe_like", "stability_score",
    }
    assert required.issubset(metrics)
    assert metrics["net_profit"] == pytest.approx(4.5)


def test_spike_detection_metrics_identify_hit_and_lead_time():
    df = pd.DataFrame(
        {
            "epoch": [i * 60 for i in range(10)],
            "close": [100, 100, 100, 100.1, 100.3, 102.0, 102.1, 102.0, 102.0, 102.0],
            "high": [100.1, 100.1, 100.1, 100.2, 100.4, 102.2, 102.2, 102.1, 102.1, 102.1],
            "low": [99.9, 99.9, 99.9, 100.0, 100.2, 101.8, 102.0, 101.9, 101.9, 101.9],
            "is_spike": [False, False, False, False, False, True, False, False, False, False],
            "future_move_atr": [0, 0, 0, 0, 0, 3.0, 0, 0, 0, 0],
        }
    )
    profile = SimpleNamespace(pre_spike_window=4, test_window_bars=4)
    out = spike_detection_metrics(df, [2], profile, "UP")
    assert out["total_spikes"] == 1
    assert out["predicted_spikes"] == 1
    assert out["false_spike_signals"] == 0
    assert out["spike_precision"] == 1
    assert out["spike_recall"] == 1
    assert out["average_lead_time_before_spike"] == 180


def test_qualification_gate_uses_multiple_dimensions():
    metrics = {
        "total_trades": 50,
        "profit_factor": 1.5,
        "max_drawdown": 2.0,
        "expectancy": 0.4,
        "stability_score": 0.8,
    }
    spike = {"spike_precision": 0.6, "spike_recall": 0.5, "false_alert_rate": 0.2}
    assert qualification_gate(metrics, spike, "oos")["passed"] is True
    weak = dict(metrics, profit_factor=0.2)
    assert qualification_gate(weak, spike, "oos")["passed"] is False


def test_oos_split_is_chronological_and_held_out():
    df = pd.DataFrame(sample_candles(500))
    train, validation, oos = chronological_split(df, 0.6, 0.2)
    assert len(train) == 300 and len(validation) == 100 and len(oos) == 100
    assert train.epoch.max() < validation.epoch.min() < oos.epoch.min()


def test_leaderboard_score_rewards_spike_quality_not_net_profit_alone():
    metrics = {
        "profit_factor": 1.4,
        "expectancy": 0.2,
        "average_loss": -0.5,
        "net_profit": 4.0,
        "max_drawdown": 2.0,
        "stability_score": 0.7,
    }
    weak_spike = {"spike_precision": 0.2, "spike_recall": 0.2, "false_alert_rate": 0.7}
    strong_spike = {"spike_precision": 0.8, "spike_recall": 0.7, "false_alert_rate": 0.2}
    weak_score, _ = calculate_composite(metrics, weak_spike, 50, 30)
    strong_score, _ = calculate_composite(metrics, strong_spike, 50, 30)
    assert strong_score > weak_score


def test_safety_flags_and_order_execution_are_fail_closed():
    with pytest.raises(RuntimeError):
        Settings(live_trading=True).assert_safe()
    with pytest.raises(RuntimeError):
        Settings(ready_for_live=True).assert_safe()
    with pytest.raises(PermissionError):
        reject_order_execution("buy")
    assert not hasattr(DerivClient, "buy")
    assert not hasattr(DerivClient, "sell")
    assert not hasattr(DerivClient, "purchase")


def test_arena_api_endpoints_and_backtest_path():
    with TestClient(app) as client:
        safety = client.get("/api/arena/safety")
        assert safety.status_code == 200
        assert safety.json()["LIVE_TRADING"] is False
        assert safety.json()["orders_opened"] == 0

        agents = client.get("/api/arena/agents")
        assert agents.status_code == 200 and len(agents.json()) == 6

        strategies = client.get("/api/arena/strategies")
        assert strategies.status_code == 200 and len(strategies.json()) >= 12

        stats = client.get("/api/arena/stats")
        assert stats.status_code == 200
        assert stats.json()["active_agents"] == 6

        leaderboard = client.get("/api/arena/leaderboard")
        assert leaderboard.status_code == 200
        assert "formula" in leaderboard.json() and "weights" in leaderboard.json()

        strategy_id = strategies.json()[0]["id"]
        detail = client.get(f"/api/arena/strategies/{strategy_id}")
        assert detail.status_code == 200
        assert detail.json()["id"] == strategy_id

        result = client.post("/api/arena/backtest", json={"strategy_id": strategy_id, "candles": sample_candles(500)})
        assert result.status_code == 200, result.text
        body = result.json()
        assert "in_sample_results" in body
        assert "validation_results" in body
        assert "qualification" in body

        # The synthetic smoke data will normally fail qualification; the OOS route must fail closed,
        # not bypass the lifecycle gate.
        oos = client.post("/api/arena/oos-test", json={"strategy_id": strategy_id, "candles": sample_candles(500)})
        assert oos.status_code in {200, 400}

        shadow_status = client.get("/api/arena/shadow/status", params={"strategy_id": strategy_id})
        assert shadow_status.status_code == 200
        shadow_start = client.post("/api/arena/shadow/start", json={"strategy_id": strategy_id, "duration_hours": 24})
        if body["strategy"]["status"] != "forward_testing":
            assert shadow_start.status_code == 400

        stopped = client.post("/api/arena/shadow/stop", params={"strategy_id": strategy_id})
        assert stopped.status_code == 200

        activity = client.get("/api/arena/activity")
        profiles = client.get("/api/arena/profiles")
        assert activity.status_code == 200
        assert profiles.status_code == 200 and len(profiles.json()) == 10


def test_spike_label_marks_arrival_not_future_start():
    profile = SimpleNamespace(spike_window=2, spike_atr_multiple=2.0, pre_spike_window=3)
    df = pd.DataFrame(
        {
            "close": [100.0, 100.0, 100.0, 100.2, 103.0, 103.1],
            "atr": [1.0] * 6,
        }
    )
    out = label_spikes(df, profile, "UP")
    assert out.loc[2, "is_spike"] == False
    assert out.loc[4, "is_spike"] == True
    assert out.loc[3, "pre_spike"] == True


def test_completed_mtf_context_does_not_use_incomplete_future_bucket():
    profile = get_profile("BOOM500")
    candles = pd.DataFrame(sample_candles(180, seed=123))
    base = prepare_arena_frame(candles, profile, with_spikes=False)

    changed = candles.copy()
    changed.loc[155:159, ["open", "high", "low", "close"]] = changed.loc[155:159, ["open", "high", "low", "close"]] + 1000
    altered = prepare_arena_frame(changed, profile, with_spikes=False)

    # At index 154 (end of the previous completed M5 bucket), later bucket values must not leak backward.
    assert base.loc[154, "m5_trend"] == altered.loc[154, "m5_trend"]
    assert base.loc[154, "m15_trend"] == altered.loc[154, "m15_trend"]
