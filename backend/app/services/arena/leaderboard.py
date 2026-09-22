from __future__ import annotations

from sqlalchemy import select

from ...config import settings
from ...models import ForwardRun, LeaderboardEntry, StrategyCandidate, StrategyMetric
from .qualification import quality_score


FORMULA_DESCRIPTION = (
    "composite_score = 100 * weighted_sum(PF_norm, expectancy_norm, drawdown_quality, "
    "spike_precision, spike_recall, 1-false_alert_rate, OOS_quality, forward_quality, stability). "
    "Net profit is displayed but is not a direct ranking weight."
)


def _latest_metric(db, strategy_id: str, stage: str):
    return db.execute(
        select(StrategyMetric)
        .where(StrategyMetric.strategy_id == strategy_id, StrategyMetric.stage == stage)
        .order_by(StrategyMetric.id.desc())
        .limit(1)
    ).scalar_one_or_none()


def _latest_forward(db, strategy_id: str):
    return db.execute(
        select(ForwardRun)
        .where(ForwardRun.strategy_id == strategy_id)
        .order_by(ForwardRun.id.desc())
        .limit(1)
    ).scalar_one_or_none()


def composite_components(metrics: dict, spike: dict, oos_score: float = 0.0, forward_score: float = 0.0) -> dict:
    pf = min(max(float(metrics.get("profit_factor", 0)), 0.0) / 2.5, 1.0)
    avg_loss = abs(float(metrics.get("average_loss", 0) or 0))
    expectancy = float(metrics.get("expectancy", 0) or 0)
    expectancy_norm = min(max(expectancy / (avg_loss + 1e-9), 0.0), 1.0) if expectancy > 0 else 0.0
    net = max(float(metrics.get("net_profit", 0) or 0), 0.0)
    dd = max(float(metrics.get("max_drawdown", 0) or 0), 0.0)
    drawdown_quality = net / (net + dd + 1e-9) if net > 0 else 0.0
    return {
        "profit_factor": pf,
        "expectancy": expectancy_norm,
        "drawdown": drawdown_quality,
        "spike_precision": min(max(float(spike.get("spike_precision", 0)), 0.0), 1.0),
        "spike_recall": min(max(float(spike.get("spike_recall", 0)), 0.0), 1.0),
        "false_alert": 1.0 - min(max(float(spike.get("false_alert_rate", 1)), 0.0), 1.0),
        "oos": min(max(oos_score / 100.0, 0.0), 1.0),
        "forward": min(max(forward_score / 100.0, 0.0), 1.0),
        "stability": min(max(float(metrics.get("stability_score", 0)), 0.0), 1.0),
    }


def calculate_composite(metrics: dict, spike: dict, oos_score: float = 0.0, forward_score: float = 0.0) -> tuple[float, dict]:
    components = composite_components(metrics, spike, oos_score, forward_score)
    weights = settings.arena_composite_weights
    total_weight = sum(float(weights.get(k, 0)) for k in components) or 1.0
    score = 100.0 * sum(components[k] * float(weights.get(k, 0)) for k in components) / total_weight
    return round(score, 4), components


def refresh_leaderboard(db, strategy_id: str) -> LeaderboardEntry | None:
    candidate = db.get(StrategyCandidate, strategy_id)
    if not candidate:
        return None
    base_row = _latest_metric(db, strategy_id, "validation") or _latest_metric(db, strategy_id, "in_sample")
    if not base_row:
        return None

    oos_row = _latest_metric(db, strategy_id, "out_of_sample")
    oos_score = quality_score(oos_row.metrics, oos_row.spike_metrics) if oos_row else 0.0
    forward = _latest_forward(db, strategy_id)
    forward_score = float((forward.metrics or {}).get("forward_score", 0)) if forward else 0.0

    score, components = calculate_composite(base_row.metrics, base_row.spike_metrics, oos_score, forward_score)
    entry = db.execute(select(LeaderboardEntry).where(LeaderboardEntry.strategy_id == strategy_id)).scalar_one_or_none()
    if not entry:
        entry = LeaderboardEntry(strategy_id=strategy_id, symbol=candidate.symbol, composite_score=score, components=components)
        db.add(entry)
    else:
        entry.symbol = candidate.symbol
        entry.composite_score = score
        entry.components = components
    db.flush()
    return entry


def leaderboard_rows(db, symbol: str | None = None, agent: str | None = None, status: str | None = None) -> list[dict]:
    stmt = select(StrategyCandidate)
    if symbol:
        stmt = stmt.where(StrategyCandidate.symbol == symbol.upper().replace(" ", "").replace("INDEX", ""))
    if agent:
        stmt = stmt.where(StrategyCandidate.agent_name == agent)
    if status:
        stmt = stmt.where(StrategyCandidate.status == status)

    candidates = db.execute(stmt).scalars().all()
    rows = []
    for candidate in candidates:
        entry = db.execute(select(LeaderboardEntry).where(LeaderboardEntry.strategy_id == candidate.id)).scalar_one_or_none()
        if not entry:
            entry = refresh_leaderboard(db, candidate.id)
        base = _latest_metric(db, candidate.id, "validation") or _latest_metric(db, candidate.id, "in_sample")
        oos = _latest_metric(db, candidate.id, "out_of_sample")
        forward = _latest_forward(db, candidate.id)
        metrics = base.metrics if base else {}
        spike = base.spike_metrics if base else {}
        rows.append(
            {
                "strategy_id": candidate.id,
                "strategy": candidate.name,
                "agent": candidate.agent_name,
                "symbol": candidate.symbol,
                "status": candidate.status,
                "trades": metrics.get("total_trades", 0),
                "net_profit": metrics.get("net_profit", 0),
                "profit_factor": metrics.get("profit_factor", 0),
                "expectancy": metrics.get("expectancy", 0),
                "max_drawdown": metrics.get("max_drawdown", 0),
                "spike_precision": spike.get("spike_precision", 0),
                "spike_recall": spike.get("spike_recall", 0),
                "false_alert_rate": spike.get("false_alert_rate", 0),
                "oos_score": quality_score(oos.metrics, oos.spike_metrics) if oos else 0,
                "forward_score": float((forward.metrics or {}).get("forward_score", 0)) if forward else 0,
                "stability": metrics.get("stability_score", 0),
                "composite_score": entry.composite_score if entry else 0,
            }
        )
    return sorted(rows, key=lambda r: r["composite_score"], reverse=True)
