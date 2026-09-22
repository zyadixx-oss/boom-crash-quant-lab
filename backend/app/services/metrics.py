import math
from statistics import median

import numpy as np


def classification_metrics(y_true, y_pred):
    tp = sum(1 for a, b in zip(y_true, y_pred) if a and b)
    fp = sum(1 for a, b in zip(y_true, y_pred) if not a and b)
    fn = sum(1 for a, b in zip(y_true, y_pred) if a and not b)
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    false_rate = fp / (tp + fp) if tp + fp else 0.0
    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "false_signal_rate": false_rate,
        "spike_precision": precision,
        "spike_recall": recall,
        "false_alert_rate": false_rate,
    }


def _stability_score(values: np.ndarray) -> float:
    if len(values) < 4:
        return 0.0
    chunks = [x for x in np.array_split(values, min(4, len(values))) if len(x)]
    positive_periods = sum(float(np.mean(x)) > 0 for x in chunks) / len(chunks)
    mean = float(np.mean(values))
    std = float(np.std(values, ddof=1)) if len(values) > 1 else 0.0
    dispersion = 1.0 if std == 0 and mean > 0 else max(0.0, 1.0 - min(std / (abs(mean) + 1e-9), 1.0))
    return round((0.75 * positive_periods + 0.25 * dispersion) if mean > 0 else 0.0, 6)


def trading_metrics(pnls: list[float]):
    if not pnls:
        return {
            "total_trades": 0,
            "winning_trades": 0,
            "losing_trades": 0,
            "wins": 0,
            "losses": 0,
            "win_rate": 0.0,
            "net_profit": 0.0,
            "gross_profit": 0.0,
            "gross_loss": 0.0,
            "profit_factor": 0.0,
            "expectancy": 0.0,
            "max_drawdown": 0.0,
            "average_drawdown": 0.0,
            "average_win": 0.0,
            "average_loss": 0.0,
            "risk_reward": 0.0,
            "consecutive_losses": 0,
            "max_consecutive_losses": 0,
            "recovery_factor": 0.0,
            "sharpe": 0.0,
            "sharpe_like": 0.0,
            "sortino": 0.0,
            "stability_score": 0.0,
        }

    a = np.asarray(pnls, dtype=float)
    wins = a[a > 0]
    losses = a[a <= 0]
    gross_profit = float(wins.sum())
    gross_loss = float(abs(losses.sum()))
    net_profit = float(a.sum())

    equity = np.cumsum(a)
    peaks = np.maximum.accumulate(np.r_[0.0, equity])[1:]
    drawdowns = peaks - equity
    max_drawdown = float(drawdowns.max(initial=0.0))
    average_drawdown = float(drawdowns.mean()) if len(drawdowns) else 0.0

    max_consecutive = current = 0
    for value in a:
        current = current + 1 if value <= 0 else 0
        max_consecutive = max(max_consecutive, current)

    std = float(a.std(ddof=1)) if len(a) > 1 else 0.0
    downside = a[a < 0]
    downside_std = float(downside.std(ddof=1)) if len(downside) > 1 else 0.0
    expectancy = float(a.mean())
    average_win = float(wins.mean()) if len(wins) else 0.0
    average_loss = float(losses.mean()) if len(losses) else 0.0
    risk_reward = average_win / abs(average_loss) if average_loss else (999.0 if average_win else 0.0)
    sharpe = expectancy / std * math.sqrt(len(a)) if std else 0.0
    sortino = expectancy / downside_std * math.sqrt(len(a)) if downside_std else 0.0

    return {
        "total_trades": int(len(a)),
        "winning_trades": int(len(wins)),
        "losing_trades": int(len(losses)),
        "wins": int(len(wins)),
        "losses": int(len(losses)),
        "win_rate": float(len(wins) / len(a)),
        "net_profit": net_profit,
        "gross_profit": gross_profit,
        "gross_loss": gross_loss,
        "profit_factor": float(gross_profit / gross_loss) if gross_loss else (999.0 if gross_profit else 0.0),
        "expectancy": expectancy,
        "max_drawdown": max_drawdown,
        "average_drawdown": average_drawdown,
        "average_win": average_win,
        "average_loss": average_loss,
        "risk_reward": float(risk_reward),
        "consecutive_losses": int(max_consecutive),
        "max_consecutive_losses": int(max_consecutive),
        "recovery_factor": float(net_profit / max_drawdown) if max_drawdown else 0.0,
        "sharpe": float(sharpe),
        "sharpe_like": float(sharpe),
        "sortino": float(sortino),
        "stability_score": _stability_score(a),
    }


def spike_detection_metrics(df, signal_indexes: list[int], profile, direction: str):
    spike_indexes = [int(i) for i in df.index[df["is_spike"].fillna(False)].tolist()]
    captured_spikes: set[int] = set()
    true_signals = 0
    false_signals = 0
    lead_times: list[float] = []
    mae_values: list[float] = []
    mfe_values: list[float] = []
    events: list[dict] = []

    for signal_i in signal_indexes:
        future_spikes = [j for j in spike_indexes if signal_i < j <= signal_i + profile.pre_spike_window]
        target = future_spikes[0] if future_spikes else None
        entry = float(df.iloc[signal_i]["close"])
        atr = float(df.iloc[signal_i].get("atr", 0) or 0)

        horizon_end = min(len(df) - 1, signal_i + profile.test_window_bars)
        horizon = df.iloc[signal_i : horizon_end + 1]
        if direction == "UP":
            mfe = float(horizon["high"].max() - entry)
        else:
            mfe = float(entry - horizon["low"].min())
        mfe_values.append(mfe)

        if target is None:
            false_signals += 1
            continue

        true_signals += 1
        captured_spikes.add(target)
        signal_epoch = int(df.iloc[signal_i]["epoch"])
        spike_epoch = int(df.iloc[target]["epoch"])
        lead = max(0, spike_epoch - signal_epoch)
        lead_times.append(float(lead))

        path = df.iloc[signal_i : target + 1]
        if direction == "UP":
            mae = max(0.0, entry - float(path["low"].min()))
        else:
            mae = max(0.0, float(path["high"].max()) - entry)
        mae_values.append(mae)

        events.append(
            {
                "epoch": spike_epoch,
                "signal_epoch": signal_epoch,
                "side": direction,
                "lead_time_seconds": int(lead),
                "magnitude_atr": float(df.iloc[target].get("future_move_atr", 0) or 0),
                "mae_before_spike": mae,
                "mfe_after_signal": mfe,
                "entry_atr": atr,
            }
        )

    total_spikes = len(spike_indexes)
    total_signals = len(signal_indexes)
    predicted_spikes = len(captured_spikes)
    missed_spikes = max(0, total_spikes - predicted_spikes)
    precision = true_signals / total_signals if total_signals else 0.0
    recall = predicted_spikes / total_spikes if total_spikes else 0.0
    false_alert_rate = false_signals / total_signals if total_signals else 0.0
    capture = 100.0 * (0.4 * precision + 0.4 * recall + 0.2 * (1.0 - false_alert_rate if total_signals else 0.0))

    return {
        "total_spikes": total_spikes,
        "predicted_spikes": predicted_spikes,
        "missed_spikes": missed_spikes,
        "false_spike_signals": false_signals,
        "spike_precision": precision,
        "spike_recall": recall,
        "false_alert_rate": false_alert_rate,
        "average_lead_time_before_spike": float(np.mean(lead_times)) if lead_times else 0.0,
        "median_lead_time_before_spike": float(median(lead_times)) if lead_times else 0.0,
        "MAE_before_spike": float(np.mean(mae_values)) if mae_values else 0.0,
        "MFE_after_signal": float(np.mean(mfe_values)) if mfe_values else 0.0,
        "spike_capture_score": round(capture, 4),
        "events": events,
    }
