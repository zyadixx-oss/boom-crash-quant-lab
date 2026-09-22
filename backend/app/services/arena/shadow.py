from __future__ import annotations

import asyncio
from datetime import datetime
import math
import time
from types import SimpleNamespace

import pandas as pd
from sqlalchemy import select

from ...config import settings
from ...db import SessionLocal
from ...models import ForwardRun, StrategyCandidate, StrategyStatusHistory, SystemEvent
from ...symbol_profiles import get_profile
from ..candles import aggregate_ticks
from ..deriv_client import DerivClient
from ..safety import assert_market_data_only
from .evaluator import prepare_arena_frame, score_candidate_row
from .leaderboard import refresh_leaderboard


ALLOWED_DURATIONS = {24, 72, 168}


def _candidate_copy(row: StrategyCandidate):
    return SimpleNamespace(
        id=row.id,
        name=row.name,
        agent_name=row.agent_name,
        symbol=row.symbol,
        version=row.version,
        parameters=dict(row.parameters or {}),
        rules=list(row.rules or []),
        indicators_used=list(row.indicators_used or []),
        timeframes=list(row.timeframes or []),
    )


class ArenaShadowManager:
    def __init__(self):
        self.tasks: dict[int, asyncio.Task] = {}

    async def start(self, strategy_id: str, duration_hours: int = 24) -> dict:
        assert_market_data_only()
        if duration_hours not in ALLOWED_DURATIONS:
            raise ValueError("duration_hours must be one of 24, 72, 168")

        with SessionLocal() as db:
            candidate = db.get(StrategyCandidate, strategy_id)
            if not candidate:
                raise KeyError("Strategy not found")
            if candidate.status not in {"forward_testing", "qualified"}:
                raise ValueError(f"Strategy is not eligible for shadow testing; current status={candidate.status}")

            existing = db.execute(
                select(ForwardRun)
                .where(ForwardRun.strategy_id == strategy_id, ForwardRun.status == "running")
                .order_by(ForwardRun.id.desc())
                .limit(1)
            ).scalar_one_or_none()
            if existing and existing.id in self.tasks and not self.tasks[existing.id].done():
                return self._run_dict(existing, active=True)

            run = ForwardRun(
                strategy_id=strategy_id,
                mode=settings.arena_mode,
                duration_hours=duration_hours,
                status="running",
                metrics={
                    "total_signals": 0,
                    "evaluated_signals": 0,
                    "spike_hits": 0,
                    "false_signals": 0,
                    "spike_precision": 0.0,
                    "false_alert_rate": 0.0,
                    "average_lead_time": 0.0,
                    "average_future_movement": 0.0,
                    "forward_score": 0.0,
                },
            )
            db.add(run)
            db.add(
                SystemEvent(
                    level="INFO",
                    event_type="arena_shadow_start",
                    message=f"{strategy_id} {duration_hours}h",
                    payload={"strategy_id": strategy_id, "mode": settings.arena_mode, "duration_hours": duration_hours},
                )
            )
            db.commit()
            db.refresh(run)
            run_id = run.id

        task = asyncio.create_task(self._run(run_id), name=f"arena-shadow-{run_id}")
        self.tasks[run_id] = task
        return {"id": run_id, "strategy_id": strategy_id, "status": "running", "mode": settings.arena_mode, "duration_hours": duration_hours, "active": True}

    async def stop(self, strategy_id: str) -> dict:
        assert_market_data_only()
        with SessionLocal() as db:
            run = db.execute(
                select(ForwardRun)
                .where(ForwardRun.strategy_id == strategy_id, ForwardRun.status == "running")
                .order_by(ForwardRun.id.desc())
                .limit(1)
            ).scalar_one_or_none()
            if not run:
                return {"strategy_id": strategy_id, "status": "not_running", "active": False}
            task = self.tasks.get(run.id)
            if task and not task.done():
                task.cancel()
            run.status = "stopped"
            run.ended_at = datetime.utcnow()
            db.add(
                SystemEvent(
                    level="INFO",
                    event_type="arena_shadow_stop",
                    message=strategy_id,
                    payload={"strategy_id": strategy_id, "run_id": run.id},
                )
            )
            db.commit()
            return self._run_dict(run, active=False)

    def status(self, strategy_id: str | None = None) -> list[dict]:
        with SessionLocal() as db:
            stmt = select(ForwardRun)
            if strategy_id:
                stmt = stmt.where(ForwardRun.strategy_id == strategy_id)
            rows = db.execute(stmt.order_by(ForwardRun.id.desc()).limit(100)).scalars().all()
            return [self._run_dict(r, active=(r.id in self.tasks and not self.tasks[r.id].done())) for r in rows]

    def _run_dict(self, run: ForwardRun, active: bool) -> dict:
        return {
            "id": run.id,
            "strategy_id": run.strategy_id,
            "mode": run.mode,
            "duration_hours": run.duration_hours,
            "status": run.status,
            "metrics": run.metrics or {},
            "last_error": run.last_error,
            "started_at": run.started_at.isoformat() if run.started_at else None,
            "ended_at": run.ended_at.isoformat() if run.ended_at else None,
            "active": active,
        }

    def _record_signal(self, run_id: int) -> None:
        with SessionLocal() as db:
            run = db.get(ForwardRun, run_id)
            if not run:
                return
            metrics = dict(run.metrics or {})
            metrics["total_signals"] = int(metrics.get("total_signals", 0)) + 1
            run.metrics = metrics
            db.commit()

    def _record_result(self, run_id: int, result_payload: dict) -> None:
        with SessionLocal() as db:
            run = db.get(ForwardRun, run_id)
            if not run:
                return
            metrics = dict(run.metrics or {})
            evaluated = int(metrics.get("evaluated_signals", 0)) + 1
            hits = int(metrics.get("spike_hits", 0)) + int(bool(result_payload["spike_occurred"]))
            false_signals = int(metrics.get("false_signals", 0)) + int(bool(result_payload["false_signal"]))
            prior_lead_total = float(metrics.get("average_lead_time", 0)) * int(metrics.get("evaluated_signals", 0))
            lead = float(result_payload.get("lead_time") or 0)
            prior_move_total = float(metrics.get("average_future_movement", 0)) * int(metrics.get("evaluated_signals", 0))
            movement = float(result_payload.get("actual_future_movement") or 0)

            precision = hits / evaluated if evaluated else 0.0
            false_rate = false_signals / evaluated if evaluated else 0.0
            forward_score = 100.0 * (0.60 * precision + 0.40 * (1.0 - false_rate))

            metrics.update(
                {
                    "evaluated_signals": evaluated,
                    "spike_hits": hits,
                    "false_signals": false_signals,
                    "spike_precision": precision,
                    "false_alert_rate": false_rate,
                    "average_lead_time": (prior_lead_total + lead) / evaluated,
                    "average_future_movement": (prior_move_total + movement) / evaluated,
                    "forward_score": round(forward_score, 4),
                }
            )
            run.metrics = metrics
            refresh_leaderboard(db, run.strategy_id)
            db.commit()

    def _complete_run(self, run_id: int, failed: bool = False, error: str | None = None) -> None:
        with SessionLocal() as db:
            run = db.get(ForwardRun, run_id)
            if not run:
                return
            if run.status == "stopped":
                return
            run.status = "failed" if failed else "completed"
            run.last_error = error
            run.ended_at = datetime.utcnow()

            candidate = db.get(StrategyCandidate, run.strategy_id)
            if candidate and not failed:
                metrics = run.metrics or {}
                evaluated = int(metrics.get("evaluated_signals", 0))
                passed = (
                    evaluated >= settings.arena_minimum_trades
                    and float(metrics.get("spike_precision", 0)) >= settings.arena_minimum_spike_precision
                    and float(metrics.get("false_alert_rate", 1)) <= settings.arena_maximum_false_alert_rate
                )
                old = candidate.status
                candidate.status = "qualified" if passed else "retired"
                db.add(
                    StrategyStatusHistory(
                        strategy_id=candidate.id,
                        from_status=old,
                        to_status=candidate.status,
                        reason="forward/shadow gate passed" if passed else "forward/shadow gate failed or insufficient observations",
                    )
                )
                refresh_leaderboard(db, candidate.id)

            db.add(
                SystemEvent(
                    level="ERROR" if failed else "INFO",
                    event_type="arena_shadow_complete",
                    message=f"run={run_id} status={run.status}",
                    payload={"strategy_id": run.strategy_id, "run_id": run_id, "error": error},
                )
            )
            db.commit()

    async def _run(self, run_id: int) -> None:
        assert_market_data_only()
        with SessionLocal() as db:
            run = db.get(ForwardRun, run_id)
            if not run:
                return
            candidate_row = db.get(StrategyCandidate, run.strategy_id)
            if not candidate_row:
                self._complete_run(run_id, failed=True, error="Strategy not found")
                return
            candidate = _candidate_copy(candidate_row)
            duration_hours = run.duration_hours

        profile = get_profile(candidate.symbol)
        direction = "UP" if candidate.symbol.startswith("BOOM") else "DOWN"
        threshold = float(candidate.parameters.get("min_signal_score", profile.min_signal_score))
        client = DerivClient()
        pending: list[dict] = []
        buffer: list[dict] = []
        last_closed_epoch = None
        last_signal_epoch = -10**12
        deadline = time.monotonic() + duration_hours * 3600

        try:
            await client.connect()
            api_symbol = await client.resolve_symbol(profile.display_name, profile.api_symbol)
            if not api_symbol:
                raise RuntimeError(f"Could not resolve Deriv symbol for {profile.display_name}")

            async for tick in client.subscribe_ticks(api_symbol):
                if time.monotonic() >= deadline:
                    break
                epoch = int(tick["epoch"])
                price = float(tick["quote"])
                buffer.append({"epoch": epoch, "price": price})
                buffer = buffer[-30000:]

                # Evaluate pending paper signals against future market movement.
                for item in pending:
                    if item.get("done"):
                        continue
                    signed_move = price - item["price"] if direction == "UP" else item["price"] - price
                    if item.get("spike_epoch") is None and item["atr"] > 0 and signed_move >= item["atr"] * profile.spike_atr_multiple:
                        item["spike_epoch"] = epoch
                    if epoch < item["epoch"] + profile.test_window_bars * 60:
                        continue

                    spike_epoch = item.get("spike_epoch")
                    spike_occurred = spike_epoch is not None
                    payload = {
                        "timestamp": epoch,
                        "symbol": candidate.symbol,
                        "strategy_id": candidate.id,
                        "signal_direction": direction,
                        "signal_score": item["score"],
                        "predicted_spike_side": direction,
                        "price_at_signal": item["price"],
                        "actual_future_movement": signed_move,
                        "spike_occurred": spike_occurred,
                        "lead_time": (spike_epoch - item["epoch"]) if spike_occurred else None,
                        "false_signal": not spike_occurred,
                        "outcome": "SPIKE_HIT" if spike_occurred else "NO_SPIKE",
                        "mode": settings.arena_mode,
                    }
                    with SessionLocal() as db:
                        db.add(SystemEvent(level="INFO", event_type="arena_shadow_result", message=candidate.id, payload=payload))
                        db.commit()
                    self._record_result(run_id, payload)
                    item["done"] = True

                candles = aggregate_ticks(pd.DataFrame(buffer), "M1")
                closed = candles[candles["epoch"] + 60 <= epoch]
                if len(closed) < 80:
                    continue
                current_closed_epoch = int(closed.iloc[-1]["epoch"])
                if current_closed_epoch == last_closed_epoch:
                    continue
                last_closed_epoch = current_closed_epoch

                window = closed.tail(500).reset_index(drop=True)
                frame = prepare_arena_frame(window, profile, with_spikes=False)
                row = frame.iloc[-1]
                score, reasons = score_candidate_row(row, candidate, profile, direction)
                if score < threshold:
                    continue
                if current_closed_epoch - last_signal_epoch < profile.cooldown_bars * 60:
                    continue

                atr = float(row.get("atr", 0) or 0)
                if not math.isfinite(atr):
                    atr = 0.0
                signal_payload = {
                    "timestamp": current_closed_epoch,
                    "symbol": candidate.symbol,
                    "strategy_id": candidate.id,
                    "signal_direction": direction,
                    "signal_score": score,
                    "predicted_spike_side": direction,
                    "price_at_signal": float(row["close"]),
                    "actual_future_movement": None,
                    "spike_occurred": None,
                    "lead_time": None,
                    "false_signal": None,
                    "outcome": "PENDING",
                    "mode": settings.arena_mode,
                    "reasons": reasons,
                }
                with SessionLocal() as db:
                    db.add(SystemEvent(level="INFO", event_type="arena_shadow_signal", message=candidate.id, payload=signal_payload))
                    db.commit()
                self._record_signal(run_id)
                pending.append({"epoch": current_closed_epoch, "price": float(row["close"]), "score": score, "atr": atr, "done": False})
                pending = pending[-500:]
                last_signal_epoch = current_closed_epoch

            self._complete_run(run_id)
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            self._complete_run(run_id, failed=True, error=str(exc))
        finally:
            await client.close()
            self.tasks.pop(run_id, None)


shadow_manager = ArenaShadowManager()
