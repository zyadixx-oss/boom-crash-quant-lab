import argparse
import asyncio
import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "backend"))

from app.config import settings
from app.db import SessionLocal, init_db
from app.services.arena.pipeline import ArenaPipeline, bootstrap_arena
from app.services.arena.shadow import shadow_manager


def load_csv(path: str) -> list[dict]:
    file_path = Path(path)
    if not file_path.exists():
        raise SystemExit(f"Missing candle CSV: {file_path}")
    return pd.read_csv(file_path).to_dict(orient="records")


def list_strategies() -> None:
    with SessionLocal() as db:
        rows = ArenaPipeline.list_strategies(db)
    for row in rows:
        print(f"{row['id']}  {row['symbol']:8}  {row['status']:15}  {row['agent_name']}")


def run_backtest(strategy_id: str, csv_path: str) -> None:
    with SessionLocal() as db:
        result = ArenaPipeline.backtest(db, strategy_id, load_csv(csv_path))
    print(json.dumps(result, indent=2, default=str))


def run_oos(strategy_id: str, csv_path: str) -> None:
    with SessionLocal() as db:
        result = ArenaPipeline.oos_test(db, strategy_id, load_csv(csv_path))
    print(json.dumps(result, indent=2, default=str))


async def run_shadow(strategy_id: str, hours: int) -> None:
    started = await shadow_manager.start(strategy_id, hours)
    print(json.dumps(started, indent=2))
    task = shadow_manager.tasks.get(started["id"])
    if task:
        await task


def main() -> None:
    settings.assert_safe()
    init_db()
    bootstrap_arena()

    parser = argparse.ArgumentParser(description="AI Strategy Arena CLI (research/shadow only; no order execution)")
    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser("list", help="List Strategy Candidates")

    bt = sub.add_parser("backtest", help="Run train + validation historical backtest")
    bt.add_argument("strategy_id")
    bt.add_argument("csv")

    oos = sub.add_parser("oos", help="Run untouched OOS + robustness test")
    oos.add_argument("strategy_id")
    oos.add_argument("csv")

    shadow = sub.add_parser("shadow", help="Run Deriv market-data-only forward/shadow test")
    shadow.add_argument("strategy_id")
    shadow.add_argument("--hours", type=int, choices=[24, 72, 168], default=24)

    args = parser.parse_args()
    if args.command == "list":
        list_strategies()
    elif args.command == "backtest":
        run_backtest(args.strategy_id, args.csv)
    elif args.command == "oos":
        run_oos(args.strategy_id, args.csv)
    elif args.command == "shadow":
        asyncio.run(run_shadow(args.strategy_id, args.hours))


if __name__ == "__main__":
    main()
