import asyncio
import json
import math
import random
from dataclasses import dataclass
from datetime import timezone
from pathlib import Path

import numpy as np
import pandas as pd
import websockets

# Research-only CRT backtest. No auth, no order endpoints, no trading.
SYMBOLS = [
    "Boom 300 Index", "Boom 500 Index", "Boom 600 Index", "Boom 900 Index", "Boom 1000 Index",
    "Crash 300 Index", "Crash 500 Index", "Crash 600 Index", "Crash 900 Index", "Crash 1000 Index",
]
GRANULARITY = 300  # M5
BATCHES = 3        # up to 15k M5 candles ~= 52 days
BATCH_SIZE = 5000
HORIZON = 24       # 2 hours after entry
SPIKE_HORIZON = 12 # 1 hour for favorable excursion test
SEED = 42


def norm_name(s: str) -> str:
    return " ".join(str(s).lower().replace("index", "").split())


async def connect():
    endpoints = [
        "wss://ws.derivws.com/websockets/v3?app_id=1089",
        "wss://ws.binaryws.com/websockets/v3?app_id=1089",
    ]
    last = None
    for ep in endpoints:
        try:
            return await websockets.connect(ep, ping_interval=20, ping_timeout=20, max_size=8_000_000)
        except Exception as e:
            last = e
    raise RuntimeError(f"Could not connect to Deriv WebSocket: {last}")


async def request(ws, payload):
    await ws.send(json.dumps(payload))
    while True:
        msg = json.loads(await ws.recv())
        if "error" in msg:
            raise RuntimeError(msg["error"].get("message", str(msg["error"])))
        # sequential requests: first non-subscription response belongs to us
        return msg


async def resolve_symbols(ws):
    data = await request(ws, {"active_symbols": "brief"})
    active = data.get("active_symbols", [])
    by_name = {norm_name(x.get("display_name", "")): x.get("symbol") for x in active}
    out = {}
    for display in SYMBOLS:
        key = norm_name(display)
        if key in by_name and by_name[key]:
            out[display] = by_name[key]
    return out


async def fetch_candles(ws, api_symbol):
    all_rows = []
    end = "latest"
    for _ in range(BATCHES):
        data = await request(ws, {
            "ticks_history": api_symbol,
            "adjust_start_time": 1,
            "count": BATCH_SIZE,
            "end": end,
            "style": "candles",
            "granularity": GRANULARITY,
        })
        rows = data.get("candles", [])
        if not rows:
            break
        parsed = [{
            "epoch": int(x["epoch"]),
            "open": float(x["open"]),
            "high": float(x["high"]),
            "low": float(x["low"]),
            "close": float(x["close"]),
        } for x in rows]
        all_rows.extend(parsed)
        oldest = min(x["epoch"] for x in parsed)
        if len(parsed) < 100:
            break
        end = oldest - 1
        await asyncio.sleep(0.15)
    if not all_rows:
        return pd.DataFrame(columns=["epoch", "open", "high", "low", "close"])
    df = pd.DataFrame(all_rows).drop_duplicates("epoch").sort_values("epoch").reset_index(drop=True)
    return df


def add_features(df):
    x = df.copy()
    prev_close = x["close"].shift(1)
    tr = pd.concat([
        x["high"] - x["low"],
        (x["high"] - prev_close).abs(),
        (x["low"] - prev_close).abs(),
    ], axis=1).max(axis=1)
    x["atr"] = tr.rolling(14, min_periods=14).mean()
    x["dt"] = pd.to_datetime(x["epoch"], unit="s", utc=True)
    x["hour"] = x["dt"].dt.floor("1h")

    h1 = x.set_index("dt").resample("1h").agg(
        ref_open=("open", "first"),
        ref_high=("high", "max"),
        ref_low=("low", "min"),
        ref_close=("close", "last"),
        bars=("close", "count"),
    ).dropna()
    h1 = h1[h1["bars"] >= 10].copy()
    h1["hour"] = h1.index
    h1["join_hour"] = h1["hour"] + pd.Timedelta(hours=1)
    refs = h1[["join_hour", "ref_open", "ref_high", "ref_low", "ref_close"]].copy()
    x = x.merge(refs, left_on="hour", right_on="join_hour", how="left")
    return x.reset_index(drop=True)


def trade_path(x, entry_i, direction, entry, stop, target, horizon=HORIZON):
    risk = (entry - stop) if direction == "UP" else (stop - entry)
    if not np.isfinite(risk) or risk <= 0:
        return None

    last = min(len(x) - 1, entry_i + horizon)
    outcome = "TIME"
    exit_price = float(x.iloc[last]["close"])
    exit_i = last

    for j in range(entry_i + 1, last + 1):
        hi = float(x.iloc[j]["high"])
        lo = float(x.iloc[j]["low"])
        if direction == "UP":
            hit_sl = lo <= stop
            hit_tp = hi >= target
        else:
            hit_sl = hi >= stop
            hit_tp = lo <= target

        # Conservative OHLC ambiguity handling: if both touched in same bar, stop first.
        if hit_sl:
            outcome, exit_price, exit_i = "SL", stop, j
            break
        if hit_tp:
            outcome, exit_price, exit_i = "TP", target, j
            break

    pnl = (exit_price - entry) if direction == "UP" else (entry - exit_price)
    r = pnl / risk

    spike_last = min(len(x) - 1, entry_i + SPIKE_HORIZON)
    atr = float(x.iloc[entry_i]["atr"])
    if direction == "UP":
        mfe = float(x.iloc[entry_i + 1: spike_last + 1]["high"].max() - entry) if spike_last > entry_i else 0.0
    else:
        mfe = float(entry - x.iloc[entry_i + 1: spike_last + 1]["low"].min()) if spike_last > entry_i else 0.0
    mfe_atr = mfe / atr if np.isfinite(atr) and atr > 0 else np.nan

    return {
        "entry_i": entry_i,
        "epoch": int(x.iloc[entry_i]["epoch"]),
        "entry": entry,
        "stop": stop,
        "target": target,
        "risk": risk,
        "outcome": outcome,
        "r": float(r),
        "mfe_atr": float(mfe_atr) if np.isfinite(mfe_atr) else None,
        "target_r": float(abs(target - entry) / risk),
        "hold_bars": int(exit_i - entry_i),
    }


def raw_crt_signals(x, direction):
    trades = []
    seen_hours = set()
    for i in range(20, len(x) - HORIZON - 1):
        row = x.iloc[i]
        if not np.isfinite(row.get("atr", np.nan)) or pd.isna(row.get("ref_low")):
            continue
        hour = row["hour"]
        if hour in seen_hours:
            continue
        atr = float(row["atr"])

        if direction == "UP":
            swept = float(row["low"]) < float(row["ref_low"]) and float(row["close"]) > float(row["ref_low"])
            if not swept:
                continue
            entry = float(row["close"])
            stop = float(row["low"]) - 0.15 * atr
            target = float(row["ref_high"])
            if target <= entry:
                continue
        else:
            swept = float(row["high"]) > float(row["ref_high"]) and float(row["close"]) < float(row["ref_high"])
            if not swept:
                continue
            entry = float(row["close"])
            stop = float(row["high"]) + 0.15 * atr
            target = float(row["ref_low"])
            if target >= entry:
                continue

        t = trade_path(x, i, direction, entry, stop, target)
        if t:
            t["setup_i"] = i
            trades.append(t)
            seen_hours.add(hour)
    return trades


def filtered_crt_signals(x, direction, require_fvg=False):
    trades = []
    seen_hours = set()
    for i in range(20, len(x) - HORIZON - 4):
        row = x.iloc[i]
        if not np.isfinite(row.get("atr", np.nan)) or pd.isna(row.get("ref_low")):
            continue
        hour = row["hour"]
        if hour in seen_hours:
            continue
        atr = float(row["atr"])

        if direction == "UP":
            swept = float(row["low"]) < float(row["ref_low"]) and float(row["close"]) > float(row["ref_low"])
        else:
            swept = float(row["high"]) > float(row["ref_high"]) and float(row["close"]) < float(row["ref_high"])
        if not swept:
            continue

        confirm_i = None
        for j in range(i + 1, min(i + 4, len(x) - HORIZON)):
            c = x.iloc[j]
            body = abs(float(c["close"]) - float(c["open"]))
            a = float(c["atr"]) if np.isfinite(c["atr"]) else atr
            displacement = a > 0 and body >= 0.50 * a
            if direction == "UP":
                swing = float(x.iloc[j-3:j]["high"].max())
                mss = float(c["close"]) > swing
                fvg = j >= 2 and float(c["low"]) > float(x.iloc[j-2]["high"])
            else:
                swing = float(x.iloc[j-3:j]["low"].min())
                mss = float(c["close"]) < swing
                fvg = j >= 2 and float(c["high"]) < float(x.iloc[j-2]["low"])
            if mss and displacement and (fvg or not require_fvg):
                confirm_i = j
                break
        if confirm_i is None:
            continue

        c = x.iloc[confirm_i]
        if direction == "UP":
            entry = float(c["close"])
            stop = float(row["low"]) - 0.15 * atr
            target = float(row["ref_high"])
            if target <= entry:
                continue
        else:
            entry = float(c["close"])
            stop = float(row["high"]) + 0.15 * atr
            target = float(row["ref_low"])
            if target >= entry:
                continue

        t = trade_path(x, confirm_i, direction, entry, stop, target)
        if t:
            t["setup_i"] = i
            trades.append(t)
            seen_hours.add(hour)
    return trades


def max_drawdown(rs):
    if not rs:
        return 0.0
    eq = np.cumsum(np.array(rs, dtype=float))
    peak = np.maximum.accumulate(np.r_[0.0, eq])
    dd = peak[1:] - eq
    return float(np.max(dd)) if len(dd) else 0.0


def metrics(trades):
    if not trades:
        return {
            "n": 0, "win_rate": None, "tp_rate": None, "profit_factor": None,
            "expectancy_r": None, "median_r": None, "max_drawdown_r": None,
            "mfe_2atr_rate": None, "median_mfe_atr": None,
        }
    rs = np.array([t["r"] for t in trades], dtype=float)
    pos = rs[rs > 0].sum()
    neg = -rs[rs < 0].sum()
    pf = float(pos / neg) if neg > 0 else (999.0 if pos > 0 else 0.0)
    mfes = np.array([t["mfe_atr"] for t in trades if t["mfe_atr"] is not None], dtype=float)
    return {
        "n": len(trades),
        "win_rate": float(np.mean(rs > 0)),
        "tp_rate": float(np.mean([t["outcome"] == "TP" for t in trades])),
        "profit_factor": pf,
        "expectancy_r": float(np.mean(rs)),
        "median_r": float(np.median(rs)),
        "max_drawdown_r": max_drawdown(rs.tolist()),
        "mfe_2atr_rate": float(np.mean(mfes >= 2.0)) if len(mfes) else None,
        "median_mfe_atr": float(np.median(mfes)) if len(mfes) else None,
    }


def baseline_mfe(x, direction, n, seed=SEED):
    eligible = x.index[
        x["atr"].notna() & x["ref_low"].notna()
    ].tolist()
    eligible = [i for i in eligible if i < len(x) - SPIKE_HORIZON - 1 and i > 20]
    if not eligible or n <= 0:
        return None
    rng = random.Random(seed)
    sample = rng.sample(eligible, min(n, len(eligible)))
    hits = []
    for i in sample:
        entry = float(x.iloc[i]["close"])
        atr = float(x.iloc[i]["atr"])
        last = min(len(x) - 1, i + SPIKE_HORIZON)
        if direction == "UP":
            mfe = float(x.iloc[i+1:last+1]["high"].max() - entry)
        else:
            mfe = float(entry - x.iloc[i+1:last+1]["low"].min())
        hits.append(mfe / atr if atr > 0 else np.nan)
    a = np.array([v for v in hits if np.isfinite(v)], dtype=float)
    if not len(a):
        return None
    return {
        "n": int(len(a)),
        "mfe_2atr_rate": float(np.mean(a >= 2.0)),
        "median_mfe_atr": float(np.median(a)),
    }


def split_oos(trades):
    if not trades:
        return [], []
    ts = sorted(trades, key=lambda t: t["epoch"])
    cut = max(1, int(len(ts) * 0.70))
    return ts[:cut], ts[cut:]


def verdict(oos, baseline):
    if oos["n"] < 20:
        return "INSUFFICIENT"
    pf = oos["profit_factor"] or 0
    ex = oos["expectancy_r"] or 0
    lift = None
    if baseline and baseline["mfe_2atr_rate"] and oos["mfe_2atr_rate"] is not None:
        lift = oos["mfe_2atr_rate"] / baseline["mfe_2atr_rate"]
    if oos["n"] >= 30 and pf >= 1.30 and ex >= 0.15 and (lift is None or lift >= 1.15):
        return "EXCELLENT_CANDIDATE"
    if pf >= 1.10 and ex > 0:
        return "PROMISING"
    return "WEAK_OR_FAIL"


def pct(v):
    return "—" if v is None else f"{100*v:.1f}%"


def num(v, nd=2):
    return "—" if v is None else f"{v:.{nd}f}"


async def main():
    ws = await connect()
    try:
        resolved = await resolve_symbols(ws)
        print("Resolved symbols:", resolved)
        results = {}
        for display, api_symbol in resolved.items():
            print(f"\n=== {display} [{api_symbol}] ===", flush=True)
            raw = await fetch_candles(ws, api_symbol)
            print(f"Fetched {len(raw)} M5 candles", flush=True)
            if len(raw) < 1000:
                results[display] = {"api_symbol": api_symbol, "error": "too_few_candles", "candles": len(raw)}
                continue
            x = add_features(raw)
            direction = "UP" if display.startswith("Boom") else "DOWN"
            variants = {
                "CRT_RAW": raw_crt_signals(x, direction),
                "CRT_MSS": filtered_crt_signals(x, direction, require_fvg=False),
                "CRT_MSS_FVG": filtered_crt_signals(x, direction, require_fvg=True),
            }
            symbol_out = {
                "api_symbol": api_symbol,
                "candles": len(raw),
                "from_epoch": int(raw["epoch"].min()),
                "to_epoch": int(raw["epoch"].max()),
                "direction": direction,
                "variants": {},
            }
            for name, trades in variants.items():
                train, test = split_oos(trades)
                all_m = metrics(trades)
                oos_m = metrics(test)
                base = baseline_mfe(x, direction, max(200, oos_m["n"] * 5 if oos_m["n"] else 200), seed=SEED + len(name))
                v = verdict(oos_m, base)
                symbol_out["variants"][name] = {
                    "all": all_m,
                    "oos": oos_m,
                    "baseline": base,
                    "verdict": v,
                }
                print(
                    f"{name}: all n={all_m['n']} PF={num(all_m['profit_factor'])} "
                    f"Exp={num(all_m['expectancy_r'])}R | OOS n={oos_m['n']} "
                    f"PF={num(oos_m['profit_factor'])} Exp={num(oos_m['expectancy_r'])}R "
                    f"TP={pct(oos_m['tp_rate'])} 2ATR={pct(oos_m['mfe_2atr_rate'])} "
                    f"baseline2ATR={pct(base['mfe_2atr_rate'] if base else None)} => {v}",
                    flush=True
                )
            results[display] = symbol_out

        out = Path("crt_backtest_results.json")
        out.write_text(json.dumps(results, indent=2), encoding="utf-8")

        # compact markdown summary
        lines = [
            "# CRT Backtest Summary",
            "",
            "Research only. Public Deriv market data; no authentication and no trades.",
            "",
            "| Symbol | Variant | OOS n | PF | Exp (R) | TP rate | 2ATR hit | Baseline 2ATR | Verdict |",
            "|---|---:|---:|---:|---:|---:|---:|---:|---|",
        ]
        for display, s in results.items():
            if "variants" not in s:
                continue
            for name, d in s["variants"].items():
                o = d["oos"]
                b = d["baseline"]
                lines.append(
                    f"| {display} | {name} | {o['n']} | {num(o['profit_factor'])} | "
                    f"{num(o['expectancy_r'])} | {pct(o['tp_rate'])} | {pct(o['mfe_2atr_rate'])} | "
                    f"{pct(b['mfe_2atr_rate'] if b else None)} | {d['verdict']} |"
                )
        Path("crt_backtest_summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
        print("\n" + "\n".join(lines), flush=True)
    finally:
        await ws.close()


if __name__ == "__main__":
    asyncio.run(main())
