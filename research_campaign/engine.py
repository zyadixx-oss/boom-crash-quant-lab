"""Offline, one-position research accounting with explicit execution limits.

Candle execution is an assumption scenario.  It can produce numerical research
metrics but can never pass the execution gate without original tick replay.
No API here can send orders, authorize an account, or enable live trading.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from hashlib import sha256
import json
import math
from typing import Any, Iterable

import numpy as np
import pandas as pd

from .strategies import (StrategyContext, StrategySpec, TIMEFRAMES,
                         direction_for_symbol, normalize_bars, prepare_context)

SAFETY = {"LIVE_TRADING": False, "READY_FOR_LIVE": False, "LIVE_ALLOWED": False,
          "OPENED_TRADES": False, "ready_for_live": False, "live_allowed": False,
          "opened_trades": False}
RISK_FRACTION = .005


@dataclass(frozen=True)
class SplitPlan:
    start: int
    train_end: int
    validation_end: int
    end: int

    def __post_init__(self) -> None:
        if not self.start < self.train_end < self.validation_end < self.end:
            raise ValueError("Nonempty chronological partitions required")

    @classmethod
    def from_calendar(cls, start_epoch: int, end_epoch: int) -> "SplitPlan":
        """One fixed UTC calendar for all frames; never resampled row quantiles."""
        span = int(end_epoch)-int(start_epoch)
        return cls(int(start_epoch), int(start_epoch)+span*3//5,
                   int(start_epoch)+span*4//5, int(end_epoch))

    def bounds(self, partition: str) -> tuple[int, int]:
        return {"train": (self.start, self.train_end),
                "validation": (self.train_end, self.validation_end),
                "final": (self.validation_end, self.end)}[partition]

    def to_dict(self) -> dict[str, Any]:
        return {**asdict(self), "fractions": [.6, .2, .2],
                "calendar_shared_between_frames": True}

    @classmethod
    def from_dict(cls, values: dict[str, Any]) -> "SplitPlan":
        return cls(**{k: int(values[k]) for k in ("start","train_end","validation_end","end")})


@dataclass(frozen=True)
class CostModel:
    """Explicit *assumed* round-trip quote-price cost by default.

    ATR fractions are scenario assumptions, not Deriv fees or spread estimates.
    Verification flags need external historical contract/cost evidence.  Even
    those flags cannot override absent per-trade tick execution in this engine.
    """
    round_trip_atr: float = .10
    round_trip_price: float = 0.0
    mapping_verified: bool = False
    historical_costs_verified: bool = False
    execution_verified: bool = False
    contract_type: str = "UNVERIFIED_CFD_MAPPING"
    evidence: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if not math.isfinite(self.round_trip_atr) or not math.isfinite(self.round_trip_price):
            raise ValueError("Non-finite costs")
        if min(self.round_trip_atr, self.round_trip_price)<0:
            raise ValueError("Negative costs are invalid")
        flags=(self.mapping_verified,self.historical_costs_verified,self.execution_verified)
        if any(type(v) is not bool for v in flags):
            raise ValueError("Verification flags must be booleans")
        if any(flags) and (not self.evidence or not all(isinstance(x,str) and x.strip() for x in self.evidence)):
            raise ValueError("Verification flags require named evidence")

    @property
    def verified(self) -> bool:
        return bool(self.mapping_verified and self.historical_costs_verified and self.execution_verified and self.evidence)

    def to_dict(self) -> dict[str, Any]:
        return {**asdict(self), "status": "VERIFIED" if self.verified else "ASSUMED_NOT_VERIFIED"}

    @classmethod
    def from_dict(cls, values: dict[str, Any]) -> "CostModel":
        payload = {k:v for k,v in values.items() if k!="status"}
        payload["evidence"] = tuple(payload.get("evidence",()))
        return cls(**payload)


def validate_bars(bars: pd.DataFrame, timeframe: str) -> dict[str, Any]:
    if timeframe not in TIMEFRAMES:
        raise ValueError("Unsupported timeframe")
    frame = normalize_bars(bars)
    epoch = frame["epoch"].to_numpy(dtype=np.int64)
    step = TIMEFRAMES[timeframe]
    deltas = np.diff(epoch)
    misaligned = int(np.count_nonzero(epoch % step))
    nonmultiples = int(np.count_nonzero(deltas % step))
    missing = int(np.maximum(deltas//step-1,0).sum())
    return {"rows":len(frame), "missing_bars":missing,
            "misaligned_timestamps":misaligned, "nonmultiple_gaps":nonmultiples,
            "duplicates":0, "invalid_ohlc":0,
            "ok": bool(len(frame)>0 and missing==0 and misaligned==0 and nonmultiples==0)}


def _finite(value: float) -> float | None:
    return float(value) if math.isfinite(float(value)) else None


def _drawdown(equity: np.ndarray) -> float:
    if not len(equity):
        return 0.0
    peaks = np.maximum.accumulate(equity)
    return float(np.max((peaks-equity)/np.maximum(peaks,1e-12)))


def _metrics(trades: list[dict[str, Any]], equity: list[dict[str, Any]],
             initial_equity: float, start: int, end: int) -> dict[str, Any]:
    returns = np.asarray([x["net_R"] for x in trades], dtype=float)
    cash = np.asarray([x["net_pnl"] for x in trades], dtype=float)
    wins = float(cash[cash>0].sum())
    losses = -float(cash[cash<0].sum())
    profit_factor = wins/losses if losses>0 else float("nan")
    final_equity = float(equity[-1]["equity"]) if equity else initial_equity
    dd = _drawdown(np.asarray([x["equity"] for x in equity]))
    active_days = len({int(x["entry_epoch"])//86400 for x in trades})
    return {"completed_trades":len(trades), "sample_size":len(trades),
            "win_rate":float((cash>0).mean()) if len(cash) else None,
            "profit_factor":_finite(profit_factor), "net_profit":float(cash.sum()),
            "net_return":final_equity/initial_equity-1,
            "initial_equity":initial_equity, "final_equity":final_equity,
            "max_drawdown":dd, "mean_net_R":float(returns.mean()) if len(returns) else None,
            "median_holding_seconds":float(np.median([t["exit_epoch"]-t["entry_epoch"] for t in trades])) if trades else None,
            "active_days":active_days, "period_days":(end-start)/86400,
            "units":"assumed_account_units; CFD cash mapping NOT VERIFIED"}


def block_uncertainty(trades: list[dict[str, Any]], seed: int = 1729,
                      repetitions: int = 99999) -> dict[str, Any]:
    """Resample whole fixed UTC weeks; centered one-sided netR null test.

    This accounts for within-week dependence, not unlimited adaptive search.
    The caller must apply Holm across *all* frozen final hypotheses.
    """
    if repetitions<99:
        raise ValueError("At least 99 bootstrap draws required")
    weeks = sorted({t["entry_epoch"]//604800 for t in trades})
    if len(weeks)<8 or len(trades)<100:
        return {"status":"INSUFFICIENT_SAMPLE", "weeks":len(weeks),
                "profit_factor_ci95":None,"mean_net_R_ci95":None,"p_value":1.0,
                "multiplicity_adjusted":False}
    index = {w:i for i,w in enumerate(weeks)}
    count = np.zeros(len(weeks)); total=np.zeros(len(weeks)); pos=np.zeros(len(weeks)); neg=np.zeros(len(weeks))
    for t in trades:
        i=index[t["entry_epoch"]//604800]; r=t["net_R"]
        count[i]+=1; total[i]+=r
        pos[i]+=max(t["net_pnl"],0); neg[i]+=max(-t["net_pnl"],0)
    rng=np.random.default_rng(seed)
    draws=rng.integers(0,len(weeks),(repetitions,len(weeks)))
    ns=count[draws].sum(axis=1)
    means=total[draws].sum(axis=1)/ns
    denominator=neg[draws].sum(axis=1)
    ratios=np.divide(pos[draws].sum(axis=1),denominator,
                     out=np.full(repetitions,np.nan),where=denominator>0)
    observed=total.sum()/count.sum()
    centered=total-observed*count
    null=centered[draws].sum(axis=1)/ns
    p=(1+int(np.count_nonzero(null>=observed)))/(repetitions+1)
    finite=ratios[np.isfinite(ratios)]
    return {"status":"TESTED", "method":"fixed UTC week block percentile bootstrap; centered null",
            "weeks":len(weeks), "repetitions":repetitions,"seed":seed,
            "p_value_resolution":1/(repetitions+1),
            "profit_factor_ci95":[float(x) for x in np.quantile(finite,[.025,.975])] if len(finite)==repetitions else None,
            "mean_net_R_ci95":[float(x) for x in np.quantile(means,[.025,.975])],
            "p_value":p,"multiplicity_adjusted":False}


def _reference(ctx: StrategyContext, symbol: str, start: int, end: int,
               step: int, costs: CostModel, initial: float) -> dict[str, Any]:
    frame, a=ctx.bars,ctx.arrays
    epochs=frame["epoch"].to_numpy(dtype=np.int64)
    idx=np.flatnonzero((epochs>=start)&(epochs+step<=end))
    if not len(idx):
        return {"buy_and_hold":None,"native_direction_hold":None,"appropriate_for_cfd":False}
    first,last=int(idx[0]),int(idx[-1]); entry=a["open"][first]; exit_=a["close"][last]
    # Benchmark enters at this candle's OPEN: its own high/low/close cannot
    # determine a supposedly already-known ATR cost at that open.
    atr=a["atr"][first-1] if first>0 and epochs[first-1]+step==epochs[first] else float("nan")
    if not np.isfinite(atr):
        return {"buy_and_hold":None,"native_direction_hold":None,"appropriate_for_cfd":False}
    cost=costs.round_trip_atr*atr+costs.round_trip_price
    longret=(exit_-entry-cost)/entry
    nativeret=(direction_for_symbol(symbol)*(exit_-entry)-cost)/entry
    common={"entry_epoch":int(epochs[first]),"exit_epoch":int(epochs[last]+step),
            "entry_price":float(entry),"exit_price":float(exit_),"round_trip_cost_price":float(cost),
            "position_model":"fully funded one-unit-price exposure; no verified CFD margin/swap mapping"}
    return {"buy_and_hold":{**common,"direction":1,"net_return":float(longret),"net_profit":float(initial*longret)},
            "native_direction_hold":{**common,"direction":direction_for_symbol(symbol),"net_return":float(nativeret),"net_profit":float(initial*nativeret)},
            "appropriate_for_cfd":False,
            "limitation":"Original buy-and-hold hurdle retained; cash/leveraged CFD comparability NOT VERIFIED"}


def run_backtest(bars: pd.DataFrame | StrategyContext, signals: np.ndarray,
                 spec: StrategySpec, symbol: str, timeframe: str,
                 start: int, end: int, *, partition: str = "validation",
                 costs: CostModel | None = None, initial_equity: float = 10000.0,
                 data_quality_ok: bool = True, provenance: dict[str, Any] | None = None,
                 ticks: pd.DataFrame | None = None, bootstrap_repetitions: int = 99999) -> dict[str, Any]:
    """Replay a fixed policy with signal at close and next-bar-open entry.

    Signals near a partition end are purged by the entire *planned* holding
    horizon, even if they would exit early.  One position at a time.  Candle
    stop/target ambiguity resolves stop first and is always disclosed.  Gaps
    through a stop fill at the observed open, never at a clipped stop price.
    Tick mode requires an exact entry quote and every expected second; unknown
    paths are logged and cannot contribute invented completed returns.
    """
    if partition not in {"train","validation","walk_forward","final"}:
        raise ValueError("Unknown partition")
    if timeframe not in TIMEFRAMES or end<=start or initial_equity<=0:
        raise ValueError("Invalid period/frame/capital")
    costs=costs or CostModel()
    ctx=bars if isinstance(bars,StrategyContext) else prepare_context(bars,timeframe)
    frame,a=ctx.bars,ctx.arrays
    quality=validate_bars(frame,timeframe)
    epochs=frame["epoch"].to_numpy(dtype=np.int64); step=TIMEFRAMES[timeframe]
    s=np.asarray(signals)
    if s.shape!=(len(frame),) or not np.isin(s,[-1,0,1]).all():
        raise ValueError("Signals must be a matching one-dimensional {-1,0,1} array")
    native=direction_for_symbol(symbol)
    if np.any((s!=0)&(s!=native)):
        raise ValueError("Only this symbol's native spike direction is permitted")
    tick_epoch=tick_price=None
    if ticks is not None:
        tc="epoch" if "epoch" in ticks else "timestamp"
        pc="quote" if "quote" in ticks else "price"
        raw_epoch=ticks[tc].to_numpy(dtype=float)
        if not np.isfinite(raw_epoch).all() or np.any(raw_epoch!=np.floor(raw_epoch)):
            raise ValueError("Tick timestamps must be finite integer UTC seconds")
        tick_epoch=raw_epoch.astype(np.int64); tick_price=ticks[pc].to_numpy(dtype=float)
        if len(tick_epoch)!=len(tick_price) or np.any(np.diff(tick_epoch)<=0) or not np.isfinite(tick_price).all():
            raise ValueError("Invalid tick chronology/quotes")
        if np.any(tick_price<=0):
            raise ValueError("Invalid nonpositive tick quote")
    cash=float(initial_equity); occupied_until=-1
    trades=[]; dispositions=[]; equity=[{"epoch":int(start),"equity":cash,"kind":"initial"}]
    ambiguous=0; unknown=0; purged=0
    for i in np.flatnonzero(s):
        i=int(i); j=i+1
        signal_epoch=int(epochs[i]+step)
        if signal_epoch<start or signal_epoch>=end:
            continue
        entry_epoch=int(epochs[j]) if j<len(epochs) else signal_epoch
        planned_end=entry_epoch+spec.max_hold_bars*step
        row={"signal_epoch":signal_epoch,"entry_epoch":entry_epoch,"planned_end_epoch":planned_end}
        if j>=len(epochs) or entry_epoch!=signal_epoch:
            dispositions.append({**row,"status":"UNKNOWN_MISSING_NEXT_OPEN"}); unknown+=1
            occupied_until=planned_end
            continue
        if planned_end>end:
            dispositions.append({**row,"status":"PURGED_FULL_PLANNED_HORIZON"}); purged+=1
            continue
        if entry_epoch<occupied_until:
            dispositions.append({**row,"status":"OCCUPIED"}); continue
        if cash<=0:
            dispositions.append({**row,"status":"ACCOUNT_DEPLETED"}); continue
        atr=float(a["atr"][i])
        if not math.isfinite(atr) or atr<=0:
            dispositions.append({**row,"status":"INVALID_ATR"}); continue
        k_end=j+spec.max_hold_bars
        if k_end>len(epochs) or np.any(np.diff(epochs[j:k_end])!=step):
            dispositions.append({**row,"status":"UNKNOWN_MISSING_HOLDING_BARS"}); unknown+=1
            occupied_until=planned_end; continue
        entry=float(a["open"][j]); stop=entry-native*spec.stop_atr*atr
        target=entry+native*spec.target_atr*atr
        risk_cash=cash*RISK_FRACTION; unit_risk=spec.stop_atr*atr
        quantity=risk_cash/unit_risk
        cost_price=costs.round_trip_atr*atr+costs.round_trip_price
        exit_price=float(a["close"][k_end-1]); exit_epoch=planned_end; reason="TIMEOUT"
        path=[]; trade_ambiguous=False
        if tick_epoch is not None:
            lo=int(np.searchsorted(tick_epoch,entry_epoch)); hi=int(np.searchsorted(tick_epoch,planned_end,side="left"))
            te=tick_epoch[lo:hi]; tp=tick_price[lo:hi]
            if not len(te) or te[0]!=entry_epoch or not math.isclose(float(tp[0]),entry,rel_tol=1e-10,abs_tol=1e-9) or len(te)!=planned_end-entry_epoch or np.any(np.diff(te)!=1):
                dispositions.append({**row,"status":"UNKNOWN_TICK_EXECUTION"}); unknown+=1
                occupied_until=planned_end; continue
            stops=native*(tp-stop)<=0; targets=native*(tp-target)>=0
            hits=np.flatnonzero(stops|targets)
            hit=int(hits[0]) if len(hits) else len(te)-1
            exit_price=float(tp[hit]); exit_epoch=int(te[hit]); reason=("STOP" if stops[hit] else "TARGET") if len(hits) else "TIMEOUT_LAST_OBSERVED_QUOTE"
            path_values=cash+quantity*(native*(tp[:hit+1]-entry)-cost_price)
            peak=np.maximum.accumulate(np.r_[cash,path_values])[1:]
            # Retain all extrema capable of changing drawdown, without millions
            # of JSON points.  Full tick-vector drawdown remains in this path.
            draw=(peak-path_values)/np.maximum(peak,1e-12)
            worst=int(np.argmax(draw))
            prior_peak=int(np.argmax(path_values[:worst+1]))
            candidates={0,hit,int(np.argmin(path_values)),int(np.argmax(path_values)),worst,prior_peak}
            for q in sorted(candidates):
                path.append((int(te[q]),float(path_values[q]),"tick_extremum"))
        else:
            for k in range(j,k_end):
                op,hi_,lo_,cl=(float(a[x][k]) for x in ("open","high","low","close"))
                gap_stop=native*(op-stop)<=0; gap_target=native*(op-target)>=0
                hit_stop=(lo_<=stop) if native==1 else (hi_>=stop)
                hit_target=(hi_>=target) if native==1 else (lo_<=target)
                if gap_stop or gap_target:
                    exit_price=op; exit_epoch=int(epochs[k]); reason="GAP_STOP" if gap_stop else "GAP_TARGET"
                    path.append((exit_epoch,cash+quantity*(native*(op-entry)-cost_price),"gap_exit")); break
                if hit_stop or hit_target:
                    trade_ambiguous=bool(hit_stop and hit_target)
                    exit_price=stop if hit_stop else target; exit_epoch=int(epochs[k]+step)
                    reason="AMBIGUOUS_STOP_FIRST" if trade_ambiguous else ("STOP" if hit_stop else "TARGET")
                    # Exact time inside the candle is unobserved.  Block reentry
                    # until that candle closes; never claim a tick fill.
                    adverse=exit_price if hit_stop else (lo_ if native==1 else hi_)
                    favorable=min(hi_,target) if native==1 else max(lo_,target)
                    path.append((exit_epoch,cash+quantity*(native*(favorable-entry)-cost_price),"conservative_candle_favorable_before_adverse"))
                    path.append((exit_epoch,cash+quantity*(native*(adverse-entry)-cost_price),"candle_adverse_envelope")); break
                adverse=lo_ if native==1 else hi_
                favorable=hi_ if native==1 else lo_
                path.append((int(epochs[k]+step),cash+quantity*(native*(favorable-entry)-cost_price),"conservative_candle_favorable_before_adverse"))
                path.append((int(epochs[k]+step),cash+quantity*(native*(adverse-entry)-cost_price),"candle_adverse_envelope"))
                path.append((int(epochs[k]+step),cash+quantity*(native*(cl-entry)-cost_price),"candle_close"))
        gross_R=native*(exit_price-entry)/unit_risk; net_R=gross_R-cost_price/unit_risk
        net_pnl=risk_cash*net_R; cash_after=cash+net_pnl
        for t,v,kind in path:
            equity.append({"epoch":t,"equity":v,"kind":kind})
        equity.append({"epoch":exit_epoch,"equity":cash_after,"kind":"realized"})
        trades.append({**row,"exit_epoch":exit_epoch,"direction":native,
                       "signal_bar_index":i,"entry_bar_index":j,"entry_price":entry,
                       "exit_price":exit_price,"stop_price":stop,"target_price":target,
                       "entry_atr":atr,"quantity":quantity,"risk_fraction":RISK_FRACTION,
                       "risk_cash":risk_cash,"gross_R":gross_R,"net_R":net_R,
                       "round_trip_cost_price":cost_price,"net_pnl":net_pnl,
                       "equity_before":cash,"equity_after":cash_after,"exit_reason":reason,
                       "ambiguous_candle":trade_ambiguous,
                       "execution_mode":"original_tick_observations" if tick_epoch is not None else "assumed_candle_stop_first"})
        dispositions.append({**row,"status":"COMPLETED","trade_index":len(trades)-1})
        ambiguous+=int(trade_ambiguous); cash=cash_after; occupied_until=exit_epoch
    metrics=_metrics(trades,equity,initial_equity,start,end)
    reference=_reference(ctx,symbol,start,end,step,costs,initial_equity)
    monthly={}
    for trade in trades:
        month=pd.Timestamp(trade["exit_epoch"],unit="s",tz="UTC").strftime("%Y-%m")
        value=monthly.setdefault(month,{"month":month,"completed_trades":0,"net_profit":0.0,"sum_net_R":0.0,
                                       "equity_start":trade["equity_before"],"equity_end":trade["equity_before"],"net_return":0.0})
        value["completed_trades"]+=1; value["net_profit"]+=trade["net_pnl"]; value["sum_net_R"]+=trade["net_R"]
        value["equity_end"]=trade["equity_after"];value["net_return"]=value["equity_end"]/value["equity_start"]-1
    identity={"strategy":spec.to_dict(),"symbol":symbol,"timeframe":timeframe,"start":start,"end":end,"partition":partition,"costs":costs.to_dict()}
    result_id=sha256(json.dumps(identity,sort_keys=True,separators=(",",":")).encode()).hexdigest()
    uncertainty=block_uncertainty(trades,int(result_id[:8],16),bootstrap_repetitions) if partition=="final" else {"status":"NOT_TESTED_UNTIL_FINAL","p_value":1.0,"multiplicity_adjusted":False}
    result={"id":result_id,**identity,"strategy_id":spec.id,"metrics":metrics,
            "trades":trades,"equity":equity,"monthly":list(monthly.values()),
            "reference":reference,"data_quality":quality,"provenance":provenance or {},
            "dispositions":dispositions,"unknown_paths":unknown,"purged_signals":purged,
            "ambiguous_candles":ambiguous,"uncertainty":uncertainty,
            "safety":dict(SAFETY),"realized_cash_profit":"NOT_TESTED",
            "reproducibility":{"deterministic":True,"signal_causality_verified":False,
                               "raw_source_hash_verified":False},
            "requested_data_quality_ok":bool(data_quality_ok),
            "execution_tick_replayed":tick_epoch is not None,
            "drawdown_model":"observed tick extrema" if tick_epoch is not None else "conservative OHLC envelope, favorable before adverse; chronology unverified"}
    result["gates"]=evaluate_final(result)
    result["qualified"]=all(result["gates"].values())
    result["rejection_reasons"]=[k for k,v in result["gates"].items() if not v]
    return result


def risk_adjusted_score(result: dict[str, Any]) -> float:
    """Validation ranking only; never choose settings from the final test."""
    if result["partition"] not in {"validation","walk_forward"}:
        raise ValueError("Only validation/walk-forward results may rank parameters")
    m=result["metrics"]
    if m["completed_trades"]<10 or m["net_return"]<=0 or result["unknown_paths"]:
        return -1e12
    # Return/drawdown with a sample penalty: neither win rate nor PF alone ranks.
    score=m["net_return"]/max(m["max_drawdown"],.005)
    return float(score*math.sqrt(m["completed_trades"]/(m["completed_trades"]+100)))


def evaluate_final(result: dict[str, Any]) -> dict[str, bool]:
    m=result["metrics"]; u=result["uncertainty"]; refs=result["reference"]
    cost=CostModel.from_dict(result["costs"])
    pf=m["profit_factor"]
    bh=refs.get("buy_and_hold"); native=refs.get("native_direction_hold")
    mean_ci=u.get("mean_net_R_ci95"); pf_ci=u.get("profit_factor_ci95")
    repro=result["reproducibility"]
    return {"final_partition":result["partition"]=="final",
            "minimum_100_completed":m["completed_trades"]>=100,
            "maximum_drawdown_20pct":m["max_drawdown"]<=.20,
            "profit_factor_1_10":pf is not None and math.isfinite(pf) and pf>=1.10,
            "positive_net_profit":m["net_profit"]>0,
            "outperforms_buy_and_hold":bool(bh and m["net_return"]>bh["net_return"]),
            "outperforms_native_reference":bool(native and m["net_return"]>native["net_return"]),
            "data_quality":bool(result["data_quality"]["ok"] and result["requested_data_quality_ok"] and not result["unknown_paths"]),
            "future_leak_checked":bool(repro["signal_causality_verified"]),
            "reproducible_real_source":bool(repro["deterministic"] and repro["raw_source_hash_verified"] and result["provenance"].get("real_deriv_data",False)),
            "actual_contract_costs_verified":cost.verified,
            "original_tick_execution_verified":bool(result["execution_tick_replayed"] and cost.execution_verified and not result["unknown_paths"]),
            "statistical_uncertainty":bool(mean_ci and pf_ci and mean_ci[0]>0 and pf_ci[0]>1),
            "multiplicity_adjusted":bool(u.get("multiplicity_adjusted") and u.get("holm_p_value",1)<=.05),
            "all_live_gates_false":set(result["safety"])==set(SAFETY) and all(v is False for v in result["safety"].values())}


def finalize_multiplicity(results: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    """Apply Holm to the entire frozen final family; missing tests receive p=1."""
    rows=list(results)
    if any(r["partition"]!="final" for r in rows):
        raise ValueError("Multiplicity family must contain final results only")
    ordered=sorted(enumerate(rows),key=lambda x:(float(x[1]["uncertainty"].get("p_value",1)),x[1]["id"]))
    last=0.0
    for rank,(_,r) in enumerate(ordered):
        p=float(r["uncertainty"].get("p_value",1))
        if not math.isfinite(p) or not 0<=p<=1:
            raise ValueError("Invalid p value")
        last=max(last,min(1.0,(len(rows)-rank)*p))
        resolution=r["uncertainty"].get("p_value_resolution")
        r["uncertainty"].update({"holm_p_value":last,"multiplicity_adjusted":True,"family_size":len(rows),
                                 "resolution_can_reject_first_holm":bool(resolution is not None and resolution<=.05/max(len(rows),1))})
        r["gates"]=evaluate_final(r); r["qualified"]=all(r["gates"].values())
        r["rejection_reasons"]=[k for k,v in r["gates"].items() if not v]
    return rows
