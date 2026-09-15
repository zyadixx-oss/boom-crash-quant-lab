from dataclasses import dataclass, asdict
import math

@dataclass
class SignalDecision:
    symbol:str; epoch:int; direction:str; score:float; reason_codes:list[str]; features_snapshot:dict; entry_reference_price:float; expected_test_window:int
    def to_dict(self): return asdict(self)

def _ok(v): return v is not None and not (isinstance(v,float) and math.isnan(v))

def score_row(row, profile, direction:str)->SignalDecision:
    w=profile.weights; flags=profile.feature_flags; score=0.0; reasons=[]
    def add(key,cond,reason):
        nonlocal score
        if cond: score+=w.get(key,0); reasons.append(reason)
    if flags.get('bollinger',True): add('bollinger_squeeze', _ok(row.get('bb_ratio')) and row['bb_ratio']<profile.bb_squeeze_threshold, 'BB_SQUEEZE')
    if flags.get('atr',True): add('atr_compression', _ok(row.get('atr_ratio')) and row['atr_ratio']<profile.atr_compression_threshold, 'ATR_COMPRESSION')
    add('candle_structure', _ok(row.get('range')) and _ok(row.get('atr')) and row['range']<row['atr']*0.8, 'TIGHT_CANDLE')
    if flags.get('ict',True):
        sweep = row.get('liquidity_sweep_down',False) if direction=='UP' else row.get('liquidity_sweep_up',False)
        add('liquidity_sweep', bool(sweep), 'LIQUIDITY_SWEEP')
    sr = (_ok(row.get('support')) and row['close'] <= row['support']*1.003) if direction=='UP' else (_ok(row.get('resistance')) and row['close'] >= row['resistance']*0.997)
    add('support_resistance', sr, 'SR_ALIGNMENT')
    if flags.get('rsi',True):
        mom = (_ok(row.get('rsi')) and row['rsi']<45) if direction=='UP' else (_ok(row.get('rsi')) and row['rsi']>55)
        add('rsi_stoch', mom, 'MOMENTUM_FILTER')
    if flags.get('tick_velocity',True): add('tick_acceleration', _ok(row.get('tick_acceleration')) and abs(row['tick_acceleration'])>profile.tick_velocity_threshold, 'TICK_ACCELERATION')
    # MTF inputs are optional; absence earns no points rather than fabricated confirmation.
    add('m5_confirmation', bool(row.get('m5_confirm',False)), 'M5_CONFIRM')
    if flags.get('m15',True): add('m15_context', bool(row.get('m15_confirm',False)), 'M15_CONTEXT')
    return SignalDecision(profile.symbol,int(row['epoch']),direction,round(min(score,100.0),2),reasons,{k:(None if not _ok(row.get(k)) else float(row[k]) if hasattr(row.get(k),'item') or isinstance(row.get(k),(int,float)) else row.get(k)) for k in ['atr','atr_ratio','bb_ratio','rsi','stoch','tick_velocity','tick_acceleration']},float(row['close']),profile.test_window_bars)
