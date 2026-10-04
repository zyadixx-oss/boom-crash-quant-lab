import asyncio, json, math, os, sys, time
from datetime import timezone
import numpy as np
import pandas as pd
import websockets

WS_URL = "wss://ws.derivws.com/websockets/v3?app_id=1089"
SYMBOLS = {"BOOM1000": "boom", "CRASH1000": "crash"}
TARGET_MINUTES = int(os.getenv("CRT_TARGET_MINUTES", "30000"))
PAGE = 5000
PRIMARY_ATR_MULT = 2.0
HORIZON_MIN = 15

async def request(ws, payload):
    await ws.send(json.dumps(payload))
    while True:
        msg = json.loads(await asyncio.wait_for(ws.recv(), timeout=30))
        if msg.get("error"):
            raise RuntimeError(msg["error"])
        if msg.get("msg_type") in ("candles", "history"):
            return msg

async def fetch_m1(symbol, target=TARGET_MINUTES):
    rows = []
    end = "latest"
    async with websockets.connect(WS_URL, ping_interval=20, ping_timeout=20, max_size=8_000_000) as ws:
        while len(rows) < target:
            count = min(PAGE, target - len(rows))
            payload = {
                "ticks_history": symbol,
                "style": "candles",
                "granularity": 60,
                "count": count,
                "end": end,
                "adjust_start_time": 1,
            }
            msg = await request(ws, payload)
            candles = msg.get("candles", [])
            if not candles:
                break
            rows.extend(candles)
            oldest = min(int(c["epoch"]) for c in candles)
            end = oldest - 1
            if len(candles) < count:
                break
            await asyncio.sleep(0.25)
    df = pd.DataFrame(rows)
    if df.empty:
        raise RuntimeError(f"No candles returned for {symbol}")
    for c in ["open","high","low","close"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df["epoch"] = pd.to_numeric(df["epoch"], errors="coerce").astype("int64")
    df = df.dropna().drop_duplicates("epoch").sort_values("epoch")
    df["time"] = pd.to_datetime(df["epoch"], unit="s", utc=True)
    df = df.set_index("time")[["open","high","low","close"]]
    return df

def atr(df, n=14):
    prev = df["close"].shift(1)
    tr = pd.concat([
        df["high"]-df["low"],
        (df["high"]-prev).abs(),
        (df["low"]-prev).abs()
    ], axis=1).max(axis=1)
    return tr.rolling(n, min_periods=n).mean()

def resample_ohlc(df, rule):
    return df.resample(rule, label="left", closed="left").agg(
        open=("open","first"), high=("high","max"), low=("low","min"), close=("close","last")
    ).dropna()

def label_event(m1, signal_time, entry, direction, atr_mult):
    pos = m1.index.searchsorted(signal_time)
    if pos <= 14 or pos >= len(m1):
        return None
    a = float(m1["atr14"].iloc[pos-1])
    if not np.isfinite(a) or a <= 0:
        return None
    end_time = signal_time + pd.Timedelta(minutes=HORIZON_MIN)
    fut = m1.iloc[pos:m1.index.searchsorted(end_time)]
    if fut.empty:
        return None
    if direction == "boom":
        excursion = float(fut["high"].max() - entry)
    else:
        excursion = float(entry - fut["low"].min())
    hit = excursion >= atr_mult*a
    # earliest time to threshold
    tts = None
    threshold = atr_mult*a
    if direction == "boom":
        mask = fut["high"] - entry >= threshold
    else:
        mask = entry - fut["low"] >= threshold
    if bool(mask.any()):
        first = fut.index[np.flatnonzero(mask.to_numpy())[0]]
        tts = (first - signal_time).total_seconds()/60.0
    return bool(hit), excursion/a, tts

def find_candidates(m1_raw, direction):
    m1 = m1_raw.copy()
    m1["atr14"] = atr(m1,14)
    m5 = resample_ohlc(m1, "5min")
    m5["atr14"] = atr(m5,14)
    m5["body"] = (m5["close"]-m5["open"]).abs()
    m5["body_med20"] = m5["body"].rolling(20, min_periods=20).median()
    h1 = resample_ohlc(m1, "1h")
    h1map = h1.to_dict("index")

    variants = {"CRT": [], "CRT_MSS": [], "CRT_MSS_DISP": [], "CRT_MSS_DISP_FVG": []}
    used_ref = {k:set() for k in variants}

    for i in range(25, len(m5)-5):
        t = m5.index[i]
        ref_t = t.floor("h") - pd.Timedelta(hours=1)
        if ref_t not in h1map:
            continue
        ref = h1map[ref_t]
        a = float(m5["atr14"].iloc[i])
        if not np.isfinite(a) or a <= 0:
            continue
        if direction == "boom":
            depth = ref["low"] - m5["low"].iloc[i]
        else:
            depth = m5["high"].iloc[i] - ref["high"]
        if depth < 0.10*a or depth > 0.65*a:
            continue

        # reclaim within current + next 2 completed M5 candles
        reclaim = None
        for j in range(i, min(i+3, len(m5))):
            ok = (m5["close"].iloc[j] > ref["low"]) if direction=="boom" else (m5["close"].iloc[j] < ref["high"])
            if ok:
                reclaim = j
                break
        if reclaim is None:
            continue

        def add(name, idx):
            if ref_t in used_ref[name]:
                return
            used_ref[name].add(ref_t)
            variants[name].append({
                "time": m5.index[idx] + pd.Timedelta(minutes=5),
                "entry": float(m5["close"].iloc[idx]),
                "ref_hour": ref_t,
                "sweep_depth_atr": float(depth/a),
            })

        add("CRT", reclaim)

        mss_idx = None
        for k in range(reclaim, min(reclaim+4, len(m5))):
            if k < 5:
                continue
            if direction == "boom":
                level = float(m5["high"].iloc[k-5:k].max())
                ok = float(m5["close"].iloc[k]) > level
            else:
                level = float(m5["low"].iloc[k-5:k].min())
                ok = float(m5["close"].iloc[k]) < level
            if ok:
                mss_idx = k
                break
        if mss_idx is None:
            continue
        add("CRT_MSS", mss_idx)

        med = float(m5["body_med20"].iloc[mss_idx])
        body = float(m5["body"].iloc[mss_idx])
        if not np.isfinite(med) or med <= 0 or body < 1.30*med:
            continue
        add("CRT_MSS_DISP", mss_idx)

        if mss_idx < 2:
            continue
        if direction == "boom":
            fvg = float(m5["low"].iloc[mss_idx]) > float(m5["high"].iloc[mss_idx-2])
        else:
            fvg = float(m5["high"].iloc[mss_idx]) < float(m5["low"].iloc[mss_idx-2])
        if fvg:
            add("CRT_MSS_DISP_FVG", mss_idx)

    return m1, m5, h1, variants

def eval_signals(m1, signals, direction, split_time, atr_mult):
    out=[]
    for s in signals:
        lab=label_event(m1,s["time"],s["entry"],direction,atr_mult)
        if lab is None:
            continue
        hit, excursion_atr, tts = lab
        out.append({**s,"hit":hit,"excursion_atr":excursion_atr,"tts":tts,
                    "split":"train" if s["time"] < split_time else "test"})
    return out

def base_rate(m1, m5, direction, split_time, atr_mult):
    vals={"train":[],"test":[]}
    # Evaluate every M5 close as a naive timing baseline.
    for i in range(5, len(m5)-4):
        st=m5.index[i]+pd.Timedelta(minutes=5)
        entry=float(m5["close"].iloc[i])
        lab=label_event(m1,st,entry,direction,atr_mult)
        if lab is None: continue
        sp="train" if st<split_time else "test"
        vals[sp].append(bool(lab[0]))
    return {k:(sum(v)/len(v) if v else None) for k,v in vals.items()}

def wilson(k,n,z=1.96):
    if n==0:return [None,None]
    p=k/n
    d=1+z*z/n
    c=(p+z*z/(2*n))/d
    h=z*math.sqrt((p*(1-p)+z*z/(4*n))/n)/d
    return [max(0,c-h),min(1,c+h)]

def summarize(rows, base):
    n=len(rows); k=sum(1 for r in rows if r["hit"])
    p=k/n if n else None
    tts=[r["tts"] for r in rows if r["tts"] is not None]
    ex=[r["excursion_atr"] for r in rows]
    return {
        "signals":n,"hits":k,"precision":p,"base_rate":base,
        "lift":(p/base if p is not None and base not in (None,0) else None),
        "precision_ci95":wilson(k,n),
        "median_time_to_spike_min":float(np.median(tts)) if tts else None,
        "median_max_excursion_atr":float(np.median(ex)) if ex else None,
        "sample_status":"OK" if n>=30 else "INSUFFICIENT_SAMPLE"
    }

async def main():
    result={"experiment":{
        "name":"CRT core falsification test",
        "data_source":"Deriv public WebSocket historical M1 candles",
        "target_minutes_per_symbol":TARGET_MINUTES,
        "primary_label":f"directional excursion >= {PRIMARY_ATR_MULT} x M1 ATR(14) within {HORIZON_MIN} minutes",
        "rules":{"range":"previous fully closed H1","sweep":"0.10-0.65 x M5 ATR14","reclaim":"0-2 M5 bars",
                 "MSS":"close breaks previous 5 M5 bars","displacement":"body >=1.30 x median body20","FVG":"3-candle gap"},
        "note":"Signal uses only data known by confirmation close; future prices are used only for outcome labeling."
    },"symbols":{}}
    for symbol,direction in SYMBOLS.items():
        print(f"FETCH {symbol}", flush=True)
        raw=await fetch_m1(symbol)
        print(f"{symbol}: {len(raw)} M1 candles {raw.index.min()} -> {raw.index.max()}", flush=True)
        m1,m5,h1,variants=find_candidates(raw,direction)
        split_time=m1.index.min()+(m1.index.max()-m1.index.min())*0.70
        symres={"rows_m1":len(m1),"from":str(m1.index.min()),"to":str(m1.index.max()),"split_time":str(split_time),"thresholds":{}}
        for mult in [1.5,2.0,3.0]:
            base=base_rate(m1,m5,direction,split_time,mult)
            tres={"base_rate":base,"variants":{}}
            for name,sigs in variants.items():
                rows=eval_signals(m1,sigs,direction,split_time,mult)
                tres["variants"][name]={}
                for sp in ["train","test"]:
                    rr=[r for r in rows if r["split"]==sp]
                    tres["variants"][name][sp]=summarize(rr,base[sp])
            symres["thresholds"][str(mult)]=tres
        result["symbols"][symbol]=symres
    with open("crt_results.json","w",encoding="utf-8") as f:
        json.dump(result,f,indent=2,default=str)
    print("===CRT_RESULTS_BEGIN===")
    print(json.dumps(result,indent=2,default=str))
    print("===CRT_RESULTS_END===")

if __name__=="__main__":
    asyncio.run(main())
