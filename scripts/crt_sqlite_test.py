import json, math, sqlite3, sys
from pathlib import Path
import numpy as np
import pandas as pd

DB = Path(sys.argv[1] if len(sys.argv) > 1 else "external/tradlab/market_data.db")
HORIZON_BARS = 3
PRIMARY_MULT = 2.0

def atr(df, n=14):
    prev=df["close"].shift(1)
    tr=pd.concat([
        df["high"]-df["low"],
        (df["high"]-prev).abs(),
        (df["low"]-prev).abs(),
    ],axis=1).max(axis=1)
    return tr.rolling(n,min_periods=n).mean()

def load_bars(conn,symbol):
    q="""SELECT time,open,high,low,close
         FROM bars WHERE symbol=? AND timeframe=5
         ORDER BY time ASC"""
    df=pd.read_sql_query(q,conn,params=(symbol,))
    if df.empty:
        return df
    for c in ["open","high","low","close"]:
        df[c]=pd.to_numeric(df[c],errors="coerce")
    df=df.dropna().drop_duplicates("time").sort_values("time")
    df["dt"]=pd.to_datetime(df["time"],unit="s",utc=True)
    df=df.set_index("dt")[["open","high","low","close"]]
    # Keep only true 5-minute grid rows; duplicates removed above.
    df["atr14"]=atr(df,14)
    df["body"]=(df["close"]-df["open"]).abs()
    df["body_med20"]=df["body"].rolling(20,min_periods=20).median()
    return df

def h1_ranges(m5):
    return m5.resample("1h",label="left",closed="left").agg(
        open=("open","first"),high=("high","max"),low=("low","min"),close=("close","last"),
        count=("close","count")
    )

def build_candidates(m5,direction):
    h1=h1_ranges(m5)
    hmap=h1.to_dict("index")
    names=["CRT","CRT_MSS","CRT_MSS_DISP","CRT_MSS_DISP_FVG"]
    out={n:[] for n in names}
    used={n:set() for n in names}

    def add(name,idx,ref_t,depth_atr):
        if ref_t in used[name]:
            return
        used[name].add(ref_t)
        out[name].append({
            "idx":int(idx),
            "time":m5.index[idx]+pd.Timedelta(minutes=5),
            "entry":float(m5["close"].iloc[idx]),
            "atr":float(m5["atr14"].iloc[idx]),
            "ref_hour":ref_t,
            "sweep_depth_atr":float(depth_atr),
        })

    for i in range(25,len(m5)-HORIZON_BARS-5):
        t=m5.index[i]
        ref_t=t.floor("h")-pd.Timedelta(hours=1)
        ref=hmap.get(ref_t)
        if ref is None or int(ref.get("count",0)) < 10:
            continue
        a=float(m5["atr14"].iloc[i])
        if not np.isfinite(a) or a<=0:
            continue
        if direction=="boom":
            depth=float(ref["low"]-m5["low"].iloc[i])
        else:
            depth=float(m5["high"].iloc[i]-ref["high"])
        if depth < 0.10*a or depth > 0.65*a:
            continue

        reclaim=None
        for j in range(i,min(i+3,len(m5))):
            if direction=="boom":
                ok=float(m5["close"].iloc[j]) > float(ref["low"])
            else:
                ok=float(m5["close"].iloc[j]) < float(ref["high"])
            if ok:
                reclaim=j
                break
        if reclaim is None:
            continue

        add("CRT",reclaim,ref_t,depth/a)

        mss=None
        for k in range(reclaim,min(reclaim+4,len(m5))):
            if k<5:
                continue
            if direction=="boom":
                level=float(m5["high"].iloc[k-5:k].max())
                ok=float(m5["close"].iloc[k])>level
            else:
                level=float(m5["low"].iloc[k-5:k].min())
                ok=float(m5["close"].iloc[k])<level
            if ok:
                mss=k
                break
        if mss is None:
            continue
        add("CRT_MSS",mss,ref_t,depth/a)

        med=float(m5["body_med20"].iloc[mss])
        body=float(m5["body"].iloc[mss])
        if not np.isfinite(med) or med<=0 or body<1.30*med:
            continue
        add("CRT_MSS_DISP",mss,ref_t,depth/a)

        if mss<2:
            continue
        if direction=="boom":
            fvg=float(m5["low"].iloc[mss])>float(m5["high"].iloc[mss-2])
        else:
            fvg=float(m5["high"].iloc[mss])<float(m5["low"].iloc[mss-2])
        if fvg:
            add("CRT_MSS_DISP_FVG",mss,ref_t,depth/a)
    return out

def event_at(m5,idx,direction,mult):
    if idx<14 or idx+HORIZON_BARS>=len(m5):
        return None
    a=float(m5["atr14"].iloc[idx])
    if not np.isfinite(a) or a<=0:
        return None
    entry=float(m5["close"].iloc[idx])
    fut=m5.iloc[idx+1:idx+1+HORIZON_BARS]
    if len(fut)<HORIZON_BARS:
        return None
    if direction=="boom":
        excursions=(fut["high"]-entry)/a
    else:
        excursions=(entry-fut["low"])/a
    hit=bool((excursions>=mult).any())
    tts=None
    if hit:
        first=int(np.flatnonzero((excursions>=mult).to_numpy())[0])+1
        tts=float(first*5)
    return hit,float(excursions.max()),tts

def wilson(k,n,z=1.96):
    if n==0:return [None,None]
    p=k/n; d=1+z*z/n
    c=(p+z*z/(2*n))/d
    h=z*math.sqrt((p*(1-p)+z*z/(4*n))/n)/d
    return [max(0,c-h),min(1,c+h)]

def base_rate(m5,direction,mult,start_idx,end_idx):
    vals=[]
    for i in range(max(20,start_idx),min(end_idx,len(m5)-HORIZON_BARS)):
        ev=event_at(m5,i,direction,mult)
        if ev is not None: vals.append(ev[0])
    return float(np.mean(vals)) if vals else None, len(vals)

def summarize(m5,signals,direction,mult,start_idx,end_idx,base):
    rows=[]
    for s in signals:
        i=s["idx"]
        if not (start_idx<=i<end_idx):
            continue
        ev=event_at(m5,i,direction,mult)
        if ev is None: continue
        hit,ex,tts=ev
        rows.append((hit,ex,tts))
    n=len(rows); k=sum(int(r[0]) for r in rows)
    p=k/n if n else None
    tts=[r[2] for r in rows if r[2] is not None]
    ex=[r[1] for r in rows]
    return {
        "signals":n,"hits":k,"precision":p,
        "base_rate":base,
        "lift":(p/base if p is not None and base not in (None,0) else None),
        "precision_ci95":wilson(k,n),
        "median_time_to_event_min":float(np.median(tts)) if tts else None,
        "median_max_excursion_atr":float(np.median(ex)) if ex else None,
        "sample_status":"OK" if n>=30 else "INSUFFICIENT_SAMPLE"
    }

def main():
    if not DB.exists():
        raise SystemExit(f"DB not found: {DB}")
    con=sqlite3.connect(f"file:{DB}?mode=ro",uri=True)
    inv=pd.read_sql_query("""
        SELECT symbol,timeframe,COUNT(*) AS n,MIN(time) AS first_time,MAX(time) AS last_time
        FROM bars
        WHERE lower(symbol) LIKE 'boom %' OR lower(symbol) LIKE 'crash %'
        GROUP BY symbol,timeframe
        ORDER BY symbol,timeframe
    """,con)
    print("=== INVENTORY ===")
    print(inv.to_string(index=False))

    syms=pd.read_sql_query("""
        SELECT symbol,COUNT(*) AS n
        FROM bars
        WHERE timeframe=5 AND (lower(symbol) LIKE 'boom %' OR lower(symbol) LIKE 'crash %')
        GROUP BY symbol HAVING COUNT(*) >= 500
        ORDER BY symbol
    """,con)
    result={
      "experiment":{
        "source_repo":"Graeza/tradlab public market_data.db",
        "source_file":"market_data.db",
        "timeframe":"M5",
        "range":"previous fully closed H1 derived from M5",
        "sweep":"0.10 to 0.65 x M5 ATR(14)",
        "reclaim":"same or next 2 M5 bars",
        "MSS":"close beyond previous 5 M5 highs/lows",
        "displacement":"body >= 1.30 x median body(20)",
        "FVG":"3-candle gap",
        "primary_outcome":"directional excursion >= 2.0 x signal-time M5 ATR(14) during next 3 M5 bars",
        "split":"chronological 70/30, no parameter fitting",
        "caution":"Public third-party historical DB; results verify this dataset, not Deriv's official server history."
      },
      "symbols":{}
    }
    for symbol in syms["symbol"].tolist():
        direction="boom" if symbol.lower().startswith("boom") else "crash"
        m5=load_bars(con,symbol)
        if len(m5)<500: continue
        cands=build_candidates(m5,direction)
        cut=int(len(m5)*0.70)
        sres={
          "bars":len(m5),
          "from":str(m5.index.min()),
          "to":str(m5.index.max()),
          "split_time":str(m5.index[cut]),
          "thresholds":{}
        }
        for mult in [1.5,2.0,3.0]:
            br_train,nbt=base_rate(m5,direction,mult,0,cut)
            br_test,nbx=base_rate(m5,direction,mult,cut,len(m5))
            t={"base_rate":{"train":br_train,"test":br_test},
               "baseline_samples":{"train":nbt,"test":nbx},
               "variants":{}}
            for name,sigs in cands.items():
                t["variants"][name]={
                  "train":summarize(m5,sigs,direction,mult,0,cut,br_train),
                  "test":summarize(m5,sigs,direction,mult,cut,len(m5),br_test)
                }
            sres["thresholds"][str(mult)]=t
        result["symbols"][symbol]=sres

    with open("crt_sqlite_results.json","w",encoding="utf-8") as f:
        json.dump(result,f,indent=2,default=str)
    print("===CRT_RESULTS_BEGIN===")
    print(json.dumps(result,indent=2,default=str))
    print("===CRT_RESULTS_END===")

if __name__=="__main__":
    main()
