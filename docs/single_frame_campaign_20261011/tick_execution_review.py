"""Predeclared public-tick diagnostic for four fixed development policies.

No accounts, orders, proposals, final holdout, parameter search, or cost
promotion.  Raw wire pages are retained even when the provider returns errors.
"""
from __future__ import annotations

import argparse
import asyncio
import gzip
from hashlib import sha256
import json
import math
from pathlib import Path
import platform
import ssl
import time

import certifi
import numpy as np
import pandas as pd
import websockets

from research_campaign.data import ENDPOINT, load_m1
from research_campaign.engine import CostModel, SplitPlan, run_backtest
from research_campaign.source_audit import audit_catalogue, audit_source_row
from research_campaign.state import atomic_json, digest, encoded, guard, read_json, utc
from research_campaign.strategies import candidates_for_round, generate_signals, prepare_context

ROOT=Path(__file__).resolve().parents[2]
TARGETS=("BOOM500","CRASH500")
FAMILIES=("pre_spike","post_spike")
MAX_SIGNALS=50
MAX_REQUESTS=500
SPACING_SECONDS=2
PAGE_SECONDS=1000
SOURCE_FILES=("research_campaign/engine.py","research_campaign/strategies.py",
              "research_campaign/data.py","research_campaign/source_audit.py",
              "research_campaign/state.py",str(Path(__file__).relative_to(ROOT)))


def _pins():
    return {name:digest(ROOT/name) for name in SOURCE_FILES}


def _runtime():
    return {"python":platform.python_version(),"numpy":np.__version__,"pandas":pd.__version__,
            "websockets":websockets.__version__}


def select_indices(context, signals, spec, start, end, limit=MAX_SIGNALS):
    """Chronological raw signals; no future trade outcomes determine selection."""
    epochs=context.bars.epoch.to_numpy(dtype=np.int64)
    atr=context.arrays["atr"]
    selected=[]
    for i in np.flatnonzero(signals):
        i=int(i);j=i+1;last=j+spec.max_hold_bars
        if j>=len(epochs) or last>len(epochs):
            continue
        availability=int(epochs[i]+60)
        if availability<start or availability>=end or epochs[j]!=availability:
            continue
        if availability+spec.max_hold_bars*60>end or np.any(np.diff(epochs[j:last])!=60):
            continue
        if not math.isfinite(float(atr[i])) or atr[i]<=0:
            continue
        selected.append(i)
        if len(selected)==limit:
            break
    return selected


def merge_windows(windows):
    """Half-open windows, merged before fixed-width chronological pagination."""
    merged=[]
    for start,end in sorted(set((int(a),int(b)) for a,b in windows)):
        if end<=start:
            raise ValueError("Positive fixed tick window required")
        if merged and start<=merged[-1][1]:
            merged[-1][1]=max(merged[-1][1],end)
        else:
            merged.append([start,end])
    return merged


def request_plan(rows, symbols):
    requests=[]
    for symbol in TARGETS:
        windows=[(r["entry_epoch"],r["planned_end_epoch"]) for row in rows if row["symbol"]==symbol for r in row["signals"]]
        for start,end in merge_windows(windows):
            for left in range(start,end,PAGE_SECONDS):
                right=min(end,left+PAGE_SECONDS)
                request={"ticks_history":symbols[symbol],"style":"ticks","count":right-left,
                         "start":left,"end":right-1,"req_id":len(requests)+1}
                requests.append({"symbol":symbol,"request":request})
    if len(requests)>MAX_REQUESTS:
        raise ValueError("Declared tick request budget exceeded; never silently truncate")
    return requests


def validate_request(request, declared):
    if set(request)!={"ticks_history","style","count","start","end","req_id"}:
        raise ValueError("Only exact unauthenticated public historical-tick requests are allowed")
    if any(type(request[k]) is not int for k in ("count","start","end","req_id")):
        raise ValueError("Exact integer tick request parameters required")
    if request!=declared or request["style"]!="ticks" or not 1<=request["count"]<=PAGE_SECONDS:
        raise ValueError("Request differs from the frozen diagnostic pagination plan")
    if request["end"]-request["start"]+1!=request["count"]:
        raise ValueError("Tick request must cover exactly its declared second window")


def parse_page(response, request):
    if response.get("req_id")!=request["req_id"] or response.get("echo_req")!=request:
        raise ValueError("Original tick response identity differs from its exact request")
    if response.get("error"):
        return {"status":"API_ERROR","error":response["error"],"ticks":[],"missing_seconds":request["count"]}
    history=response.get("history",{})
    epochs=history.get("times",[]);raw_prices=history.get("prices",[])
    if len(epochs)!=len(raw_prices) or any(type(t) is not int for t in epochs):
        raise ValueError("Original tick arrays must have equal lengths and integer epochs")
    prices=[float(p) for p in raw_prices]
    if any(isinstance(p,bool) for p in raw_prices) or not all(math.isfinite(p) and p>0 for p in prices):
        raise ValueError("Original tick prices must be finite and positive")
    if len(set(epochs))!=len(epochs) or any(b<=a for a,b in zip(epochs,epochs[1:])):
        raise ValueError("Original tick page has duplicate/unordered seconds")
    if any(t<request["start"] or t>request["end"] for t in epochs):
        raise ValueError("Provider returned ticks outside the fixed requested window")
    expected=list(range(request["start"],request["end"]+1))
    return {"status":"EXACT_1S_WINDOW" if epochs==expected else "PARTIAL_WINDOW",
            "returned_ticks":len(epochs),"missing_seconds":len(set(expected)-set(epochs)),
            "historical_bid_ask_present":{"bid","ask"}.issubset(history),
            "ticks":[[t,p] for t,p in zip(epochs,prices)]}


def _verify(directory):
    declaration=read_json(directory/"protocol.json")
    if declaration["source_pins"]!=_pins() or declaration["runtime"]!=_runtime():
        raise ValueError("Frozen diagnostic source/runtime changed")
    if declaration["safety"]!=guard() or digest(Path(declaration["manifest_file"]))!=declaration["manifest_sha256"]:
        raise ValueError("Frozen diagnostic public source or safety changed")
    for row in declaration["datasets"].values():
        if digest(Path(row["requested_file"]))!=row["requested_sha256"]:
            raise ValueError("Frozen diagnostic market prices changed")
    selection=read_json(directory/"selection.json")
    if selection["protocol_sha256"]!=digest(directory/"protocol.json"):
        raise ValueError("Diagnostic selection no longer matches premeasurement protocol")
    if digest(directory/"selection.json")!=read_json(directory/"selection_seal.json")["sha256"]:
        raise ValueError("Premeasurement diagnostic signal selection changed")
    return declaration,selection


def _contexts(declaration):
    contexts={}
    for symbol,row in declaration["datasets"].items():
        split=SplitPlan.from_dict(row["split"])
        bars=load_m1(row["requested_file"])
        bars=bars.loc[bars.epoch+60<=split.validation_end].copy()
        contexts[symbol]=prepare_context(bars,"M1")
    return contexts


def prepare(manifest_file, directory):
    guard();directory=Path(directory).resolve();directory.mkdir(parents=True,exist_ok=False)
    manifest_file=Path(manifest_file).resolve();manifest=read_json(manifest_file)
    if manifest.get("source")!="Official Deriv public M1 candles" or manifest.get("endpoint")!=ENDPOINT:
        raise ValueError("Only original official Deriv historical candles are permitted")
    symbols=audit_catalogue(manifest["catalogue"])
    rows={}
    for symbol in TARGETS:
        row=manifest["symbols"][symbol]
        audit=audit_source_row(row,catalogue_provenance=manifest["catalogue"])
        split=SplitPlan.from_calendar(audit["effective_start_epoch"],audit["effective_end_exclusive_epoch"])
        rows[symbol]={"requested_file":row["requested_file"],"requested_sha256":row["requested_sha256"],
                      "api_symbol":symbols[symbol],"source_audit":audit,"split":split.to_dict()}
    specs=[s for s in candidates_for_round(0) if s.family in FAMILIES]
    protocol={"declared_at":utc(),"manifest_file":str(manifest_file),"manifest_sha256":digest(manifest_file),
              "source_pins":_pins(),"runtime":_runtime(),"datasets":rows,"specs":[s.to_dict() for s in specs],
              "frame":"M1","partition":"validation","selection":"first 50 chronological raw eligible signals per fixed policy, never trade outcomes",
              "eligibility":"native closed-bar signal, positive known ATR, exact next open, all planned four holding bars present, full horizon within validation",
              "maximum_signals_per_policy":MAX_SIGNALS,"maximum_requests":MAX_REQUESTS,"spacing_seconds":SPACING_SECONDS,
              "page_seconds":PAGE_SECONDS,"fixed_holding_seconds":240,"costs":CostModel().to_dict(),
              "tick_rule":"exact one quote per second across full planned window and exact entry-open match; unknown is never filled or imputed",
              "stop_target_rule":"unchanged frozen engine, fixed SL/TP, next open, uncapped observed tick gaps, one open position",
              "source_authenticity":"original M1 wire reconstruction and independently resolved catalogue audited before freezing",
              "final_holdout_accessed":False,"qualification_allowed":False,"cost_promotion_allowed":False,
              "selection_bias":"bounded first-signals diagnostic, not representative profitability or final OOS evidence",
              "safety":guard(),"authentication_accounts_proposals_orders_used":False}
    atomic_json(directory/"protocol.json",protocol,exclusive=True)
    contexts=_contexts(protocol);selected=[]
    for symbol in TARGETS:
        ctx=contexts[symbol];split=SplitPlan.from_dict(rows[symbol]["split"])
        for spec in specs:
            signals=generate_signals(ctx,spec,symbol)
            indices=select_indices(ctx,signals,spec,split.train_end,split.validation_end)
            selected.append({"symbol":symbol,"spec":spec.to_dict(),"signals":[
                {"signal_index":i,"signal_epoch":int(ctx.bars.epoch.iloc[i]+60),
                 "entry_epoch":int(ctx.bars.epoch.iloc[i+1]),"planned_end_epoch":int(ctx.bars.epoch.iloc[i+1]+240)} for i in indices],
                "full_prefix_signal_sha256":sha256(signals.astype(np.int8).tobytes()).hexdigest()})
    selection={"selected_at":utc(),"protocol_sha256":digest(directory/"protocol.json"),"policies":selected,
               "requests":request_plan(selected,symbols),"outcomes_measured":False,"safety":guard()}
    atomic_json(directory/"selection.json",selection,exclusive=True)
    atomic_json(directory/"selection_seal.json",{"sealed_at":utc(),"sha256":digest(directory/"selection.json"),"outcomes_measured":False},exclusive=True)
    text=("# بروتوكول تحقق تنفيذ تاريخي بالتكات\n\n"
          "أربع سياسات ثابتة: Boom 500 وCrash 500، على M1، فرضيتا ما قبل الانفجار وما بعده بإعدادات الجولة صفر. "
          "نختار أول 50 إشارة مؤهلة زمنيًا لكل سياسة قبل حساب نتائج الصفقات. الإشارات الخام تشمل ما قد ترفضه قاعدة صفقة واحدة لاحقًا.\n\n"
          "التقسيم نفسه 60/20/20، والتشخيص على جزء التحقق فقط؛ الاختبار النهائي مغلق. نطلب نافذة الاحتفاظ الكاملة، أربع دقائق، لكل إشارة. "
          "التغطية المطلوبة تكة أصلية لكل ثانية وتطابق سعر الدخول مع افتتاح الشمعة. نحتفظ بالأخطاء والفجوات والنتائج المجهولة.\n\n"
          "الطلبات محددة قبل القياس ومجزأة إلى 1000 ثانية كحد أقصى، وبفاصل ثانيتين على الأقل؛ الحد الأعلى 500 طلب، والتوقف عند خطأ API دون إعادة محاولة تلقائية. "
          "يحفظ كل رد أصلي وطلب وإيصال.\n\n"
          "المقارنة تشخيص لاختيار التنفيذ، وليست اختبار ربحية مؤهلًا. التكلفة 0.10 ATR مفترضة دون إثبات CFD أو bid/ask أو تنفيذ حساب. "
          "لا تغيير لإعدادات الاستراتيجية، ولا ترقيتها، ولا أمر تداول، وجميع بوابات التداول الحقيقي false.\n")
    (directory/"PROTOCOL.ar.md").write_text(text)
    return {"protocol_sha256":digest(directory/"protocol.json"),"selection_sha256":digest(directory/"selection.json"),
            "policies":len(selected),"selected_signals":sum(len(r["signals"]) for r in selected),
            "planned_requests":len(selection["requests"]),"expected_requested_seconds":sum(v["request"]["count"] for v in selection["requests"])}


async def collect(directory):
    directory=Path(directory).resolve();declaration,selection=_verify(directory)
    rawdir=directory/"raw";rawdir.mkdir(exist_ok=True)
    context=ssl.create_default_context(cafile=certifi.where())
    reports=[];last_sent=0.0;errors=[]
    async with websockets.connect(ENDPOINT,ssl=context,open_timeout=30,ping_interval=20,ping_timeout=20,max_size=2_000_000) as connection:
        for page in selection["requests"]:
            request=page["request"];number=request["req_id"];base=rawdir/f"page_{number:04d}"
            validate_request(request,page["request"])
            if base.with_suffix(".receipt.json").exists():
                receipt=read_json(base.with_suffix(".receipt.json"))
                if digest(base.with_suffix(".wire"))!=receipt["response_sha256"] or receipt["request"]!=request:
                    raise ValueError("Preserved tick receipt/raw wire changed")
                summary=parse_page(json.loads(base.with_suffix(".wire").read_bytes()),request)
                reports.append({"number":number,"symbol":page["symbol"],**{k:v for k,v in summary.items() if k!="ticks"}})
                if summary["status"]=="API_ERROR":break
                continue
            if base.with_suffix(".sent.json").exists():
                errors.append({"number":number,"status":"UNKNOWN_PRIOR_SEND_NO_RESPONSE","reissued":False});break
            await asyncio.sleep(max(0,SPACING_SECONDS-(time.monotonic()-last_sent)))
            atomic_json(base.with_suffix(".intent.json"),{"request":request,"canonical_symbol":page["symbol"],
                "endpoint":ENDPOINT,"issued_before_send_at":utc(),"authentication_used":False,"safety":guard()},exclusive=True)
            try:
                await connection.send(json.dumps(request));last_sent=time.monotonic()
                atomic_json(base.with_suffix(".sent.json"),{"request":request,"sent_at":utc(),"safety":guard()},exclusive=True)
                wire=await asyncio.wait_for(connection.recv(),timeout=40)
                raw=wire.encode() if isinstance(wire,str) else wire
                with base.with_suffix(".wire").open("xb") as stream:stream.write(raw)
                receipt={"request":request,"canonical_symbol":page["symbol"],"endpoint":ENDPOINT,
                         "received_at":utc(),"response_sha256":sha256(raw).hexdigest(),"authentication_used":False,"safety":guard()}
                atomic_json(base.with_suffix(".receipt.json"),receipt,exclusive=True)
                summary=parse_page(json.loads(raw),request)
                atomic_json(base.with_suffix(".summary.json"),{k:v for k,v in summary.items() if k!="ticks"},exclusive=True)
                reports.append({"number":number,"symbol":page["symbol"],**{k:v for k,v in summary.items() if k!="ticks"}})
                print(json.dumps({"received_page":number,"planned_pages":len(selection["requests"]),"status":summary["status"],"ticks":summary.get("returned_ticks",0)}),flush=True)
                if summary["status"]=="API_ERROR":break
            except Exception as error:
                errors.append({"number":number,"error":f"{type(error).__name__}: {error}","retry":False});break
    result={"completed_at":utc(),"protocol_sha256":digest(directory/"protocol.json"),"selection_sha256":digest(directory/"selection.json"),
            "planned_requests":len(selection["requests"]),"actual_requests_sent":len(list(rawdir.glob("*.sent.json"))),
            "actual_responses_saved":len(list(rawdir.glob("*.wire"))),"reports":reports,"errors":errors,"safety":guard()}
    atomic_json(directory/"collection_result.json",result)
    return result


def _observed_ticks(directory,selection):
    values={s:[] for s in TARGETS};page_errors=[]
    for page in selection["requests"]:
        request=page["request"];base=directory/"raw"/f"page_{request['req_id']:04d}"
        if not base.with_suffix(".receipt.json").exists():
            page_errors.append({"number":request["req_id"],"status":"NOT_COLLECTED"});continue
        receipt=read_json(base.with_suffix(".receipt.json"))
        if receipt["request"]!=request or digest(base.with_suffix(".wire"))!=receipt["response_sha256"]:
            raise ValueError("Original tick page identity/hash changed")
        try:
            parsed=parse_page(json.loads(base.with_suffix(".wire").read_bytes()),request)
        except ValueError as error:
            page_errors.append({"number":request["req_id"],"status":"INVALID_PAGE","error":str(error)});continue
        if parsed["status"]=="API_ERROR":
            page_errors.append({"number":request["req_id"],"status":"API_ERROR","error":parsed["error"]});continue
        values[page["symbol"]].extend(parsed["ticks"])
    frames={}
    for symbol,quotes in values.items():
        frame=pd.DataFrame(quotes,columns=["epoch","quote"])
        if len(frame) and (frame.epoch.duplicated().any() or not frame.epoch.is_monotonic_increasing):
            raise ValueError("Fixed nonoverlapping pages produced duplicate or unordered tick seconds")
        frames[symbol]=frame
    return frames,page_errors


def evaluate(directory):
    directory=Path(directory).resolve();declaration,selection=_verify(directory)
    contexts=_contexts(declaration);ticks,page_errors=_observed_ticks(directory,selection)
    policies=[]
    for selected in selection["policies"]:
        symbol=selected["symbol"];ctx=contexts[symbol]
        spec=next(s for s in candidates_for_round(0) if s.to_dict()==selected["spec"])
        original=generate_signals(ctx,spec,symbol)
        if sha256(original.astype(np.int8).tobytes()).hexdigest()!=selected["full_prefix_signal_sha256"]:
            raise ValueError("Sealed diagnostic signals changed")
        split=SplitPlan.from_dict(declaration["datasets"][symbol]["split"])
        indices=select_indices(ctx,original,spec,split.train_end,split.validation_end)
        if indices!=[v["signal_index"] for v in selected["signals"]]:
            raise ValueError("Sealed first chronological signal selection changed")
        limited=np.zeros(len(original),dtype=np.int8);limited[indices]=original[indices]
        common={"partition":"validation","costs":CostModel.from_dict(declaration["costs"]),
                "data_quality_ok":declaration["datasets"][symbol]["source_audit"]["internal_missing_minutes"]==0,
                "provenance":{"real_deriv_data":True,"diagnostic":True,"final_holdout_accessed":False,
                              "source_sha256":declaration["datasets"][symbol]["requested_sha256"]}}
        args=(ctx,limited,spec,symbol,"M1",split.train_end,split.validation_end)
        candle=run_backtest(*args,**common);tick=run_backtest(*args,ticks=ticks[symbol],**common)
        for measured in (candle,tick):
            measured["gates"]["diagnostic_cannot_qualify"]=False;measured["qualified"]=False
            measured["rejection_reasons"]=[k for k,v in measured["gates"].items() if not v]
        c={t["signal_epoch"]:t for t in candle["trades"]};t={t["signal_epoch"]:t for t in tick["trades"]}
        paired=[{"signal_epoch":epoch,"candle_exit_price":c[epoch]["exit_price"],"tick_exit_price":t[epoch]["exit_price"],
                 "candle_net_R":c[epoch]["net_R"],"tick_net_R":t[epoch]["net_R"],"delta_net_R":t[epoch]["net_R"]-c[epoch]["net_R"],
                 "candle_exit_reason":c[epoch]["exit_reason"],"tick_exit_reason":t[epoch]["exit_reason"]} for epoch in sorted(set(c)&set(t))]
        row={"symbol":symbol,"family":spec.family,"selected_raw_signals":len(indices),"candle":candle,"tick":tick,
             "paired":paired,"paired_completed":len(paired),
             "paired_mean_delta_net_R":float(np.mean([p["delta_net_R"] for p in paired])) if paired else None,
             "paired_exit_price_changed":sum(not math.isclose(p["candle_exit_price"],p["tick_exit_price"],rel_tol=0,abs_tol=1e-12) for p in paired),
             "paired_exit_reason_changed":sum(p["candle_exit_reason"]!=p["tick_exit_reason"] for p in paired)}
        artifact=directory/f"{symbol}_{spec.family}_replay.json.gz"
        with artifact.open("xb") as stream:stream.write(gzip.compress(encoded(row),mtime=0))
        policies.append({k:v for k,v in row.items() if k not in {"candle","tick","paired"}}|{
            "candle_metrics":candle["metrics"],"tick_metrics":tick["metrics"],
            "candle_ambiguous":candle["ambiguous_candles"],"tick_unknown_paths":tick["unknown_paths"],
            "tick_unknown_dispositions":[v for v in tick["dispositions"] if v["status"].startswith("UNKNOWN")],
            "artifact_file":str(artifact),"artifact_sha256":digest(artifact)})
    _verify(directory)
    result={"completed_at":utc(),"protocol_sha256":digest(directory/"protocol.json"),"selection_sha256":digest(directory/"selection.json"),
            "collection_result_sha256":digest(directory/"collection_result.json"),"policies":policies,"page_errors":page_errors,
            "actual_original_ticks":{s:len(f) for s,f in ticks.items()},"qualification_allowed":False,"qualified_policies":0,
            "costs_verified":False,"actual_cfd_fills_verified":False,"final_holdout_accessed":False,"safety":guard()}
    atomic_json(directory/"result.json",result,exclusive=True)
    lines=["# تشخيص تنفيذ الاستراتيجيات بالتكات الأصلية", "", "هذه مقارنة تنفيذ محدودة على بيانات التحقق؛ ليست استراتيجية مؤهلة أو دليل ربح CFD. اختيرت الإشارات قبل القياس، دون لمس الاختبار النهائي.", "",
           "| المؤشر | السياسة | إشارات خام | مكتملة بالشموع | مكتملة بالتكات | مجهولة بالتكات | متوسط R شموع | متوسط R تكات | متوسط فرق R للأزواج |", "|---|---|---:|---:|---:|---:|---:|---:|---:|"]
    def fmt(value):return "NOT TESTED" if value is None else f"{value:.6f}"
    for p in policies:
        lines.append(f"| {p['symbol']} | {p['family']} | {p['selected_raw_signals']} | {p['candle_metrics']['completed_trades']} | {p['tick_metrics']['completed_trades']} | {p['tick_unknown_paths']} | {fmt(p['candle_metrics']['mean_net_R'])} | {fmt(p['tick_metrics']['mean_net_R'])} | {fmt(p['paired_mean_delta_net_R'])} |")
    lines += ["",f"التكات الأصلية المحفوظة: {result['actual_original_ticks']}. جميع الصفقات الافتراضية ومسارات رأس المال والتصرفات محفوظة في ملفات replay المضغوطة؛ الردود الأصلية في raw.","",
              "لا تقارن المتوسطين كعينة واحدة عند اختلاف عدد الصفقات؛ فرق الأزواج يحتسب فقط الإشارات المكتملة بالطريقتين. قاعدة صفقة واحدة والفجوات قد تغير الإشارات المنفذة لاحقًا.","",
              "يختبر التشخيص تسلسل أسعار المؤشر العام، ولا يثبت أسعار bid/ask أو ملء CFD أو الانزلاق والسبريد التاريخيين. التكلفة 0.10 ATR فرضية معلنة. التوقف المتحرك ممنوع. لا ترقيات أو بحث إعدادات هنا، والعينة الأولى صغيرة وغير ممثلة لكل التاريخ.","",
              "LIVE_TRADING=false، READY_FOR_LIVE=false، LIVE_ALLOWED=false، OPENED_TRADES=false. لا أوامر حقيقية. الاختبار النهائي بقي مغلقًا."]
    (directory/"REPORT.ar.md").write_text("\n".join(lines)+"\n")
    return result


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command",choices=("prepare","collect","evaluate"))
    parser.add_argument("--output-dir",type=Path,required=True)
    parser.add_argument("--manifest",type=Path)
    args=parser.parse_args()
    if args.command=="prepare":
        if args.manifest is None:raise ValueError("Original initial manifest required")
        result=prepare(args.manifest,args.output_dir)
    elif args.command=="collect":result=asyncio.run(collect(args.output_dir))
    else:result=evaluate(args.output_dir)
    print(json.dumps(result,sort_keys=True,allow_nan=False,indent=2))


if __name__=="__main__":main()
