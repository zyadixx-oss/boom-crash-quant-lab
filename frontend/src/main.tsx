import React,{useEffect,useMemo,useState} from 'react';
import {createRoot} from 'react-dom/client';
import {
  Activity,AlertTriangle,BarChart3,BrainCircuit,Database,FlaskConical,
  Gauge,Layers3,Radio,ShieldCheck,Trophy
} from 'lucide-react';
import './index.css';

type ArenaStats={
  total_strategies:number;qualified:number;rejected:number;oos_testing:number;
  forward_testing:number;active_agents:number;backtests_today:number;shadow_signals_today:number;
};
type Safety={LIVE_TRADING:boolean;ready_for_live:boolean;live_allowed:boolean;opened_trades:boolean;orders_opened:number;mode:string};
type Agent={id:number;name:string;description:string;capabilities:string[];enabled:boolean};
type Strategy={
  id:string;name:string;agent_name:string;symbol:string;version:string;parameters:Record<string,unknown>;
  rules:string[];indicators_used:string[];timeframes:string[];status:string;creation_time?:string;
};
type LeaderRow={
  strategy_id:string;strategy:string;agent:string;symbol:string;status:string;trades:number;net_profit:number;
  profit_factor:number;expectancy:number;max_drawdown:number;spike_precision:number;spike_recall:number;
  false_alert_rate:number;oos_score:number;forward_score:number;stability:number;composite_score:number;
};
type Profile={
  symbol:string;display_name:string;min_signal_score:number;atr_period:number;rsi_period:number;bb_period:number;
  bb_std_dev:number;bb_squeeze_threshold:number;support_resistance_lookback:number;false_signal_threshold:number;
  spike_atr_multiple:number;spike_window:number;pre_spike_window:number;timeframe_weights:Record<string,number>;
};

const API_BASE=(import.meta.env.VITE_API_BASE_URL||'/api').replace(/\/$/,'');
const WS_BASE=(import.meta.env.VITE_WS_BASE_URL||`${location.protocol==='https:'?'wss:':'ws:'}//${location.host}`).replace(/\/$/,'');
const apiUrl=(path:string)=>`${API_BASE}${path}`;
const wsUrl=(path:string)=>`${WS_BASE}${path}`;

const fmt=(v:unknown,d=2)=>{
  const n=Number(v);
  return Number.isFinite(n)?n.toFixed(d):'—';
};
const pct=(v:unknown)=>`${fmt(Number(v||0)*100,1)}%`;

function StatCard({label,value,icon:Icon}:{label:string;value:React.ReactNode;icon:React.ElementType}){
  return <div className="arena-card stat-card"><div><p className="eyebrow">{label}</p><div className="stat-value">{value}</div></div><Icon size={20}/></div>;
}

function Badge({value}:{value:string}){
  const cls=value==='qualified'?'badge good-bg':value==='rejected'||value==='retired'?'badge bad-bg':value.includes('testing')?'badge warn-bg':'badge';
  return <span className={cls}>{value.replaceAll('_',' ')}</span>;
}

function MetricGrid({data}:{data:Record<string,unknown>|undefined}){
  if(!data) return <p className="muted">No measured results for this stage yet.</p>;
  const keys=[
    'total_trades','winning_trades','losing_trades','win_rate','net_profit','gross_profit','gross_loss',
    'profit_factor','expectancy','max_drawdown','average_drawdown','average_win','average_loss','risk_reward',
    'consecutive_losses','sharpe_like','stability_score'
  ];
  return <div className="metric-grid">{keys.map(k=><div className="metric" key={k}><span>{k.replaceAll('_',' ')}</span><b>{k==='win_rate'||k==='stability_score'?pct(data[k]):fmt(data[k],3)}</b></div>)}</div>;
}

function SpikeGrid({data}:{data:Record<string,unknown>|undefined}){
  if(!data) return <p className="muted">No spike metrics recorded yet.</p>;
  const keys=['total_spikes','predicted_spikes','missed_spikes','false_spike_signals','spike_precision','spike_recall','false_alert_rate','average_lead_time_before_spike','median_lead_time_before_spike','MAE_before_spike','MFE_after_signal','spike_capture_score'];
  return <div className="metric-grid">{keys.map(k=><div className="metric" key={k}><span>{k.replaceAll('_',' ')}</span><b>{['spike_precision','spike_recall','false_alert_rate'].includes(k)?pct(data[k]):fmt(data[k],3)}</b></div>)}</div>;
}

function ForwardGrid({data}:{data:Record<string,unknown>|undefined}){
  if(!data) return <p className="muted">No forward/shadow results recorded yet.</p>;
  const keys=['total_signals','evaluated_signals','spike_hits','false_signals','spike_precision','false_alert_rate','average_lead_time','average_future_movement','forward_score'];
  return <div className="metric-grid">{keys.map(k=><div className="metric" key={k}><span>{k.replaceAll('_',' ')}</span><b>{['spike_precision','false_alert_rate'].includes(k)?pct(data[k]):fmt(data[k],3)}</b></div>)}</div>;
}

function RobustnessGrid({data}:{data:any}){
  const metrics=data?.metrics;
  if(!metrics) return <p className="muted">Robustness test has not run yet.</p>;
  return <div className="metric-grid">
    <div className="metric"><span>passed</span><b>{String(metrics.passed??false).toUpperCase()}</b></div>
    <div className="metric"><span>stability score</span><b>{pct(metrics.stability_score)}</b></div>
    <div className="metric"><span>sensitivity</span><b>{pct(metrics.sensitivity_pct)}</b></div>
    <div className="metric"><span>parameters tested</span><b>{metrics.parameters_tested?.length??0}</b></div>
  </div>;
}

function Curve({points,keyName,label}:{points:any[]|undefined;keyName:string;label:string}){
  if(!points?.length) return <div className="curve-empty">No {label.toLowerCase()} data yet.</div>;
  const values=points.map(p=>Number(p[keyName])).filter(Number.isFinite);
  if(!values.length) return <div className="curve-empty">No {label.toLowerCase()} data yet.</div>;
  const min=Math.min(...values),max=Math.max(...values),span=max-min||1;
  const coords=values.map((v,i)=>`${(i/(Math.max(1,values.length-1)))*100},${95-((v-min)/span)*85}`).join(' ');
  return <div><div className="curve-label">{label}</div><svg viewBox="0 0 100 100" preserveAspectRatio="none" className="curve"><polyline points={coords} fill="none" stroke="currentColor" strokeWidth="1.5"/></svg></div>;
}

function App(){
  const [stats,setStats]=useState<ArenaStats|null>(null);
  const [safety,setSafety]=useState<Safety|null>(null);
  const [agents,setAgents]=useState<Agent[]>([]);
  const [strategies,setStrategies]=useState<Strategy[]>([]);
  const [board,setBoard]=useState<LeaderRow[]>([]);
  const [formula,setFormula]=useState('');
  const [weights,setWeights]=useState<Record<string,number>>({});
  const [activity,setActivity]=useState<any[]>([]);
  const [profiles,setProfiles]=useState<Profile[]>([]);
  const [selectedId,setSelectedId]=useState<string>('');
  const [detail,setDetail]=useState<any>(null);
  const [symbol,setSymbol]=useState('ALL');
  const [agent,setAgent]=useState('ALL');
  const [status,setStatus]=useState('ALL');
  const [timeframe,setTimeframe]=useState('ALL');
  const [testType,setTestType]=useState<'validation'|'backtest'|'oos'>('validation');

  const load=async()=>{
    const [s,sa,a,st,lb,ac,p]=await Promise.all([
      fetch(apiUrl('/api/arena/stats')).then(r=>r.json()),
      fetch(apiUrl('/api/arena/safety')).then(r=>r.json()),
      fetch(apiUrl('/api/arena/agents')).then(r=>r.json()),
      fetch(apiUrl('/api/arena/strategies')).then(r=>r.json()),
      fetch(apiUrl('/api/arena/leaderboard')).then(r=>r.json()),
      fetch(apiUrl('/api/arena/activity?limit=50')).then(r=>r.json()),
      fetch(apiUrl('/api/arena/profiles')).then(r=>r.json()),
    ]);
    setStats(s);setSafety(sa);setAgents(a);setStrategies(st);setBoard(lb.rows||[]);setFormula(lb.formula||'');setWeights(lb.weights||{});setActivity(ac);setProfiles(p);
    if(!selectedId && st?.length) setSelectedId(st[0].id);
  };

  useEffect(()=>{
    load().catch(console.error);
    const ws=new WebSocket(wsUrl('/ws/arena'));
    ws.onmessage=e=>{
      try{
        const m=JSON.parse(e.data);
        if(m.stats)setStats(m.stats);
        if(m.safety)setSafety(m.safety);
        if(m.activity)setActivity(m.activity);
        if(m.leaderboard)setBoard(m.leaderboard);
      }catch{}
    };
    return()=>ws.close();
  },[]);

  useEffect(()=>{
    if(!selectedId){setDetail(null);return;}
    fetch(apiUrl(`/api/arena/strategies/${selectedId}`)).then(r=>r.ok?r.json():null).then(setDetail).catch(()=>setDetail(null));
  },[selectedId,activity]);

  const strategyMap=useMemo(()=>Object.fromEntries(strategies.map(s=>[s.id,s])),[strategies]);
  const filtered=useMemo(()=>board.filter(r=>{
    const s=strategyMap[r.strategy_id];
    return (symbol==='ALL'||r.symbol===symbol)
      &&(agent==='ALL'||r.agent===agent)
      &&(status==='ALL'||r.status===status)
      &&(timeframe==='ALL'||s?.timeframes?.includes(timeframe));
  }),[board,strategyMap,symbol,agent,status,timeframe]);

  const selectedResult=testType==='oos'?detail?.oos_results:testType==='backtest'?detail?.backtest_results:detail?.validation_results;
  const spikeData=selectedResult?.spike_metrics || detail?.validation_results?.spike_metrics || detail?.backtest_results?.spike_metrics;
  const forward=detail?.forward_results?.[0];

  return <main className="arena-shell">
    <header className="topbar">
      <div><p className="eyebrow">Deriv Boom & Crash · Research only</p><h1>AI Strategy Arena</h1><p className="muted">Strategy factory · OOS validation · robustness · real_24h_shadow_v2</p></div>
      <div className="safety-chip"><ShieldCheck size={18}/><span>MARKET DATA / SHADOW ONLY</span></div>
    </header>

    <section className="safety-panel">
      <div><span>LIVE TRADING</span><b className="good">DISABLED</b></div>
      <div><span>READY FOR LIVE</span><b className="good">{String(safety?.ready_for_live??false).toUpperCase()}</b></div>
      <div><span>ORDERS OPENED</span><b className="good">{safety?.orders_opened??0}</b></div>
      <div><span>MODE</span><b>{safety?.mode||'...'}</b></div>
    </section>

    <section className="stats-grid">
      <StatCard label="Total Strategies" value={stats?.total_strategies??'—'} icon={Layers3}/>
      <StatCard label="Qualified" value={stats?.qualified??'—'} icon={Trophy}/>
      <StatCard label="Rejected" value={stats?.rejected??'—'} icon={AlertTriangle}/>
      <StatCard label="OOS Testing" value={stats?.oos_testing??'—'} icon={FlaskConical}/>
      <StatCard label="Forward Testing" value={stats?.forward_testing??'—'} icon={Radio}/>
      <StatCard label="Active Agents" value={stats?.active_agents??'—'} icon={BrainCircuit}/>
      <StatCard label="Backtests Today" value={stats?.backtests_today??'—'} icon={Database}/>
      <StatCard label="Shadow Signals Today" value={stats?.shadow_signals_today??'—'} icon={Activity}/>
    </section>

    <section className="arena-card">
      <div className="section-title"><div><p className="eyebrow">Filters</p><h2>Strategy Search</h2></div><Gauge/></div>
      <div className="filters">
        <select value={symbol} onChange={e=>setSymbol(e.target.value)}><option value="ALL">All symbols</option>{profiles.map(p=><option key={p.symbol}>{p.symbol}</option>)}</select>
        <select value={agent} onChange={e=>setAgent(e.target.value)}><option value="ALL">All agents</option>{agents.map(a=><option key={a.id}>{a.name}</option>)}</select>
        <select value={status} onChange={e=>setStatus(e.target.value)}><option value="ALL">All statuses</option>{['created','backtesting','qualified','rejected','oos_testing','forward_testing','retired'].map(s=><option key={s}>{s}</option>)}</select>
        <select value={timeframe} onChange={e=>setTimeframe(e.target.value)}><option value="ALL">All timeframes</option><option>M1</option><option>M5</option><option>M15</option></select>
        <select value={testType} onChange={e=>setTestType(e.target.value as any)}><option value="validation">Validation</option><option value="backtest">In-sample</option><option value="oos">Out-of-sample</option></select>
      </div>
    </section>

    <section>
      <div className="section-title"><div><p className="eyebrow">1 · AI Agents</p><h2>Specialist Strategy Agents</h2></div><BrainCircuit/></div>
      <div className="agent-grid">{agents.map(a=><article className="arena-card agent-card" key={a.id}><div className="agent-head"><b>{a.name}</b><span className={a.enabled?'dot on':'dot'}/></div><p className="muted">{a.description}</p><div className="chips">{a.capabilities.map(c=><span key={c}>{c.replaceAll('_',' ')}</span>)}</div></article>)}</div>
    </section>

    <section className="arena-card">
      <div className="section-title"><div><p className="eyebrow">2 · Strategy Leaderboard</p><h2>Risk & spike-aware ranking</h2></div><Trophy/></div>
      <p className="muted formula">{formula}</p>
      <div className="chips">{Object.entries(weights).map(([k,v])=><span key={k}>{k}: {fmt(v,2)}</span>)}</div>
      <div className="table-wrap"><table><thead><tr><th>Strategy</th><th>Agent</th><th>Symbol</th><th>Status</th><th>Trades</th><th>PF</th><th>Expectancy</th><th>Max DD</th><th>Spike P</th><th>Spike R</th><th>False</th><th>OOS</th><th>Forward</th><th>Stability</th><th>Composite</th></tr></thead>
      <tbody>{filtered.map(r=><tr key={r.strategy_id} className={selectedId===r.strategy_id?'selected-row':''} onClick={()=>setSelectedId(r.strategy_id)}>
        <td><b>{r.strategy}</b></td><td>{r.agent}</td><td>{r.symbol}</td><td><Badge value={r.status}/></td><td>{r.trades}</td><td>{fmt(r.profit_factor)}</td><td>{fmt(r.expectancy,3)}</td><td>{fmt(r.max_drawdown)}</td><td>{pct(r.spike_precision)}</td><td>{pct(r.spike_recall)}</td><td>{pct(r.false_alert_rate)}</td><td>{fmt(r.oos_score,1)}</td><td>{fmt(r.forward_score,1)}</td><td>{pct(r.stability)}</td><td><b>{fmt(r.composite_score,1)}</b></td>
      </tr>)}</tbody></table>{!filtered.length&&<p className="muted empty-pad">No measured leaderboard rows match the current filters.</p>}</div>
    </section>

    <section className="details-grid">
      <article className="arena-card details-main">
        <div className="section-title"><div><p className="eyebrow">3 · Strategy Details</p><h2>{detail?.name||'Select a strategy'}</h2></div>{detail?.status&&<Badge value={detail.status}/>}</div>
        {detail?<>
          <div className="detail-meta"><span><b>Agent</b>{detail.agent_name}</span><span><b>Symbol</b>{detail.symbol}</span><span><b>Version</b>{detail.version}</span><span><b>Timeframes</b>{detail.timeframes?.join(' / ')}</span></div>
          <h3>Parameters</h3><pre>{JSON.stringify(detail.parameters,null,2)}</pre>
          <h3>Rules</h3><ul>{detail.rules?.map((r:string)=><li key={r}>{r}</li>)}</ul>
        </>:<p className="muted">Choose a strategy from the leaderboard.</p>}
      </article>
      <article className="arena-card">
        <div className="section-title"><div><p className="eyebrow">4 · Curves</p><h2>Equity & Drawdown</h2></div><BarChart3/></div>
        <Curve points={selectedResult?.equity_curve} keyName="equity" label="Equity Curve"/>
        <Curve points={selectedResult?.drawdown_curve} keyName="drawdown" label="Drawdown Curve"/>
      </article>
    </section>

    <section className="result-grid">
      <article className="arena-card"><p className="eyebrow">5 · Backtest Results</p><h2>In-Sample</h2><MetricGrid data={detail?.backtest_results?.metrics}/></article>
      <article className="arena-card"><p className="eyebrow">5 · Backtest Results</p><h2>Validation</h2><MetricGrid data={detail?.validation_results?.metrics}/></article>
      <article className="arena-card"><p className="eyebrow">6 · OOS Results</p><h2>Out-of-Sample</h2><MetricGrid data={detail?.oos_results?.metrics}/></article>
      <article className="arena-card"><p className="eyebrow">6 · Robustness</p><h2>Parameter Sensitivity</h2><RobustnessGrid data={detail?.robustness_results}/></article>
      <article className="arena-card"><p className="eyebrow">7 · Forward Results</p><h2>{forward?.mode||'Shadow'}</h2><ForwardGrid data={forward?.metrics}/></article>
    </section>

    <section className="arena-card">
      <div className="section-title"><div><p className="eyebrow">8 · Spike Detection Metrics</p><h2>Does it actually predict spikes?</h2></div><Activity/></div>
      <SpikeGrid data={spikeData}/>
    </section>

    <section className="details-grid">
      <article className="arena-card">
        <div className="section-title"><div><p className="eyebrow">Strategy Audit</p><h2>Recent Signals</h2></div><Radio/></div>
        <div className="activity-list">{detail?.recent_signals?.map((e:any)=><div className="audit-row" key={e.id}><span className="mono">{e.created_at||'—'}</span><span>{e.event_type}</span><span className="muted">{e.payload?.outcome||'PENDING'} · score {fmt(e.payload?.signal_score,1)}</span></div>)}{!detail?.recent_signals?.length&&<p className="muted">No shadow signals for this strategy yet.</p>}</div>
      </article>
      <article className="arena-card">
        <div className="section-title"><div><p className="eyebrow">Strategy Audit</p><h2>Status History</h2></div><Layers3/></div>
        <div className="activity-list">{detail?.status_history?.map((h:any,i:number)=><div className="audit-row" key={i}><span className="mono">{h.created_at||'—'}</span><span>{h.from||'new'} → {h.to}</span><span className="muted">{h.reason}</span></div>)}{!detail?.status_history?.length&&<p className="muted">No status transitions recorded yet.</p>}</div>
      </article>
    </section>

    <section className="arena-card">
      <div className="section-title"><div><p className="eyebrow">9 · Live Shadow Activity</p><h2>Measured events only</h2></div><Radio/></div>
      <div className="activity-list">{activity.slice(0,30).map(e=><div className="activity-row" key={e.id}><span className="mono">{e.created_at||'—'}</span><Badge value={e.event_type.replace('arena_','')}/><span>{e.message}</span><span className="muted">{e.payload?.strategy_id||''}</span></div>)}{!activity.length&&<p className="muted">No Arena activity recorded yet.</p>}</div>
    </section>

    <section>
      <div className="section-title"><div><p className="eyebrow">10 · Symbol Profiles</p><h2>Independent Boom / Crash parameters</h2></div><Database/></div>
      <div className="profile-grid">{profiles.map(p=><article className="arena-card" key={p.symbol}><div className="profile-head"><b>{p.display_name}</b><span>{p.symbol}</span></div><div className="profile-lines"><span>Signal score <b>{p.min_signal_score}</b></span><span>ATR / RSI <b>{p.atr_period} / {p.rsi_period}</b></span><span>BB <b>{p.bb_period} × {p.bb_std_dev}</b></span><span>S/R lookback <b>{p.support_resistance_lookback}</b></span><span>Spike ATR <b>{p.spike_atr_multiple}</b></span><span>Pre-spike <b>{p.pre_spike_window} bars</b></span></div></article>)}</div>
    </section>

    <section className="arena-card safety-bottom">
      <div className="section-title"><div><p className="eyebrow">11 · Safety Status</p><h2>Fail-closed constraints</h2></div><ShieldCheck/></div>
      <div className="safety-big"><b>LIVE TRADING: DISABLED</b><b>READY FOR LIVE: FALSE</b><b>ORDERS OPENED: {safety?.orders_opened??0}</b></div>
      <p className="muted">Historical backtesting · Demo data · Deriv market-data WebSocket · shadow/paper analysis only. No buy/sell order path is exposed.</p>
    </section>
  </main>;
}

createRoot(document.getElementById('root')!).render(<React.StrictMode><App/></React.StrictMode>);
