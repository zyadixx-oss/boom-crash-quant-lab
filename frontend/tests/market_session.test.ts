import assert from 'node:assert/strict';
import {test} from 'node:test';
import {createDerivMarketSession, PUBLIC_ENDPOINT} from '../src/lib/deriv.ts';
import {getCRT} from '../src/lib/crt.ts';
import type {MarketState} from '../src/lib/deriv.ts';

const NOW = 1700002800;
const ACTIVE = [
  {underlying_symbol:'BOOM500',underlying_symbol_name:'Boom 500 Index'},
  {underlying_symbol:'CRASH500',underlying_symbol_name:'Crash 500 Index'},
];

class FakeSocket {
  static OPEN = 1;
  static CLOSING = 2;
  static instances: FakeSocket[] = [];
  readyState = 0;
  requests: Record<string, any>[] = [];
  closes = 0;
  onopen: (() => void) | null = null;
  onclose: (() => void) | null = null;
  onerror: (() => void) | null = null;
  onmessage: ((event: {data: string}) => void) | null = null;
  url: string;
  constructor(url: string) {this.url=url; FakeSocket.instances.push(this);}
  open() {this.readyState = 1; this.onopen?.();}
  send(data: string) {assert.equal(this.readyState,1); this.requests.push(JSON.parse(data));}
  close() {this.closes++; this.readyState = 3; this.onclose?.();}
  message(data: Record<string, any>) {this.onmessage?.({data:JSON.stringify(data)});}
  active(rows = ACTIVE) {this.message({msg_type:'active_symbols',req_id:this.requests[0].req_id,active_symbols:rows});}
  history(requestIndex: number, id: string) {
    const request = this.requests[requestIndex];
    const size = request.granularity as number;
    this.message({msg_type:'candles',req_id:request.req_id,subscription:{id},candles:[
      {epoch:NOW-size,open:101,high:102,low:100,close:101.4},
      {epoch:NOW,open:101.4,high:102,low:101,close:101.6},
    ]});
  }
  tick(requestIndex: number, id: string, epoch = NOW + 1, symbol = 'BOOM500') {
    const request = this.requests[requestIndex];
    this.message({msg_type:'ohlc',req_id:request.req_id,subscription:{id},ohlc:{
      id,symbol,epoch,open_time:Math.floor(epoch/request.granularity)*request.granularity,granularity:request.granularity,open:'101.4',high:'102',low:'101',close:'101.7',
    }});
  }
}

function initial(): MarketState {
 return {symbols:[],bars:[],analysisBars:[],price:null,lastTickEpoch:null,status:'connecting',error:null,isStale:true};
}
function harness(t: any, symbol = 'BOOM500', frame: 'M1'|'M5'|'M15'|'H1' = 'M1') {
 const Original = globalThis.WebSocket;
 FakeSocket.instances = [];
 globalThis.WebSocket = FakeSocket as unknown as typeof WebSocket;
 t.mock.timers.enable({apis:['setTimeout','setInterval','Date'],now:NOW*1000});
 let state = initial();
 const dispose = createDerivMarketSession(symbol,frame,update => {state = update(state);});
 t.after(() => {dispose(); globalThis.WebSocket = Original; t.mock.timers.reset();});
 return {socket:FakeSocket.instances[0],dispose,state:()=>state};
}

test('one public socket sends only allowed public data requests and routes both candle streams', t => {
 const h = harness(t);
 assert.equal(h.socket.url,PUBLIC_ENDPOINT);
 h.socket.open(); h.socket.active();
 assert.equal(h.socket.requests.length,3);
 assert.equal(h.socket.requests[1].granularity,60); assert.equal(h.socket.requests[2].granularity,300);
 assert.ok(h.socket.requests.every(request => request.active_symbols || request.ticks_history));
 assert.ok(h.socket.requests.every(request => !('authorize' in request || 'buy' in request || 'sell' in request || 'proposal' in request)));
 h.socket.history(1,'chart'); h.socket.history(2,'analysis'); h.socket.tick(1,'chart'); h.socket.tick(2,'analysis');
 assert.equal(h.state().status,'connected'); assert.equal(h.state().isStale,false);
 assert.equal(h.state().bars.length,2); assert.equal(h.state().analysisBars.length,1);
 assert.equal(h.state().analysisBars[0].time,NOW-300); assert.equal(h.state().price,101.7);
});
test('M5 reuses a single subscription for chart and closed-bar analysis', t => {
 const h = harness(t,'BOOM500','M5'); h.socket.open(); h.socket.active();
 assert.equal(h.socket.requests.length,2);
 h.socket.history(1,'m5'); h.socket.tick(1,'m5');
 assert.equal(h.state().bars.length,2); assert.equal(h.state().analysisBars.length,1);
});
test('selection cleanup closes the old socket and ignores late history and ticks', t => {
 const h = harness(t); h.socket.open(); h.socket.active(); h.dispose();
 const snapshot = h.state(); h.socket.history(1,'old'); h.socket.tick(1,'old');
 assert.strictEqual(h.state(),snapshot); assert.equal(h.socket.closes,1);
 t.mock.timers.tick(60000); assert.equal(FakeSocket.instances.length,1);
});
test('a new symbol and timeframe requests its own data without retaining old candles', t => {
 const h = harness(t); h.socket.open(); h.socket.active(); h.socket.history(1,'old'); h.socket.history(2,'old-m5');
 h.dispose();
 let state = h.state();
 const closeNew = createDerivMarketSession('CRASH500','H1',update => {state=update(state);}); t.after(closeNew);
 assert.equal(state.bars.length,0); assert.equal(state.price,null); assert.equal(state.lastTickEpoch,null);
 const ws = FakeSocket.instances[1]; ws.open(); ws.active();
 assert.equal(ws.requests[1].ticks_history,'CRASH500'); assert.equal(ws.requests[1].granularity,3600);
 assert.equal(ws.requests[2].granularity,300);
});
test('an unavailable symbol emits an honest error and does not invent prices or requests', t => {
 const h = harness(t,'BOOM_FAKE'); h.socket.open(); h.socket.active();
 assert.equal(h.socket.requests.length,1); assert.equal(h.state().price,null);
 assert.equal(h.state().isStale,true); assert.ok(h.state().error?.includes('غير موجود'));
});
test('pending request timeout closes the socket and retries with one new socket', t => {
 const h = harness(t); h.socket.open();
 t.mock.timers.tick(15000); assert.equal(h.socket.closes,1); assert.equal(h.state().status,'reconnecting');
 t.mock.timers.tick(1000); assert.equal(FakeSocket.instances.length,2);
});
test('a quiet market feed becomes stale, then reconnects after the receive timeout', t => {
 const h = harness(t); h.socket.open(); h.socket.active(); h.socket.history(1,'chart'); h.socket.history(2,'analysis'); h.socket.tick(1,'chart'); h.socket.tick(2,'analysis');
 assert.equal(h.state().isStale,false);
 t.mock.timers.tick(24000); assert.equal(h.state().isStale,true);
 t.mock.timers.tick(10000); assert.equal(h.socket.closes,1);
 t.mock.timers.tick(1000); assert.equal(FakeSocket.instances.length,2);
});
test('ticks for an unrelated symbol are ignored', t => {
 const h = harness(t); h.socket.open(); h.socket.active(); h.socket.history(1,'chart'); h.socket.history(2,'analysis');
 const snapshot = h.state(); h.socket.tick(1,'chart',NOW+1,'CRASH500');
 assert.strictEqual(h.state(),snapshot); assert.equal(h.state().lastTickEpoch,null);
});
test('API errors do not erase the error or leak pending request timers after cleanup', t => {
 const h = harness(t); h.socket.open(); h.socket.active();
 h.socket.message({error:{code:'RateLimit',message:'limit'},req_id:h.socket.requests[1].req_id});
 assert.equal(h.socket.closes,1); assert.ok(h.state().error);
 h.dispose(); const snapshot=h.state(); t.mock.timers.tick(60000);
 assert.strictEqual(h.state(),snapshot); assert.equal(FakeSocket.instances.length,1);
});

test('reconnection does not report connected until both new histories and a live tick arrive', t => {
 const h = harness(t); h.socket.open(); h.socket.active(); h.socket.history(1,'chart'); h.socket.history(2,'analysis'); h.socket.tick(1,'chart');
 h.socket.close(); t.mock.timers.tick(1000);
 const ws=FakeSocket.instances[1]; ws.open(); ws.active(); ws.history(1,'new-chart'); ws.tick(1,'new-chart');
 assert.equal(h.state().status,'reconnecting');
 ws.history(2,'new-analysis'); assert.equal(h.state().status,'reconnecting'); assert.equal(h.state().isStale,true);
 ws.tick(2,'new-analysis'); assert.equal(h.state().status,'connected'); assert.equal(h.state().isStale,false);
});
test('a late candle stream cannot move the last tick and live price backwards', t => {
 const h = harness(t); h.socket.open(); h.socket.active(); h.socket.history(1,'chart'); h.socket.history(2,'analysis');
 h.socket.tick(1,'chart',NOW+5);
 const snapshot=h.state();
 const request=h.socket.requests[2];
 h.socket.message({msg_type:'ohlc',req_id:request.req_id,subscription:{id:'analysis'},ohlc:{
  id:'analysis',symbol:'BOOM500',epoch:NOW+3,open_time:NOW,granularity:300,open:'101.4',high:'102',low:'101',close:'101.5',
 }});
 assert.equal(h.state().lastTickEpoch,NOW+5); assert.equal(h.state().price,snapshot.price);
});
test('older ticks on the same stream cannot rewrite the displayed candle', t => {
 const h = harness(t); h.socket.open(); h.socket.active(); h.socket.history(1,'chart'); h.socket.history(2,'analysis');
 h.socket.tick(1,'chart',NOW+5); const snapshot=h.state(); h.socket.tick(1,'chart',NOW+3);
 assert.strictEqual(h.state(),snapshot);
});
test('repeated connection failure stops after the bounded retry budget', t => {
 const h = harness(t);
 for (let i=0;i<9;i++) {
  t.mock.timers.tick(15000);
  if (i<8) t.mock.timers.tick(Math.min(1000*2**i,30000));
 }
 assert.equal(FakeSocket.instances.length,9); assert.equal(h.state().status,'error'); assert.ok(h.state().error?.includes('تعذر استعادة'));
 t.mock.timers.tick(60000); assert.equal(FakeSocket.instances.length,9);
});

function rangeHistory(ws: FakeSocket) {
 const rows = Array.from({length:72},(_,i)=>({epoch:NOW-(72-i)*300,open:101,high:102,low:100,close:101.4}));
 rows.push({epoch:NOW,open:100.2,high:101.4,low:99.5,close:100.8});
 ws.message({msg_type:'candles',req_id:ws.requests[2].req_id,subscription:{id:'analysis'},candles:rows});
}
function analysisSweepTick(ws: FakeSocket,epoch: number) {
 ws.message({msg_type:'ohlc',req_id:ws.requests[2].req_id,subscription:{id:'analysis'},ohlc:{
  id:'analysis',symbol:'BOOM500',epoch,open_time:NOW,granularity:300,open:'100.2',high:'101.4',low:'99.5',close:'100.8',
 }});
}
test('a chart tick after M5 close cannot promote the partial analysis-history snapshot', t => {
 const h=harness(t); h.socket.open(); h.socket.active(); h.socket.history(1,'chart'); rangeHistory(h.socket);
 t.mock.timers.setTime((NOW+301)*1000); h.socket.tick(1,'chart',NOW+301);
 assert.equal(h.state().isStale,true); assert.notEqual(h.state().status,'connected');
 assert.equal(h.state().analysisBars.length,72); assert.ok(h.state().analysisBars.every(bar=>bar.time<NOW));
 const crt=getCRT(h.state().analysisBars,'boom',NOW+301)!;
 assert.equal(crt.state,'waiting'); assert.equal(crt.events.length,0);
});
test('the first analysis tick in a later bucket does not certify an unseen earlier partial snapshot', t => {
 const h=harness(t); h.socket.open(); h.socket.active(); h.socket.history(1,'chart'); rangeHistory(h.socket);
 t.mock.timers.setTime((NOW+301)*1000); h.socket.tick(1,'chart',NOW+301); h.socket.tick(2,'analysis',NOW+301);
 assert.equal(h.state().isStale,false); assert.equal(h.state().status,'connected');
 assert.equal(h.state().analysisBars.length,72); assert.ok(h.state().analysisBars.every(bar=>bar.time!==NOW));
 assert.equal(getCRT(h.state().analysisBars,'boom',NOW+301)?.state,'waiting');
});
test('an observed M5 is closed only by a later tick on its own analysis stream', t => {
 const h=harness(t); h.socket.open(); h.socket.active(); h.socket.history(1,'chart'); rangeHistory(h.socket);
 t.mock.timers.setTime((NOW+299)*1000); h.socket.tick(1,'chart',NOW+299); analysisSweepTick(h.socket,NOW+299);
 assert.equal(h.state().isStale,false); assert.ok(h.state().analysisBars.every(bar=>bar.time<NOW));
 t.mock.timers.setTime((NOW+301)*1000); h.socket.tick(1,'chart',NOW+301);
 assert.ok(h.state().analysisBars.every(bar=>bar.time<NOW));
 h.socket.tick(2,'analysis',NOW+301);
 assert.equal(h.state().analysisBars.length,73);
 assert.equal(getCRT(h.state().analysisBars,'boom',NOW+301)?.state,'reclaimed');
});
test('fresh chart ticks cannot hide stale analysis or prevent its receive timeout', t => {
 const h=harness(t); h.socket.open(); h.socket.active(); h.socket.history(1,'chart'); h.socket.history(2,'analysis');
 h.socket.tick(1,'chart'); h.socket.tick(2,'analysis'); assert.equal(h.state().isStale,false);
 t.mock.timers.tick(24000); h.socket.tick(1,'chart',NOW+24);
 assert.equal(h.state().lastTickEpoch,NOW+24); assert.equal(h.state().isStale,true);
 t.mock.timers.tick(8000); assert.equal(h.socket.closes,1); assert.equal(h.state().status,'reconnecting');
});

test('the shared M5 chart stream advances its own closed-candle watermark', t => {
 const h=harness(t,'BOOM500','M5'); h.socket.open(); h.socket.active(); h.socket.history(1,'m5');
 t.mock.timers.setTime((NOW+299)*1000); h.socket.tick(1,'m5',NOW+299);
 assert.equal(h.state().isStale,false); assert.equal(h.state().analysisBars.length,1);
 t.mock.timers.setTime((NOW+301)*1000); h.socket.tick(1,'m5',NOW+301);
 assert.equal(h.state().isStale,false); assert.equal(h.state().analysisBars.length,2);
 assert.equal(h.state().analysisBars.at(-1)?.time,NOW);
});
