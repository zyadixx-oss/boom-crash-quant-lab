import asyncio, json, logging, time
from collections.abc import AsyncIterator
import websockets
from ..config import settings

log=logging.getLogger(__name__)
class DerivClient:
    def __init__(self,url=None,app_id=None): self.url=url or settings.deriv_ws_url; self.app_id=app_id or settings.deriv_app_id; self.ws=None; self._last_request=0.0
    @property
    def endpoint(self): return f"{self.url}?app_id={self.app_id}"
    async def connect(self): self.ws=await websockets.connect(self.endpoint,ping_interval=20,ping_timeout=20,max_queue=1000); return self
    async def close(self):
        if self.ws: await self.ws.close(); self.ws=None
    async def _send(self,payload):
        if self.ws is None: raise RuntimeError('not connected')
        wait=max(0,0.08-(time.monotonic()-self._last_request))
        if wait: await asyncio.sleep(wait)
        self._last_request=time.monotonic(); await self.ws.send(json.dumps(payload)); raw=json.loads(await self.ws.recv())
        if 'error' in raw: raise RuntimeError(raw['error'].get('message','Deriv API error'))
        return raw
    async def active_symbols(self): return await self._send({'active_symbols':'brief','product_type':'basic'})
    async def resolve_symbol(self,display_name:str,fallback:str|None=None):
        data=await self.active_symbols(); target=display_name.lower().replace(' index','').strip()
        for s in data.get('active_symbols',[]):
            name=str(s.get('display_name','')).lower().replace(' index','').strip()
            if name==target: return s.get('symbol')
        return fallback
    async def history(self,symbol,count=1000,granularity=60):
        return await self._send({'ticks_history':symbol,'adjust_start_time':1,'count':min(int(count),5000),'end':'latest','style':'candles','granularity':granularity})
    async def candles(self,symbol,count=1000,granularity=60):
        data=await self.history(symbol,count,granularity); rows=[]
        for c in data.get('candles',[]): rows.append({'epoch':int(c['epoch']),'open':float(c['open']),'high':float(c['high']),'low':float(c['low']),'close':float(c['close']),'tick_count':0})
        return rows
    async def subscribe_ticks(self,symbol)->AsyncIterator[dict]:
        if self.ws is None: raise RuntimeError('not connected')
        await self.ws.send(json.dumps({'ticks':symbol,'subscribe':1}))
        while True:
            msg=json.loads(await self.ws.recv())
            if 'error' in msg: raise RuntimeError(msg['error'].get('message','Deriv error'))
            if msg.get('msg_type')=='tick': yield msg['tick']
