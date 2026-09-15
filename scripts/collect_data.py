import asyncio,sys
from pathlib import Path
import pandas as pd
ROOT=Path(__file__).resolve().parents[1]; sys.path.insert(0,str(ROOT/'backend'))
from app.symbol_profiles import get_profile
from app.services.deriv_client import DerivClient

async def main():
    symbol=sys.argv[1] if len(sys.argv)>1 else 'BOOM500'; count=int(sys.argv[2]) if len(sys.argv)>2 else 5000
    profile=get_profile(symbol); c=DerivClient()
    try:
        await c.connect(); api_symbol=await c.resolve_symbol(profile.display_name,profile.api_symbol)
        if not api_symbol: raise RuntimeError(f'{profile.display_name} not found in active_symbols')
        rows=await c.candles(api_symbol,count,60); out=ROOT/'data'/f'{profile.symbol.lower()}_m1.csv'; pd.DataFrame(rows).to_csv(out,index=False)
        print(f'Saved {len(rows)} real Deriv M1 candles to {out}')
    finally: await c.close()
asyncio.run(main())
