import asyncio,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]; sys.path.insert(0,str(ROOT/'backend'))
from app.services.deriv_client import DerivClient
async def main():
    c=DerivClient()
    try:
        await c.connect(); data=await c.active_symbols()
        rows=[(s.get('symbol'),s.get('display_name')) for s in data.get('active_symbols',[]) if 'Boom' in s.get('display_name','') or 'Crash' in s.get('display_name','')]
        for symbol,name in rows: print(f'{symbol}\t{name}')
        if not rows: print('No Boom/Crash symbols returned.')
    except Exception as e:
        raise SystemExit(f'Deriv connectivity check failed: {e}')
    finally:
        await c.close()
asyncio.run(main())
