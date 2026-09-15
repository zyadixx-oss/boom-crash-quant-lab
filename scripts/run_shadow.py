import asyncio,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]; sys.path.insert(0,str(ROOT/'backend'))
from app.config import settings
from app.services.shadow import ShadowRunner
settings.assert_safe()
symbol=sys.argv[1] if len(sys.argv)>1 else 'BOOM500'
print(f'Starting SHADOW-ONLY collector for {symbol}; no trade execution endpoint exists.')
asyncio.run(ShadowRunner(symbol).run())
