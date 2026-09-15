from pydantic import BaseModel, Field
class BacktestRequest(BaseModel):
    symbol:str; candles:list[dict]; min_signal_score:float|None=None
class OptimizeRequest(BaseModel):
    symbol:str; candles:list[dict]; search_space:dict[str,list[float]]|None=None
class HealthResponse(BaseModel):
    status:str='ok'; mode:str='SHADOW_ONLY'; live_trading:bool=False; ready_for_live:bool=False; live_allowed:bool=False; opened_trades:bool=False
