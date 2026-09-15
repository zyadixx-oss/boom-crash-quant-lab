from pathlib import Path
from pydantic_settings import BaseSettings, SettingsConfigDict

BASE_DIR = Path(__file__).resolve().parents[2]

class Settings(BaseSettings):
    app_env: str = "development"
    database_url: str = f"sqlite:///{BASE_DIR / 'data' / 'signals.db'}"
    deriv_ws_url: str = "wss://ws.derivws.com/websockets/v3"
    deriv_app_id: str = "1089"
    live_trading: bool = False
    ready_for_live: bool = False
    live_allowed: bool = False
    opened_trades: bool = False
    frontend_origin: str = "http://localhost:5173"
    model_config = SettingsConfigDict(env_file=BASE_DIR / '.env', extra='ignore')

    def assert_safe(self) -> None:
        if any([self.live_trading, self.ready_for_live, self.live_allowed, self.opened_trades]):
            raise RuntimeError("Safety invariant violated: live trading flags must remain false")

settings = Settings()
settings.assert_safe()
