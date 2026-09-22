from pathlib import Path
from typing import Any

from pydantic import Field
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

    # AI Strategy Arena configuration. All values are environment-overridable.
    arena_mode: str = "real_24h_shadow_v2"
    arena_mvp_symbols: list[str] = Field(default_factory=lambda: ["CRASH500", "BOOM500"])
    arena_train_ratio: float = 0.60
    arena_validation_ratio: float = 0.20
    arena_parameter_sensitivity_pct: float = 0.05
    arena_max_optimized_parameters: int = 3

    arena_minimum_trades: int = 20
    arena_minimum_profit_factor: float = 1.05
    arena_maximum_drawdown: float = 20.0
    arena_minimum_expectancy: float = 0.0
    arena_minimum_spike_precision: float = 0.20
    arena_minimum_spike_recall: float = 0.10
    arena_maximum_false_alert_rate: float = 0.80
    arena_minimum_oos_stability: float = 0.35

    arena_composite_weights: dict[str, float] = Field(
        default_factory=lambda: {
            "profit_factor": 0.15,
            "expectancy": 0.10,
            "drawdown": 0.10,
            "spike_precision": 0.15,
            "spike_recall": 0.15,
            "false_alert": 0.10,
            "oos": 0.10,
            "forward": 0.05,
            "stability": 0.10,
        }
    )

    model_config = SettingsConfigDict(env_file=BASE_DIR / ".env", extra="ignore")

    def assert_safe(self) -> None:
        if any([self.live_trading, self.ready_for_live, self.live_allowed, self.opened_trades]):
            raise RuntimeError("Safety invariant violated: live trading flags must remain false")

    def safety_snapshot(self) -> dict[str, Any]:
        self.assert_safe()
        return {
            "LIVE_TRADING": False,
            "ready_for_live": False,
            "live_allowed": False,
            "opened_trades": False,
            "orders_opened": 0,
            "mode": self.arena_mode,
        }


settings = Settings()
settings.assert_safe()
