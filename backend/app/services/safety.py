from ..config import settings


FORBIDDEN_ORDER_ACTIONS = {
    "buy",
    "sell",
    "order",
    "purchase",
    "proposal_open_contract",
    "open_position",
    "execute_trade",
}


def assert_market_data_only() -> None:
    settings.assert_safe()


def reject_order_execution(action: str = "order") -> None:
    settings.assert_safe()
    raise PermissionError(
        f"Blocked forbidden action '{action}'. This project is market-data, backtest, and shadow-analysis only."
    )


def safety_snapshot() -> dict:
    return settings.safety_snapshot()
