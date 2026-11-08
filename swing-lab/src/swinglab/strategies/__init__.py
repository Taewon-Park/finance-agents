from swinglab.config import UniverseConfig
from swinglab.strategies.base import Strategy, price_panel
from swinglab.strategies.buy_hold import BuyAndHold
from swinglab.strategies.market_filter import MarketFilter
from swinglab.strategies.rsi2_pullback import Rsi2Pullback
from swinglab.strategies.sma_cross import SmaCross

STRATEGY_NAMES: tuple[str, ...] = ("buy_hold", "sma_cross", "rsi2_pullback")


def build_strategy(name: str, universe: UniverseConfig) -> Strategy:
    """이름으로 전략을 만든다. 매매 전략의 대상은 벤치마크를 뺀 유니버스 전체."""
    if name == "buy_hold":
        return BuyAndHold(ticker=universe.benchmark)
    if name == "sma_cross":
        return SmaCross(tickers=universe.tradable)
    if name == "rsi2_pullback":
        return Rsi2Pullback(tickers=universe.tradable)
    raise ValueError(f"알 수 없는 전략: {name} (가능: {', '.join(STRATEGY_NAMES)})")


__all__ = [
    "STRATEGY_NAMES",
    "BuyAndHold",
    "MarketFilter",
    "Rsi2Pullback",
    "SmaCross",
    "Strategy",
    "build_strategy",
    "price_panel",
]
