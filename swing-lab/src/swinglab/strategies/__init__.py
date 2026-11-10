from swinglab.config import UniverseConfig
from swinglab.strategies.base import Strategy, price_panel
from swinglab.strategies.breakout import Breakout
from swinglab.strategies.buy_hold import BuyAndHold
from swinglab.strategies.dual_momentum import SECTOR_ETFS, DualMomentum
from swinglab.strategies.market_filter import MarketFilter
from swinglab.strategies.rsi2_pullback import Rsi2Pullback
from swinglab.strategies.rsi2_swing import Rsi2Swing
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


_PARAMETRIZED = {
    "rsi2_v2": Rsi2Swing,
    "dual_momentum": DualMomentum,
    "breakout": Breakout,
    "sma_cross": SmaCross,
    "rsi2_pullback": Rsi2Pullback,
}


def build_parametrized(
    name: str,
    params: dict,
    universe: UniverseConfig,
    tickers: list[str] | None = None,
    market_filter: bool = False,
) -> Strategy:
    """1단계 실험용: 이름·파라미터로 전략을 만든다. tickers가 없으면 벤치마크를 뺀 유니버스 전체."""
    if name not in _PARAMETRIZED:
        raise ValueError(f"알 수 없는 전략: {name} (가능: {', '.join(_PARAMETRIZED)})")
    strategy = _PARAMETRIZED[name](tickers=list(tickers or universe.tradable), **params)
    return MarketFilter(strategy, benchmark=universe.benchmark) if market_filter else strategy


__all__ = [
    "SECTOR_ETFS",
    "STRATEGY_NAMES",
    "Breakout",
    "BuyAndHold",
    "DualMomentum",
    "MarketFilter",
    "Rsi2Pullback",
    "Rsi2Swing",
    "SmaCross",
    "Strategy",
    "build_parametrized",
    "build_strategy",
    "price_panel",
]
