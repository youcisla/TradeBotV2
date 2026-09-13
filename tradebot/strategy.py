"""Strategies.

A strategy is a hypothesis, and it stays a hypothesis until `measure` says
otherwise.  The rules for writing one here:

* **Few parameters.**  Every free parameter is a chance to fit noise.  Three or
  four is already a lot for a few thousand samples.
* **State the hypothesis in the docstring.**  If you cannot write down what you
  expect to happen and why, you are curve-fitting, not hypothesising.
* **Never look past ``view.now``.**  You cannot: ``MarketView`` offers only
  cutoff-bound accessors.  Do not work around it by importing MT5 directly.
* **Market entry at the last close.**  The measurement engine requires limit
  entries to be touched, so a limit that price never returns to correctly
  produces no trade rather than a phantom fill.
"""

from __future__ import annotations

from dataclasses import dataclass

from tradebot.domain import Candle, Direction, Intent, Signal
from tradebot.measure import MarketView, Strategy  # noqa: F401  (re-export)


def atr(candles: list[Candle], period: int = 14) -> float:
    """Average true range over the last *period* bars."""
    if len(candles) < period + 1:
        return 0.0
    ranges = [
        max(
            candles[i].high - candles[i].low,
            abs(candles[i].high - candles[i - 1].close),
            abs(candles[i].low - candles[i - 1].close),
        )
        for i in range(1, len(candles))
    ]
    recent = ranges[-period:]
    return sum(recent) / len(recent) if recent else 0.0


def _signal(
    view: MarketView,
    direction: Direction,
    entry: float,
    risk: float,
    r_multiples: tuple[float, ...],
    name: str,
    reason: str,
) -> Signal:
    sign = 1.0 if direction.is_buy else -1.0
    targets = [entry + sign * risk * m for m in r_multiples]
    return Signal(
        symbol=view.symbol,
        direction=direction,
        entry_price=round(entry, 6),
        stop_loss=round(entry - sign * risk, 6),
        take_profit_1=round(targets[0], 6),
        take_profit_2=round(targets[1], 6) if len(targets) > 1 else 0.0,
        take_profit_3=round(targets[2], 6) if len(targets) > 2 else 0.0,
        strategy=name,
        confidence=0.5,
        intent=Intent.STRATEGY,
        reason=reason,
    )


@dataclass(slots=True)
class BreakoutStrategy:
    """Donchian breakout on H1, ATR-scaled stop, fixed R multiples out.

    **Hypothesis.**  When H1 closes beyond the range of the previous *lookback*
    bars, the move has participants behind it and tends to extend before it
    reverts.  The stop sits far enough away (an ATR multiple) that ordinary
    noise does not reach it, and the first target is beyond 1R so the trade
    does not need a high win rate merely to break even.

    Deliberately the simplest thing that could work.  It is here to be
    falsified, not to impress: if a plain breakout cannot clear its own
    transaction costs on this broker, nothing built on top of it will either.
    """

    lookback: int = 55
    atr_period: int = 14
    stop_atr: float = 1.5
    r_multiples: tuple[float, ...] = (1.5, 2.5, 4.0)
    name: str = "breakout"

    def detect(self, view: MarketView) -> Signal | None:
        candles = view.candles
        if len(candles) < max(self.lookback, self.atr_period) + 1:
            return None

        prior = candles[:-1][-self.lookback:]   # never the bar we trade into
        if len(prior) < self.lookback:
            return None

        last = candles[-1]
        band_high = max(c.high for c in prior)
        band_low = min(c.low for c in prior)
        atr_value = atr(candles, self.atr_period)
        if atr_value <= 0:
            return None

        # A genuine close beyond the band, not merely a wick.
        if last.close > band_high:
            direction = Direction.BUY
        elif last.close < band_low:
            direction = Direction.SELL
        else:
            return None

        risk = self.stop_atr * atr_value
        # A stop nearer than the spread is not a stop, it is a guaranteed loss:
        # the position opens already behind it.
        if risk <= view.spread:
            return None

        return _signal(
            view, direction, last.close, risk, self.r_multiples, self.name,
            f"close {last.close:.5f} beyond {self.lookback}-bar band",
        )


@dataclass(slots=True)
class MeanReversionStrategy:
    """Fade excursions beyond the band, same ATR stop discipline.

    **Hypothesis.**  In the absence of a trend, H1 excursions beyond a moving
    band revert.

    Included mainly as the exact counterpart to :class:`BreakoutStrategy`.  It
    creates a decisive comparison: if a momentum rule and its precise opposite
    both lose money, the deficit is execution cost and not direction — which is
    a different diagnosis with a different fix, and worth being able to tell
    apart cheaply.
    """

    lookback: int = 40
    atr_period: int = 14
    stop_atr: float = 1.5
    r_multiples: tuple[float, ...] = (1.0, 1.75, 3.0)
    name: str = "mean_reversion"

    def detect(self, view: MarketView) -> Signal | None:
        candles = view.candles
        if len(candles) < max(self.lookback, self.atr_period) + 1:
            return None

        prior = candles[:-1][-self.lookback:]
        if len(prior) < self.lookback:
            return None

        last = candles[-1]
        band_high = max(c.high for c in prior)
        band_low = min(c.low for c in prior)
        atr_value = atr(candles, self.atr_period)
        if atr_value <= 0:
            return None

        # The inverse of the breakout condition.
        if last.close < band_low:
            direction = Direction.BUY
        elif last.close > band_high:
            direction = Direction.SELL
        else:
            return None

        risk = self.stop_atr * atr_value
        if risk <= view.spread:
            return None

        return _signal(
            view, direction, last.close, risk, self.r_multiples, self.name,
            f"close {last.close:.5f} outside {self.lookback}-bar band (fade)",
        )


def _self_check() -> None:
    """Fire beyond the band, stay quiet inside it, and keep valid geometry."""
    from datetime import UTC, datetime, timedelta

    from tradebot.measure import AsOfSeries

    t0 = datetime(2026, 1, 7, tzinfo=UTC)
    flat = [
        Candle(t0 + timedelta(hours=i), 1.1000, 1.1010, 1.0990, 1.1000, spread=0.0001)
        for i in range(80)
    ]
    inside = MarketView("EURUSD", flat[-1].timestamp, AsOfSeries({"H1": flat}))
    assert BreakoutStrategy(lookback=55).detect(inside) is None, "inside band must not fire"
    assert MeanReversionStrategy(lookback=40).detect(inside) is None

    # Push the last close above the band.
    hot = flat[:-1] + [
        Candle(flat[-1].timestamp, 1.1000, 1.1080, 1.0995, 1.1075, spread=0.0001)
    ]
    hot_view = MarketView("EURUSD", hot[-1].timestamp, AsOfSeries({"H1": hot}))

    breakout = BreakoutStrategy(lookback=55).detect(hot_view)
    assert breakout is not None and breakout.direction is Direction.BUY, breakout
    assert breakout.geometry_error() is None, breakout.geometry_error()
    assert breakout.stop_loss < breakout.entry_price < breakout.take_profit_1

    fade = MeanReversionStrategy(lookback=40).detect(hot_view)
    assert fade is not None and fade.direction is Direction.SELL, fade
    assert fade.geometry_error() is None, fade.geometry_error()

    print("strategy.py self-check: OK")


if __name__ == "__main__":
    _self_check()
