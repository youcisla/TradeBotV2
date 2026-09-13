"""Measurement engine — the only thing that earns a number trust.

Every rule here exists because its absence invalidated a real result in the
previous system.  Read them as scars, not preferences:

1. **As-of access only.**  A strategy receives a :class:`MarketView` built with
   a timestamp cutoff and has no other data source, so look-ahead is
   *impossible to express* rather than merely discouraged.  The old harness
   built higher-timeframe bars from the whole array and returned the tail, so
   every decision was partly made from the future.
2. **Real costs.**  Each fill pays the bar's actual spread plus commission,
   expressed in R so it scales with the trade's own risk.  A forex result at
   zero spread is fiction.
3. **Touched entries.**  A limit entry counts only if a later bar actually
   reached the level.  Counting unfilled limits as trades inflates the sample
   with trades that never happened.
4. **True scale-out.**  Targets are taken 40/30/30, not the whole position at
   the furthest target.
5. **Per-position statistics.**  One record per position.  Counting each leg
   as a trade triples *n* with correlated observations and flatters Sharpe.
6. **Frozen holdout.**  The last slice of time is reported separately and never
   used to choose anything.

Where the intrabar order is unknowable from OHLC, the stop is assumed to hit
first.  Optimism is available via ``pessimistic=False`` for sensitivity
analysis — never for reporting.
"""

from __future__ import annotations

import bisect
import logging
import math
from dataclasses import dataclass
from datetime import datetime
from typing import Protocol, Sequence

from tradebot.domain import Candle, Direction, Outcome, Signal
from tradebot.symbols import spec as symbol_spec

logger = logging.getLogger(__name__)

WINDOW = 150            # H1 bars of history visible to a strategy
MAX_HOLD_BARS = 48      # H1 bars a simulated position may stay open
COOLDOWN_BARS = 5       # H1 bars between signals from one strategy
TP_SPLITS = (0.40, 0.30, 0.30)


# ── As-of data access ────────────────────────────────────────────────────────

class AsOfSeries:
    """Candle store that can only return bars at or before a timestamp."""

    def __init__(self, series: dict[str, Sequence[Candle]]) -> None:
        self._series = {
            tf: sorted(candles, key=lambda c: c.timestamp) for tf, candles in series.items()
        }
        self._times = {tf: [c.timestamp for c in cs] for tf, cs in self._series.items()}

    def upto(self, timeframe: str, ts: datetime, count: int) -> list[Candle]:
        times = self._times.get(timeframe)
        if not times:
            return []
        idx = bisect.bisect_right(times, ts)
        return self._series[timeframe][max(0, idx - count):idx]

    def timeframes(self) -> tuple[str, ...]:
        return tuple(self._series)


class MarketView:
    """Everything a strategy may see at one decision point.

    Deliberately narrow.  A strategy holding this object cannot reach past its
    own decision time, because the only methods it has are cutoff-bound.
    """

    __slots__ = ("symbol", "now", "_series", "_window")

    def __init__(self, symbol: str, now: datetime, series: AsOfSeries, window: int = WINDOW):
        self.symbol = symbol
        self.now = now
        self._series = series
        self._window = window

    @property
    def candles(self) -> list[Candle]:
        """The H1 window ending at the last *closed* bar."""
        return self._series.upto("H1", self.now, self._window)

    def higher(self, timeframe: str, count: int = 200) -> list[Candle]:
        return self._series.upto(timeframe, self.now, count)

    @property
    def spread(self) -> float:
        candles = self.candles
        return candles[-1].spread if candles else 0.0


class Strategy(Protocol):
    """Anything that turns a :class:`MarketView` into a signal or nothing."""

    name: str

    def detect(self, view: MarketView) -> Signal | None: ...


# ── Simulated position ───────────────────────────────────────────────────────

@dataclass(slots=True)
class SimTrade:
    """One resolved position.  The unit of measurement — never a single leg."""

    symbol: str
    strategy: str
    direction: Direction
    opened_at: datetime
    entry: float
    stop_loss: float
    targets: list[float]
    outcome: Outcome = Outcome.INVALID
    net_r: float = 0.0
    bars_held: int = 0
    filled: bool = False

    @property
    def is_counted(self) -> bool:
        """NO_FILL and INVALID are not results — they consumed no capital."""
        return self.outcome not in (Outcome.NO_FILL, Outcome.INVALID)


def _splits(n_targets: int) -> tuple[float, ...]:
    if n_targets <= 1:
        return (1.0,)
    if n_targets == 2:
        return (0.40, 0.60)
    return TP_SPLITS


def _close(
    trade: SimTrade,
    gross_r: float,
    spread_cost_r: float,
    commission_r: float,
    outcome: Outcome,
    bars: int,
) -> SimTrade:
    trade.outcome = outcome
    trade.net_r = round(gross_r - spread_cost_r - commission_r, 6)
    trade.bars_held = bars
    return trade


def simulate(
    signal: Signal,
    opened_at: datetime,
    future: Sequence[Candle],
    *,
    max_hold: int = MAX_HOLD_BARS,
    pessimistic: bool = True,
) -> SimTrade:
    """Resolve one signal into one position, net of spread and commission."""
    trade = SimTrade(
        symbol=signal.symbol,
        strategy=signal.strategy,
        direction=signal.direction,
        opened_at=opened_at,
        entry=signal.entry_price,
        stop_loss=signal.stop_loss,
        targets=signal.targets,
    )

    r_dist = signal.risk_distance
    if r_dist <= 0 or not future:
        trade.outcome = Outcome.INVALID
        return trade

    sspec = symbol_spec(signal.symbol)
    # Costs in R, so they scale with the trade's own risk rather than being a
    # fixed drag that flatters tight stops.
    spread_cost_r = (future[0].spread or 0.0) / r_dist
    commission_r = sspec.round_trip_commission(1.0) / max(sspec.contract_size * r_dist, 1e-12)

    is_buy = signal.direction.is_buy
    entry, stop = signal.entry_price, signal.stop_loss
    targets = trade.targets
    fractions = _splits(len(targets))

    filled = False
    remaining = 1.0
    realised = 0.0
    hit = 0
    last = 0

    for i, candle in enumerate(future[:max_hold]):
        high, low = candle.high, candle.low
        last = i + 1

        if not filled:
            touched = (low <= entry) if is_buy else (high >= entry)
            if not touched:
                continue
            filled = True
            trade.filled = True

        stop_hit = (low <= stop) if is_buy else (high >= stop)

        # Intrabar order is not recoverable from OHLC.  Default: the stop got
        # there first.  Assuming the opposite flatters every result.
        if pessimistic and stop_hit:
            return _close(trade, realised - remaining, spread_cost_r, commission_r,
                          Outcome.SL, last)

        while hit < len(targets):
            target = targets[hit]
            reached = (high >= target) if is_buy else (low <= target)
            if not reached:
                break
            leg_r = ((target - entry) / r_dist) if is_buy else ((entry - target) / r_dist)
            realised += fractions[hit] * leg_r
            remaining -= fractions[hit]
            hit += 1

        if remaining <= 1e-9:
            return _close(trade, realised, spread_cost_r, commission_r, Outcome.TP, last)

        if stop_hit:
            return _close(trade, realised - remaining, spread_cost_r, commission_r,
                          Outcome.SL, last)

    if not filled:
        trade.outcome = Outcome.NO_FILL
        trade.bars_held = last
        return trade

    # Still open at the horizon — mark out at the final close.
    exit_candle = future[min(last, len(future)) - 1]
    exit_r = (
        (exit_candle.close - entry) / r_dist
        if is_buy
        else (entry - exit_candle.close) / r_dist
    )
    return _close(trade, realised + remaining * exit_r, spread_cost_r, commission_r,
                  Outcome.TIMEOUT, last)


# ── Statistics ───────────────────────────────────────────────────────────────

@dataclass(slots=True)
class Metrics:
    label: str = ""
    positions: int = 0
    no_fills: int = 0
    win_rate: float = 0.0
    avg_r: float = 0.0
    profit_factor: float = 0.0
    max_drawdown: float = 0.0
    sharpe: float = 0.0
    total_return: float = 0.0
    buy_hold: float = 0.0
    has_baseline: bool = False
    span_days: float = 0.0

    def render(self) -> str:
        bh = f"{self.buy_hold:>+8.2f}%" if self.has_baseline else "     n/a"
        return (
            f"{self.label:<14} n={self.positions:<5} nofill={self.no_fills:<4} "
            f"win={self.win_rate:>6.1%} avgR={self.avg_r:>+7.3f} "
            f"PF={self.profit_factor:>5.2f} maxDD={self.max_drawdown:>6.1%} "
            f"Sharpe={self.sharpe:>6.2f} ret={self.total_return:>+8.2f}% B&H={bh}"
        )


def _max_drawdown(equity: Sequence[float]) -> float:
    peak = equity[0] if equity else 0.0
    worst = 0.0
    for value in equity:
        peak = max(peak, value)
        if peak > 0:
            worst = max(worst, (peak - value) / peak)
    return worst


def compute_metrics(
    trades: Sequence[SimTrade],
    *,
    label: str = "",
    risk_fraction: float,
    first_close: float = 0.0,
    last_close: float = 0.0,
) -> Metrics:
    m = Metrics(label=label)
    counted = [t for t in trades if t.is_counted]
    m.no_fills = sum(1 for t in trades if t.outcome is Outcome.NO_FILL)
    m.positions = len(counted)
    if not counted:
        return m

    m.win_rate = sum(1 for t in counted if t.net_r > 0) / len(counted)
    m.avg_r = sum(t.net_r for t in counted) / len(counted)

    gross_win = sum(t.net_r for t in counted if t.net_r > 0)
    gross_loss = abs(sum(t.net_r for t in counted if t.net_r < 0))
    m.profit_factor = gross_win / gross_loss if gross_loss > 0 else float("inf")

    equity = [1.0]
    for t in counted:
        equity.append(equity[-1] * (1.0 + risk_fraction * t.net_r))
    m.total_return = (equity[-1] - 1.0) * 100.0
    m.max_drawdown = _max_drawdown(equity)

    returns = [risk_fraction * t.net_r for t in counted]
    mean = sum(returns) / len(returns)
    variance = sum((r - mean) ** 2 for r in returns) / len(returns)
    sd = math.sqrt(variance)
    span_days = (counted[-1].opened_at - counted[0].opened_at).total_seconds() / 86400.0
    m.span_days = span_days
    if sd > 0 and span_days > 0:
        years = max(span_days / 365.25, 1e-9)
        m.sharpe = (mean / sd) * math.sqrt(len(counted) / years)

    if first_close > 0 and last_close > 0:
        m.buy_hold = (last_close / first_close - 1.0) * 100.0
        m.has_baseline = True
    return m


# ── Driver ───────────────────────────────────────────────────────────────────

def run_series(
    symbol: str,
    series: dict[str, Sequence[Candle]],
    strategy: Strategy,
    *,
    window: int = WINDOW,
    max_hold: int = MAX_HOLD_BARS,
    cooldown: int = COOLDOWN_BARS,
    pessimistic: bool = True,
) -> list[SimTrade]:
    """Walk one symbol forward, resolving a position per signal.

    Bars are consumed as *closed* only: the decision at index ``i`` uses bars up
    to ``i-1`` and trades into bar ``i`` and beyond.
    """
    h1 = list(series.get("H1") or [])
    if len(h1) < window + max_hold + 2:
        logger.warning("%s: only %d H1 bars — skipping", symbol, len(h1))
        return []

    store = AsOfSeries(series)
    trades: list[SimTrade] = []
    last_signal = -10_000

    for i in range(window, len(h1) - max_hold):
        if i - last_signal < cooldown:
            continue
        cutoff = h1[i - 1].timestamp
        view = MarketView(symbol, cutoff, store, window)

        try:
            signal = strategy.detect(view)
        except Exception as exc:                       # noqa: BLE001
            logger.debug("%s: strategy raised at bar %d (%s)", symbol, i, exc)
            continue
        if signal is None:
            continue

        error = signal.geometry_error()
        if error:
            logger.debug("%s: rejected at bar %d — %s", symbol, i, error)
            trades.append(
                SimTrade(symbol, signal.strategy, signal.direction, cutoff,
                         signal.entry_price, signal.stop_loss, signal.targets,
                         outcome=Outcome.INVALID)
            )
            continue

        last_signal = i
        trades.append(
            simulate(signal, cutoff, h1[i: i + max_hold + 1],
                     max_hold=max_hold, pessimistic=pessimistic)
        )
    return trades


def evaluate(
    per_symbol: dict[str, tuple[dict[str, Sequence[Candle]], list[SimTrade]]],
    *,
    risk_fraction: float,
    folds: int = 5,
    holdout_fraction: float = 0.20,
) -> dict[str, Metrics]:
    """Summarise per symbol, pooled, by fold, and on a frozen holdout."""
    results: dict[str, Metrics] = {}

    for symbol, (series, trades) in per_symbol.items():
        h1 = series.get("H1") or []
        results[symbol] = compute_metrics(
            trades,
            label=symbol,
            risk_fraction=risk_fraction,
            first_close=h1[0].close if h1 else 0.0,
            last_close=h1[-1].close if h1 else 0.0,
        )

    pooled = sorted(
        (t for _, trades in per_symbol.values() for t in trades),
        key=lambda t: t.opened_at,
    )
    results["pooled"] = compute_metrics(pooled, label="POOLED", risk_fraction=risk_fraction)

    if folds > 1 and pooled:
        size = max(1, len(pooled) // folds)
        for i in range(folds):
            chunk = pooled[i * size: (i + 1) * size]
            if chunk:
                results[f"fold{i + 1}"] = compute_metrics(
                    chunk, label=f"fold {i + 1}", risk_fraction=risk_fraction
                )

    if pooled:
        cut = int(len(pooled) * (1.0 - holdout_fraction))
        results["holdout"] = compute_metrics(
            pooled[cut:], label="HOLDOUT", risk_fraction=risk_fraction
        )
    return results


# ── Self-check ───────────────────────────────────────────────────────────────

def _self_check() -> None:
    """Assertions on the rules that decide money. Run: python -m tradebot.measure."""
    from datetime import UTC, timedelta

    t0 = datetime(2026, 1, 7, 14, 0, tzinfo=UTC)

    def bar(o: float, h: float, lo: float, c: float, spread: float = 0.0, i: int = 0) -> Candle:
        return Candle(t0 + timedelta(hours=i), o, h, lo, c, spread=spread)

    def sig(**kw) -> Signal:
        base = dict(symbol="EURUSD", direction=Direction.BUY, entry_price=1.1000,
                    stop_loss=1.0950, take_profit_1=1.1050, strategy="t")
        base.update(kw)
        return Signal(**base)

    # 1. A limit price never returns to is not a trade.
    never = [bar(1.1060, 1.1070, 1.1055, 1.1065, i=i) for i in range(10)]
    assert simulate(sig(), t0, never).outcome is Outcome.NO_FILL

    # 2. Touched, then straight to stop = exactly -1R (no costs on the bar).
    trade = simulate(sig(), t0, [bar(1.1000, 1.1005, 1.0940, 1.0950)])
    assert trade.outcome is Outcome.SL and abs(trade.net_r + 1.0) < 1e-9, trade

    # 3. Touched, then the single target = the full R, not a fraction of it.
    trade = simulate(sig(), t0, [bar(1.1000, 1.1060, 1.0995, 1.1055)])
    assert trade.outcome is Outcome.TP and abs(trade.net_r - 1.0) < 1e-9, trade

    # 4. Three targets reached scale out 40/30/30, not 100% at the last one.
    ladder = sig(take_profit_1=1.1050, take_profit_2=1.1100, take_profit_3=1.1150)
    bars = [
        bar(1.1000, 1.1050, 1.0999, 1.1049, i=0),
        bar(1.1049, 1.1100, 1.1040, 1.1099, i=1),
        bar(1.1099, 1.1150, 1.1090, 1.1149, i=2),
    ]
    trade = simulate(ladder, t0, bars)
    expected = 0.40 * 1.0 + 0.30 * 2.0 + 0.30 * 3.0
    assert abs(trade.net_r - expected) < 1e-9, (trade.net_r, expected)

    # 5. Spread is charged, and in R — 2 pips against a 50-pip stop = 0.04R.
    free = simulate(sig(), t0, [bar(1.1000, 1.1060, 1.0995, 1.1055)])
    costed = simulate(sig(), t0, [bar(1.1000, 1.1060, 1.0995, 1.1055, spread=0.0002)])
    assert abs(costed.net_r - (free.net_r - 0.0002 / 0.0050)) < 1e-9, (free, costed)

    # 6. A bar spanning both levels resolves to the stop, not the target.
    spanning = [bar(1.1000, 1.1060, 1.0940, 1.1000)]
    assert simulate(sig(), t0, spanning).outcome is Outcome.SL

    # 7. As-of access cannot see the future.
    series = AsOfSeries({"H1": [bar(1.0, 1.0, 1.0, 1.0, i=i) for i in range(10)]})
    window = series.upto("H1", t0 + timedelta(hours=4), 100)
    assert len(window) == 5, len(window)
    assert window[-1].timestamp == t0 + timedelta(hours=4)

    print("measure.py self-check: OK")


if __name__ == "__main__":
    _self_check()
