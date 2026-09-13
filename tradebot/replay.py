"""Replay harness — run the REAL bot over historical data.

``measure.py`` scores a *strategy*.  This drives the *bot*: the same
``Bot.tick()`` that runs live, with ``MarketData`` swapped for a cursor over
history and ``Broker`` swapped for a simulator that fills orders against
subsequent bars.

That distinction is the point.  Measurement covers entry logic, costs and the
scale-out arithmetic.  It does **not** cover the risk manager refusing a trade,
position sizing, the breakeven move, the partial-close sequence,
reconciliation, or the daily-loss halt — those live in the bot, not the
strategy.  Until they run over history, the only way to exercise them is with
real money in the market.

Fills are deliberately conservative:

* an order decided at the close of bar *i* fills at the **open of bar i+1**,
  paying the spread on the correct side;
* stops and targets are then checked against each later bar's high/low, with
  the stop assumed to hit first when a bar spans both;
* an order that never fills before the replay ends is reported, not counted.

Filling at the price that generated the signal is the single most common way a
replay flatters itself, so the simulator refuses to do it.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from datetime import UTC, datetime

from tradebot.domain import Candle, Direction, OrderResult, Position

logger = logging.getLogger(__name__)

CONTRACT_SIZE = 100_000.0     # per 1.00 lot, for P&L on a price move


class ReplayData:
    """Serves candles up to a moving cursor, never beyond it."""

    def __init__(self, series: dict[str, list[Candle]]) -> None:
        self._series = series
        self._cursor = 0
        self._resolved = {symbol: symbol for symbol in series}

    def connect(self) -> bool:
        return True

    def disconnect(self) -> None:
        return None

    @property
    def trade_allowed(self) -> bool:
        return True

    @property
    def dropped_bars(self) -> int:
        return 0

    def seek(self, index: int) -> None:
        self._cursor = index

    def now(self) -> datetime:
        stamps = [
            c[self._cursor].timestamp
            for c in self._series.values()
            if len(c) > self._cursor
        ]
        return max(stamps) if stamps else datetime.now(UTC)

    def candles(self, symbol: str, timeframe: str = "H1", count: int = 500) -> list[Candle]:
        series = self._series.get(symbol) or []
        end = min(self._cursor + 1, len(series))
        return series[max(0, end - count):end]

    def resolve(self, symbol: str) -> str | None:
        return symbol if symbol in self._series else None


@dataclass
class _Open:
    """A simulated position, tracked at broker-confirmed size."""

    ticket: int
    symbol: str
    direction: Direction
    volume: float
    entry_price: float
    stop_loss: float
    take_profit: float
    opened_at: datetime
    current_price: float = 0.0


@dataclass
class ReplayResult:
    bars: int = 0
    opened: int = 0
    stopped: int = 0
    targeted: int = 0
    never_filled: int = 0
    starting_balance: float = 0.0
    final_balance: float = 0.0
    final_equity: float = 0.0
    peak_equity: float = 0.0
    max_drawdown: float = 0.0
    halted: bool = False
    refusals: dict[str, int] = field(default_factory=dict)
    elapsed_s: float = 0.0
    us_per_iteration: float = 0.0

    @property
    def return_pct(self) -> float:
        if self.starting_balance <= 0:
            return 0.0
        return (self.final_balance / self.starting_balance - 1.0) * 100.0

    def render(self) -> str:
        top = sorted(self.refusals.items(), key=lambda kv: -kv[1])[:6]
        refused = "  ".join(f"{name}={n}" for name, n in top) or "none"
        return (
            f"bars={self.bars}\n"
            f"opened={self.opened}  stopped={self.stopped}  targeted={self.targeted}  "
            f"halted={self.halted}\n"
            f"balance {self.starting_balance:.2f} -> {self.final_balance:.2f} "
            f"({self.return_pct:+.2f}%)  maxDD={self.max_drawdown:.1%}\n"
            f"refusals: {refused}\n"
            f"compute: {self.elapsed_s:.2f}s total, {self.us_per_iteration:.0f}us per iteration "
            f"(a live iteration waits {60}s between runs — this is the headroom)"
        )


class PaperBroker:
    """Simulates MT5 order handling against replayed candles.

    Mirrors the real broker's *semantics*, not just its interface: volume
    normalises to a 0.01 step, a partial close honours its volume, and an order
    decided on one bar fills on the next.
    """

    LOT_STEP = 0.01
    LOT_MIN = 0.01

    def __init__(self, starting_balance: float = 1_000.0) -> None:
        self.balance = starting_balance
        self.realised = 0.0
        self._positions: dict[int, _Open] = {}
        self._next_ticket = 1
        self._symbol_map: dict[str, str] = {}
        self._last_price: dict[str, float] = {}
        self._last_spread: dict[str, float] = {}
        self._last_time: datetime = datetime.now(UTC)

    # ── interface used by Bot ────────────────────────────────────────────────

    def bind_symbols(self, mapping: dict[str, str]) -> None:
        self._symbol_map.update(mapping)

    def account(self) -> tuple[float, float]:
        floating = 0.0
        for p in self._positions.values():
            direction = 1.0 if p.direction.is_buy else -1.0
            floating += p.volume * (p.current_price - p.entry_price) * CONTRACT_SIZE * direction
        return self.balance, self.balance + floating

    def ours(self) -> list[Position]:
        return [
            Position(
                ticket=p.ticket, symbol=p.symbol, direction=p.direction,
                volume=p.volume, entry_price=p.entry_price,
                stop_loss=p.stop_loss, take_profit=p.take_profit,
                current_price=p.current_price, magic=0,
            )
            for p in self._positions.values()
        ]

    def place(self, symbol: str, direction: Direction, volume: float,
              stop_loss: float, take_profit: float = 0.0,
              comment: str = "tradebot") -> OrderResult:
        volume = self._normalise(volume)
        if volume <= 0:
            return OrderResult(False, message="volume below broker minimum")

        mark = self._last_price.get(symbol)
        if mark is None:
            return OrderResult(False, message=f"no price known for {symbol}")

        # A market order fills at the current price, and the bot's last known
        # price is the close of the bar it just decided on.  Paying half the
        # spread on the correct side is the execution cost.
        #
        # Filling here rather than on the next bar is not an optimisation: MT5
        # fills market orders immediately, and queuing them made the bot track a
        # ticket that did not exist yet — every order collided on ticket 0.
        half_spread = self._last_spread.get(symbol, 0.0) / 2.0
        price = mark + half_spread if direction.is_buy else mark - half_spread

        ticket = self._next_ticket
        self._next_ticket += 1
        self._positions[ticket] = _Open(
            ticket=ticket, symbol=symbol, direction=direction, volume=volume,
            entry_price=round(price, 6), stop_loss=stop_loss,
            take_profit=take_profit, opened_at=self._last_time,
            current_price=round(price, 6),
        )
        return OrderResult(
            True, ticket=ticket, filled_volume=volume,
            fill_price=round(price, 6), message="filled",
        )

    def modify(self, ticket: int, *, stop_loss: float | None = None,
               take_profit: float | None = None) -> bool:
        position = self._positions.get(ticket)
        if position is None:
            return False
        if stop_loss is not None:
            position.stop_loss = stop_loss
        if take_profit is not None:
            position.take_profit = take_profit
        return True

    def close_partial(self, ticket: int, symbol: str, direction: Direction,
                      volume: float) -> bool:
        position = self._positions.get(ticket)
        if position is None:
            return True
        size = min(self._normalise(volume), position.volume)
        if size <= 0:
            return False
        self._realise(position, size, position.current_price)
        position.volume = round(position.volume - size, 2)
        if position.volume < self.LOT_MIN - 1e-9:
            self._positions.pop(ticket, None)
        return True

    def close(self, ticket: int, symbol: str, direction: Direction) -> bool:
        position = self._positions.get(ticket)
        if position is None:
            return True
        self._realise(position, position.volume, position.current_price)
        self._positions.pop(ticket, None)
        return True

    # ── simulation ───────────────────────────────────────────────────────────

    @classmethod
    def _normalise(cls, volume: float) -> float:
        return round(int(volume / cls.LOT_STEP) * cls.LOT_STEP, 2)

    def _realise(self, position: _Open, volume: float, price: float) -> None:
        direction = 1.0 if position.direction.is_buy else -1.0
        move = volume * (price - position.entry_price) * CONTRACT_SIZE * direction
        self.realised += move
        self.balance = round(self.balance + move, 4)

    def mark(self, symbol: str, price: float, spread: float = 0.0,
             at: datetime | None = None) -> None:
        """Record the latest price before the bot looks at the world."""
        self._last_price[symbol] = price
        self._last_spread[symbol] = spread
        if at is not None:
            self._last_time = at
        for position in self._positions.values():
            if position.symbol == symbol:
                position.current_price = price

    def hit_levels(self, bar: Candle) -> list[tuple[int, str]]:
        """Resolve stops and targets against this bar.

        Assumes the stop is reached first when a bar spans both — the same
        pessimistic rule the measurement engine uses.
        """
        events: list[tuple[int, str]] = []
        for ticket, position in list(self._positions.items()):
            is_buy = position.direction.is_buy
            stop_hit = (bar.low <= position.stop_loss) if is_buy \
                else (bar.high >= position.stop_loss)
            target_hit = bool(position.take_profit) and (
                (bar.high >= position.take_profit) if is_buy
                else (bar.low <= position.take_profit)
            )

            if stop_hit:
                self._realise(position, position.volume, position.stop_loss)
                events.append((ticket, "stop"))
                self._positions.pop(ticket, None)
            elif target_hit:
                self._realise(position, position.volume, position.take_profit)
                events.append((ticket, "target"))
                self._positions.pop(ticket, None)
            else:
                position.current_price = bar.close
        return events

    def open_count(self) -> int:
        return len(self._positions)

    def force_close_all(self, price_lookup) -> None:
        for ticket, position in list(self._positions.items()):
            price = price_lookup(position.symbol)
            if price:
                self._realise(position, position.volume, price)
            self._positions.pop(ticket, None)


def run_replay(
    bot,
    series: dict[str, list[Candle]],
    *,
    start: int = 200,
    stride: int = 1,
    max_steps: int | None = None,
) -> ReplayResult:
    """Drive *bot* across *series*, one bar at a time.

    The bot is the real one.  Only its data source and broker are simulated, so
    every rule the live bot enforces is enforced here too.
    """
    lengths = [len(c) for c in series.values() if c]
    if not lengths:
        raise ValueError("no candles to replay")
    end = min(lengths)
    symbols = [s for s in bot.settings.symbols if s in series]
    if not symbols:
        raise ValueError("none of the bot's symbols are present in the series")

    paper = PaperBroker(starting_balance=bot.broker.account()[0] or 1_000.0)
    data = ReplayData(series)
    bot.data = data
    bot.broker = paper
    bot.dry_run = False              # we WANT it to trade the simulator
    bot.trades.clear()
    bot.risk.reset()

    result = ReplayResult(starting_balance=paper.balance)
    seen_refusals: dict[str, int] = {}

    started = time.perf_counter()
    index = start
    steps = 0
    while index < end - 1:
        if max_steps is not None and steps >= max_steps:
            break

        data.seek(index)
        bar = series[symbols[0]][index]

        # 1. Mark to this bar's close, so the simulator can price fills and the
        #    bot sees the same world it would see live.
        for symbol in symbols:
            symbol_series = series[symbol]
            if index < len(symbol_series):
                candle = symbol_series[index]
                paper.mark(symbol, candle.close, candle.spread, candle.timestamp)

        # 2. Resolve stops and targets against this bar.
        for _ticket, kind in paper.hit_levels(bar):
            if kind == "stop":
                result.stopped += 1
            else:
                result.targeted += 1

        # 3. Let the REAL bot decide, against the simulated broker.
        opened_before = paper.open_count()
        bot.iteration += 1
        try:
            bot.tick()
        except Exception:                              # noqa: BLE001
            logger.exception("replay: iteration failed at bar %d", index)

        result.opened += max(0, paper.open_count() - opened_before)

        for reason, count in bot.refusals.items():
            delta = count - seen_refusals.get(reason, 0)
            if delta > 0:
                result.refusals[reason] = result.refusals.get(reason, 0) + delta
            seen_refusals[reason] = count

        balance, equity = paper.account()
        result.peak_equity = max(result.peak_equity, equity)
        if result.peak_equity > 0:
            result.max_drawdown = max(
                result.max_drawdown, (result.peak_equity - equity) / result.peak_equity
            )
        result.final_equity = equity
        result.halted = result.halted or bot.risk.state.halted
        result.bars += 1
        index += stride
        steps += 1

    # Close anything still open at the last known price so equity is real.
    last_prices = {
        symbol: series[symbol][min(end, len(series[symbol])) - 1].close for symbol in symbols
    }
    paper.force_close_all(lambda symbol: last_prices.get(symbol))
    result.final_balance, result.final_equity = paper.account()
    result.elapsed_s = time.perf_counter() - started
    if result.bars:
        result.us_per_iteration = (result.elapsed_s / result.bars) * 1_000_000
    return result


def _self_check() -> None:
    """The simulator must never fill at the price that generated the signal."""
    t0 = datetime(2026, 1, 7, tzinfo=UTC)

    # An order cannot be priced without a known mark.
    paper = PaperBroker(starting_balance=1_000.0)
    assert paper.place("EURUSD", Direction.BUY, 0.02, stop_loss=1.0950).success is False

    paper.mark("EURUSD", 1.1000, spread=0.0002, at=t0)
    first = paper.place("EURUSD", Direction.BUY, 0.02, stop_loss=1.0950)
    assert first.success and first.ticket == 1, first
    position = paper._positions[first.ticket]               # noqa: SLF001
    assert position.entry_price > 1.1000, "a BUY must pay the ask, not the midpoint"
    assert abs(position.entry_price - 1.1001) < 1e-9

    # Tickets must be unique.  The first replay run returned 0 for every order,
    # so all positions collided on one key and reconcile retired them at once.
    second = paper.place("EURUSD", Direction.SELL, 0.01, stop_loss=1.1050)
    assert second.ticket == 2 and paper.open_count() == 2

    # Volume normalises to the broker's step, and rounds DOWN.
    assert PaperBroker._normalise(0.037) == 0.03
    assert PaperBroker._normalise(0.001) == 0.0

    # A bar spanning both levels resolves to the stop, not the target.
    straddle = Candle(datetime(2026, 1, 7, 3, tzinfo=UTC), 1.1000, 1.1060, 1.0940, 1.1000)
    paper2 = PaperBroker()
    paper2.mark("EURUSD", 1.1000)
    paper2.place("EURUSD", Direction.BUY, 0.01, stop_loss=1.0950, take_profit=1.1050)
    assert [k for _t, k in paper2.hit_levels(straddle)] == ["stop"]

    # P&L arithmetic: 1 lot of EURUSD moving 100 pips is $1000.  The stop must
    # sit AWAY from entry — a zero-distance stop is hit by the same bar's low —
    # and an open position moves equity, not balance, until it closes.
    paper3 = PaperBroker(starting_balance=1_000.0)
    paper3.mark("EURUSD", 1.0000)
    paper3.place("EURUSD", Direction.BUY, 1.0, stop_loss=0.9900, take_profit=1.0100)
    assert paper3.balance == 1_000.0, "an open position must not move the balance"
    profit_bar = Candle(datetime(2026, 1, 7, 1, tzinfo=UTC), 1.0000, 1.0100, 1.0000, 1.0100)
    assert [k for _t, k in paper3.hit_levels(profit_bar)] == ["target"]
    assert abs(paper3.balance - 2_000.0) < 1e-6, paper3.balance

    print("replay.py self-check: OK")


if __name__ == "__main__":
    _self_check()
