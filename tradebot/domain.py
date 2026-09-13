"""Core domain types.

Pure data.  No I/O, no MT5, no database, no network — so every rule in here is
testable in isolation and cannot be accidentally coupled to a broker.

Two invariants live in this module because they are the ones that lose money
when violated:

* a :class:`Signal` must have sane geometry (stop on the losing side of entry,
  targets on the winning side) — see :meth:`Signal.geometry_error`;
* a :class:`Trade` must never claim a size or a stop the broker has not
  confirmed — see :class:`Trade`'s docstring.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, datetime
from enum import StrEnum


class Direction(StrEnum):
    BUY = "BUY"
    SELL = "SELL"

    @property
    def is_buy(self) -> bool:
        return self is Direction.BUY

    @property
    def opposite(self) -> "Direction":
        return Direction.SELL if self.is_buy else Direction.BUY


class Intent(StrEnum):
    """How much authority a signal carries.

    A signal from outside the system (a webhook alert, another bot) arrives as
    ``STRATEGY`` — an idea to be evaluated — never as ``EXECUTION``.  Only the
    pipeline's own gated output may be ``EXECUTION``.  Getting this wrong is
    how an external source ends up placing trades directly.
    """

    DISCUSSION = "DISCUSSION"
    STRATEGY = "STRATEGY"
    EXECUTION = "EXECUTION"


class Outcome(StrEnum):
    OPEN = "OPEN"
    TP = "TP"
    SL = "SL"
    TIMEOUT = "TIMEOUT"
    MANUAL = "MANUAL"


@dataclass(frozen=True, slots=True)
class Candle:
    """One OHLC bar.  ``spread`` is in price units, never points."""

    timestamp: datetime
    open: float
    high: float
    low: float
    close: float
    volume: float = 0.0
    spread: float = 0.0

    def __post_init__(self) -> None:
        if self.high < self.low:
            raise ValueError(f"candle high {self.high} below low {self.low}")
        if not (self.low <= self.open <= self.high):
            raise ValueError(
                f"candle open outside [low, high]: "
                f"o={self.open} h={self.high} l={self.low}"
            )
        if not (self.low <= self.close <= self.high):
            raise ValueError(
                f"candle close outside [low, high]: "
                f"c={self.close} h={self.high} l={self.low}"
            )


@dataclass(slots=True)
class Signal:
    """A trade idea.  Not yet authorized to trade — see :class:`Intent`."""

    symbol: str
    direction: Direction
    entry_price: float
    stop_loss: float
    take_profit_1: float = 0.0
    take_profit_2: float = 0.0
    take_profit_3: float = 0.0
    strategy: str = ""
    confidence: float = 0.5
    intent: Intent = Intent.STRATEGY
    reason: str = ""
    created_at: datetime = field(default_factory=lambda: datetime.now(UTC))

    @property
    def risk_distance(self) -> float:
        """Price distance from entry to stop — the definition of 1R."""
        return abs(self.entry_price - self.stop_loss)

    @property
    def targets(self) -> list[float]:
        return [t for t in (self.take_profit_1, self.take_profit_2, self.take_profit_3) if t > 0]

    def geometry_error(self) -> str | None:
        """Return why this signal cannot become a trade, or None if it can.

        Checked at the trust boundary — strategies can be wrong, and externally
        supplied signals definitely will be.
        """
        if not self.symbol.strip():
            return "symbol is empty"
        if self.entry_price <= 0:
            return "entry_price must be positive"
        if self.stop_loss <= 0:
            return "stop_loss must be positive"

        buy = self.direction.is_buy
        if buy and self.stop_loss >= self.entry_price:
            return "BUY stop must sit below entry"
        if not buy and self.stop_loss <= self.entry_price:
            return "SELL stop must sit above entry"

        for i, target in enumerate(self.targets, start=1):
            if buy and target <= self.entry_price:
                return f"BUY target {i} must sit above entry"
            if not buy and target >= self.entry_price:
                return f"SELL target {i} must sit below entry"

        # Targets must be ordered outwards; a nearer target listed after a
        # further one means the scale-out would fire out of sequence.
        if buy and self.targets != sorted(self.targets):
            return "BUY targets are not ordered furthest-last"
        if not buy and self.targets != sorted(self.targets, reverse=True):
            return "SELL targets are not ordered furthest-last"

        return None


@dataclass(slots=True)
class Position:
    """A live position as the broker reports it.

    This is the source of truth for size and stops.  Local bookkeeping exists
    only to describe what the broker already holds; when the two disagree the
    broker wins and the local record is wrong.
    """

    ticket: int
    symbol: str
    direction: Direction
    volume: float
    entry_price: float
    stop_loss: float = 0.0
    take_profit: float = 0.0
    current_price: float = 0.0
    profit: float = 0.0
    magic: int = 0
    comment: str = ""

    @property
    def is_long(self) -> bool:
        return self.direction.is_buy


@dataclass(slots=True)
class OrderResult:
    """What came back from an order attempt.

    ``ambiguous`` exists because MT5 can time out *after* the broker accepted
    the request.  An ambiguous result is not a failure — treating it as one is
    how positions end up live at the broker but invisible to the bot.
    """

    success: bool
    ticket: int = 0
    filled_volume: float = 0.0
    fill_price: float = 0.0
    message: str = ""
    ambiguous: bool = False


@dataclass(slots=True)
class Trade:
    """A tracked position.

    Invariant: ``volume`` and ``stop_loss`` reflect the **broker-confirmed**
    values, not the requested ones.  The broker normalises volume to its own
    lot step and may reject a stop modification; booking the requested value
    instead makes the local record claim protection the account does not hold.
    """

    ticket: int
    symbol: str
    direction: Direction
    volume: float
    entry_price: float
    stop_loss: float
    take_profit_1: float = 0.0
    take_profit_2: float = 0.0
    take_profit_3: float = 0.0
    opened_at: datetime = field(default_factory=lambda: datetime.now(UTC))
    outcome: Outcome = Outcome.OPEN
    closed_at: datetime | None = None
    pnl: float = 0.0
    strategy: str = ""
    spread_at_entry: float = 0.0
    # Fraction of the position already taken off at each target.
    closed_fraction: float = 0.0

    @property
    def risk_distance(self) -> float:
        return abs(self.entry_price - self.stop_loss)

    @property
    def remaining_fraction(self) -> float:
        return max(0.0, 1.0 - self.closed_fraction)

    @property
    def is_open(self) -> bool:
        return self.outcome is Outcome.OPEN
