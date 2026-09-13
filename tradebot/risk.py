"""Risk enforcement.

:data:`tradebot.config.RISK` is a frozen declaration.  This module is what
actually *refuses* things.  The distinction matters: a limit that is declared
and never checked is a comment, and a comment has never stopped a trade.

Order of checks is deliberate — the cheapest and most absolute come first, so a
halted account does no work and an over-sized trade never reaches the broker.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import UTC, date, datetime

from tradebot.config import RISK, RiskEnvelope, Settings
from tradebot.domain import Position, Signal
from tradebot.symbols import spec as symbol_spec

logger = logging.getLogger(__name__)


@dataclass(slots=True)
class RiskState:
    """Per-day tracking.  Reset when the UTC date rolls over."""

    day: date = field(default_factory=lambda: datetime.now(UTC).date())
    day_start_equity: float = 0.0
    peak_equity: float = 0.0
    halted: bool = False
    halt_reason: str = ""

    def roll(self, equity: float) -> None:
        today = datetime.now(UTC).date()
        if today != self.day:
            logger.info("Risk state rolling over to %s (equity %.2f)", today, equity)
            self.day = today
            self.day_start_equity = equity
            # A daily halt clears at the rollover; a drawdown halt does not —
            # losing a fifth of the account is not resolved by the clock.
            if self.halted and self.halt_reason.startswith("daily"):
                self.halted = False
                self.halt_reason = ""
        if self.day_start_equity <= 0:
            self.day_start_equity = equity
        self.peak_equity = max(self.peak_equity, equity)


def position_size(signal: Signal, equity: float, envelope: RiskEnvelope = RISK) -> float:
    """Lots such that a stop-out costs ``risk_per_trade`` of equity.

    Sizing from the stop distance is what makes 1% actually mean 1%: a wider
    stop gets fewer lots, so every trade risks the same amount regardless of how
    volatile the instrument is.  Without this, "1% risk" quietly becomes "1% on
    quiet trades and 4% on volatile ones".
    """
    risk_distance = signal.risk_distance
    if risk_distance <= 0 or equity <= 0:
        return 0.0

    money_at_risk = equity * envelope.risk_per_trade
    money_per_lot_at_stop = risk_distance * symbol_spec(signal.symbol).contract_size
    if money_per_lot_at_stop <= 0:
        return 0.0

    return max(0.0, min(money_at_risk / money_per_lot_at_stop, envelope.max_volume_per_trade))


class RiskManager:
    """Refuses trades that breach the envelope."""

    def __init__(self, envelope: RiskEnvelope = RISK, settings: Settings | None = None):
        self.envelope = envelope
        self.settings = settings
        self.state = RiskState()

    # ── checks ───────────────────────────────────────────────────────────────

    def check(
        self,
        signal: Signal,
        *,
        open_positions: list[Position],
        balance: float,
        equity: float,
        spread: float,
    ) -> tuple[bool, str]:
        """Return (allowed, reason).  Reason is empty when allowed."""
        self.state.roll(equity)

        # 1. Any halt is absolute.
        if self.state.halted:
            return False, f"halted: {self.state.halt_reason}"

        # 2. Capital already lost today.
        if self.state.day_start_equity > 0:
            daily_loss = (self.state.day_start_equity - equity) / self.state.day_start_equity
            if daily_loss >= self.envelope.max_daily_loss:
                self.state.halted = True
                self.state.halt_reason = f"daily loss {daily_loss:.1%}"
                return False, f"daily loss limit hit ({daily_loss:.1%})"

        # 3. Capital lost from the peak — the harder stop; needs a human.
        if self.state.peak_equity > 0:
            drawdown = (self.state.peak_equity - equity) / self.state.peak_equity
            if drawdown >= self.envelope.max_drawdown:
                self.state.halted = True
                self.state.halt_reason = f"drawdown {drawdown:.1%}"
                return False, f"max drawdown hit ({drawdown:.1%}) — manual reset required"

        # 4. Concurrency.
        cap = self.envelope.max_concurrent_positions
        if len(open_positions) >= cap:
            return False, f"{len(open_positions)} positions open (cap {cap})"

        # 5. One position per symbol: two on one instrument is one position at
        #    double size, and a single stop cannot protect both.
        if any(p.symbol == signal.symbol for p in open_positions):
            return False, f"already holding {signal.symbol}"

        # 6. Geometry.
        error = signal.geometry_error()
        if error:
            return False, f"bad geometry: {error}"

        reward_risk = self._reward_risk(signal)
        if reward_risk < self.envelope.min_reward_risk:
            return False, (
                f"reward:risk {reward_risk:.2f} below {self.envelope.min_reward_risk:.2f}"
            )

        # 7. Cost against risk.  A spread eating a quarter of 1R means the
        #    trade starts a quarter of its risk behind.
        risk_distance = signal.risk_distance
        if risk_distance > 0 and spread > 0:
            fraction = spread / risk_distance
            cap_fraction = self.envelope.max_spread_fraction_of_risk
            if fraction > cap_fraction:
                return False, f"spread is {fraction:.0%} of risk (cap {cap_fraction:.0%})"

        # 8. Size.
        if position_size(signal, equity, self.envelope) <= 0:
            return False, "position size rounds to zero at this equity"

        return True, ""

    @staticmethod
    def _reward_risk(signal: Signal) -> float:
        """Reward:risk to the furthest target, in R."""
        risk_distance = signal.risk_distance
        if risk_distance <= 0 or not signal.targets:
            return 0.0
        furthest = max(signal.targets) if signal.direction.is_buy else min(signal.targets)
        return abs(furthest - signal.entry_price) / risk_distance

    # ── emergency ────────────────────────────────────────────────────────────

    def halt(self, reason: str) -> None:
        self.state.halted = True
        self.state.halt_reason = reason
        logger.error("RISK HALT: %s", reason)

    def reset(self) -> None:
        """Clear a halt.  Manual only — anything that can clear itself is not a halt."""
        logger.warning("Risk halt cleared manually (was: %s)", self.state.halt_reason)
        self.state.halted = False
        self.state.halt_reason = ""
        self.state.peak_equity = 0.0


def _self_check() -> None:
    """Sizing is arithmetic; assert that 1% really is 1%."""
    from tradebot.domain import Direction

    def sig(stop: float, entry: float = 1.1000) -> Signal:
        return Signal("EURUSD", Direction.BUY, entry, stop, entry + 0.02)

    # Use an equity small enough that the absolute lot ceiling does not bind,
    # so these assertions test the sizing formula rather than the clamp.
    # 1% of 2,000 is 20. A 50-pip stop on EURUSD is $500/lot -> 0.04 lots.
    assert abs(position_size(sig(1.0950), 2_000.0) - 0.04) < 1e-9
    # A stop twice as wide risks the same money in half the size.
    assert abs(position_size(sig(1.0900), 2_000.0) - 0.02) < 1e-9
    # Volatility does not change the money at risk — that is the whole point.
    assert abs(
        position_size(sig(1.0950), 2_000.0) * 0.0050 * 100_000
        - position_size(sig(1.0900), 2_000.0) * 0.0100 * 100_000
    ) < 1e-6
    # The absolute ceiling binds, and binds DOWNWARD.  Risking less than
    # intended is safe; the clamp must never increase size.
    assert position_size(sig(1.0950), 1_000_000.0) == RISK.max_volume_per_trade
    # Degenerate inputs produce no trade rather than a guessed size.
    assert position_size(sig(1.1000), 10_000.0) == 0.0
    assert position_size(sig(1.0950), 0.0) == 0.0

    # A daily loss past the cap halts, and stays halted.
    rm = RiskManager()
    rm.state.day_start_equity = 10_000.0
    rm.state.peak_equity = 10_000.0
    ok, reason = rm.check(
        sig(1.0950), open_positions=[], balance=9_400.0, equity=9_400.0, spread=0.0001
    )
    assert ok is False and "daily loss" in reason, reason
    assert rm.state.halted is True

    # A spread eating a quarter of the risk is refused.
    rm2 = RiskManager()
    rm2.state.day_start_equity = 10_000.0
    rm2.state.peak_equity = 10_000.0
    ok, reason = rm2.check(
        sig(1.0950), open_positions=[], balance=10_000.0, equity=10_000.0, spread=0.0020
    )
    assert ok is False and "spread" in reason, reason

    print("risk.py self-check: OK")


if __name__ == "__main__":
    _self_check()
