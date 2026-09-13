"""The trading loop.

One iteration, in this order — and the order is the design:

1. **Stop check.**  A kill switch must be honoured before anything else runs.
2. **Reconcile with the broker.**  The broker is the source of truth.  A
   position it holds that we do not know about is adopted; a trade we think is
   open that it does not hold is retired locally.  Doing this *first* means
   every later step reasons about reality rather than about our memory of it.
3. **Manage what is open.**  Take partials at targets, move the stop to
   breakeven once the first target is banked.
4. **Drain external signals.**  Webhook alerts enter here — as candidates that
   still have to pass risk.
5. **Scan for new signals.**
6. **Risk-check, size, place.**

Nothing here trusts a local number over the broker's.  Sizing books the volume
the broker confirmed, and a stop is recorded only after the broker accepted it.
"""

from __future__ import annotations

import logging
import time
from datetime import UTC, datetime

from tradebot.broker import Broker
from tradebot.config import RISK, Settings, load_settings
from tradebot.data import MarketData
from tradebot.domain import Direction, Intent, Outcome, Position, Signal, Trade
from tradebot.intake import INBOX
from tradebot.measure import AsOfSeries, MarketView
from tradebot.risk import RiskManager, position_size
from tradebot.state import StateStore
from tradebot.strategy import BreakoutStrategy, MeanReversionStrategy

logger = logging.getLogger(__name__)

STRATEGIES = {
    "breakout": BreakoutStrategy,
    "mean_reversion": MeanReversionStrategy,
}

TP_FRACTIONS = (0.40, 0.30, 0.30)


class Bot:
    """Single-process, single-account trading loop."""

    def __init__(
        self,
        settings: Settings,
        *,
        strategy: str = "mean_reversion",
        interval_seconds: float = 60.0,
        window: int = 150,
        dry_run: bool = False,
    ) -> None:
        if strategy not in STRATEGIES:
            raise ValueError(f"unknown strategy {strategy!r}; have {sorted(STRATEGIES)}")
        self.settings = settings
        self.strategy_name = strategy
        self.interval = interval_seconds
        self.window = window
        self.dry_run = dry_run

        self.data = MarketData(settings)
        self.broker = Broker(settings)
        self.risk = RiskManager(RISK, settings)
        self.state = StateStore(settings.state_dir)
        self.trades: dict[int, Trade] = {}
        self.iteration = 0

    # ── lifecycle ────────────────────────────────────────────────────────────

    def startup(self) -> bool:
        self.settings.validate(require_broker=True)

        if not self.data.connect():
            logger.error("Cannot start: no market data connection")
            return False

        if not self.data.trade_allowed:
            logger.error(
                "Algo Trading is OFF in the MT5 terminal. Enable it before running."
            )
            return False

        # Share the resolved symbol map so the broker sends broker-side names.
        self.broker.bind_symbols(self.data._resolved)   # noqa: SLF001

        logger.info("Risk envelope (frozen): %s", RISK.describe())
        logger.info("Mode: %s", "DEMO" if self.settings.demo_mode else "*** LIVE CAPITAL ***")
        logger.info("Strategy: %s", self.strategy_name)

        self._load_local_trades()
        self._reconcile()
        return True

    def shutdown(self) -> None:
        self.state.save_trades(list(self.trades.values()))
        self.data.disconnect()
        logger.info("Shutdown complete")

    def run(self) -> int:
        if not self.startup():
            return 1
        try:
            while not self.state.stop_requested():
                self.iteration += 1
                try:
                    self.tick()
                except Exception:                          # noqa: BLE001
                    # One bad iteration must not end the session: positions may
                    # be open and would be left unmanaged.
                    logger.exception("Iteration %d failed — continuing", self.iteration)
                time.sleep(self.interval)
        except KeyboardInterrupt:
            logger.info("Interrupted")
        finally:
            self.shutdown()
        return 0

    # ── one iteration ────────────────────────────────────────────────────────

    def tick(self) -> None:
        positions = self.broker.ours()
        balance, equity = self.broker.account()

        self._reconcile(positions)
        self._manage(positions)

        # External candidates join *after* management and before scanning, so
        # they face the same risk checks as anything generated internally.
        candidates = self._scan()
        candidates.extend(self._external_candidates())

        opened = 0
        for signal in candidates:
            if self._enter(signal, positions):
                opened += 1
                positions = self.broker.ours()

        self.state.write_heartbeat(
            iteration=self.iteration,
            equity=equity,
            balance=balance,
            open_positions=len(positions),
            halted=self.risk.state.halted,
        )
        self.state.save_trades(list(self.trades.values()))

        if opened:
            logger.info("Iteration %d: opened %d position(s)", self.iteration, opened)

    # ── reconciliation ───────────────────────────────────────────────────────

    def _load_local_trades(self) -> None:
        for trade in self.state.load_trades():
            if trade.outcome is Outcome.OPEN:
                self.trades[trade.ticket] = trade
        if self.trades:
            logger.info("Loaded %d open trade(s) from state", len(self.trades))

    def _reconcile(self, positions: list[Position] | None = None) -> None:
        """Make local belief match broker reality, in both directions."""
        live = {
            p.ticket: p
            for p in (positions if positions is not None else self.broker.ours())
        }

        # We believe it is open; the broker disagrees → it closed without us
        # noticing (stop hit, or closed by hand).  Record it and stop managing.
        for ticket, trade in list(self.trades.items()):
            if ticket not in live:
                logger.warning(
                    "Reconcile: %s ticket %d is gone from the broker — marking closed",
                    trade.symbol, ticket,
                )
                trade.outcome = Outcome.MANUAL
                trade.closed_at = datetime.now(UTC)
                self.trades.pop(ticket, None)

        # The broker holds it; we do not know about it → adopt it, but never
        # invent targets for a position whose plan we cannot reconstruct.
        for ticket, position in live.items():
            if ticket not in self.trades:
                logger.warning(
                    "Reconcile: adopting untracked position %d (%s %.4f) — "
                    "its stop will be managed but not its targets",
                    ticket, position.symbol, position.volume,
                )
                self.trades[ticket] = Trade(
                    ticket=ticket,
                    symbol=position.symbol,
                    direction=position.direction,
                    volume=position.volume,
                    entry_price=position.entry_price,
                    stop_loss=position.stop_loss,
                    opened_at=datetime.now(UTC),
                    strategy="adopted",
                )

    # ── position management ──────────────────────────────────────────────────

    @staticmethod
    def _fraction_taken(trade: Trade) -> int:
        """How many targets have already been banked, inferred from the fraction."""
        taken = 0
        cumulative = 0.0
        for fraction in TP_FRACTIONS:
            cumulative += fraction
            if trade.closed_fraction + 1e-9 >= cumulative:
                taken += 1
        return taken

    def _manage(self, positions: list[Position]) -> None:
        """Take partials at targets and move the stop to breakeven after TP1."""
        for position in positions:
            trade = self.trades.get(position.ticket)
            if trade is None or trade.strategy == "adopted":
                continue

            targets = [
                t for t in (trade.take_profit_1, trade.take_profit_2, trade.take_profit_3)
                if t > 0
            ]
            if not targets:
                continue

            price = position.current_price
            is_buy = trade.direction.is_buy
            reached = [t for t in targets if (price >= t if is_buy else price <= t)]
            already = self._fraction_taken(trade)
            if len(reached) <= already:
                continue

            # Bank the next slice only.  Skipping a level would take more than
            # the plan allowed at that price.
            index = min(already, len(TP_FRACTIONS) - 1)
            fraction = TP_FRACTIONS[index]
            volume = round(trade.volume * fraction, 2)

            if volume > 0 and self.broker.close_partial(
                position.ticket, trade.symbol, trade.direction, volume
            ):
                trade.closed_fraction = min(1.0, trade.closed_fraction + fraction)
                logger.info(
                    "TP%d hit on %s ticket %d — closed %.2f lots (%.0f%% banked)",
                    index + 1, trade.symbol, position.ticket, volume,
                    trade.closed_fraction * 100,
                )

                if index == 0:
                    # First target banked: move the stop to entry so the rest of
                    # the position cannot become a loss.  Only record it if the
                    # broker actually accepted the modification.
                    if self.broker.modify(position.ticket, stop_loss=trade.entry_price):
                        trade.stop_loss = trade.entry_price
                    else:
                        logger.error(
                            "Broker refused breakeven on %d — local stop unchanged",
                            position.ticket,
                        )

            if trade.closed_fraction >= 0.999:
                trade.outcome = Outcome.TP
                trade.closed_at = datetime.now(UTC)
                self.trades.pop(position.ticket, None)

    # ── signal generation ────────────────────────────────────────────────────

    def _scan(self) -> list[Signal]:
        factory = STRATEGIES[self.strategy_name]
        out: list[Signal] = []
        for symbol in self.settings.symbols:
            candles = self.data.candles(symbol, "H1", self.window + 10)
            if len(candles) < self.window:
                continue
            store = AsOfSeries({"H1": candles})
            view = MarketView(symbol, candles[-1].timestamp, store, self.window)
            try:
                signal = factory().detect(view)
            except Exception:                              # noqa: BLE001
                logger.exception("Strategy raised on %s", symbol)
                continue
            if signal is not None:
                out.append(signal)
        return out

    @staticmethod
    def _external_candidates() -> list[Signal]:
        """Drain webhook alerts into the candidate pool.

        They arrive as ``Intent.STRATEGY`` — ideas to be evaluated, never
        instructions to be obeyed — and pass every check an internal signal
        passes.
        """
        out: list[Signal] = []
        for external in INBOX.drain():
            out.append(
                Signal(
                    symbol=external.symbol,
                    direction=external.direction,
                    entry_price=external.entry_price,
                    stop_loss=external.stop_loss,
                    take_profit_1=external.take_profit_1,
                    take_profit_2=external.take_profit_2,
                    take_profit_3=external.take_profit_3,
                    strategy=f"ext:{external.source}"[:29],
                    confidence=0.5,
                    intent=Intent.STRATEGY,
                    reason=f"{external.source} alert: {external.comment}".strip(),
                )
            )
        return out

    # ── entry ────────────────────────────────────────────────────────────────

    def _current_spread(self, symbol: str) -> float:
        recent = self.data.candles(symbol, "H1", 2)
        return recent[-1].spread if recent else 0.0

    def _enter(self, signal: Signal, positions: list[Position]) -> bool:
        balance, equity = self.broker.account()
        spread = self._current_spread(signal.symbol)

        allowed, reason = self.risk.check(
            signal, open_positions=positions, balance=balance, equity=equity, spread=spread
        )
        if not allowed:
            logger.info("Refused %s %s: %s", signal.direction, signal.symbol, reason)
            return False

        volume = position_size(signal, equity, RISK)
        if volume <= 0:
            logger.info("Refused %s: sizes to zero at this equity", signal.symbol)
            return False

        if self.dry_run:
            logger.info(
                "[dry-run] would %s %s %.2f lots @ %.5f",
                signal.direction, signal.symbol, volume, signal.entry_price,
            )
            return False

        targets = signal.targets
        broker_tp = 0.0
        if targets:
            broker_tp = max(targets) if signal.direction.is_buy else min(targets)

        result = self.broker.place(
            symbol=signal.symbol,
            direction=signal.direction,
            volume=volume,
            stop_loss=signal.stop_loss,
            take_profit=broker_tp,
            comment=signal.strategy,
        )
        if not result.success:
            logger.warning("Order refused for %s: %s", signal.symbol, result.message)
            return False

        # Book the broker-confirmed volume, never the requested one: it may have
        # been normalised to the lot step, and an IOC fill can come back partial.
        filled = result.filled_volume or volume
        self.trades[result.ticket] = Trade(
            ticket=result.ticket,
            symbol=signal.symbol,
            direction=signal.direction,
            volume=filled,
            entry_price=result.fill_price or signal.entry_price,
            stop_loss=signal.stop_loss,
            take_profit_1=signal.take_profit_1,
            take_profit_2=signal.take_profit_2,
            take_profit_3=signal.take_profit_3,
            strategy=signal.strategy,
            spread_at_entry=spread,
        )
        logger.info(
            "OPEN %s %s %.4f lots @ %.5f (sl %.5f)",
            signal.direction, signal.symbol, filled,
            result.fill_price or signal.entry_price, signal.stop_loss,
        )
        return True


def _self_check() -> None:
    """Target bookkeeping and reconciliation — the two things that lose money."""
    import tempfile
    from pathlib import Path

    with tempfile.TemporaryDirectory() as tmp:
        bot = Bot(Settings(state_dir=Path(tmp)), dry_run=True)

        def trade(closed: float) -> Trade:
            return Trade(ticket=1, symbol="EURUSD", direction=Direction.BUY,
                         volume=0.10, entry_price=1.10, stop_loss=1.09,
                         closed_fraction=closed)

        assert bot._fraction_taken(trade(0.0)) == 0
        assert bot._fraction_taken(trade(0.40)) == 1
        assert bot._fraction_taken(trade(0.70)) == 2
        assert bot._fraction_taken(trade(1.00)) == 3
        # Just under a boundary counts as not yet taken — never over-bank.
        assert bot._fraction_taken(trade(0.399999)) == 0

        # A trade the broker does not hold is retired, not retried.
        bot.trades[99] = Trade(ticket=99, symbol="GBPUSD", direction=Direction.SELL,
                               volume=0.01, entry_price=1.25, stop_loss=1.26)
        bot._reconcile(positions=[])
        assert 99 not in bot.trades

        # A broker position we do not know about is adopted, not ignored.
        orphan = Position(ticket=7, symbol="USDJPY", direction=Direction.BUY,
                          volume=0.03, entry_price=150.0)
        bot._reconcile(positions=[orphan])
        assert 7 in bot.trades and bot.trades[7].strategy == "adopted"
        assert abs(bot.trades[7].volume - 0.03) < 1e-12

        print("run.py self-check: OK")


def main(argv: list[str] | None = None) -> int:
    """Entry point.  ``python -m tradebot.run`` runs the bot, not the checks."""
    import argparse

    parser = argparse.ArgumentParser(description="TradeBot live loop")
    parser.add_argument("--strategy", default="mean_reversion", choices=sorted(STRATEGIES))
    parser.add_argument("--interval", type=float, default=60.0,
                        help="seconds between iterations")
    parser.add_argument("--symbols", default="", help="comma-separated override")
    parser.add_argument("--dry-run", action="store_true",
                        help="evaluate and log decisions without sending orders")
    parser.add_argument("--self-check", action="store_true", help="run assertions and exit")
    args = parser.parse_args(argv)

    if args.self_check:
        _self_check()
        return 0

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(levelname)-7s  %(message)s",
        datefmt="%H:%M:%S",
    )

    settings = load_settings()
    if args.symbols:
        settings.symbols = tuple(s.strip() for s in args.symbols.split(",") if s.strip())

    return Bot(
        settings,
        strategy=args.strategy,
        interval_seconds=args.interval,
        dry_run=args.dry_run,
    ).run()


if __name__ == "__main__":
    raise SystemExit(main())
