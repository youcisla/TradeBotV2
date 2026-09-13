"""Reconciliation — why do the two engines disagree?

    python scripts/reconcile.py

``measure.py`` scores the strategy and finds no edge.  ``replay.py`` drives the
real bot over the same bars and finishes well ahead.  Both cannot be right, and
"they model different things" is a description, not a resolution.

So this runs the measurement engine four times over the same history the replay
used, switching one difference on at a time.  If the gap is explained, the
variants converge on the ledger.  If it is not, the unexplained remainder is
itself the finding — and it is the number that decides whether this bot is
allowed near money.

The switches are sensitivity knobs only.  Every reported figure in
``scripts/measure.py`` still runs with both off.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parent.parent
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from tradebot.agents import r_multiple  # noqa: E402
from tradebot.config import RISK, load_settings  # noqa: E402
from tradebot.data import MarketData  # noqa: E402
from tradebot.measure import MAX_HOLD_BARS, evaluate, run_series  # noqa: E402
from tradebot.state import StateStore  # noqa: E402
from tradebot.strategy import MeanReversionStrategy  # noqa: E402

logging.basicConfig(
    level=logging.WARNING, format="%(asctime)s %(levelname)-7s %(message)s", datefmt="%H:%M:%S"
)
logger = logging.getLogger("reconcile")

RULE = "-" * 104

# One difference at a time, then every difference we know of at once.  The
# order is for reading, not for arithmetic: these are independent runs, not a
# cumulative pipeline.
LONG_HOLD = 10_000

VARIANTS: tuple[tuple[str, dict[str, object]], ...] = (
    ("measure, as reported", {"breakeven": False}),
    ("+ stop to breakeven after TP1", {"breakeven": True}),
    ("+ market entry instead of limit", {"breakeven": False, "market_entry": True}),
    ("+ no 48-bar hold limit",
     {"breakeven": False, "max_hold": LONG_HOLD, "horizon": MAX_HOLD_BARS}),
    ("+ all of the above",
     {"breakeven": True, "market_entry": True,
      "max_hold": LONG_HOLD, "horizon": MAX_HOLD_BARS}),
)


def ledger_stats(ledger: list) -> tuple[int, float, float, float]:
    """(n, win rate, average R, profit factor) for the replay's own record."""
    priced = [t for t in ledger if t.pnl_recorded]
    if not priced:
        return 0, 0.0, 0.0, 0.0

    wins = [t for t in priced if t.pnl > 0]
    gross_win = sum(t.pnl for t in wins)
    gross_loss = abs(sum(t.pnl for t in priced if t.pnl < 0))
    rs = [r for r in (r_multiple(t) for t in priced) if r is not None]

    return (
        len(priced),
        len(wins) / len(priced),
        sum(rs) / len(rs) if rs else 0.0,
        gross_win / gross_loss if gross_loss else float("inf"),
    )


def main() -> int:
    parser = argparse.ArgumentParser(description="Reconcile measure.py against replay.py")
    parser.add_argument("--symbols", default="EURUSD,GBPUSD")
    parser.add_argument("--candles", type=int, default=5000,
                        help="must match the replay run being compared against")
    args = parser.parse_args()

    settings = load_settings()
    settings.validate(require_broker=True)
    symbols = tuple(s.strip() for s in args.symbols.split(",") if s.strip())

    data = MarketData(settings)
    if not data.connect():
        logger.error("MT5 is required to fetch the history both engines used")
        return 1

    try:
        # The same call the replay makes, so both are reading identical bars.
        history = {
            symbol: {"H1": data.candles(symbol, "H1", args.candles)} for symbol in symbols
        }
    finally:
        data.disconnect()

    history = {s: series for s, series in history.items() if series["H1"]}
    if not history:
        logger.error("no H1 history fetched")
        return 1

    print()
    print("=== MEASUREMENT ENGINE, ONE DIFFERENCE AT A TIME ===")
    print(f"symbols: {', '.join(history)}   bars: {args.candles} H1   "
          f"strategy: mean_reversion   intrabar: pessimistic")
    print(RULE)
    print(f"{'variant':<34}{'n':>6}{'nofill':>8}{'win':>8}{'avgR':>9}{'PF':>7}{'maxDD':>8}")
    print(RULE)

    for label, flags in VARIANTS:
        per_symbol = {}
        for symbol, series in history.items():
            trades = run_series(symbol, series, MeanReversionStrategy(), **flags)
            per_symbol[symbol] = (series, trades)

        pooled = evaluate(per_symbol, risk_fraction=RISK.risk_per_trade)["pooled"]
        pf = "inf" if pooled.profit_factor == float("inf") else f"{pooled.profit_factor:.2f}"
        print(
            f"{label:<34}{pooled.positions:>6}{pooled.no_fills:>8}"
            f"{pooled.win_rate:>8.1%}{pooled.avg_r:>+9.3f}{pf:>7}"
            f"{pooled.max_drawdown:>8.1%}"
        )

    # The replay's own ledger, scored the same way.  Average R is per-trade and
    # therefore comparable across engines regardless of how many trades each took.
    ledger = StateStore(_REPO / "state" / "replay").load_closed()
    n, win, avg_r, pf = ledger_stats(ledger)
    print(RULE)
    if n:
        print(
            f"{'REPLAY LEDGER (actual record)':<34}{n:>6}{'-':>8}"
            f"{win:>8.1%}{avg_r:>+9.3f}{pf:>7.2f}{'-':>8}"
        )
    else:
        print(f"{'REPLAY LEDGER (actual record)':<34}{'empty — run main.py replay':>50}")
    print(RULE)

    print()
    print("Read the difference between the last two rows.  A variant that lands near")
    print("the ledger has explained the gap; anything still between them is")
    print("unexplained, and unexplained means unverified.")
    print()
    return 0


if __name__ == "__main__":
    sys.exit(main())
