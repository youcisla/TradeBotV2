"""Measurement CLI — run strategies over real MT5 history and report honestly.

    python scripts/measure.py
    python scripts/measure.py --strategies breakout,mean_reversion
    python scripts/measure.py --symbols EURUSD,GBPUSD --candles 20000
    python scripts/measure.py --optimistic        # sensitivity check ONLY

Every number printed is out-of-sample in the sense that matters: nothing here
tunes a parameter.  The holdout block is separated by time so it can stay
untouched while parameters are explored elsewhere.

Reporting uses pessimistic intrabar resolution (stop wins when a bar spans both
levels).  ``--optimistic`` exists to bound the result in the other direction —
if a strategy is still unprofitable under the most favourable assumption
available, no amount of intrabar detail will rescue it.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parent.parent
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from tradebot.config import RISK, load_settings  # noqa: E402
from tradebot.data import MarketData  # noqa: E402
from tradebot.measure import evaluate, run_series  # noqa: E402
from tradebot.strategy import BreakoutStrategy, MeanReversionStrategy  # noqa: E402

logging.basicConfig(
    level=logging.WARNING, format="%(asctime)s %(levelname)-7s %(message)s", datefmt="%H:%M:%S"
)
logger = logging.getLogger("measure")

STRATEGIES = {
    "breakout": BreakoutStrategy,
    "mean_reversion": MeanReversionStrategy,
}

RULE = "-" * 128


def main() -> int:
    parser = argparse.ArgumentParser(description="Measure strategies on real history")
    parser.add_argument("--strategies", default="breakout,mean_reversion")
    parser.add_argument("--symbols", default="")
    parser.add_argument("--candles", type=int, default=20_000)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--holdout", type=float, default=0.20)
    parser.add_argument("--optimistic", action="store_true",
                        help="Assume targets hit before stops (sensitivity only)")
    args = parser.parse_args()

    settings = load_settings()
    settings.validate(require_broker=True)
    symbols = tuple(s.strip() for s in args.symbols.split(",") if s.strip()) or settings.symbols
    wanted = [s.strip() for s in args.strategies.split(",") if s.strip()]

    data = MarketData(settings)
    if not data.connect():
        logger.error("MT5 connection failed — is the terminal running and logged in?")
        return 1

    print()
    print(f"Risk envelope (frozen): {RISK.describe()}")
    print(f"Symbols: {', '.join(symbols)}")
    print(f"Candles: {args.candles} H1 bars   Intrabar: "
          f"{'OPTIMISTIC (sensitivity)' if args.optimistic else 'pessimistic'}")
    print(RULE)

    try:
        for strategy_name in wanted:
            factory = STRATEGIES.get(strategy_name)
            if factory is None:
                logger.warning("Unknown strategy %r — skipping", strategy_name)
                continue

            per_symbol: dict[str, tuple[dict, list]] = {}
            for symbol in symbols:
                series = data.bundle(symbol, args.candles)
                if not series.get("H1"):
                    logger.warning("%s: no H1 data — skipping", symbol)
                    continue
                trades = run_series(
                    symbol, series, factory(), pessimistic=not args.optimistic
                )
                per_symbol[symbol] = (series, trades)

            if not per_symbol:
                print(f"\n{strategy_name}: no data for any symbol.")
                continue

            print(f"\n=== {strategy_name.upper()} ===")
            results = evaluate(
                per_symbol,
                risk_fraction=RISK.risk_per_trade,
                folds=args.folds,
                holdout_fraction=args.holdout,
            )
            for key, metrics in results.items():
                if key.startswith("fold") and metrics.positions == 0:
                    continue
                print(metrics.render())
    finally:
        if data.dropped_bars:
            print(f"\n({data.dropped_bars} malformed bars were quarantined)")
        data.disconnect()

    print()
    return 0


if __name__ == "__main__":
    sys.exit(main())
