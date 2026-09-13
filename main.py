"""TradeBot entry point.

    python main.py run                  # trade the loop in the foreground
    python main.py forever              # trade under supervision (restarts on crash)
    python main.py run --dry-run        # decide and log, send no orders
    python main.py run --strategy breakout
    python main.py measure              # measure strategies on real history
    python main.py webhook              # run the loop plus the TradingView listener
    python main.py stop                 # set the kill switch
    python main.py clear                # remove the kill switch
    python main.py status               # heartbeat, queue, risk envelope
    python main.py check                # run every module self-check

One entry point on purpose: a bot with four ways to start is a bot whose
failure modes nobody can enumerate.
"""

from __future__ import annotations

import argparse
import logging
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

LOG_FORMAT = "%(asctime)s  %(levelname)-7s  %(message)s"
LOG_DATEFMT = "%H:%M:%S"

SELF_CHECK_MODULES = (
    "tradebot.measure",
    "tradebot.strategy",
    "tradebot.broker",
    "tradebot.risk",
    "tradebot.state",
    "tradebot.intake",
    "tradebot.supervise",
    "tradebot.replay",
    "tradebot.dashboard",
    "tradebot.run",
)

logger = logging.getLogger(__name__)


def _configure_logging(verbose: bool = False) -> None:
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format=LOG_FORMAT,
        datefmt=LOG_DATEFMT,
    )


def cmd_run(args: argparse.Namespace) -> int:
    from tradebot.run import main as run_main

    argv = ["--strategy", args.strategy, "--interval", str(args.interval)]
    if args.symbols:
        argv += ["--symbols", args.symbols]
    if args.dry_run:
        argv.append("--dry-run")
    return run_main(argv)


def cmd_forever(args: argparse.Namespace) -> int:
    """Supervise, so a crash never leaves positions unmanaged."""
    from tradebot.config import load_settings
    from tradebot.supervise import Supervisor, SupervisorConfig

    load_settings().validate(require_broker=True)

    command = [
        sys.executable, "-m", "tradebot.run",
        "--strategy", args.strategy,
        "--interval", str(args.interval),
    ]
    if args.symbols:
        command += ["--symbols", args.symbols]
    if args.dry_run:
        command.append("--dry-run")

    cfg = SupervisorConfig(
        cwd=REPO_ROOT,
        state_dir=REPO_ROOT / "state",
        stale_after_s=args.stale_after,
        command=command,
    )
    cfg.state_dir.mkdir(parents=True, exist_ok=True)
    return Supervisor(cfg).run()


def cmd_webhook(args: argparse.Namespace) -> int:
    """Run the loop with the TradingView listener alongside it."""
    import os

    from tradebot.config import load_settings
    from tradebot.intake import serve
    from tradebot.run import Bot

    secret = os.environ.get("TRADINGVIEW_WEBHOOK_SECRET", "").strip()
    serve(host=args.host, port=args.port, secret=secret)
    return Bot(
        load_settings(),
        strategy=args.strategy,
        interval_seconds=args.interval,
        dry_run=args.dry_run,
    ).run()


def cmd_measure(args: argparse.Namespace) -> int:
    script = REPO_ROOT / "scripts" / "measure.py"
    argv = [sys.executable, str(script)]
    if args.symbols:
        argv += ["--symbols", args.symbols]
    if args.strategy:
        argv += ["--strategies", args.strategy]
    return subprocess.call(argv, cwd=str(REPO_ROOT))  # noqa: S603


def cmd_stop(_: argparse.Namespace) -> int:
    from tradebot.state import StateStore

    StateStore(REPO_ROOT / "state").request_stop("operator")
    print("Kill switch set. A running bot stops at the top of its next iteration.")
    return 0


def cmd_clear(_: argparse.Namespace) -> int:
    from tradebot.state import StateStore

    StateStore(REPO_ROOT / "state").clear_stop()
    print("Kill switch cleared.")
    return 0


def cmd_status(_: argparse.Namespace) -> int:
    from tradebot.config import RISK, load_settings
    from tradebot.intake import INBOX
    from tradebot.state import StateStore

    settings = load_settings()
    store = StateStore(REPO_ROOT / "state")

    print(f"Risk envelope (frozen): {RISK.describe()}")
    print(f"Mode:        {'DEMO' if settings.demo_mode else '*** LIVE CAPITAL ***'}")
    print(f"Symbols:     {', '.join(settings.symbols)}")
    print(
        "Broker:      "
        + ("configured" if settings.broker.configured else "MISSING")
        + f"  ({settings.broker.server or 'no server'})"
    )
    print(
        "Kill switch: "
        + ("PRESENT — the bot will stop" if store.stop_requested() else "clear")
    )

    beat = store.heartbeat()
    age = store.heartbeat_age_seconds()
    if beat is None:
        print("Heartbeat:   none (bot not running)")
    else:
        halted = f"  HALTED: {beat.halted}" if beat.halted else ""
        print(
            f"Heartbeat:   {age:.0f}s ago  iteration={beat.iteration}  "
            f"equity={beat.equity:.2f}  open={beat.open_positions}{halted}"
        )

    print(f"Open trades: {len(store.load_trades())} tracked locally")
    print(f"Intake:      {INBOX.stats()}")
    return 0


def cmd_check(_: argparse.Namespace) -> int:
    """Run every module's self-check.  These assert the money rules."""
    failures = 0
    for module in SELF_CHECK_MODULES:
        result = subprocess.run(  # noqa: S603
            [sys.executable, "-c", f"import {module} as m; m._self_check()"],
            cwd=str(REPO_ROOT), capture_output=True, text=True,
        )
        if result.returncode == 0:
            print(f"  ok    {module}")
        else:
            failures += 1
            print(f"  FAIL  {module}")
            detail = (result.stderr.strip().splitlines() or ["unknown error"])[-1]
            print(f"        {detail}")
    print()
    print(
        "All self-checks passed."
        if not failures
        else f"{failures} self-check(s) failed."
    )
    return 1 if failures else 0


def cmd_replay(args: argparse.Namespace) -> int:
    """Drive the REAL bot over historical data — no orders reach the broker.

    History is fetched from MT5 once, then the bot runs against a cursor and a
    simulated broker.  Every rule the live bot enforces is enforced here, so
    this answers "would the bot have traded, and what would it have done?"
    rather than "is the strategy profitable?".
    """
    from tradebot.config import load_settings
    from tradebot.data import MarketData
    from tradebot.replay import run_replay
    from tradebot.run import Bot

    settings = load_settings()
    settings.validate(require_broker=True)
    # Replay must never touch live state.
    settings.state_dir = REPO_ROOT / "state" / "replay"

    symbols = [s.strip() for s in args.symbols.split(",") if s.strip()] or list(settings.symbols)
    settings.symbols = tuple(symbols)

    data = MarketData(settings)
    if not data.connect():
        logger.error("MT5 is required to fetch history for a replay")
        return 1
    series: dict[str, list] = {}
    try:
        for symbol in symbols:
            candles = data.candles(symbol, "H1", args.candles)
            if len(candles) >= args.start + 10:
                series[symbol] = candles
            else:
                logger.warning("%s: only %d bars — skipping", symbol, len(candles))
    finally:
        data.disconnect()

    if not series:
        logger.error("no usable history fetched")
        return 1

    bot = Bot(settings, strategy=args.strategy, interval_seconds=0.0)
    result = run_replay(bot, series, start=args.start, stride=args.stride)

    print()
    print(
        f"REPLAY — strategy '{args.strategy}' over {len(series)} symbol(s), "
        f"{args.candles} H1 bars"
    )
    print("-" * 108)
    print(result.render())
    print()
    print(
        "This exercises the bot's own logic (risk envelope, sizing, partials,\n"
        "breakeven, reconciliation).  It is not a profitability measure — use\n"
        "'main.py measure' for that, and treat neither as a substitute for a\n"
        "forward test on the demo account."
    )
    print()
    return 0


def cmd_dashboard(args: argparse.Namespace) -> int:
    """Serve the live monitor and keep trading alongside it.

    The dashboard is a view of the running bot, not a separate mode: if it were
    separate you would be watching something other than what trades.
    """
    import webbrowser

    from tradebot.config import load_settings
    from tradebot.dashboard import serve
    from tradebot.run import Bot

    bot = Bot(
        load_settings(),
        strategy=args.strategy,
        interval_seconds=args.interval,
        dry_run=args.dry_run,
    )
    serve(bot, host=args.host, port=args.port)
    url = f"http://{args.host}:{args.port}"
    print(f"Dashboard: {url}")
    if not args.no_browser:
        try:
            webbrowser.open(url)
        except Exception:                                      # noqa: BLE001
            pass
    return bot.run()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="main.py", description="TradeBot")
    parser.add_argument("-v", "--verbose", action="store_true")
    sub = parser.add_subparsers(dest="command", required=True)

    def add_trading_flags(p: argparse.ArgumentParser) -> None:
        p.add_argument("--strategy", default="mean_reversion")
        p.add_argument("--interval", type=float, default=60.0)
        p.add_argument("--symbols", default="")
        p.add_argument("--dry-run", action="store_true")

    run = sub.add_parser("run", help="trade in the foreground")
    add_trading_flags(run)
    run.set_defaults(func=cmd_run)

    forever = sub.add_parser("forever", help="trade under a supervisor")
    add_trading_flags(forever)
    forever.add_argument("--stale-after", type=float, default=300.0,
                         help="seconds without a heartbeat before a restart")
    forever.set_defaults(func=cmd_forever)

    webhook = sub.add_parser("webhook", help="trade plus the TradingView listener")
    add_trading_flags(webhook)
    webhook.add_argument("--host", default="127.0.0.1")
    webhook.add_argument("--port", type=int, default=8765)
    webhook.set_defaults(func=cmd_webhook)

    measure = sub.add_parser("measure", help="measure strategies on real history")
    measure.add_argument("--strategy", default="", help="comma-separated; blank = all")
    measure.add_argument("--symbols", default="")
    measure.set_defaults(func=cmd_measure)

    dash = sub.add_parser("dashboard", help="trade plus a live browser monitor")
    add_trading_flags(dash)
    dash.add_argument("--host", default="127.0.0.1")
    dash.add_argument("--port", type=int, default=8787)
    dash.add_argument("--no-browser", action="store_true")
    dash.set_defaults(func=cmd_dashboard)

    replay = sub.add_parser("replay", help="drive the real bot over historical data")
    replay.add_argument("--strategy", default="mean_reversion")
    replay.add_argument("--symbols", default="")
    replay.add_argument("--candles", type=int, default=5000, help="H1 bars to fetch")
    replay.add_argument("--start", type=int, default=200, help="first bar index to act on")
    replay.add_argument("--stride", type=int, default=1, help="bars advanced per step")
    replay.set_defaults(func=cmd_replay)

    sub.add_parser("stop", help="set the kill switch").set_defaults(func=cmd_stop)
    sub.add_parser("clear", help="clear the kill switch").set_defaults(func=cmd_clear)
    sub.add_parser("status", help="show state").set_defaults(func=cmd_status)
    sub.add_parser("check", help="run every self-check").set_defaults(func=cmd_check)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    _configure_logging(args.verbose)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
