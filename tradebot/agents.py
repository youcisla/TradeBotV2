"""Agent roster — declared authority, enforced by structure.

Multi-agent products popularised treating AI workers as employees: each gets a
role, a contract, and a fence for what it may not do alone.  The pattern is
sound.  The fence usually is not, because it lives in a prompt the model can
drift from, and drift is silent.

Here the fence is structural.  An agent declares the capabilities it needs, and
:func:`build_context` hands it *only* those — an undeclared capability arrives as
``None``, never as an empty list, so "not permitted" and "nothing to report"
cannot be confused.  No context carries a broker and no module here imports one,
so "the analyst cannot trade" is a property of the types rather than a promise
in prose.

Agents observe.  The trading loop decides.  That split is the whole point: an
agent that could place an order would be a second, unaudited path to the
broker, and the risk envelope would no longer be the only thing standing
between a wrong opinion and your money.
"""

from __future__ import annotations

import logging
from collections import Counter
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from enum import StrEnum
from pathlib import Path

from tradebot.config import Settings
from tradebot.domain import Candle, Outcome, Trade
from tradebot.symbols import spec as symbol_spec

logger = logging.getLogger(__name__)

# An H1 bar older than this means the feed, not the strategy, is the problem.
MAX_BAR_AGE_SECONDS = 6 * 3600


class Capability(StrEnum):
    """What an agent may touch.

    There is deliberately no order capability.  Adding one would require an
    order path on :class:`Context`, which every agent receives — so the absence
    is load-bearing, not an oversight.
    """

    READ_TRADES = "read_trades"
    READ_MARKET = "read_market"
    WRITE_REPORT = "write_report"


@dataclass(frozen=True)
class Role:
    """A job contract.

    ``owns`` / ``good_output`` / ``never`` mirror the four-part description a
    multi-agent product asks you to write.  The difference is that ``never`` is
    documentation of a structural fact, not the enforcement itself.
    """

    name: str
    title: str
    owns: str
    good_output: str
    never: tuple[str, ...]
    may: frozenset[Capability]


@dataclass
class Report:
    role: str
    title: str
    body: str
    generated_at: datetime = field(default_factory=lambda: datetime.now(UTC))


@dataclass
class Context:
    """Everything one agent can reach, assembled from its own declaration."""

    role: Role
    trades: list[Trade] | None = None
    closed: list[Trade] | None = None
    market: object | None = None


def build_context(
    role: Role,
    *,
    trades: list[Trade] | None = None,
    closed: list[Trade] | None = None,
    market: object | None = None,
) -> Context:
    """Hand *role* only what it declared.  This function is the fence."""
    reads = Capability.READ_TRADES in role.may
    return Context(
        role=role,
        trades=trades if reads else None,
        closed=closed if reads else None,
        market=market if Capability.READ_MARKET in role.may else None,
    )


def r_multiple(trade: Trade) -> float | None:
    """A result expressed in units of the risk the trade actually took.

    Raw currency is not comparable across instruments: 50 units made on gold
    and 50 lost on EURUSD are the same number and nothing like the same bet.
    R is what makes a ledger's trades add up to one story.

    The denominator is the **initial** stop, not the current one.  The bot moves
    the stop to breakeven once the first target is banked, so measuring from the
    live field would give zero risk to every trade that worked, return None, and
    leave the average R computed over the losers alone — a statistic that looks
    authoritative and describes the opposite of the truth.
    """
    stop = trade.initial_stop_loss or trade.stop_loss
    risk = abs(trade.entry_price - stop) * symbol_spec(trade.symbol).money_per_price_unit(
        trade.volume
    )
    if risk <= 0:
        return None
    return trade.pnl / risk


class Agent:
    """Base class.  Subclasses set ``role`` and implement ``run``."""

    role: Role

    def run(self, ctx: Context) -> Report:
        raise NotImplementedError

    def execute(self, ctx: Context) -> Report:
        """Run the agent, refusing if the context was not built for its role.

        A caller that forgets :func:`build_context` gets a loud failure rather
        than an agent quietly reading a capability it never declared.
        """
        if ctx.role.name != self.role.name:
            raise PermissionError(
                f"{self.role.name} was handed a context built for {ctx.role.name!r}"
            )
        return self.run(ctx)


# ── the roster ───────────────────────────────────────────────────────────────


class Analyst(Agent):
    """Reports what the bot actually did, from the trades it recorded."""

    role = Role(
        name="analyst",
        title="Performance Analyst",
        owns="Reading closed trades and reporting outcomes, symbols and hold times.",
        good_output="Counts a stranger could check by hand, plus an explicit list "
        "of what is not yet measurable.",
        never=(
            "never opens, closes or modifies a position",
            "never recommends a position size or a direction",
            "never reports a number the trade log does not support",
        ),
        may=frozenset({Capability.READ_TRADES, Capability.WRITE_REPORT}),
    )

    def run(self, ctx: Context) -> Report:
        if ctx.trades is None:
            raise PermissionError("analyst needs READ_TRADES")

        live = ctx.trades
        open_now = [t for t in live if t.outcome is Outcome.OPEN]
        closed = list(ctx.closed or [])
        recorded = [t for t in closed if t.pnl_recorded]
        unrecorded = [t for t in closed if not t.pnl_recorded]

        head = f"- ledger: **{len(closed)}** closed, **{len(open_now)}** open"
        if unrecorded:
            head += f", {len(unrecorded)} with no recoverable P&L"
        lines = [head]

        if recorded:
            wins = [t for t in recorded if t.pnl > 0]
            losses = [t for t in recorded if t.pnl < 0]
            gross_win = sum(t.pnl for t in wins)
            gross_loss = abs(sum(t.pnl for t in losses))

            lines.append(
                f"- win rate: **{len(wins) / len(recorded):.1%}** "
                f"({len(wins)}/{len(recorded)})"
            )
            lines.append(
                f"- profit factor: **{gross_win / gross_loss:.2f}**"
                if gross_loss
                else "- profit factor: no losing trade recorded yet"
            )
            lines.append(f"- net realised: **{sum(t.pnl for t in recorded):+.2f}**")

            rs = [r for r in (r_multiple(t) for t in recorded) if r is not None]
            if rs:
                lines.append(
                    f"- average R: **{sum(rs) / len(rs):+.2f}** over {len(rs)} trade(s)"
                )
                if len(rs) < len(recorded):
                    # Never hide the denominator: an average R over a subset
                    # reads exactly like an average R over everything.
                    lines.append(
                        f"- {len(recorded) - len(rs)} trade(s) left out of R — no "
                        "entry-to-stop distance on record to divide by"
                    )

            # Drawdown measured on the realised curve, which is the only curve
            # the ledger can defend.  Peak-to-trough on closed trades, so it
            # cannot be improved by simply holding a loser open.
            curve = peak = worst = 0.0
            for trade in sorted(recorded, key=lambda t: t.closed_at or t.opened_at):
                curve += trade.pnl
                peak = max(peak, curve)
                worst = max(worst, peak - curve)
            lines.append(f"- worst realised drawdown: **{worst:.2f}** from a running peak")

            by_symbol: Counter[str] = Counter()
            for trade in recorded:
                by_symbol[trade.symbol] += trade.pnl
            lines.append(
                "- net by symbol: "
                + ", ".join(f"{s} {v:+.2f}" for s, v in by_symbol.most_common())
            )

            outcomes = Counter(str(t.outcome) for t in recorded)
            lines.append(
                "- how they ended: "
                + ", ".join(f"{name} {n}" for name, n in outcomes.most_common())
            )
        else:
            lines.append("")
            lines.append("### Not yet measurable")
            if closed:
                lines.append(
                    "- closed trades exist but none carries a recorded result, so "
                    "there is no win rate, no average R and no profit factor to "
                    "report. A number invented here is the exact failure this "
                    "ledger exists to prevent."
                )
            else:
                lines.append(
                    "- the ledger is empty: nothing has closed yet, so there is "
                    "nothing to score. This is a count of zero trades, not a "
                    "failure to record them."
                )

        if unrecorded:
            lines.append("")
            lines.append(f"### {len(unrecorded)} trade(s) with no recoverable P&L")
            lines.append(
                "- excluded from every statistic above rather than counted as "
                "break-even, which is the flattering direction to be wrong in."
            )

        return Report(role=self.role.name, title=self.role.title, body="\n".join(lines))


class Sentinel(Agent):
    """Reports whether the market data is fit to trade on."""

    role = Role(
        name="sentinel",
        title="Data Sentinel",
        owns="Checking that every symbol resolves and its newest bar is current.",
        good_output="One line per symbol: bar count, age of the newest bar, verdict.",
        never=(
            "never opens, closes or modifies a position",
            "never substitutes an older bar for a missing one",
        ),
        may=frozenset({Capability.READ_MARKET, Capability.WRITE_REPORT}),
    )

    def __init__(self, settings: Settings) -> None:
        self._settings = settings

    def run(self, ctx: Context) -> Report:
        if ctx.market is None:
            raise PermissionError("sentinel has no market data in this context")

        market = ctx.market
        now = datetime.now(UTC)
        lines: list[str] = []
        stale: list[str] = []
        current: list[str] = []

        for symbol in self._settings.symbols:
            try:
                candles = market.candles(symbol, "H1", 2)
            except Exception as exc:                       # noqa: BLE001
                lines.append(f"- **{symbol}**: read failed — {exc}")
                stale.append(symbol)
                continue
            if not candles:
                lines.append(f"- **{symbol}**: no data (unresolved, or feed down)")
                stale.append(symbol)
                continue
            newest = candles[-1]
            age = (now - newest.timestamp).total_seconds()
            verdict = "stale" if age > MAX_BAR_AGE_SECONDS else "current"
            (stale if verdict == "stale" else current).append(symbol)
            lines.append(
                f"- **{symbol}**: {verdict}, newest H1 bar {age / 3600:.1f}h old "
                f"(close {newest.close})"
            )

        dropped = getattr(market, "dropped_bars", 0)
        if dropped:
            lines.append(f"- {dropped} bar(s) quarantined for failing validation")

        # Whether staleness is a fault depends on how many instruments show it.
        # A closed market and a dead feed look identical on any single symbol, so
        # judging one symbol in isolation would cry wolf every weekend — and a
        # warning that fires when nothing is wrong is one people learn to ignore.
        lines.append("")
        if not stale:
            lines.append("**Verdict: data current.**")
        elif not current:
            lines.append(
                "**Verdict: no instrument is streaming.** The market may be closed, or "
                "the feed may be down; nothing here can tell those apart, so check the "
                "terminal before trusting either."
            )
        else:
            lines.append(
                f"**Verdict: do not trade.** {', '.join(stale)} stale while "
                f"{', '.join(current)} is current — on a closed market every symbol "
                "stops together, so a split like this is a feed fault."
            )
        return Report(role=self.role.name, title=self.role.title, body="\n".join(lines))


class Coordinator(Agent):
    """Runs the roster and merges the reports."""

    role = Role(
        name="coordinator",
        title="Coordinator",
        owns="Running every agent once and merging what they report.",
        good_output="One report where each section is attributable and nothing is "
        "silently dropped.",
        never=(
            "never trades",
            "never edits or summarises away another agent's finding",
        ),
        may=frozenset({Capability.READ_TRADES, Capability.READ_MARKET}),
    )

    def __init__(self, agents: tuple[Agent, ...]) -> None:
        self.agents = agents

    def run(self, ctx: Context) -> Report:
        parts: list[str] = []
        for agent in self.agents:
            try:
                sub = agent.execute(
                    build_context(
                        agent.role,
                        trades=ctx.trades,
                        closed=ctx.closed,
                        market=ctx.market,
                    )
                )
            except Exception as exc:                       # noqa: BLE001
                # One broken agent must not suppress the others' findings.
                logger.exception("Agent %s failed", agent.role.name)
                parts.append(f"## {agent.role.title}\n\nFAILED: {exc}")
                continue
            parts.append(f"## {sub.title}\n\n{sub.body}")

        header = f"# Roster report — {datetime.now(UTC):%Y-%m-%d %H:%M} UTC"
        return Report(
            role=self.role.name,
            title=self.role.title,
            body=header + "\n\n" + "\n\n".join(parts),
        )


def write_report(report: Report, directory: Path) -> Path:
    """Persist a report, one file per day, so history is kept not overwritten."""
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"{report.generated_at:%Y-%m-%d}.md"
    path.write_text(report.body + "\n", encoding="utf-8")
    return path


class _StubMarket:
    """Minimal stand-in so the sentinel's three verdicts are testable."""

    def __init__(self, ages_hours: dict[str, float]) -> None:
        self._ages = ages_hours
        self.dropped_bars = 0

    def candles(self, symbol: str, timeframe: str = "H1", count: int = 2) -> list[Candle]:
        hours = self._ages.get(symbol)
        if hours is None:
            return []
        stamp = datetime.now(UTC) - timedelta(hours=hours)
        return [Candle(stamp, 1.0, 1.0, 1.0, 1.0)]


def sentinel_of(settings: Settings, ages: dict[str, float]) -> Report:
    """Run a sentinel over a stub market — used by the self-check and tests."""
    sentinel = Sentinel(settings)
    return sentinel.run(build_context(sentinel.role, market=_StubMarket(ages)))


def _trade() -> Trade:
    return Trade(
        ticket=1, symbol="EURUSD", direction="BUY", volume=0.01,
        entry_price=1.10, stop_loss=1.09, initial_stop_loss=1.09,
        opened_at=datetime(2026, 1, 7, tzinfo=UTC),
        outcome=Outcome.MANUAL, closed_at=datetime(2026, 1, 7, 4, tzinfo=UTC),
    )


def _self_check() -> None:
    """The fence: undeclared capabilities arrive absent, and cannot be faked."""
    analyst = Analyst()
    sentinel = Sentinel(Settings(symbols=("EURUSD",)))

    # An agent handed a context built for another role refuses to run.
    ctx = build_context(sentinel.role, market=object())
    try:
        analyst.execute(ctx)
        raise AssertionError("analyst accepted a sentinel context")
    except PermissionError:
        pass

    # The fence is absence, not emptiness: an undeclared capability is None, so
    # "not allowed" and "nothing to report" cannot be confused.
    sentinel_ctx = build_context(sentinel.role, trades=[_trade()], market=object())
    assert sentinel_ctx.trades is None, "sentinel was given trades it never declared"
    assert sentinel_ctx.market is not None

    analyst_ctx = build_context(analyst.role, trades=[])
    assert analyst_ctx.trades == [] and analyst_ctx.market is None

    # A capability the agent did not receive is a hard error, not an empty
    # report: an empty report reads as "all clear", which is the dangerous
    # direction to be wrong in.
    try:
        sentinel.run(build_context(sentinel.role))
        raise AssertionError("sentinel ran without READ_MARKET")
    except PermissionError:
        pass

    # No context ever carries a broker — the fence near money.
    assert not hasattr(analyst_ctx, "broker"), "a context exposed a broker"

    # With nothing recorded, the analyst says so rather than inventing a number.
    report = analyst.run(build_context(analyst.role, trades=[_trade()]))
    assert "Not yet measurable" in report.body, report.body
    # "nothing has closed" and "closed but unrecorded" are different problems
    # and must not read as the same sentence.
    assert "ledger is empty" in report.body, report.body
    unbooked = analyst.run(build_context(analyst.role, trades=[], closed=[_trade()]))
    assert "none carries a recorded result" in unbooked.body, unbooked.body

    # With a recorded result it produces real statistics.  EURUSD, 0.01 lots,
    # a 100-pip stop: 1R is 10.00, so +20.00 is +2R.
    booked = _trade()
    booked.pnl = 20.0
    booked.pnl_recorded = True
    assert abs(r_multiple(booked) - 2.0) < 1e-9, r_multiple(booked)

    # A stop dragged to breakeven must not erase the risk the trade took.  Read
    # from the live stop this is a division by zero, and every trade that went
    # well silently leaves the average — which then describes only the losers.
    trailed = _trade()
    trailed.pnl = -10.0
    trailed.pnl_recorded = True
    trailed.stop_loss = trailed.entry_price          # breakeven, as the bot sets it
    assert abs(r_multiple(trailed) + 1.0) < 1e-9, r_multiple(trailed)

    stats = analyst.run(build_context(analyst.role, trades=[], closed=[booked]))
    assert "win rate" in stats.body and "100.0%" in stats.body, stats.body
    assert "profit factor" in stats.body, stats.body
    assert "+20.00" in stats.body, stats.body

    # An unrecorded result is excluded from the statistics, not counted as a
    # break-even trade — counting it would flatter the win rate.
    blank = _trade()
    mixed = analyst.run(build_context(analyst.role, trades=[], closed=[booked, blank]))
    assert "1 trade(s) with no recoverable P&L" in mixed.body, mixed.body
    assert "100.0%" in mixed.body, mixed.body

    coordinator = Coordinator((analyst,))
    merged = coordinator.run(build_context(coordinator.role, trades=[_trade()]))
    assert "Performance Analyst" in merged.body

    # A closed market and a dead feed look the same on one symbol, so the
    # verdict must depend on the split across instruments. Every symbol stale
    # together is a closed market; one stale among fresh ones is a fault.
    pair = Settings(symbols=("EURUSD", "GBPUSD"))
    both_fresh = sentinel_of(pair, {"EURUSD": 0.1, "GBPUSD": 0.2})
    assert "data current" in both_fresh.body, both_fresh.body

    both_stale = sentinel_of(pair, {"EURUSD": 41.0, "GBPUSD": 41.0})
    assert "no instrument is streaming" in both_stale.body, both_stale.body

    split = sentinel_of(pair, {"EURUSD": 0.1, "GBPUSD": 41.0})
    assert "feed fault" in split.body, split.body
    assert "do not trade" in split.body, split.body

    # Structural guard. The whole safety argument rests on this module having no
    # route to the broker, so the day someone adds one the suite must object
    # rather than the reviewer having to notice. The needle is assembled at
    # runtime so this assertion's own text cannot satisfy the search.
    import inspect
    import sys

    forbidden = "tradebot" + ".broker"
    assert forbidden not in inspect.getsource(sys.modules[__name__]), (
        "agents.py reached the broker — the fence is gone"
    )
    assert not [c for c in Capability if "ORDER" in c.name], (
        "an order capability was added — agents may observe, never execute"
    )

    print("agents.py self-check: OK")


if __name__ == "__main__":
    _self_check()
