"""MetaTrader 5 market data.

:class:`~tradebot.domain.Candle` is strict — it raises on an impossible bar.
That is correct for measurement, where one malformed bar means the whole result
is untrustworthy.  It is *wrong* for the live path, where a raise inside the
scan would kill the trading loop and leave open positions unmanaged.

So this module is the boundary, and it resolves the tension the only safe way:
validate every bar, and on failure log it loudly and drop it.  The type stays
strict; the session survives.
"""

from __future__ import annotations

import logging
import time
from datetime import UTC, datetime
from typing import Any, cast

import MetaTrader5 as _mt5

from tradebot.config import Settings
from tradebot.domain import Candle
from tradebot.symbols import (
    BROKER_SUFFIXES,
    Instrument,
    base_symbol,
    category_of,
    rank_universe,
)

mt5 = cast(Any, _mt5)

logger = logging.getLogger(__name__)

TIMEFRAMES: dict[str, int] = {
    "M1": mt5.TIMEFRAME_M1,
    "M5": mt5.TIMEFRAME_M5,
    "M15": mt5.TIMEFRAME_M15,
    "M30": mt5.TIMEFRAME_M30,
    "H1": mt5.TIMEFRAME_H1,
    "H4": mt5.TIMEFRAME_H4,
    "D1": mt5.TIMEFRAME_D1,
    "W1": mt5.TIMEFRAME_W1,
}

# A freshly subscribed symbol carries no tick until the terminal's feed catches
# up.  Two seconds is measured, not guessed: reading at zero seconds saw 161
# priced instruments where reading at two saw 1,021.
SETTLE_SECONDS = 2.0


def _quote(info: Any) -> float:
    """Price carried on a SymbolInfo snapshot, or 0.0 when it has none."""
    return float(getattr(info, "bid", 0.0) or 0.0) or float(getattr(info, "ask", 0.0) or 0.0)


class MarketData:
    """Thin, validated wrapper over the MT5 market-data IPC."""

    def __init__(self, settings: Settings) -> None:
        self._settings = settings
        self._resolved: dict[str, str] = {}
        self._point: dict[str, float] = {}
        self._dropped_bars = 0

    # ── connection ───────────────────────────────────────────────────────────

    def connect(self) -> bool:
        broker = self._settings.broker
        if not broker.configured:
            logger.error("MT5 connect refused: broker credentials missing")
            return False
        ok = mt5.initialize(
            login=int(broker.account), password=broker.password, server=broker.server
        )
        if not ok:
            logger.error("MT5 initialize failed: %s", mt5.last_error())
            return False
        info = mt5.account_info()
        if info is not None:
            logger.info(
                "MT5 connected: account=%s server=%s %s balance=%.2f",
                info.login, info.server, info.currency, info.balance,
            )
        return True

    def disconnect(self) -> None:
        try:
            mt5.shutdown()
        except Exception:                                  # noqa: BLE001
            pass

    @property
    def trade_allowed(self) -> bool:
        """False when 'Algo Trading' is off in the terminal — a hard blocker."""
        info = mt5.terminal_info()
        return bool(getattr(info, "trade_allowed", False)) if info else False

    @property
    def dropped_bars(self) -> int:
        """Count of bars quarantined for failing validation."""
        return self._dropped_bars

    # ── symbols ──────────────────────────────────────────────────────────────

    def resolve(self, symbol: str) -> str | None:
        """Map a clean symbol (EURUSD) to the broker's name (EURUSD.p).

        Selects the symbol in Market Watch, because MT5 returns no data for an
        instrument the terminal has not subscribed to — the silent cause of
        "no candles" on a symbol that plainly exists.
        """
        if symbol in self._resolved:
            return self._resolved[symbol]

        text = symbol.upper().strip()
        candidates = [text] + [f"{text}{s}" for s in BROKER_SUFFIXES]

        for name in candidates:
            info = mt5.symbol_info(name)
            if info is None:
                continue
            if not info.visible:
                mt5.symbol_select(name, True)
            if mt5.symbol_info_tick(name) is not None or mt5.copy_rates_from_pos(
                name, mt5.TIMEFRAME_H1, 0, 1
            ) is not None:
                self._resolved[symbol] = name
                self._point[symbol] = float(getattr(info, "point", 0.0) or 0.0)
                if name != text:
                    logger.info("Symbol resolved: %s -> %s", text, name)
                return name

        logger.warning("Symbol not resolvable on this broker: %s", text)
        return None

    # ── universe ─────────────────────────────────────────────────────────────

    def discover(
        self, per_category: int = 10, *, subscribe: bool = True
    ) -> dict[str, list[Instrument]]:
        """Rank every instrument the broker offers by cost to trade.

        The trap this exists to avoid: MT5 returns **no price at all** for a
        symbol nobody has subscribed to in Market Watch, so filtering on "has a
        price" quietly reduces the broker's 1,109-instrument book to whatever 63
        the terminal happened to have open.  Discovery would then report a
        universe shaped by the Market Watch panel rather than by the broker.

        Subscribing is not enough on its own either — a freshly subscribed
        symbol has no tick until the feed catches up, and reading immediately
        reported 161 instruments as tradeable when the real figure is 1,021.
        Hence the settle pause.

        Anything that does not survive ranking is unsubscribed again.  Only
        symbols this function subscribed are handed back, so the operator's own
        Market Watch is left exactly as it was found.
        """
        infos = mt5.symbols_get()
        if infos is None:
            logger.error("symbols_get failed: %s", mt5.last_error())
            return {}

        # Trade-disabled symbols are dropped *before* ranking: the cheapest
        # instrument in a category is meaningless if it cannot be traded, and it
        # would silently displace a usable one.
        tradeable = [
            s
            for s in infos
            if getattr(s, "trade_mode", 0) == mt5.SYMBOL_TRADE_MODE_FULL
            and float(getattr(s, "point", 0.0) or 0.0) > 0
        ]

        borrowed: list[str] = []
        if subscribe:
            for info in tradeable:
                if _quote(info) > 0 or getattr(info, "visible", False):
                    continue
                if mt5.symbol_select(str(info.name), True):
                    borrowed.append(str(info.name))
            if borrowed:
                time.sleep(SETTLE_SECONDS)

        # One fresh snapshot rather than a symbol_info call per symbol: the
        # same data, one IPC round trip instead of a thousand.
        fresh = {str(s.name): s for s in (mt5.symbols_get() or ())}

        found: list[Instrument] = []
        for info in tradeable:
            name = str(info.name)
            current = fresh.get(name, info)
            price = _quote(current) or _quote(info)
            if price <= 0:
                continue
            category, group = category_of(getattr(info, "path", "") or "")
            found.append(
                Instrument(
                    name=name,
                    base=base_symbol(name),
                    category=category,
                    group=group,
                    price=price,
                    spread=float(getattr(current, "spread", 0) or 0)
                    * float(getattr(info, "point", 0.0) or 0.0),
                )
            )

        universe = rank_universe(found, per_category=per_category)

        kept = {instrument.name for items in universe.values() for instrument in items}
        returned = 0
        for name in borrowed:
            if name not in kept:
                mt5.symbol_select(name, False)
                returned += 1

        logger.info(
            "Discovered %d priced instrument(s) across %d category/ies "
            "(%d subscribed to read, %d handed back)",
            len(found), len(universe), len(borrowed), returned,
        )
        return universe

    # ── candles ──────────────────────────────────────────────────────────────

    def candles(self, symbol: str, timeframe: str = "H1", count: int = 500) -> list[Candle]:
        """Return validated candles, newest last.  Bad bars are dropped, loudly."""
        name = self.resolve(symbol)
        if name is None:
            return []

        tf = TIMEFRAMES.get(timeframe.upper())
        if tf is None:
            logger.warning("Unknown timeframe %r", timeframe)
            return []

        rates = mt5.copy_rates_from_pos(name, tf, 0, count)
        if rates is None or len(rates) == 0:
            logger.warning("No candle data for %s %s: %s", name, timeframe, mt5.last_error())
            return []

        point = self._point.get(symbol, 0.0)
        has_spread = "spread" in getattr(rates.dtype, "names", ())

        out: list[Candle] = []
        for row in rates:
            raw_spread = float(row["spread"]) if has_spread else 0.0
            try:
                out.append(
                    Candle(
                        timestamp=datetime.fromtimestamp(row["time"], tz=UTC),
                        open=float(row["open"]),
                        high=float(row["high"]),
                        low=float(row["low"]),
                        close=float(row["close"]),
                        volume=float(row["tick_volume"]),
                        spread=round(raw_spread * point, 8) if point else 0.0,
                    )
                )
            except ValueError as exc:
                # One bad bar costs one bar, not the session.
                self._dropped_bars += 1
                logger.warning(
                    "Dropped bar for %s %s at %s: %s", name, timeframe, row["time"], exc
                )
        return out

    def bundle(self, symbol: str, h1_count: int = 20_000) -> dict[str, list[Candle]]:
        """Fetch every timeframe the strategies read, for one symbol.

        M15 is fetched from the broker, never resampled from H1: duplicating an
        H1 bar into four identical M15 bars fabricates intra-hour structure
        that never existed, and any M15 entry logic would then trade on it.
        """
        return {
            "H1": self.candles(symbol, "H1", h1_count),
            "H4": self.candles(symbol, "H4", h1_count // 4 + 250),
            "D1": self.candles(symbol, "D1", h1_count // 24 + 120),
            "M30": self.candles(symbol, "M30", h1_count * 2 + 250),
            "M15": self.candles(symbol, "M15", h1_count * 4 + 250),
        }
