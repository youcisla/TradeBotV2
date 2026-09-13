"""MetaTrader 5 order execution.

Every rule here comes from a failure observed against a live account, not from
reading the MT5 documentation.  They are the expensive part of this codebase.

* **Normalise volume once, then use the normalised value everywhere.**  MT5
  rewrites your requested volume to its own lot step.  The old code booked the
  *requested* size locally while the broker held a different one, so TP splits,
  commission and remaining-size arithmetic were all computed from a fiction.
  Here the normalised value is returned and is the only one that exists.

* **An ambiguous result is not a failure.**  ``order_send`` returning ``None``
  means the IPC timed out — the broker may well have accepted the request.
  Reporting failure then orphans a live position: real at the broker, invisible
  to the bot, with no stop being managed.  Ambiguity is resolved by querying
  positions by magic, never by assuming.

* **A partial close must name the volume.**  Omitting it closes everything, and
  the mistake is invisible in the logs, because a full close is also a
  "success".

* **Never record a stop the broker refused.**  If ``modify`` fails, the local
  record must not pretend otherwise — else the bot believes in protection the
  account does not hold.
"""

from __future__ import annotations

import logging
import time
from typing import Any, cast

import MetaTrader5 as _mt5

from tradebot.config import Settings
from tradebot.domain import Direction, OrderResult, Position

mt5 = cast(Any, _mt5)

logger = logging.getLogger(__name__)

MAGIC = 202603
DEVIATION = 20
CLOSE_ATTEMPTS = 3
CLOSE_BACKOFF_S = (0.25, 0.5, 1.0)


def normalize_volume(symbol: str, requested: float) -> float:
    """Snap a volume to the broker's min/max/step.  Returns 0.0 if impossible."""
    if requested <= 0:
        return 0.0

    info = mt5.symbol_info(symbol)
    if info is None:
        return round(requested, 2)

    step = float(getattr(info, "volume_step", 0.0) or 0.0)
    minimum = float(getattr(info, "volume_min", 0.0) or 0.0)
    maximum = float(getattr(info, "volume_max", 0.0) or 0.0)

    volume = requested
    if step > 0:
        volume = int(volume / step) * step
    if maximum > 0 and volume > maximum:
        volume = int(maximum / step) * step if step > 0 else maximum
    if minimum > 0 and volume < minimum:
        return 0.0
    if volume <= 0:
        return 0.0

    if step > 0:
        decimals = max(0, len(f"{step:.8f}".rstrip("0").split(".")[-1]))
        return round(volume, decimals)
    return round(volume, 2)


def _volume_min(symbol: str) -> float:
    info = mt5.symbol_info(symbol)
    return float(getattr(info, "volume_min", 0.0) or 0.0) if info else 0.0


def partial_volume(symbol: str, requested: float, open_volume: float) -> float:
    """Volume to send for a partial close.

    Never more than is open.  If the requested slice rounds below the broker
    minimum, or closing it would leave a remainder below that minimum, close
    the whole position instead — a dust remainder can never be closed later, so
    leaving one strands those lots permanently.
    """
    if requested <= 0 or open_volume <= 0:
        return 0.0

    volume = normalize_volume(symbol, min(requested, open_volume))
    if volume <= 0:
        return normalize_volume(symbol, open_volume)

    remainder = open_volume - volume
    minimum = _volume_min(symbol)
    if remainder > 1e-9 and 0.0 < remainder < minimum:
        logger.info(
            "Partial close of %s: closing all %.4f rather than %.4f (remainder %.4f "
            "is below the broker minimum %.4f)",
            symbol, open_volume, volume, remainder, minimum,
        )
        return normalize_volume(symbol, open_volume)
    return volume


class Broker:
    """Order entry, modification and closure against MT5."""

    def __init__(self, settings: Settings) -> None:
        self._settings = settings
        self._symbol_map: dict[str, str] = {}

    def bind_symbols(self, mapping: dict[str, str]) -> None:
        """Share the resolved symbol map (EURUSD -> EURUSD.p) from MarketData."""
        self._symbol_map.update(mapping)

    # ── queries ──────────────────────────────────────────────────────────────

    def positions(self) -> list[Position]:
        raw = mt5.positions_get()
        if not raw:
            return []
        out: list[Position] = []
        for p in raw:
            out.append(
                Position(
                    ticket=int(p.ticket),
                    symbol=p.symbol,
                    direction=Direction.BUY if p.type == mt5.ORDER_TYPE_BUY else Direction.SELL,
                    volume=float(p.volume),
                    entry_price=float(p.price_open),
                    stop_loss=float(p.sl or 0.0),
                    take_profit=float(p.tp or 0.0),
                    current_price=float(p.price_current),
                    profit=float(p.profit),
                    magic=int(getattr(p, "magic", 0) or 0),
                    comment=str(getattr(p, "comment", "") or ""),
                )
            )
        return out

    def ours(self) -> list[Position]:
        """Only positions carrying this bot's magic number.

        Never adopt a trade placed by hand or by another EA and start managing
        its stop — that is someone else's position.
        """
        return [p for p in self.positions() if p.magic == MAGIC]

    def account(self) -> tuple[float, float]:
        """(balance, equity).  Equity includes floating P&L; balance does not."""
        info = mt5.account_info()
        if info is None:
            return 0.0, 0.0
        return float(info.balance), float(info.equity)

    # ── order entry ──────────────────────────────────────────────────────────

    def place(
        self,
        symbol: str,
        direction: Direction,
        volume: float,
        stop_loss: float,
        take_profit: float = 0.0,
        comment: str = "tradebot",
    ) -> OrderResult:
        broker_symbol = self._symbol_map.get(symbol, symbol)
        mt5.symbol_select(broker_symbol, True)

        tick = mt5.symbol_info_tick(broker_symbol)
        if tick is None:
            return OrderResult(False, message=f"no tick for {broker_symbol}")

        volume = normalize_volume(broker_symbol, volume)
        if volume <= 0:
            return OrderResult(False, message="volume unusable after normalisation")

        info = mt5.symbol_info(broker_symbol)
        point = float(getattr(info, "point", 0.00001) or 0.00001)
        min_stop = max(float(getattr(info, "trade_stops_level", 0.0) or 0.0), 1.0) * point
        price = tick.ask if direction.is_buy else tick.bid

        # A stop the broker will reject is worse than no stop: it fails the
        # whole order.  Detect it here and say so plainly.
        if stop_loss > 0:
            if direction.is_buy and stop_loss >= price - min_stop:
                return OrderResult(False, message="stop too close to market for a BUY")
            if not direction.is_buy and stop_loss <= price + min_stop:
                return OrderResult(False, message="stop too close to market for a SELL")

        request = {
            "action": mt5.TRADE_ACTION_DEAL,
            "symbol": broker_symbol,
            "volume": volume,
            "type": mt5.ORDER_TYPE_BUY if direction.is_buy else mt5.ORDER_TYPE_SELL,
            "price": price,
            "sl": float(stop_loss) if stop_loss > 0 else 0.0,
            "deviation": DEVIATION,
            "magic": MAGIC,
            "comment": _safe_comment(comment),
            "type_time": mt5.ORDER_TIME_GTC,
            "type_filling": mt5.ORDER_FILLING_IOC,
        }
        if take_profit > 0:
            request["tp"] = float(take_profit)

        result = mt5.order_send(request)

        if result is None:
            return self._resolve_ambiguity(broker_symbol, direction)

        if result.retcode == mt5.TRADE_RETCODE_DONE_PARTIAL:
            # A real position exists at a smaller size.  Track it at that size.
            logger.warning(
                "Partial fill on %s: requested %.4f, filled %.4f",
                symbol, volume, result.volume,
            )
            return OrderResult(
                True,
                ticket=int(result.order),
                filled_volume=float(result.volume),
                fill_price=float(result.price),
                message=f"partial fill {result.volume}/{volume}",
            )

        if result.retcode != mt5.TRADE_RETCODE_DONE:
            return OrderResult(
                False, message=f"MT5 refused order ({result.retcode}): {result.comment}"
            )

        return OrderResult(
            True,
            ticket=int(result.order),
            filled_volume=float(result.volume),
            fill_price=float(result.price),
            message="ok",
        )

    def _resolve_ambiguity(self, broker_symbol: str, direction: Direction) -> OrderResult:
        """Find a position we may have opened despite an ambiguous order_send."""
        for position in self.ours():
            if position.symbol == broker_symbol and position.direction is direction:
                logger.error(
                    "order_send returned None for %s but position %d exists — adopting it",
                    broker_symbol, position.ticket,
                )
                return OrderResult(
                    True,
                    ticket=position.ticket,
                    filled_volume=position.volume,
                    fill_price=position.entry_price,
                    message="recovered after ambiguous order_send",
                    ambiguous=True,
                )
        logger.error("order_send returned None for %s: %s", broker_symbol, mt5.last_error())
        return OrderResult(
            False, message=f"order_send returned None: {mt5.last_error()}", ambiguous=True
        )

    # ── modification ─────────────────────────────────────────────────────────

    def modify(self, ticket: int, *, stop_loss: float | None = None,
               take_profit: float | None = None) -> bool:
        """Move a position's stop or target.  False means the broker refused —
        the caller must NOT record the new value locally."""
        found = mt5.positions_get(ticket=ticket)
        if not found:
            return False
        position = found[0]

        result = mt5.order_send({
            "action": mt5.TRADE_ACTION_SLTP,
            "position": ticket,
            "symbol": position.symbol,
            "sl": float(stop_loss) if stop_loss is not None else position.sl,
            "tp": float(take_profit) if take_profit is not None else position.tp,
        })
        if result is None or result.retcode != mt5.TRADE_RETCODE_DONE:
            logger.error(
                "modify failed for %d: %s",
                ticket, result.comment if result else mt5.last_error(),
            )
            return False
        return True

    # ── closure ──────────────────────────────────────────────────────────────

    def close(self, ticket: int, symbol: str, direction: Direction) -> bool:
        """Close an entire position.  Idempotent: already-gone counts as done."""
        found = mt5.positions_get(ticket=ticket)
        if not found:
            return True
        return self._send_close(ticket, symbol, direction, float(found[0].volume))

    def close_partial(self, ticket: int, symbol: str, direction: Direction,
                      volume: float) -> bool:
        """Close part of a position.  ``volume`` is honoured, never ignored."""
        found = mt5.positions_get(ticket=ticket)
        if not found:
            return True
        broker_symbol = found[0].symbol
        target = partial_volume(broker_symbol, volume, float(found[0].volume))
        if target <= 0:
            logger.error("partial close of %d: no usable volume from %.4f", ticket, volume)
            return False
        return self._send_close(ticket, broker_symbol, direction, target)

    def _send_close(self, ticket: int, symbol: str, direction: Direction,
                    volume: float) -> bool:
        broker_symbol = self._symbol_map.get(symbol, symbol)
        backoff = CLOSE_BACKOFF_S

        for attempt in range(1, CLOSE_ATTEMPTS + 1):
            if not mt5.positions_get(ticket=ticket):
                return True                       # gone: nothing left to close
            tick = mt5.symbol_info_tick(broker_symbol)
            if tick is None:
                time.sleep(backoff[min(attempt - 1, len(backoff) - 1)])
                continue

            result = mt5.order_send({
                "action": mt5.TRADE_ACTION_DEAL,
                "symbol": broker_symbol,
                "volume": volume,
                "type": mt5.ORDER_TYPE_SELL if direction.is_buy else mt5.ORDER_TYPE_BUY,
                "position": ticket,
                "price": tick.bid if direction.is_buy else tick.ask,
                "deviation": DEVIATION,
                "magic": MAGIC,
                "comment": _safe_comment("close"),
                "type_time": mt5.ORDER_TIME_GTC,
                "type_filling": mt5.ORDER_FILLING_IOC,
            })
            if result is not None and result.retcode == mt5.TRADE_RETCODE_DONE:
                logger.info("Closed %d (%.4f %s) on attempt %d",
                            ticket, volume, broker_symbol, attempt)
                return True
            logger.warning(
                "Close of %d attempt %d/%d failed: %s",
                ticket, attempt, CLOSE_ATTEMPTS,
                result.comment if result else mt5.last_error(),
            )
            if attempt < CLOSE_ATTEMPTS:
                time.sleep(backoff[min(attempt - 1, len(backoff) - 1)])

        logger.error("Gave up closing %d after %d attempts", ticket, CLOSE_ATTEMPTS)
        return False


def _safe_comment(text: str | None) -> str:
    """MT5 accepts a short ASCII comment; anything else is silently dropped."""
    cleaned = "".join(
        ch for ch in str(text or "").encode("ascii", "ignore").decode("ascii")
        if 32 <= ord(ch) <= 126
    )
    return (" ".join(cleaned.split()) or "tradebot")[:29]


def _self_check() -> None:
    """Volume rules are pure arithmetic; assert them without a broker."""
    assert partial_volume("NOPE", 0.0, 1.0) == 0.0
    assert partial_volume("NOPE", 1.0, 0.0) == 0.0
    # Without symbol info the helpers fall back to plain rounding.
    assert normalize_volume("NOPE", 0.5) == 0.5
    assert normalize_volume("NOPE", -1.0) == 0.0
    # A partial never exceeds what is open.
    assert partial_volume("NOPE", 99.0, 0.5) <= 0.5
    # Non-ASCII is DROPPED, not transliterated: "é" and "ö" vanish rather than
    # becoming "e" and "o".  MT5 rejects non-ASCII either way and the comment
    # is cosmetic, so dropping is correct — worth asserting so nobody later
    # "fixes" it into a transliteration dependency.
    assert _safe_comment("héllo wörld") == "hllo wrld", _safe_comment("héllo wörld")
    assert len(_safe_comment("x" * 100)) == 29
    assert _safe_comment("") == "tradebot"
    print("broker.py self-check: OK")


if __name__ == "__main__":
    _self_check()
