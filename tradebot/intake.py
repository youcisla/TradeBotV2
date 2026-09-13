"""External signal intake — TradingView alerts and other webhooks.

The HTTP listener runs on its own thread, so alerts cannot be handed to the
trading loop by direct call.  They land in a thread-safe bounded queue that the
loop drains once per iteration.

**An alert is a suggestion, not a command.**  Nothing here places an order.
Each alert becomes a candidate signal that travels the same risk checks as any
internally generated one.  Routing alerts straight to the executor would mean
anyone who learned the webhook URL could trade the account.

Security, because this listens on a network:

* authenticated by a shared secret, compared in constant time;
* **fail closed** — with no secret configured the endpoint refuses everything,
  rather than accepting anything it cannot authenticate;
* rate limited, so one misconfigured alert cannot flood the loop;
* answers 202 Accepted, never 200, and says "candidate" in the body, so the
  response cannot be misread as a fill confirmation.

Standard library only — the dependency list stays at MT5 and python-dotenv
rather than growing a web framework for a single endpoint.
"""

from __future__ import annotations

import hmac
import json
import logging
import threading
import time
from collections import deque
from dataclasses import dataclass, field
from datetime import UTC, datetime
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any

from tradebot.domain import Direction

logger = logging.getLogger(__name__)

MAX_QUEUE = 200
RATE_LIMIT = 30
RATE_WINDOW_S = 60.0
MAX_BODY_BYTES = 8192


@dataclass(slots=True)
class ExternalSignal:
    """A candidate from an outside source, awaiting risk review."""

    source: str
    symbol: str
    direction: Direction
    entry_price: float
    stop_loss: float = 0.0
    take_profit_1: float = 0.0
    take_profit_2: float = 0.0
    take_profit_3: float = 0.0
    comment: str = ""
    received_at: datetime = field(default_factory=lambda: datetime.now(UTC))

    def validate(self) -> tuple[bool, str]:
        """Strict, because these values come from outside the system."""
        if not self.symbol.strip():
            return False, "symbol is empty"
        if self.entry_price <= 0:
            return False, "entry_price must be positive"
        if self.stop_loss <= 0:
            return False, "stop_loss must be positive (an alert without a stop is not a trade)"

        buy = self.direction.is_buy
        if buy and self.stop_loss >= self.entry_price:
            return False, "BUY stop must sit below entry"
        if not buy and self.stop_loss <= self.entry_price:
            return False, "SELL stop must sit above entry"

        for i, level in enumerate(
            (self.take_profit_1, self.take_profit_2, self.take_profit_3), start=1
        ):
            if level <= 0:
                continue
            if buy and level <= self.entry_price:
                return False, f"BUY target {i} must sit above entry"
            if not buy and level >= self.entry_price:
                return False, f"SELL target {i} must sit below entry"
        return True, ""


class SignalInbox:
    """Thread-safe bounded queue.  The HTTP thread writes; the loop reads."""

    def __init__(self, maxlen: int = MAX_QUEUE) -> None:
        self._queue: deque[ExternalSignal] = deque(maxlen=maxlen)
        self._lock = threading.Lock()
        self._dropped = 0
        self._received = 0

    def submit(self, signal: ExternalSignal) -> bool:
        ok, reason = signal.validate()
        if not ok:
            logger.warning("Intake rejected %s %s: %s", signal.source, signal.symbol, reason)
            return False
        with self._lock:
            if len(self._queue) == self._queue.maxlen:
                self._dropped += 1
                logger.warning(
                    "Intake queue full (%d) — dropped %d so far; is an alert misconfigured?",
                    self._queue.maxlen, self._dropped,
                )
            self._queue.append(signal)
            self._received += 1
        logger.info(
            "Intake accepted %s %s %s @ %.5f",
            signal.source, signal.direction, signal.symbol, signal.entry_price,
        )
        return True

    def drain(self, max_items: int = 20) -> list[ExternalSignal]:
        out: list[ExternalSignal] = []
        with self._lock:
            while self._queue and len(out) < max_items:
                out.append(self._queue.popleft())
        return out

    def pending(self) -> int:
        with self._lock:
            return len(self._queue)

    def stats(self) -> dict[str, int]:
        with self._lock:
            return {
                "pending": len(self._queue),
                "received": self._received,
                "dropped": self._dropped,
            }


INBOX = SignalInbox()


def parse_tradingview(payload: dict[str, Any]) -> ExternalSignal:
    """Normalise a TradingView alert body into an ExternalSignal.

    Accepts the common alias spellings, so the alert message does not have to
    guess which one this endpoint wants.
    """
    raw_direction = str(payload.get("side") or payload.get("direction") or "").upper().strip()
    if raw_direction in ("LONG", "BUY"):
        direction = Direction.BUY
    elif raw_direction in ("SHORT", "SELL"):
        direction = Direction.SELL
    else:
        raise ValueError(f"unrecognised side/direction: {raw_direction!r}")

    def num(*keys: str) -> float:
        for key in keys:
            if payload.get(key) not in (None, ""):
                try:
                    return float(payload[key])
                except (TypeError, ValueError):
                    continue
        return 0.0

    return ExternalSignal(
        source="tradingview",
        symbol=str(payload.get("symbol") or "").upper().strip(),
        direction=direction,
        entry_price=num("entry", "entry_price", "price"),
        stop_loss=num("sl", "stop_loss", "stop"),
        take_profit_1=num("tp1", "take_profit_1", "tp"),
        take_profit_2=num("tp2", "take_profit_2"),
        take_profit_3=num("tp3", "take_profit_3"),
        comment=str(payload.get("comment") or payload.get("strategy") or "tradingview")[:29],
    )


class _WebhookHandler(BaseHTTPRequestHandler):
    """POST /tradingview only.  Everything else is refused."""

    secret: str = ""
    _recent: deque[float] = deque()
    _rate_lock = threading.Lock()

    server_version = "tradebot-webhook/1.0"

    def log_message(self, fmt: str, *args: Any) -> None:      # noqa: A002
        logger.debug("webhook: " + fmt, *args)

    def _reply(self, code: int, payload: dict[str, Any]) -> None:
        body = json.dumps(payload).encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    @classmethod
    def _rate_limited(cls) -> bool:
        now = time.monotonic()
        with cls._rate_lock:
            while cls._recent and now - cls._recent[0] > RATE_WINDOW_S:
                cls._recent.popleft()
            if len(cls._recent) >= RATE_LIMIT:
                return True
            cls._recent.append(now)
            return False

    def do_GET(self) -> None:                                  # noqa: N802
        if self.path.rstrip("/") == "/health":
            self._reply(200, {**INBOX.stats(), "armed": bool(self.secret)})
        else:
            self._reply(404, {"error": "not found"})

    def do_POST(self) -> None:                                 # noqa: N802
        if self.path.rstrip("/") != "/tradingview":
            self._reply(404, {"error": "not found"})
            return

        if not self.secret:
            # Cannot authenticate anyone → refuse everyone.
            self._reply(503, {"error": "webhook secret not configured"})
            return

        try:
            length = int(self.headers.get("Content-Length") or 0)
        except ValueError:
            length = 0
        if length <= 0 or length > MAX_BODY_BYTES:
            self._reply(413, {"error": "body missing or too large"})
            return

        try:
            payload = json.loads(self.rfile.read(length).decode("utf-8"))
            if not isinstance(payload, dict):
                raise ValueError("body must be a JSON object")
        except Exception as exc:                               # noqa: BLE001
            self._reply(400, {"error": f"bad JSON: {exc}"})
            return

        supplied = str(payload.get("secret") or "")
        if not hmac.compare_digest(supplied, self.secret):
            logger.warning("TradingView webhook rejected: bad secret")
            self._reply(401, {"error": "invalid secret"})
            return

        if self._rate_limited():
            self._reply(429, {"error": f"rate limit {RATE_LIMIT}/{RATE_WINDOW_S:.0f}s"})
            return

        try:
            signal = parse_tradingview(payload)
        except ValueError as exc:
            self._reply(422, {"error": str(exc)})
            return

        ok, reason = signal.validate()
        if not ok:
            self._reply(422, {"error": reason})
            return

        INBOX.submit(signal)
        self._reply(202, {
            "status": "queued",
            "note": "candidate only — subject to the risk envelope",
            "symbol": signal.symbol,
            "direction": str(signal.direction),
            "pending": INBOX.pending(),
        })


def serve(host: str = "127.0.0.1", port: int = 8765, secret: str = "") -> ThreadingHTTPServer:
    """Start the webhook listener on a background thread.  Returns the server."""
    handler = type("BoundWebhookHandler", (_WebhookHandler,), {"secret": secret})
    server = ThreadingHTTPServer((host, port), handler)
    threading.Thread(target=server.serve_forever, name="webhook", daemon=True).start()
    if not secret:
        logger.warning(
            "Webhook on %s:%d has NO SECRET — every alert will be refused until "
            "TRADINGVIEW_WEBHOOK_SECRET is set", host, port,
        )
    else:
        logger.info("Webhook listening on %s:%d", host, port)
    return server


def _self_check() -> None:
    """Validation is a trust boundary; assert it refuses bad alerts."""
    box = SignalInbox(maxlen=3)

    good = ExternalSignal("tradingview", "EURUSD", Direction.BUY, 1.1050,
                          stop_loss=1.1000, take_profit_1=1.1150)
    assert box.submit(good) is True and box.pending() == 1

    # A BUY stop above entry is nonsense.
    assert box.submit(ExternalSignal("tv", "EURUSD", Direction.BUY, 1.1050,
                                     stop_loss=1.1100)) is False
    # A BUY target at or below entry likewise.
    assert box.submit(ExternalSignal("tv", "EURUSD", Direction.BUY, 1.1050,
                                     stop_loss=1.1000, take_profit_1=1.1000)) is False
    # An alert with no stop is not a trade.
    assert box.submit(ExternalSignal("tv", "EURUSD", Direction.BUY, 1.1050)) is False

    # Bounded: oldest dropped, never grows.
    for _ in range(5):
        box.submit(ExternalSignal("tv", "GBPUSD", Direction.SELL, 1.25, stop_loss=1.26))
    assert box.pending() == 3, box.pending()
    assert box.stats()["dropped"] >= 2
    assert len(box.drain(max_items=2)) == 2 and box.pending() == 1

    # Parser accepts alias spellings and normalises direction.
    parsed = parse_tradingview({
        "symbol": "eurusd", "side": "long", "entry": "1.1050",
        "stop": "1.1000", "tp": "1.1150",
    })
    assert parsed.symbol == "EURUSD" and parsed.direction is Direction.BUY
    assert abs(parsed.stop_loss - 1.1000) < 1e-12
    assert parsed.validate()[0] is True

    try:
        parse_tradingview({"symbol": "EURUSD", "side": "sideways", "entry": 1.1})
        raise AssertionError("an unknown direction must raise")
    except ValueError:
        pass

    print("intake.py self-check: OK")


if __name__ == "__main__":
    _self_check()
