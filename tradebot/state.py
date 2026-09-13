"""Runtime state: heartbeat, open trades, kill switch.

Everything lives in ``state/`` as JSON.  Writes go through a temp file and
``os.replace`` so a reader never sees a partial document — a half-written
heartbeat would look to the supervisor exactly like a hang and trigger a
needless restart, and a half-written trade file would lose positions that are
still live at the broker.

The kill switch is a file whose *presence* is the signal.  That makes it work
from any shell, any editor, any script, with no API and no permissions model to
get wrong.
"""

from __future__ import annotations

import json
import logging
import os
import time
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from tradebot.domain import Direction, Outcome, Trade

logger = logging.getLogger(__name__)

STOP_FILE = "STOP"
HEARTBEAT_FILE = "heartbeat.json"
TRADES_FILE = "trades.json"


def _atomic_write(path: Path, payload: dict[str, Any], attempts: int = 5) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    # Atomic on POSIX, but on Windows os.replace raises Access Denied while any
    # other handle holds the target — an antivirus scan, or another thread
    # reading it.  Retrying briefly is right; failing silently is not, because a
    # lost heartbeat looks exactly like a hung bot to the supervisor.
    for attempt in range(attempts):
        try:
            os.replace(tmp, path)
            return
        except PermissionError:
            if attempt == attempts - 1:
                logger.error("Could not replace %s after %d attempts", path, attempts)
                raise
            time.sleep(0.05 * (attempt + 1))


def _read_json(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:                               # noqa: BLE001
        logger.error("State file %s unreadable: %s", path, exc)
        return None


@dataclass(slots=True)
class Heartbeat:
    ts: str
    pid: int
    iteration: int
    equity: float
    balance: float
    open_positions: int
    halted: bool = False


class StateStore:
    """All runtime files for one bot process, rooted at a directory."""

    def __init__(self, directory: Path) -> None:
        self.dir = Path(directory)
        self.dir.mkdir(parents=True, exist_ok=True)

    # ── kill switch ──────────────────────────────────────────────────────────

    @property
    def stop_path(self) -> Path:
        return self.dir / STOP_FILE

    def stop_requested(self) -> bool:
        return self.stop_path.exists()

    def request_stop(self, reason: str = "requested") -> None:
        self.stop_path.write_text(f"stop requested: {reason}\n", encoding="utf-8")

    def clear_stop(self) -> None:
        self.stop_path.unlink(missing_ok=True)

    # ── heartbeat ────────────────────────────────────────────────────────────

    def write_heartbeat(
        self, *, iteration: int, equity: float, balance: float,
        open_positions: int, halted: bool = False,
    ) -> None:
        _atomic_write(
            self.dir / HEARTBEAT_FILE,
            asdict(
                Heartbeat(
                    ts=datetime.now(UTC).isoformat(),
                    pid=os.getpid(),
                    iteration=iteration,
                    equity=equity,
                    balance=balance,
                    open_positions=open_positions,
                    halted=halted,
                )
            ),
        )

    def heartbeat(self) -> Heartbeat | None:
        raw = _read_json(self.dir / HEARTBEAT_FILE)
        if not raw:
            return None
        try:
            return Heartbeat(**raw)
        except TypeError:
            return None

    def heartbeat_age_seconds(self) -> float | None:
        beat = self.heartbeat()
        if beat is None:
            return None
        try:
            written = datetime.fromisoformat(beat.ts)
            if written.tzinfo is None:
                written = written.replace(tzinfo=UTC)
            return (datetime.now(UTC) - written).total_seconds()
        except Exception:                                  # noqa: BLE001
            return None

    # ── trades ───────────────────────────────────────────────────────────────

    def save_trades(self, trades: list[Trade]) -> None:
        records = []
        for t in trades:
            row = asdict(t)
            row["direction"] = str(t.direction)
            row["outcome"] = str(t.outcome)
            row["opened_at"] = t.opened_at.isoformat()
            row["closed_at"] = t.closed_at.isoformat() if t.closed_at else None
            records.append(row)
        _atomic_write(
            self.dir / TRADES_FILE,
            {"saved_at": datetime.now(UTC).isoformat(), "trades": records},
        )

    def load_trades(self) -> list[Trade]:
        raw = _read_json(self.dir / TRADES_FILE)
        if not raw:
            return []
        out: list[Trade] = []
        for row in raw.get("trades", []):
            try:
                out.append(
                    Trade(
                        ticket=int(row["ticket"]),
                        symbol=str(row["symbol"]),
                        direction=Direction(row["direction"]),
                        volume=float(row["volume"]),
                        entry_price=float(row["entry_price"]),
                        stop_loss=float(row["stop_loss"]),
                        take_profit_1=float(row.get("take_profit_1", 0.0)),
                        take_profit_2=float(row.get("take_profit_2", 0.0)),
                        take_profit_3=float(row.get("take_profit_3", 0.0)),
                        opened_at=datetime.fromisoformat(row["opened_at"]),
                        outcome=Outcome(row.get("outcome", "OPEN")),
                        closed_at=(
                            datetime.fromisoformat(row["closed_at"])
                            if row.get("closed_at") else None
                        ),
                        pnl=float(row.get("pnl", 0.0)),
                        strategy=str(row.get("strategy", "")),
                        spread_at_entry=float(row.get("spread_at_entry", 0.0)),
                        closed_fraction=float(row.get("closed_fraction", 0.0)),
                    )
                )
            except Exception as exc:                       # noqa: BLE001
                logger.error("Skipping unreadable trade record %s: %s", row.get("ticket"), exc)
        return out


def _self_check() -> None:
    """Atomic write, round-trip, and kill-switch semantics."""
    import tempfile
    from datetime import timedelta

    with tempfile.TemporaryDirectory() as tmp:
        store = StateStore(Path(tmp))

        # No file yet: unknown age, not a stale zero.
        assert store.heartbeat_age_seconds() is None
        assert store.heartbeat() is None
        assert store.stop_requested() is False

        store.write_heartbeat(iteration=7, equity=1000.0, balance=1000.0, open_positions=2)
        beat = store.heartbeat()
        assert beat is not None and beat.iteration == 7 and beat.open_positions == 2
        age = store.heartbeat_age_seconds()
        assert age is not None and age < 5.0, age

        # No temp file survives a write.
        assert not list(Path(tmp).glob("*.tmp")), "temp file left behind"

        trade = Trade(
            ticket=123, symbol="EURUSD", direction=Direction.BUY, volume=0.05,
            entry_price=1.1050, stop_loss=1.1000, take_profit_1=1.1125,
            opened_at=datetime.now(UTC), strategy="mean_reversion",
        )
        store.save_trades([trade])
        loaded = store.load_trades()
        assert len(loaded) == 1
        assert loaded[0].ticket == 123
        assert loaded[0].direction is Direction.BUY
        assert loaded[0].outcome is Outcome.OPEN
        assert abs(loaded[0].volume - 0.05) < 1e-12
        assert abs(loaded[0].opened_at - trade.opened_at) < timedelta(seconds=1)

        store.request_stop("test")
        assert store.stop_requested() is True
        store.clear_stop()
        assert store.stop_requested() is False

        print("state.py self-check: OK")


if __name__ == "__main__":
    _self_check()
