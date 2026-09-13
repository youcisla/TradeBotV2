"""Always-on supervision.

Runs the bot as a **subprocess**, never in-process: a crash, a segfault or a
memory blow-up inside the bot must not be able to take the supervisor with it.

What it does:

* restarts on crash, with exponential backoff, so a fast crash-loop cannot
  hammer the broker;
* detects a **hung** bot — one whose process is alive but which has stopped
  writing its heartbeat.  A hang is worse than a crash, because the process
  looks healthy while no stop is being managed;
* stops for good when the kill switch appears;
* gives up after too many restarts in an hour, because a bot that cannot stay
  up for minutes is not a bot, it is a bug with a loop around it.

A genuinely healthy run clears the crash streak, so a bot that dies after six
hours is not treated like one dying every ten seconds.
"""

from __future__ import annotations

import logging
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

from tradebot.state import StateStore

logger = logging.getLogger(__name__)


@dataclass(slots=True)
class SupervisorConfig:
    command: list[str] = field(default_factory=lambda: [sys.executable, "-m", "tradebot.run"])
    cwd: Path = field(default_factory=Path.cwd)
    state_dir: Path = field(default_factory=lambda: Path("state"))
    stale_after_s: float = 300.0
    min_uptime_s: float = 30.0
    max_restarts_per_hour: int = 6
    base_backoff_s: float = 5.0
    max_backoff_s: float = 300.0
    poll_interval_s: float = 5.0


class Supervisor:
    """Keeps one bot process alive."""

    def __init__(self, config: SupervisorConfig | None = None) -> None:
        self.cfg = config or SupervisorConfig()
        self.state = StateStore(self.cfg.state_dir)
        self._proc: subprocess.Popen | None = None
        self._restarts: list[float] = []
        self._started_at: float = 0.0

    # ── signals from outside ─────────────────────────────────────────────────

    def kill_requested(self) -> bool:
        return self.state.stop_requested()

    def is_hung(self) -> bool:
        """Alive but silent past the threshold.

        A *missing* heartbeat is not hung — the bot may simply not have started
        yet, and restarting it before it has had a chance to write one would
        loop forever on a slow first iteration.
        """
        age = self.state.heartbeat_age_seconds()
        return age is not None and age > self.cfg.stale_after_s

    # ── restart bookkeeping ──────────────────────────────────────────────────

    def _record_restart(self) -> None:
        now = time.monotonic()
        self._restarts.append(now)
        self._restarts = [t for t in self._restarts if now - t <= 3600.0]

    def budget_exhausted(self) -> bool:
        return len(self._restarts) >= self.cfg.max_restarts_per_hour

    def backoff_seconds(self) -> float:
        streak = len(self._restarts)
        if streak == 0:
            return 0.0
        return min(self.cfg.base_backoff_s * (2 ** (streak - 1)), self.cfg.max_backoff_s)

    def uptime(self) -> float:
        return time.monotonic() - self._started_at if self._proc else 0.0

    # ── process control ──────────────────────────────────────────────────────

    def start(self) -> None:
        self._proc = subprocess.Popen(  # noqa: S603
            self.cfg.command, cwd=str(self.cfg.cwd),
            stdout=sys.stdout, stderr=sys.stderr,
        )
        self._started_at = time.monotonic()
        logger.info("Supervisor: started pid=%s", self._proc.pid)

    def stop(self, *, graceful: bool = True) -> None:
        if self._proc is None or self._proc.poll() is not None:
            self._proc = None
            return
        logger.info("Supervisor: stopping pid=%s", self._proc.pid)
        try:
            if graceful:
                self._proc.terminate()
                try:
                    self._proc.wait(timeout=20)
                except subprocess.TimeoutExpired:
                    logger.warning("Supervisor: graceful stop timed out — killing")
                    self._proc.kill()
                    self._proc.wait(timeout=10)
            else:
                self._proc.kill()
                self._proc.wait(timeout=10)
        except Exception as exc:                               # noqa: BLE001
            logger.error("Supervisor: error stopping child (%s)", exc)
        finally:
            self._proc = None

    # ── loop ─────────────────────────────────────────────────────────────────

    def run(self) -> int:
        """Supervise until the kill switch appears or the budget runs out."""
        while True:
            if self.kill_requested():
                logger.warning("Supervisor: kill switch present — shutting down")
                self.stop(graceful=True)
                return 0

            self.start()

            while True:
                if self.kill_requested():
                    logger.warning("Supervisor: kill switch present — shutting down")
                    self.stop(graceful=True)
                    return 0

                code = self._proc.poll() if self._proc else 0
                if code is not None:
                    logger.error("Supervisor: child exited with code %s", code)
                    break

                if self.is_hung():
                    age = self.state.heartbeat_age_seconds()
                    logger.error(
                        "Supervisor: heartbeat stale (%.0fs > %.0fs) — child is hung; "
                        "restarting it",
                        age if age is not None else -1.0, self.cfg.stale_after_s,
                    )
                    self.stop(graceful=False)
                    break

                time.sleep(self.cfg.poll_interval_s)

            if self.uptime() >= self.cfg.min_uptime_s:
                # A healthy run clears the streak, so a late crash is not a loop.
                self._restarts.clear()

            self._record_restart()
            if self.budget_exhausted():
                logger.critical(
                    "Supervisor: %d restarts within an hour — giving up rather than "
                    "crash-looping. Inspect the logs, then remove %s to resume.",
                    self.cfg.max_restarts_per_hour, self.state.stop_path,
                )
                return 4

            wait = self.backoff_seconds()
            if wait:
                logger.info("Supervisor: backoff %.1fs before restart", wait)
                deadline = time.monotonic() + wait
                while time.monotonic() < deadline:
                    if self.kill_requested():
                        logger.warning("Supervisor: kill switch present — shutting down")
                        return 0
                    time.sleep(min(1.0, max(0.0, deadline - time.monotonic())))


def _self_check() -> None:
    """The two decisions that matter: hung detection and backoff."""
    import tempfile

    with tempfile.TemporaryDirectory() as tmp:
        cfg = SupervisorConfig(
            state_dir=Path(tmp), stale_after_s=10.0, min_uptime_s=0.0,
            base_backoff_s=5.0, max_backoff_s=60.0, max_restarts_per_hour=3,
        )
        sup = Supervisor(cfg)

        # No heartbeat yet: not hung — the bot may not have started.
        assert sup.state.heartbeat_age_seconds() is None
        assert sup.is_hung() is False
        assert sup.kill_requested() is False

        # Backoff before any restart: none, then exponential, then capped.
        assert sup.backoff_seconds() == 0.0
        sup._record_restart()
        assert sup.backoff_seconds() == 5.0
        sup._record_restart()
        assert sup.backoff_seconds() == 10.0
        sup._record_restart()
        assert sup.backoff_seconds() == 20.0
        assert sup.budget_exhausted() is True

        # A healthy run clears the streak.
        sup._restarts.clear()
        assert sup.budget_exhausted() is False and sup.backoff_seconds() == 0.0

        # A fresh heartbeat must not look hung.
        sup.state.write_heartbeat(iteration=1, equity=1.0, balance=1.0, open_positions=0)
        assert sup.is_hung() is False, "a just-written heartbeat must not look hung"

        sup.state.request_stop("test")
        assert sup.kill_requested() is True

        print("supervise.py self-check: OK")


if __name__ == "__main__":
    _self_check()
