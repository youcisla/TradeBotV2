"""Configuration.

Two kinds of value live here, and conflating them is how risk limits quietly
stop being limits:

**Operator configuration** — credentials, server, symbol universe.  Read from
the environment.  Changing it changes *where the bot connects*.

**The risk envelope** — :data:`RISK`.  A frozen dataclass, hard-coded on
purpose and read from no environment variable.  Anything an operator can widen
at runtime, or an adaptive strategy layer can rewrite, is a parameter and not a
limit.  A self-tuning bot able to edit its own position cap does not have a
position cap — it has a suggestion it is free to ignore.

If you need to trade bigger, change the source and redeploy.  That friction is
the feature.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent


def _load_dotenv() -> None:
    """Populate os.environ from .env if python-dotenv is present."""
    try:
        from dotenv import load_dotenv
    except ImportError:
        return
    load_dotenv(REPO_ROOT / ".env")


_load_dotenv()


# ── The risk envelope.  FROZEN.  Not environment-driven.  Not writable. ──────

@dataclass(frozen=True, slots=True)
class RiskEnvelope:
    """Hard limits no code path may exceed at runtime."""

    risk_per_trade: float = 0.01          # 1% of equity risked per position
    max_daily_loss: float = 0.05          # halt new entries for the day
    max_drawdown: float = 0.20            # stop everything, require a human
    max_concurrent_positions: int = 5
    max_volume_per_trade: float = 0.10    # absolute lot ceiling
    min_reward_risk: float = 1.0          # refuse targets closer than 1R
    max_spread_fraction_of_risk: float = 0.25  # spread must not eat 1/4 of 1R

    def describe(self) -> str:
        return (
            f"risk/trade={self.risk_per_trade:.2%}  "
            f"daily-loss={self.max_daily_loss:.2%}  "
            f"max-DD={self.max_drawdown:.2%}  "
            f"max-positions={self.max_concurrent_positions}  "
            f"max-lots={self.max_volume_per_trade}"
        )


RISK = RiskEnvelope()


# ── Operator configuration (environment-driven) ──────────────────────────────

@dataclass(slots=True)
class BrokerConfig:
    account: str = ""
    password: str = ""
    server: str = ""
    # PU Prime appends suffixes to most instruments.  Verified against this
    # account's 1,109-symbol list: ".p" x88, ".24H" x62, ".s" x33, ".crp" x1.
    symbol_suffixes: tuple[str, ...] = (".crp", ".24H", ".p", ".s", ".r", "_r", "m")

    @property
    def configured(self) -> bool:
        return bool(self.account and self.password and self.server)


DEFAULT_SYMBOLS: tuple[str, ...] = (
    "EURUSD", "GBPUSD", "USDJPY", "AUDUSD",
    "USDCAD", "USDCHF", "NZDUSD", "EURJPY",
)


@dataclass(slots=True)
class Settings:
    broker: BrokerConfig = field(default_factory=BrokerConfig)
    demo_mode: bool = True
    symbols: tuple[str, ...] = DEFAULT_SYMBOLS
    magic: int = 202603
    state_dir: Path = field(default_factory=lambda: REPO_ROOT / "state")

    @classmethod
    def from_env(cls) -> "Settings":
        raw_symbols = os.environ.get("SYMBOLS", "").strip()
        symbols = tuple(s.strip() for s in raw_symbols.split(",") if s.strip())
        return cls(
            broker=BrokerConfig(
                account=os.environ.get("BROKER_ACCOUNT", "").strip(),
                password=os.environ.get("BROKER_PASSWORD", "").strip(),
                server=os.environ.get("BROKER_SERVER", "").strip(),
            ),
            demo_mode=os.environ.get("DEMO_MODE", "true").strip().lower()
            not in ("false", "0", "no"),
            symbols=symbols or DEFAULT_SYMBOLS,
        )

    @property
    def live(self) -> bool:
        """True when real capital is at risk.  Demo is the safe default."""
        return not self.demo_mode

    def validate(self, *, require_broker: bool = True) -> None:
        """Fail closed at the boundary instead of deferring the failure.

        Called at startup.  Missing credentials must stop the process with a
        clear message, not surface three layers down as an opaque IPC error
        once the loop is already running and positions may exist.
        """
        if require_broker and not self.broker.configured:
            missing = [
                name
                for name, value in (
                    ("BROKER_ACCOUNT", self.broker.account),
                    ("BROKER_PASSWORD", self.broker.password),
                    ("BROKER_SERVER", self.broker.server),
                )
                if not value
            ]
            raise RuntimeError(
                f"broker credentials missing: {', '.join(missing)} — set them in .env"
            )
        if not self.symbols:
            raise RuntimeError("symbol universe is empty")


def load_settings() -> Settings:
    return Settings.from_env()
