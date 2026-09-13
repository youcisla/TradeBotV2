"""Instrument specifications.

Contract size and tick size are *inputs to measurement*, not trivia: without
them a price move cannot be converted into money, and expectancy is
uncomputable.  A backtest that does not know the contract size is measuring
pips, not pounds.

**Commission defaults to zero, and that is an assumption, not a fact.**  PU
Prime's standard accounts are spread-only on FX, so the spread carries the
whole cost — which is why `measure` charges the real per-bar spread on every
fill rather than treating costs as negligible.  If you move to a commission
account, set it here; leaving it at zero then would quietly overstate every
result.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class SymbolSpec:
    """Everything needed to turn a price move into account currency."""

    name: str
    contract_size: float          # units per 1.00 lot
    tick_size: float              # smallest price increment
    commission_per_lot: float = 0.0   # per side, in account currency

    def money_per_price_unit(self, volume: float) -> float:
        """P&L in account currency for a 1.00 price move at *volume* lots."""
        return self.contract_size * volume

    def round_trip_commission(self, volume: float) -> float:
        """Commission for opening and closing *volume* lots."""
        return 2.0 * self.commission_per_lot * volume


# 5-digit FX majors: 1 lot = 100,000 units, tick = 0.00001.
_FX_5DIGIT = ("EURUSD", "GBPUSD", "AUDUSD", "NZDUSD", "USDCAD", "USDCHF")
# 3-digit JPY crosses: 1 lot = 100,000 units, tick = 0.001.
_FX_JPY = ("USDJPY", "EURJPY", "GBPJPY", "AUDJPY", "CHFJPY", "CADJPY")

_REGISTRY: dict[str, SymbolSpec] = {}
for _name in _FX_5DIGIT:
    _REGISTRY[_name] = SymbolSpec(_name, 100_000.0, 0.00001)
for _name in _FX_JPY:
    _REGISTRY[_name] = SymbolSpec(_name, 100_000.0, 0.001)

# Metals and crypto.  Approximate contract sizes — verify against the broker
# before trusting a measurement of these, since PU Prime varies by contract.
_REGISTRY["XAUUSD"] = SymbolSpec("XAUUSD", 100.0, 0.01)
_REGISTRY["XAGUSD"] = SymbolSpec("XAGUSD", 5_000.0, 0.001)
_REGISTRY["BTCUSD"] = SymbolSpec("BTCUSD", 1.0, 0.01)
_REGISTRY["ETHUSD"] = SymbolSpec("ETHUSD", 1.0, 0.01)

DEFAULT_FX = SymbolSpec("DEFAULT", 100_000.0, 0.00001)

# Suffixes actually present on this broker's 1,109-symbol list, longest first
# so ".crp" is not partially consumed by ".p".
BROKER_SUFFIXES: tuple[str, ...] = (".crp", ".24H", ".p", ".s", ".r", "_r", "m")


def base_symbol(symbol: str, suffixes: tuple[str, ...] | None = None) -> str:
    """Strip a broker suffix and upper-case.

    Uses an explicit suffix list rather than ``str.rstrip``, which takes a
    *character set* — so it strips any trailing run of those characters and
    silently fails on suffixes whose letters fall outside the set.
    """
    text = symbol.upper().strip()
    for suffix in (suffixes if suffixes is not None else BROKER_SUFFIXES):
        if text.endswith(suffix.upper()) and len(text) > len(suffix):
            return text[: -len(suffix)]
    return text


def spec(symbol: str) -> SymbolSpec:
    """Return the spec for *symbol*, falling back to FX defaults.

    The fallback keeps measurement running on an unknown instrument, but it is
    a guess — use :func:`is_known` to refuse instead.
    """
    return _REGISTRY.get(base_symbol(symbol), DEFAULT_FX)


def is_known(symbol: str) -> bool:
    return base_symbol(symbol) in _REGISTRY


def known_symbols() -> tuple[str, ...]:
    return tuple(sorted(_REGISTRY))
