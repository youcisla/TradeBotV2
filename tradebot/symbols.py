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

from collections import defaultdict
from collections.abc import Iterable
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


# ── universe discovery ───────────────────────────────────────────────────────
#
# Ranking is by *relative cost* — spread as a fraction of price.  It is the one
# number comparable across instruments: 30 points on gold and 1 pip on EURUSD
# are both prices, and a raw spread cannot tell you which is the better market
# to trade.  Liquidity shows up here too, because a tight spread is what a
# competitive book looks like from the outside.


@dataclass(frozen=True, slots=True)
class Instrument:
    """One tradeable instrument as the broker actually lists it."""

    name: str        # broker's name, e.g. "EURUSD.p"
    base: str        # "EURUSD"
    category: str    # "Forex"
    group: str       # "Majors"
    price: float
    spread: float    # in price units, not points

    @property
    def relative_cost(self) -> float:
        if self.price <= 0:
            return float("inf")
        return self.spread / self.price


def category_of(path: str) -> tuple[str, str]:
    """Split MT5's symbol path into ``(category, group)``.

    MT5 hands back a Windows path like ``Forex\\Majors\\EURUSD.p``, but the
    separator is not guaranteed across builds, so both are accepted.
    """
    parts = [p for p in path.replace("\\", "/").split("/") if p]
    if not parts:
        return "Other", "Other"
    if len(parts) == 1:
        return parts[0], parts[0]
    return parts[0], parts[1]


def rank_universe(
    instruments: Iterable[Instrument], per_category: int = 10
) -> dict[str, list[Instrument]]:
    """Dedupe by base symbol, then keep the cheapest *per_category* per category.

    Two variants of one instrument (``EURUSD.p`` and ``EURUSD.24H``) trade the
    same underlying.  Holding both doubles the exposure to a single move and
    pays the cost of it twice, so only the cheaper listing survives.
    """
    cheapest: dict[str, Instrument] = {}
    for instrument in instruments:
        current = cheapest.get(instrument.base)
        if current is None or instrument.relative_cost < current.relative_cost:
            cheapest[instrument.base] = instrument

    by_category: dict[str, list[Instrument]] = defaultdict(list)
    for instrument in cheapest.values():
        by_category[instrument.category].append(instrument)

    return {
        category: sorted(items, key=lambda i: i.relative_cost)[:per_category]
        for category, items in sorted(by_category.items())
    }


def _self_check() -> None:
    """Dedupe, ranking and categorisation — the parts that silently mislead."""
    def inst(name: str, path: str, price: float, spread: float) -> Instrument:
        category, group = category_of(path)
        return Instrument(name, base_symbol(name), category, group, price, spread)

    # A path with no separator still categorises rather than crashing.
    assert category_of("") == ("Other", "Other")
    assert category_of("Forex\\Majors\\EURUSD.p") == ("Forex", "Majors")
    # The group is the broker's own path segment, kept verbatim.
    assert category_of("Crypto/BTCUSD.crp") == ("Crypto", "BTCUSD.crp")

    universe = rank_universe(
        [
            inst("EURUSD.p", "Forex\\Majors\\EURUSD.p", 1.1000, 0.00010),
            inst("EURUSD.s", "Forex\\Majors\\EURUSD.s", 1.1000, 0.00030),
            inst("GBPUSD.p", "Forex\\Minors\\GBPUSD.p", 1.2500, 0.00040),
            inst("XAUUSD.p", "Metals\\Gold\\XAUUSD.p", 2000.0, 0.30),
            inst("XAUUSD.crp", "Metals\\Gold\\XAUUSD.crp", 2000.0, 0.10),
            inst("BTCUSD.crp", "Crypto\\BTCUSD.crp", 60000.0, 60.0),
        ],
        per_category=10,
    )

    # One listing per underlying, and the cheaper one wins.
    assert [i.name for i in universe["Forex"]] == ["EURUSD.p", "GBPUSD.p"], universe["Forex"]
    assert [i.name for i in universe["Metals"]] == ["XAUUSD.crp"]
    assert set(universe) == {"Forex", "Metals", "Crypto"}

    # The cap applies per category, not to the whole universe.
    capped = rank_universe(
        [inst(f"AAA{i}USD.p", "Forex\\Majors\\AAA.p", 1.0, 0.001 * (i + 1)) for i in range(20)],
        per_category=10,
    )
    assert len(capped["Forex"]) == 10
    assert capped["Forex"][0].name == "AAA0USD.p", "cheapest must rank first"

    print("symbols.py self-check: OK")
