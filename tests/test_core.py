"""Tests for the invariants that decide money.

Organised by the rule each one protects, not by module.  A test here should
fail loudly if a change would let the bot lose money in a way the audit already
found once.
"""

from __future__ import annotations

import tempfile
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

from tradebot.broker import _safe_comment, normalize_volume, partial_volume
from tradebot.config import RISK, RiskEnvelope, Settings
from tradebot.domain import Candle, Direction, Intent, Outcome, Position, Signal, Trade
from tradebot.intake import ExternalSignal, SignalInbox, parse_tradingview
from tradebot.measure import AsOfSeries, MarketView, SimTrade, compute_metrics, simulate
from tradebot.risk import RiskManager, position_size
from tradebot.state import StateStore
from tradebot.strategy import BreakoutStrategy, MeanReversionStrategy, atr
from tradebot.symbols import base_symbol, is_known, spec

T0 = datetime(2026, 1, 7, 14, 0, tzinfo=UTC)


def bar(o: float, h: float, lo: float, c: float, offset: int = 0, spread: float = 0.0) -> Candle:
    return Candle(T0 + timedelta(hours=offset), o, h, lo, c, spread=spread)


def buy_signal(**kw) -> Signal:
    base = dict(symbol="EURUSD", direction=Direction.BUY, entry_price=1.1000,
                stop_loss=1.0950, take_profit_1=1.1050, strategy="test")
    base.update(kw)
    return Signal(**base)


# ── Candle: an impossible bar must never enter the system ────────────────────

class TestCandleValidation:
    def test_high_below_low_is_rejected(self):
        with pytest.raises(ValueError):
            bar(1.1, 1.0, 1.2, 1.1)

    def test_close_outside_range_is_rejected(self):
        with pytest.raises(ValueError):
            bar(1.1, 1.2, 1.0, 1.5)

    def test_open_below_low_is_rejected(self):
        with pytest.raises(ValueError):
            bar(0.9, 1.2, 1.0, 1.1)

    def test_flat_bar_is_valid(self):
        # No movement is legitimate in a thin market, not an error.
        assert bar(1.1, 1.1, 1.1, 1.1).close == 1.1

    def test_spread_defaults_to_zero(self):
        assert bar(1.1, 1.2, 1.0, 1.15).spread == 0.0


# ── Signal geometry: the trust-boundary check ────────────────────────────────

class TestSignalGeometry:
    def test_valid_buy_passes(self):
        assert buy_signal().geometry_error() is None

    def test_inverted_stop_is_caught(self):
        assert "below entry" in (buy_signal(stop_loss=1.1100).geometry_error() or "")

    def test_target_on_wrong_side_is_caught(self):
        assert "above entry" in (buy_signal(take_profit_1=1.0900).geometry_error() or "")

    def test_unordered_targets_are_caught(self):
        bad = buy_signal(take_profit_1=1.1150, take_profit_2=1.1100)
        assert "furthest-last" in (bad.geometry_error() or "")

    def test_empty_symbol_is_caught(self):
        assert "symbol" in (buy_signal(symbol="   ").geometry_error() or "")

    def test_sell_geometry_mirrors(self):
        sell = Signal("EURUSD", Direction.SELL, 1.1000, 1.1050,
                      take_profit_1=1.0950, take_profit_2=1.0900)
        assert sell.geometry_error() is None

    def test_signal_defaults_to_strategy_intent_not_execution(self):
        # Nothing is execution-grade unless the pipeline promotes it.
        assert buy_signal().intent is Intent.STRATEGY

    def test_risk_distance_is_entry_to_stop(self):
        assert abs(buy_signal().risk_distance - 0.0050) < 1e-12


# ── Symbols: the rstrip bug that silently blocked every real symbol ──────────

class TestSymbols:
    @pytest.mark.parametrize("raw,expected", [
        ("EURUSD.p", "EURUSD"),
        ("XAUUSD.crp", "XAUUSD"),
        ("BTCUSD.24H", "BTCUSD"),
        ("GBPUSD.s", "GBPUSD"),
        ("EURUSDm", "EURUSD"),
        ("EURUSD", "EURUSD"),
        ("eurusd", "EURUSD"),
    ])
    def test_suffixes_are_stripped(self, raw, expected):
        # rstrip() cannot do this: it takes a character set, so it either
        # over-strips or silently fails depending on the letters involved.
        assert base_symbol(raw) == expected

    def test_known_symbols_resolve(self):
        assert is_known("EURUSD") and is_known("EURUSD.p") and is_known("XAUUSD.crp")
        assert not is_known("NOTAPAIR")

    def test_fx_spec_values(self):
        eur = spec("EURUSD")
        assert eur.contract_size == 100_000.0 and eur.tick_size == 0.00001
        assert spec("USDJPY").tick_size == 0.001


# ── Measurement: every rule exists because its absence invalidated a result ──

class TestSimulate:
    def test_untouched_limit_is_not_a_trade(self):
        never = [bar(1.1060, 1.1070, 1.1055, 1.1065, i) for i in range(10)]
        trade = simulate(buy_signal(), T0, never)
        assert trade.outcome is Outcome.NO_FILL
        assert trade.is_counted is False

    def test_stop_hit_is_exactly_minus_one_r(self):
        trade = simulate(buy_signal(), T0, [bar(1.1000, 1.1005, 1.0940, 1.0950)])
        assert trade.outcome is Outcome.SL
        assert abs(trade.net_r + 1.0) < 1e-9

    def test_single_target_gives_full_r_not_a_fraction(self):
        trade = simulate(buy_signal(), T0, [bar(1.1000, 1.1060, 1.0995, 1.1055)])
        assert trade.outcome is Outcome.TP
        assert abs(trade.net_r - 1.0) < 1e-9

    def test_scale_out_is_40_30_30_not_all_at_the_last_target(self):
        signal = buy_signal(take_profit_1=1.1050, take_profit_2=1.1100, take_profit_3=1.1150)
        ladder = [
            bar(1.1000, 1.1050, 1.0999, 1.1049, 0),
            bar(1.1049, 1.1100, 1.1040, 1.1099, 1),
            bar(1.1099, 1.1150, 1.1090, 1.1149, 2),
        ]
        trade = simulate(signal, T0, ladder)
        assert abs(trade.net_r - (0.40 * 1 + 0.30 * 2 + 0.30 * 3)) < 1e-9

    def test_spread_is_charged_and_scales_with_risk(self):
        free = simulate(buy_signal(), T0, [bar(1.1000, 1.1060, 1.0995, 1.1055)])
        paid = simulate(buy_signal(), T0,
                        [bar(1.1000, 1.1060, 1.0995, 1.1055, spread=0.0002)])
        assert abs(paid.net_r - (free.net_r - 0.0002 / 0.0050)) < 1e-9

    def test_bar_spanning_both_levels_resolves_to_the_stop(self):
        # Intrabar order is unknowable from OHLC; the pessimistic reading is the
        # default because the optimistic one flatters every result.
        trade = simulate(buy_signal(), T0, [bar(1.1000, 1.1060, 1.0940, 1.1000)])
        assert trade.outcome is Outcome.SL

    def test_optimistic_mode_is_available_but_opt_in(self):
        trade = simulate(buy_signal(), T0, [bar(1.1000, 1.1060, 1.0940, 1.1000)],
                         pessimistic=False)
        assert trade.outcome is Outcome.TP

    def test_zero_risk_distance_is_invalid(self):
        trade = simulate(buy_signal(stop_loss=1.1000), T0, [bar(1.1, 1.1, 1.1, 1.1)])
        assert trade.outcome is Outcome.INVALID


class TestAsOf:
    def test_never_returns_a_future_bar(self):
        store = AsOfSeries({"H1": [bar(1, 1, 1, 1, i) for i in range(10)]})
        window = store.upto("H1", T0 + timedelta(hours=4), 100)
        assert len(window) == 5
        assert window[-1].timestamp == T0 + timedelta(hours=4)

    def test_unknown_timeframe_returns_nothing(self):
        store = AsOfSeries({"H1": [bar(1, 1, 1, 1, 0)]})
        assert store.upto("D1", T0, 10) == []


class TestMetrics:
    @staticmethod
    def _trade(net_r: float, hour: int) -> SimTrade:
        return SimTrade("EURUSD", "t", Direction.BUY, T0 + timedelta(hours=hour),
                        1.10, 1.09, [1.11], outcome=Outcome.TP, net_r=net_r)

    def test_no_fills_are_excluded_from_statistics(self):
        trades = [
            self._trade(1.0, 0),
            SimTrade("EURUSD", "t", Direction.BUY, T0, 1.1, 1.09, [],
                     outcome=Outcome.NO_FILL),
        ]
        m = compute_metrics(trades, risk_fraction=0.01)
        assert m.positions == 1 and m.no_fills == 1 and m.win_rate == 1.0

    def test_drawdown_and_sharpe_are_computed(self):
        trades = [self._trade(1.0, 0), self._trade(-1.0, 1), self._trade(-1.0, 2)]
        m = compute_metrics(trades, risk_fraction=0.01)
        assert 0.0 < m.max_drawdown < 1.0
        assert m.positions == 3

    def test_profit_factor_is_gross_wins_over_gross_losses(self):
        trades = [self._trade(2.0, 0), self._trade(-1.0, 1)]
        assert abs(compute_metrics(trades, risk_fraction=0.01).profit_factor - 2.0) < 1e-9


# ── Risk: the envelope must actually refuse things ───────────────────────────

class TestPositionSizing:
    def test_risk_is_exactly_the_envelope_fraction(self):
        # 0.5% of 2,000 = 10. A 50-pip stop on EURUSD is $500/lot -> 0.02 lots.
        # Equity is small so the absolute lot ceiling does not bind.
        assert abs(position_size(buy_signal(), 2_000.0) - 0.02) < 1e-9

    def test_wider_stop_gives_proportionally_smaller_size(self):
        assert abs(position_size(buy_signal(stop_loss=1.0900), 2_000.0) - 0.01) < 1e-9

    def test_money_at_risk_is_constant_across_volatility(self):
        tight = position_size(buy_signal(), 2_000.0) * 0.0050 * 100_000
        wide = position_size(buy_signal(stop_loss=1.0900), 2_000.0) * 0.0100 * 100_000
        assert abs(tight - wide) < 1e-6

    def test_lot_ceiling_only_ever_reduces_size(self):
        assert position_size(buy_signal(), 1_000_000.0) == RISK.max_volume_per_trade

    def test_degenerate_inputs_give_no_trade(self):
        assert position_size(buy_signal(stop_loss=1.1000), 10_000.0) == 0.0
        assert position_size(buy_signal(), 0.0) == 0.0


class TestRiskManager:
    @staticmethod
    def _manager() -> RiskManager:
        rm = RiskManager()
        rm.state.day_start_equity = 10_000.0
        rm.state.peak_equity = 10_000.0
        return rm

    def test_daily_loss_halts_and_stays_halted(self):
        rm = self._manager()
        ok, reason = rm.check(buy_signal(), open_positions=[], balance=9_400.0,
                              equity=9_400.0, spread=0.0001)
        assert ok is False and "daily loss" in reason
        assert rm.state.halted is True

    def test_drawdown_beyond_the_limit_halts(self):
        # Isolate the drawdown halt from the daily-loss halt by starting the day
        # already at the reduced equity, so only peak-to-now can fire.
        rm = RiskManager()
        rm.state.day_start_equity = 7_900.0
        rm.state.peak_equity = 10_000.0
        ok, reason = rm.check(buy_signal(), open_positions=[], balance=7_900.0,
                              equity=7_900.0, spread=0.0001)
        assert ok is False and "drawdown" in reason

    def test_a_one_to_one_trade_is_not_refused_by_floating_point(self):
        # (1.1050-1.1000)/(1.1000-1.0950) is 0.9999999999999778 in IEEE-754, so
        # a bare `< min_reward_risk` test rejected genuine 1:1 trades on the
        # rounding error alone.  This is the regression guard for that.
        rm = self._manager()
        signal = buy_signal(take_profit_1=1.1050)
        assert rm._reward_risk(signal) <= 1.0      # the ratio really is short
        ok, reason = rm.check(signal, open_positions=[], balance=10_000.0,
                              equity=10_000.0, spread=0.0001)
        assert ok is True, reason

    def test_spread_eating_a_quarter_of_risk_is_refused(self):
        rm = self._manager()
        ok, reason = rm.check(buy_signal(), open_positions=[], balance=10_000.0,
                              equity=10_000.0, spread=0.0020)
        assert ok is False and "spread" in reason

    def test_position_cap_is_enforced(self):
        rm = self._manager()
        held = [Position(ticket=i, symbol=f"SYM{i}", direction=Direction.BUY,
                         volume=0.01, entry_price=1.0) for i in range(10)]
        ok, reason = rm.check(buy_signal(), open_positions=held, balance=10_000.0,
                              equity=10_000.0, spread=0.0001)
        assert ok is False and "positions open" in reason

    def test_second_position_on_same_symbol_is_refused(self):
        rm = self._manager()
        held = [Position(ticket=1, symbol="EURUSD", direction=Direction.BUY,
                         volume=0.01, entry_price=1.1)]
        ok, reason = rm.check(buy_signal(), open_positions=held, balance=10_000.0,
                              equity=10_000.0, spread=0.0001)
        assert ok is False and "already holding" in reason

    def test_a_clean_trade_is_allowed(self):
        rm = self._manager()
        ok, reason = rm.check(buy_signal(), open_positions=[], balance=10_000.0,
                              equity=10_000.0, spread=0.0001)
        assert ok is True and reason == ""


# ── Broker: volume arithmetic is where the old code silently lied ────────────

class TestBrokerVolumes:
    def test_partial_never_exceeds_open_volume(self):
        assert partial_volume("NOPE", 99.0, 0.5) <= 0.5

    def test_zero_inputs_produce_no_order(self):
        assert partial_volume("NOPE", 0.0, 1.0) == 0.0
        assert partial_volume("NOPE", 1.0, 0.0) == 0.0
        assert normalize_volume("NOPE", -1.0) == 0.0

    def test_comments_are_ascii_and_bounded(self):
        # Non-ASCII is dropped, not transliterated.
        assert _safe_comment("héllo wörld") == "hllo wrld"
        assert len(_safe_comment("x" * 100)) == 29
        assert _safe_comment("") == "tradebot"


# ── State: a half-written file must never be readable ────────────────────────

class TestStateStore:
    def test_heartbeat_round_trip(self):
        with tempfile.TemporaryDirectory() as tmp:
            store = StateStore(Path(tmp))
            assert store.heartbeat() is None
            assert store.heartbeat_age_seconds() is None
            store.write_heartbeat(iteration=3, equity=1000.0, balance=990.0,
                                  open_positions=2)
            beat = store.heartbeat()
            assert beat is not None and beat.iteration == 3 and beat.open_positions == 2
            assert store.heartbeat_age_seconds() < 5.0

    def test_trade_round_trip_preserves_types(self):
        with tempfile.TemporaryDirectory() as tmp:
            store = StateStore(Path(tmp))
            store.save_trades([Trade(ticket=5, symbol="EURUSD", direction=Direction.SELL,
                                     volume=0.03, entry_price=1.10, stop_loss=1.11,
                                     opened_at=T0, strategy="mean_reversion")])
            loaded = store.load_trades()
            assert loaded[0].direction is Direction.SELL
            assert loaded[0].outcome is Outcome.OPEN
            assert abs(loaded[0].volume - 0.03) < 1e-12

    def test_no_temp_file_is_left_behind(self):
        with tempfile.TemporaryDirectory() as tmp:
            store = StateStore(Path(tmp))
            store.write_heartbeat(iteration=1, equity=1.0, balance=1.0, open_positions=0)
            assert not list(Path(tmp).glob("*.tmp"))

    def test_kill_switch_is_presence_only(self):
        with tempfile.TemporaryDirectory() as tmp:
            store = StateStore(Path(tmp))
            assert store.stop_requested() is False
            store.request_stop("test")
            assert store.stop_requested() is True
            store.clear_stop()
            assert store.stop_requested() is False


# ── Intake: a trust boundary must refuse malformed input ─────────────────────

class TestIntake:
    def test_valid_alert_is_accepted(self):
        box = SignalInbox(maxlen=5)
        assert box.submit(ExternalSignal("tv", "EURUSD", Direction.BUY, 1.1050,
                                         stop_loss=1.1000, take_profit_1=1.1150)) is True
        assert box.pending() == 1

    def test_alert_without_a_stop_is_refused(self):
        assert SignalInbox().submit(
            ExternalSignal("tv", "EURUSD", Direction.BUY, 1.1050)) is False

    def test_inverted_stop_is_refused(self):
        assert SignalInbox().submit(
            ExternalSignal("tv", "EURUSD", Direction.BUY, 1.1050, stop_loss=1.1100)) is False

    def test_queue_is_bounded_and_reports_drops(self):
        box = SignalInbox(maxlen=2)
        for _ in range(5):
            box.submit(ExternalSignal("tv", "GBPUSD", Direction.SELL, 1.25, stop_loss=1.26))
        assert box.pending() == 2
        assert box.stats()["dropped"] == 3

    def test_parser_accepts_aliases_and_case(self):
        parsed = parse_tradingview({"symbol": "eurusd", "side": "long", "entry": "1.1050",
                                    "stop": "1.1000", "tp": "1.1150"})
        assert parsed.symbol == "EURUSD"
        assert parsed.direction is Direction.BUY
        assert abs(parsed.stop_loss - 1.1000) < 1e-12

    def test_parser_rejects_an_unknown_side(self):
        with pytest.raises(ValueError):
            parse_tradingview({"symbol": "EURUSD", "side": "sideways", "entry": 1.1})


# ── Strategies: fire beyond the band, stay quiet inside it ───────────────────

class TestStrategies:
    @staticmethod
    def _view(candles: list[Candle]) -> MarketView:
        return MarketView("EURUSD", candles[-1].timestamp, AsOfSeries({"H1": candles}))

    def test_atr_of_a_constant_range(self):
        candles = [bar(1.10, 1.11, 1.10, 1.105, i) for i in range(20)]
        assert abs(atr(candles, 14) - 0.01) < 1e-9

    def test_quiet_market_fires_nothing(self):
        flat = [bar(1.1000, 1.1010, 1.0990, 1.1000, i, spread=0.0001) for i in range(80)]
        view = self._view(flat)
        assert BreakoutStrategy(lookback=55).detect(view) is None
        assert MeanReversionStrategy(lookback=40).detect(view) is None

    def test_breakout_goes_long_above_the_band_and_fade_goes_short(self):
        flat = [bar(1.1000, 1.1010, 1.0990, 1.1000, i, spread=0.0001) for i in range(80)]
        hot = flat[:-1] + [bar(1.1000, 1.1080, 1.0995, 1.1075, 79, spread=0.0001)]
        view = self._view(hot)

        long = BreakoutStrategy(lookback=55).detect(view)
        short = MeanReversionStrategy(lookback=40).detect(view)
        assert long is not None and long.direction is Direction.BUY
        assert short is not None and short.direction is Direction.SELL
        assert long.geometry_error() is None and short.geometry_error() is None

    def test_strategy_refuses_when_the_stop_would_sit_inside_the_spread(self):
        # A 3-pip range with a 1.5x ATR stop is ~4.5 pips of risk; a 9-pip
        # spread makes the trade a guaranteed loss, so it must not be proposed.
        tight = [bar(1.1000, 1.1003, 1.0997, 1.1000, i, spread=0.0009) for i in range(80)]
        assert BreakoutStrategy(lookback=55).detect(self._view(tight)) is None


# ── Config: the envelope is immutable and validating ─────────────────────────

class TestConfig:
    def test_envelope_is_frozen(self):
        from dataclasses import FrozenInstanceError
        with pytest.raises(FrozenInstanceError):
            RISK.risk_per_trade = 0.99          # type: ignore[misc]

    def test_a_custom_envelope_is_also_frozen(self):
        from dataclasses import FrozenInstanceError
        with pytest.raises(FrozenInstanceError):
            RiskEnvelope().risk_per_trade = 0.5  # type: ignore[misc]

    def test_validate_raises_when_credentials_are_missing(self):
        with pytest.raises(RuntimeError, match="credentials missing"):
            Settings().validate(require_broker=True)

    def test_validate_can_be_skipped_for_pure_measurement(self):
        Settings().validate(require_broker=False)

    def test_demo_is_the_safe_default(self):
        assert Settings().live is False
