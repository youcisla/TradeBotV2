# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

An autonomous MetaTrader 5 trading bot. Windows-only, single process, MT5-only — `MetaTrader5` publishes wheels only up to Python 3.13, and `pyproject.toml` pins `>=3.11,<3.14` for that reason. On 3.14 the broker package cannot install at all.

This is a rebuild. The prior system (~48k lines, nine agents, LSTM/LightGBM/GARCH/HMM, an LLM council) could not connect to a broker — `MT5Connector` never implemented its abstract methods, so every live path raised `TypeError` while the mock-based test suite stayed green. This rebuild inverts the order: **measurement first, automation second.** Read `README.md` for the full post-mortem.

The design contract (`tradebot/__init__.py`) is the spine of every decision here:

1. **Measurement before automation** — no strategy is trusted that has not survived an out-of-sample test with real costs.
2. **The risk envelope is frozen** — `tradebot.config.RISK`, hard-coded, no env var, no setter.
3. **The broker is the source of truth** — local records describe what the broker holds; they never assert it.
4. **External input is a suggestion** — webhook signals enter as `Intent.STRATEGY` and pass the same gate as internal ones.
5. **Fail closed** — missing config, a crashed check, or an ambiguous order result stops progress, never permits it.

## Commands

```bash
pip install -e ".[dev]"        # dev extras (ruff, mypy) are NOT installed by default

python -m pytest               # 69 tests, ~0.2s
python -m pytest tests/test_core.py::TestRiskManager::test_daily_loss_halts_and_stays_halted   # one test

python main.py check           # runs every module's _self_check() — the money-rule assertions
python main.py run             # trade in the foreground
python main.py run --dry-run   # decide and log, send no orders
python main.py forever         # trade under the supervisor (restarts on crash)
python main.py dashboard       # trade + live browser monitor on 127.0.0.1:8787
python main.py webhook         # trade + TradingView listener on 127.0.0.1:8765
python main.py status          # heartbeat, kill switch, envelope
python main.py stop            # set the kill switch (bot stops at top of next iteration)
python main.py clear           # clear it

python main.py measure         # measure strategies over real MT5 history (needs a live connection)
python main.py replay          # drive the REAL bot over historical bars; writes state/replay/
python scripts/reconcile.py    # size the gap between the measurement engine and the replay ledger
```

`ruff` and `mypy` are configured in `pyproject.toml` but absent from the venv; install the dev extras to use them.

Two commands are **not** reachable through `main.py`: `scripts/reconcile.py` must be run directly, and `scripts/measure.py`'s `--candles` / `--folds` / `--holdout` / `--optimistic` flags are only reachable by calling the script itself — `main.py measure` forwards only `--symbols` and `--strategies`.

## Architecture

### One strategy core, three consumers

This is the thing to understand before touching anything. There is exactly one strategy implementation (`tradebot/strategy.py`) and one market-view abstraction (`tradebot/measure.py`), and all three engines drive them:

| | engine | what it scores |
|---|---|---|
| `scripts/measure.py` | `measure.py` | a **strategy**, in isolation, over 20k bars |
| `main.py replay` | `replay.py` | the **bot** — over historical bars, simulated broker |
| `main.py run` | `run.py` | the **bot** — against the live broker |

The dependency arrow is the non-obvious part: `strategy.py` imports `MarketView`, `AsOfSeries` and the `Strategy` Protocol **from `measure.py`** (`strategy.py:22`). Measurement owns the view abstraction; strategies depend on measurement, not the reverse.

`Bot._scan()` (`run.py:349`) builds `AsOfSeries({"H1": candles})` and `MarketView(symbol, candles[-1].timestamp, store, self.window)` — byte-for-byte the same construction `run_series` uses (`measure.py:397-408`). Replay reaches the strategy only through `bot.tick()`, so **replay and live are identical by construction**; measure is identical by convention plus the shared classes.

`MarketView` uses `__slots__` and exposes only cutoff-bound `candles` / `higher()` accessors, so a strategy physically cannot reach a future bar. That is the no-look-ahead guarantee — structural, not conventional.

### One tick, in order

`Bot.tick()` (`run.py:135`) runs six stages in a fixed order, asserted in the module docstring:

1. read broker positions (`Broker.ours()`) and account
2. `_reconcile()` — adopt untracked broker positions, retire vanished ones
3. `_manage()` — scale out at targets, move stop to breakeven after TP1
4. `_scan()` — run strategies via `MarketView`
5. drain `INBOX` and append external candidates **after** the scan, before ranking
6. for each candidate: `_enter()` → `RiskManager.check()` → `position_size()` → `Broker.place()`

The loop is `while not self.state.stop_requested()` — the kill switch is a `STOP` file whose mere presence is the signal.

### The gate

The README and `__init__.py` say signals "pass every guard". **There is no guard module.** The referent is `RiskManager.check()` (`risk.py:80`) — a hard veto returning `(allowed, reason)`, checked before sizing and before any order call. Its checks in order: halted, daily loss, drawdown, position cap, one-per-symbol, geometry, reward:risk, spread-fraction-of-risk, non-zero size. The daily-loss and drawdown checks set `halted=True` permanently; only a daily halt clears at UTC rollover (`RiskState.roll()`), a drawdown halt requires a manual `RiskManager.reset()`.

Two structural backstops sit outside it: `Signal.geometry_error()` (`domain.py:116`, checked at the trust boundary) and `ExternalSignal.validate()` (`intake.py:63`).

### Module map

```
config.py     Settings (env-driven) + RISK (frozen).  The two must never merge.
domain.py     pure data types. No I/O — every rule here is testable in isolation.
measure.py    the measurement engine + MarketView/AsOfSeries/Strategy protocol
data.py       MT5 market data (read side)
broker.py     MT5 execution (write side)
symbols.py    instrument specs + universe ranking by relative cost
risk.py       envelope enforcement + position sizing
strategy.py   BreakoutStrategy, MeanReversionStrategy
run.py        the trading loop (Bot)
replay.py     ReplayData + PaperBroker + run_replay
state.py      StateStore — heartbeat, trades, kill switch
supervise.py  out-of-process watchdog
intake.py     TradingView webhook → INBOX
dashboard.py  HTTP monitor on :8787
agents.py     observers (Analyst/Sentinel/Coordinator) — OFF the trading path
```

### `_self_check()` is a contract

Every module in `main.py:SELF_CHECK_MODULES` exposes a `_self_check()` that asserts its money rules, and `python main.py check` runs all twelve in subprocesses. Modules with no test coverage (`replay.py`, `dashboard.py`, `strategy.py`) are covered only here. When you change a money rule, the self-check is where its assertion lives.

`tests/test_core.py` deliberately never imports `tradebot.data` and never touches MT5 — the only broker import is pure volume/string arithmetic (`_safe_comment`, `normalize_volume`, `partial_volume`). There is no `conftest.py` and no fixtures; keep it that way. A test that needs a broker is a test that cannot run here.

### Broker vs. mock: no abstraction layer

There is no ABC and no `if mock:` branch. `Broker` (`broker.py:109`) and `PaperBroker` (`replay.py:136`) are duck-typed to matching method names, and replay swaps them by **direct attribute assignment** (`replay.py:366-372`: `bot.broker = paper`, `bot.data = data`). If you add a method `Bot` calls on its broker, you must add it to `PaperBroker` too or replay breaks silently.

## Invariants that lose money when broken

- **`Trade.volume` and `Trade.stop_loss` hold broker-confirmed values, never requested ones.** MT5 normalises volume to its own lot step and may reject a stop modification. `run.py` books `result.filled_volume`. (`domain.py:194`)
- **`initial_stop_loss` is fixed at entry and never moves.** `stop_loss` is live — the bot moves it to breakeven after TP1 — so measuring risk from it afterwards silently drops every trade that did well. 1R is entry-to-`initial_stop_loss`. (`domain.py:209`)
- **`pnl` and `pnl_recorded` are two fields, not one.** An unreadable deal history must not record `0.0`, which reads back as break-even and deletes a loss from the statistics. `state.py` serialises `pnl: null` when `pnl_recorded` is false.
- **`NO_FILL` is its own outcome.** A limit entry price never returned to consumed no capital; folding it into win or loss distorts both. Excluded from statistics via `SimTrade.is_counted`.
- **`OrderResult.ambiguous` is not a failure.** MT5 can time out *after* the broker accepted. Treating it as failure is how positions end up live at the broker but invisible to the bot.
- **An external signal is never `Intent.EXECUTION`.** Only the pipeline's own gated output may be. (`domain.py:35`)

### Two deliberate asymmetries — do not "fix" them

`measure.py` charges **full spread** at the entry bar (`measure.py:229`) while `replay.py` charges **half spread** at the last mark (`replay.py:200-201`) and no commission; measure defaults `breakeven` and `market_entry` **off** while the live bot does both. These are exactly the deltas `scripts/reconcile.py` enumerates as five variants, one difference at a time. Reconciling them would erase the tool that measures the gap.

Commission is computed but structurally zero: `SymbolSpec.commission_per_lot` defaults to `0.0` and no `_REGISTRY` entry overrides it (`symbols.py:30, 48-59`). Only spread is a real cost today.

## Traps

- **`.env` is a legacy artefact.** It advertises `BOT_PROFILE`, `GUARD_PIPELINE_ENABLED` ("Pre-execution safety layer (7 guards)"), `FIXED_LOT_SIZE`, pydantic settings and a dozen integration keys — **no current code reads any of them.** The rebuild reads exactly five vars: `SYMBOLS`, `BROKER_ACCOUNT`, `BROKER_PASSWORD`, `BROKER_SERVER`, `DEMO_MODE` (`config.py:104-113`), plus `TRADINGVIEW_WEBHOOK_SECRET` (`main.py:105`, used only by `main.py webhook`). Note the webhook secret is not in `.env` — the endpoint refuses everything when it is empty. `.env` is gitignored; edit it by hand or via `main.py symbols --write`, which rewrites only the `SYMBOLS=` line.
- **`README.md`'s "Layout" and "Status" sections are stale** — they list `strategy/` as a package (it is `strategy.py`) and mark `data.py`, `broker.py`, `risk.py`, `run.py` as "(to build)". All are built.
- **`Settings.magic` is dead.** `Broker` uses the module constant `MAGIC = 202603` (`broker.py:42`); changing `settings.magic` has no effect.
- **Duplicated constants, two copies each:** `TP_SPLITS` (`measure.py:46`) vs `TP_FRACTIONS` (`run.py:45`); `WINDOW = 150` (`measure.py:43`) vs `window=150` (`run.py:62`); `STRATEGIES` (`run.py:40`) vs `scripts/measure.py:39`. Change one, change the other.
- **The holdout is not a frozen date.** `measure.py:21` calls it a "Frozen holdout" and `scripts/measure.py:9` says it is "separated by time so it can stay untouched while parameters are explored elsewhere" — but the boundary is `int(len(pooled) * 0.8)` over trades sorted by `opened_at` (`measure.py:464-465`). It is time-*ordered*, and it moves as the sample grows (`--candles` changes it). Nothing in the repo pins a date.
- **State is split by engine.** Live writes `state/`; replay writes `state/replay/` (`main.py:218`) and must never touch live state. `state/` is gitignored. `StateStore` writes atomically (`.tmp` + `os.replace`, 5 retries for Windows file-handle contention) and `closed.jsonl` is append-only.
- **`agents.py` has a runtime structural guard** asserting `tradebot.broker` is absent from its own source and no `Capability` contains `"ORDER"` (`agents.py:515-524`). Agents observe and report; they are never in the trading path, and `run.py` does not import them.
- **Live capital requires editing source.** `RISK` is a frozen dataclass read from no environment variable. That friction is the design — do not add an override.
