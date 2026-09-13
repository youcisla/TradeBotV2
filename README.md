# TradeBot

A rebuild from scratch. The previous codebase is preserved on the
`legacy/pre-rebuild-2026-09-13` branch of `github.com/youcisla/TradeBot`, and
locally as `TradeBot_legacy_2026-09-13.zip` and
`TradeBot_history_2026-09-13.bundle` in the parent directory.

## Why this was rebuilt

The previous system reached roughly 48,000 lines of Python across a nine-agent
pipeline, an LSTM, a LightGBM ensemble, GARCH and HMM regime detection, a
council of LLM experts and a debate layer — and **could not connect to a
broker.** `MT5Connector` inherited an abstract base class without implementing
its abstract methods, so instantiating it raised `TypeError`. Every live path
was dead. The test suite passed throughout because every test used a mock.

Once it could connect, the measurement was rebuilt and produced a clear answer:
the ICT strategy had **no edge**. 3,271 positions across six majors, 37.9% win
rate, average −0.431R, profit factor 0.46, 100% max drawdown. Negative in every
symbol, every fold and the frozen holdout — and still negative under maximally
optimistic intrabar fills. Inverting every signal lost *worse*, so it was not a
sign error.

The lesson is not "the strategy was bad". It is that **complexity was built on
an unmeasured foundation.** This rebuild inverts that order.

## Design contract

1. **Measurement before automation.**
   Nothing is built on a strategy that has not survived an out-of-sample test
   with real costs. `tradebot.measure` is the only way a number earns trust:
   as-of slicing on every timeframe (no look-ahead), real per-bar spread and
   commission charged on every fill, limit entries required to be touched, the
   true scale-out modelled, per-position statistics, and a time-frozen holdout.

2. **The risk envelope is frozen.**
   `tradebot.config.RISK` is a frozen dataclass, hard-coded and read from no
   environment variable. Anything an operator or an adaptive layer can widen at
   runtime is a parameter, not a limit. To trade bigger, change the source and
   redeploy — that friction is the feature.

3. **The broker is the source of truth.**
   Local records describe what the broker holds; they never assert it. Volume
   is booked from the broker's confirmed fill, and a stop is recorded only
   after the broker confirms the modification.

4. **External input is a suggestion.**
   Webhook signals (TradingView alerts) arrive as `Intent.STRATEGY`, enter the
   candidate pool after the scan and before ranking, and pass every guard an
   internal signal passes. Nothing external can reach the executor directly.

5. **Fail closed.**
   Missing config, a crashed guard, or an ambiguous order result must stop
   progress — never permit it.

## Layout

```
tradebot/
  config.py    operator settings + the frozen risk envelope
  domain.py    pure data types; no I/O
  measure.py   the measurement engine (to build)
  data.py      MT5 market data (to build)
  broker.py    MT5 execution (to build)
  strategy/    strategy base + implementations (to build)
  risk.py      envelope enforcement (to build)
  run.py       the trading loop (to build)
tests/
```

## Status

Foundation laid: `config.py`, `domain.py`. Everything else is still to build,
in the order listed above — measurement before strategy, strategy before
execution.
