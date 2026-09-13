"""TradeBot — a clean rebuild.

Design contract, in priority order:

1. **Measurement before automation.**  Nothing is built on a strategy that has
   not survived an out-of-sample test with real costs.  ``tradebot.measure``
   is the only way a number earns trust.
2. **The risk envelope is frozen.**  See ``tradebot.config.RISK``.
3. **The broker is the source of truth.**  Local records describe what the
   broker holds; they never assert it.
4. **External input is a suggestion.**  Webhook signals arrive as
   ``Intent.STRATEGY`` and pass every guard an internal signal passes.
5. **Fail closed.**  Missing config, a crashed guard or an ambiguous order
   result must stop progress, never permit it.
"""

__version__ = "0.1.0"
