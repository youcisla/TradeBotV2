"""Live dashboard — watch the bot decide, in a browser.

Standard library only, plus one CDN script for the chart.  The bot has no web
framework and should not grow one for a status page.

What it shows, and why each part earns its place:

* **Activity feed** — what the bot did each iteration: which symbols it
  scanned, which signals it found, and *why* it refused the ones it did.  A
  monitor that shows only equity cannot tell you whether the bot is working or
  merely idle.
* **Positions** — what is open at the broker right now.
* **Price chart** — TradingView's lightweight-charts rendering *our* candles
  from MT5.  Charting is not something to hand-roll.
* **Risk envelope** — the frozen limits next to current state, so a breach is
  visible rather than inferred.

The page polls ``/api/state`` every two seconds.  No websockets: the state is
small, and a poll that fails self-heals, whereas a broken socket looks exactly
like a frozen bot.
"""

from __future__ import annotations

import json
import logging
import threading
from collections import deque
from dataclasses import dataclass
from datetime import UTC, datetime
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any
from urllib.parse import parse_qs, urlparse

from tradebot.config import RISK

logger = logging.getLogger(__name__)

MAX_EVENTS = 300
LIGHTWEIGHT_CHARTS_CDN = (
    "https://unpkg.com/lightweight-charts@4.2.0/dist/lightweight-charts.standalone.production.js"
)


@dataclass(slots=True)
class Event:
    at: str
    kind: str          # scan | signal | refuse | open | close | partial | halt | info
    symbol: str
    detail: str

    def as_dict(self) -> dict[str, str]:
        return {"at": self.at, "kind": self.kind, "symbol": self.symbol, "detail": self.detail}


class ActivityLog:
    """Bounded, thread-safe record of what the bot did.

    Bounded on purpose: a monitor that grows without limit becomes the outage.
    """

    def __init__(self, maxlen: int = MAX_EVENTS) -> None:
        self._events: deque[Event] = deque(maxlen=maxlen)
        self._lock = threading.Lock()
        self._counts: dict[str, int] = {}

    def add(self, kind: str, detail: str, symbol: str = "") -> None:
        event = Event(
            at=datetime.now(UTC).strftime("%H:%M:%S"), kind=kind, symbol=symbol, detail=detail
        )
        with self._lock:
            self._events.append(event)
            self._counts[kind] = self._counts.get(kind, 0) + 1

    def recent(self, limit: int = 60) -> list[dict[str, str]]:
        with self._lock:
            return [e.as_dict() for e in list(self._events)[-limit:]][::-1]

    def counts(self) -> dict[str, int]:
        with self._lock:
            return dict(self._counts)

    def clear(self) -> None:
        with self._lock:
            self._events.clear()
            self._counts.clear()


ACTIVITY = ActivityLog()


# ── HTTP ─────────────────────────────────────────────────────────────────────

class _Handler(BaseHTTPRequestHandler):
    bot: Any = None
    server_version = "tradebot-dashboard/1.0"

    def log_message(self, fmt: str, *args: Any) -> None:      # noqa: A002
        logger.debug("dashboard: " + fmt, *args)

    def _send(self, code: int, body: bytes, content_type: str) -> None:
        self.send_response(code)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def _json(self, payload: dict[str, Any], code: int = 200) -> None:
        self._send(code, json.dumps(payload, default=str).encode("utf-8"), "application/json")

    def do_GET(self) -> None:                                  # noqa: N802
        route = urlparse(self.path)
        query = parse_qs(route.query)

        if route.path in ("/", "/index.html"):
            page = PAGE.replace("__CDN__", LIGHTWEIGHT_CHARTS_CDN)
            self._send(200, page.encode("utf-8"), "text/html; charset=utf-8")
            return
        if route.path == "/api/state":
            self._json(self._state())
            return
        if route.path == "/api/candles":
            symbol = (query.get("symbol") or ["EURUSD"])[0]
            self._json({"symbol": symbol, "candles": self._candles(symbol)})
            return
        self._json({"error": "not found"}, 404)

    # ── data ────────────────────────────────────────────────────────────────

    def _state(self) -> dict[str, Any]:
        bot = self.bot
        if bot is None:
            return {"bot": "not attached", "activity": ACTIVITY.recent()}

        store = bot.state
        beat = store.heartbeat()
        balance, equity = bot.broker.account()

        positions = []
        try:
            for position in bot.broker.ours():
                positions.append({
                    "ticket": position.ticket,
                    "symbol": position.symbol,
                    "direction": str(position.direction),
                    "volume": position.volume,
                    "entry": position.entry_price,
                    "stop": position.stop_loss,
                    "current": position.current_price,
                    "profit": round(position.profit, 2),
                })
        except Exception as exc:                               # noqa: BLE001
            logger.debug("dashboard: positions unavailable (%s)", exc)

        return {
            "mode": "DEMO" if bot.settings.demo_mode else "LIVE",
            "strategy": bot.strategy_name,
            "iteration": bot.iteration,
            "balance": round(balance, 2),
            "equity": round(equity, 2),
            "open_positions": len(positions),
            "positions": positions,
            "halted": bot.risk.state.halted,
            "halt_reason": bot.risk.state.halt_reason,
            "heartbeat_age_s": store.heartbeat_age_seconds(),
            "heartbeat_iteration": beat.iteration if beat else None,
            "stop_requested": store.stop_requested(),
            "risk": {
                "risk_per_trade": RISK.risk_per_trade,
                "max_daily_loss": RISK.max_daily_loss,
                "max_drawdown": RISK.max_drawdown,
                "max_positions": RISK.max_concurrent_positions,
                "max_lots": RISK.max_volume_per_trade,
            },
            "refusals": bot.refusals,
            "counts": ACTIVITY.counts(),
            "activity": ACTIVITY.recent(),
        }

    def _candles(self, symbol: str) -> list[dict[str, Any]]:
        bot = self.bot
        if bot is None:
            return []
        try:
            candles = bot.data.candles(symbol, "H1", 300)
        except Exception as exc:                               # noqa: BLE001
            logger.debug("dashboard: candles unavailable (%s)", exc)
            return []
        return [
            {
                "time": int(c.timestamp.timestamp()),
                "open": c.open,
                "high": c.high,
                "low": c.low,
                "close": c.close,
            }
            for c in candles
        ]


PAGE = """<!doctype html>
<html lang="en"><head>
<meta charset="utf-8"/>
<meta name="viewport" content="width=device-width,initial-scale=1"/>
<title>TradeBot — live</title>
<script src="__CDN__"></script>
<style>
  :root { --bg:#0f1115; --panel:#171a21; --line:#252a34; --fg:#e6e9ef;
          --dim:#8b94a7; --ok:#3fb950; --warn:#d29922; --bad:#f85149; --accent:#58a6ff; }
  * { box-sizing:border-box; }
  body { margin:0; background:var(--bg); color:var(--fg);
         font:13px/1.5 ui-monospace,SFMono-Regular,Menlo,monospace; }
  header { display:flex; gap:20px; align-items:center; flex-wrap:wrap;
           padding:12px 18px; border-bottom:1px solid var(--line); background:var(--panel); }
  header h1 { font-size:14px; margin:0; letter-spacing:.08em; text-transform:uppercase; }
  .kpi { display:flex; gap:6px; align-items:baseline; }
  .kpi b { font-size:16px; }
  .kpi span { color:var(--dim); font-size:11px; text-transform:uppercase; }
  main { display:grid; grid-template-columns:1fr 380px; gap:14px; padding:14px; }
  .panel { background:var(--panel); border:1px solid var(--line); border-radius:8px; }
  .panel h2 { margin:0; padding:9px 12px; font-size:11px; letter-spacing:.1em;
              text-transform:uppercase; color:var(--dim); border-bottom:1px solid var(--line); }
  .body { padding:10px 12px; }
  #chart { height:340px; }
  table { width:100%; border-collapse:collapse; }
  th,td { text-align:left; padding:5px 8px; border-bottom:1px solid var(--line); font-size:12px; }
  th { color:var(--dim); font-weight:400; text-transform:uppercase; font-size:10px; }
  #feed { max-height:330px; overflow:auto; }
  .ev { display:grid; grid-template-columns:56px 62px 1fr; gap:8px;
        padding:4px 12px; border-bottom:1px solid var(--line); font-size:12px; }
  .ev .t { color:var(--dim); }
  .ev .k { text-transform:uppercase; font-size:10px; padding-top:2px; }
  .k.refuse { color:var(--warn); } .k.open,.k.close,.k.partial { color:var(--ok); }
  .k.signal { color:var(--accent); } .k.halt { color:var(--bad); } .k.scan { color:var(--dim); }
  .bad { color:var(--bad); } .good { color:var(--ok); } .warn { color:var(--warn); }
  .dim { color:var(--dim); }
  .risk { display:grid; grid-template-columns:1fr auto; gap:2px 12px; font-size:12px; }
  .row { display:flex; justify-content:space-between; padding:2px 0; }
  #banner { display:none; padding:10px 18px; background:#3d1418; color:#ffb3b3;
            border-bottom:1px solid var(--bad); }
  @media (max-width:900px){ main { grid-template-columns:1fr; } }
</style></head><body>
<header>
  <h1>TradeBot</h1>
  <div class="kpi"><span>mode</span><b id="mode">—</b></div>
  <div class="kpi"><span>strategy</span><b id="strategy">—</b></div>
  <div class="kpi"><span>equity</span><b id="equity">—</b></div>
  <div class="kpi"><span>balance</span><b id="balance">—</b></div>
  <div class="kpi"><span>open</span><b id="open">—</b></div>
  <div class="kpi"><span>iteration</span><b id="iteration">—</b></div>
  <div class="kpi"><span>heartbeat</span><b id="beat">—</b></div>
</header>
<div id="banner"></div>
<main>
  <div>
    <div class="panel"><h2>Price — <span id="sym">EURUSD</span> H1</h2>
      <div id="chart"></div></div>
    <div class="panel" style="margin-top:14px"><h2>Positions</h2>
      <table><thead><tr><th>Ticket</th><th>Symbol</th><th>Dir</th><th>Lots</th>
      <th>Entry</th><th>Stop</th><th>Now</th><th>P&amp;L</th></tr></thead>
      <tbody id="positions"><tr><td colspan="8" class="dim">none</td></tr></tbody></table></div>
  </div>
  <div>
    <div class="panel"><h2>Activity</h2><div id="feed"></div></div>
    <div class="panel" style="margin-top:14px"><h2>Risk envelope (frozen)</h2>
      <div class="body risk" id="risk"></div></div>
    <div class="panel" style="margin-top:14px"><h2>Refusals</h2>
      <div class="body" id="refusals"></div></div>
  </div>
</main>
<script>
const chart = LightweightCharts.createChart(document.getElementById('chart'), {
  layout:{ background:{color:'#171a21'}, textColor:'#8b94a7' },
  grid:{ vertLines:{color:'#252a34'}, horzLines:{color:'#252a34'} },
  timeScale:{ timeVisible:true, borderColor:'#252a34' },
  rightPriceScale:{ borderColor:'#252a34' }, height:340,
});
const candles = chart.addCandlestickSeries({
  upColor:'#3fb950', downColor:'#f85149', borderVisible:false,
  wickUpColor:'#3fb950', wickDownColor:'#f85149',
});

async function loadChart(symbol){
  document.getElementById('sym').textContent = symbol;
  const r = await fetch('/api/candles?symbol=' + encodeURIComponent(symbol));
  const d = await r.json();
  if (d.candles && d.candles.length) { candles.setData(d.candles); chart.timeScale().fitContent(); }
}

function fmt(n){ return (n===null||n===undefined) ? '—' :
  Number(n).toLocaleString(undefined,{minimumFractionDigits:2,maximumFractionDigits:2}); }
function esc(s){ return String(s).replace(/[&<>]/g, c => ({'&':'&amp;','<':'&lt;','>':'&gt;'}[c])); }

async function tick(){
  let s;
  try { s = await (await fetch('/api/state')).json(); } catch(e){ return; }

  document.getElementById('mode').textContent = s.mode;
  document.getElementById('mode').className = s.mode === 'LIVE' ? 'bad' : 'good';
  document.getElementById('strategy').textContent = s.strategy;
  document.getElementById('equity').textContent = fmt(s.equity);
  document.getElementById('balance').textContent = fmt(s.balance);
  document.getElementById('open').textContent = s.open_positions;
  document.getElementById('iteration').textContent = s.iteration;
  const age = s.heartbeat_age_s;
  const beatEl = document.getElementById('beat');
  if (age === null || age === undefined) { beatEl.textContent = 'none'; beatEl.className = 'dim'; }
  else { beatEl.textContent = Math.round(age) + 's'; beatEl.className = age > 120 ? 'bad' : 'good'; }

  const banner = document.getElementById('banner');
  if (s.stop_requested) {
    banner.style.display='block';
    banner.textContent='KILL SWITCH SET — the bot will stop at the top of its next iteration.';
  } else if (s.halted) {
    banner.style.display='block';
    banner.textContent='RISK HALT — ' + (s.halt_reason||'') + '. No new entries.';
  } else { banner.style.display='none'; }

  const pos = s.positions || [];
  document.getElementById('positions').innerHTML = pos.length ? pos.map(p =>
    `<tr><td>${p.ticket}</td><td>${esc(p.symbol)}</td>
     <td class="${p.direction==='BUY'?'good':'bad'}">${p.direction}</td>
     <td>${p.volume}</td><td>${fmt(p.entry)}</td><td>${fmt(p.stop)}</td><td>${fmt(p.current)}</td>
     <td class="${p.profit>=0?'good':'bad'}">${fmt(p.profit)}</td></tr>`).join('')
    : '<tr><td colspan="8" class="dim">none</td></tr>';

  document.getElementById('feed').innerHTML = (s.activity||[]).map(e =>
    `<div class="ev"><span class="t">${esc(e.at)}</span>
     <span class="k ${esc(e.kind)}">${esc(e.kind)}</span>
     <span>${esc(e.symbol)} ${esc(e.detail)}</span></div>`).join('')
    || '<div class="body dim">no activity yet</div>';

  const r = s.risk || {};
  document.getElementById('risk').innerHTML = `
    <div>risk / trade</div><div>${(r.risk_per_trade*100).toFixed(2)}%</div>
    <div>max daily loss</div><div>${(r.max_daily_loss*100).toFixed(1)}%</div>
    <div>max drawdown</div><div>${(r.max_drawdown*100).toFixed(0)}%</div>
    <div>max positions</div><div>${r.max_positions}</div>
    <div>max lots</div><div>${r.max_lots}</div>`;

  const ref = s.refusals || {};
  const keys = Object.keys(ref).sort((a,b)=>ref[b]-ref[a]);
  document.getElementById('refusals').innerHTML = keys.length
    ? keys.map(k => `<div class="row"><span class="dim">${esc(k)}</span><b>${ref[k]}</b></div>`).join('')
    : '<span class="dim">nothing refused</span>';
}

loadChart('EURUSD');
tick();
setInterval(tick, 2000);
</script></body></html>
"""


def serve(bot: Any, host: str = "127.0.0.1", port: int = 8787) -> ThreadingHTTPServer:
    """Start the dashboard on a background thread.  Returns the server."""
    handler = type("BoundDashboard", (_Handler,), {"bot": bot})
    server = ThreadingHTTPServer((host, port), handler)
    threading.Thread(target=server.serve_forever, name="dashboard", daemon=True).start()
    logger.info("Dashboard: http://%s:%d", host, port)
    return server


def _self_check() -> None:
    """The log is bounded, ordered newest-first, and counts by kind."""
    log = ActivityLog(maxlen=3)
    for i in range(5):
        log.add("scan", f"event {i}", symbol="EURUSD")

    recent = log.recent()
    assert len(recent) == 3, len(recent)                     # bounded
    assert recent[0]["detail"] == "event 4", recent[0]       # newest first
    assert log.counts()["scan"] == 5                         # counts survive eviction

    log.add("refuse", "spread is 40% of risk")
    assert log.counts()["refuse"] == 1
    assert log.recent(limit=1)[0]["kind"] == "refuse"

    log.clear()
    assert log.recent() == [] and log.counts() == {}

    # The CDN placeholder must survive until render, so the page is offline-safe
    # to reason about and the URL lives in exactly one place.
    assert "__CDN__" in PAGE and "unpkg.com" not in PAGE

    print("dashboard.py self-check: OK")


if __name__ == "__main__":
    _self_check()
