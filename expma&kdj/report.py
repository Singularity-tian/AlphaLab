"""HTML report generator for EXPMA(12) & KDJ hourly strategy backtest."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


def generate_html_report(
    result: dict[str, Any],
    output_path: str | Path | None = None,
    initial_cash: float = 100_000.0,
    max_positions: int = 20,
    position_pct: float = 0.05,
    stop_loss_pct: float = 0.05,
) -> str:
    """Generate a self-contained HTML report from EXPMA&KDJ backtest results."""
    if output_path is None:
        output_path = Path("expma&kdj/results/report.html")
    else:
        output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    stats = result["stats"]
    equity_curve = result["equity_curve"]
    daily = result["daily"]
    trades_df = result["trades"]
    per_stock_pnl = result["per_stock_pnl"]

    def _safe(v):
        if isinstance(v, float) and (np.isnan(v) or np.isinf(v)):
            return None
        return round(v, 2)

    # Resample hourly to daily for charts
    eq_daily = equity_curve.resample("D").last().dropna()
    daily_resampled = daily.resample("D").last().dropna()

    eq_dates = [d.strftime("%Y-%m-%d") for d in eq_daily.index]
    eq_vals = [_safe(v) for v in eq_daily.values]

    cummax = eq_daily.cummax()
    dd = [_safe(v) for v in ((eq_daily - cummax) / cummax * 100).tolist()]

    # Sample weekly for lighter charts
    step = 5
    pos_dates = [d.strftime("%Y-%m-%d") for d in daily_resampled.index[::step]]
    pos_counts = [int(v) if not np.isnan(v) else 0 for v in daily_resampled["num_positions"].values[::step]]
    cash_vals = [_safe(v) for v in daily_resampled["cash"].values[::step]]
    inv_vals = [_safe(v) for v in daily_resampled["position_value"].values[::step]]

    pnl_tickers = per_stock_pnl["ticker"].tolist() if not per_stock_pnl.empty else []
    pnl_values = per_stock_pnl["total_pnl"].tolist() if not per_stock_pnl.empty else []

    # Limit P&L chart to top/bottom 30
    if len(pnl_tickers) > 60:
        top_n = 30
        pnl_df = per_stock_pnl.copy()
        top = pnl_df.head(top_n)
        bottom = pnl_df.tail(top_n)
        pnl_chart = pd.concat([top, bottom]).drop_duplicates(subset=["ticker"])
        pnl_tickers = pnl_chart["ticker"].tolist()
        pnl_values = pnl_chart["total_pnl"].tolist()

    trades_data = []
    if not trades_df.empty:
        for _, t in trades_df.iterrows():
            trades_data.append({
                "date": str(t["date"])[:16],  # include time for hourly
                "ticker": t["ticker"],
                "side": t["side"], "shares": int(t["shares"]),
                "price": round(t["price"], 2), "value": round(t["value"], 2),
                "commission": round(t["commission"], 2),
                "cash_impact": round(t["cash_impact"], 2),
                "cash_after": round(t["cash_after"], 2),
                "reason": t.get("reason", ""),
            })

    stock_table = []
    if not per_stock_pnl.empty:
        for _, r in per_stock_pnl.iterrows():
            stock_table.append({
                "ticker": r["ticker"], "total_cost": round(r["total_cost"], 2),
                "total_proceeds": round(r["total_proceeds"], 2),
                "unrealized": round(r["unrealized_value"], 2),
                "pnl": round(r["total_pnl"], 2), "open": int(r["open_shares"]),
                "buys": int(r["num_buys"]), "sells": int(r["num_sells"]),
            })

    # Sell reason counts
    sell_reasons = stats.get("Sell Reasons", {})
    trend_breaks = sell_reasons.get("TREND_BREAK", 0)
    stop_losses = sell_reasons.get("STOP_LOSS", 0)

    html = _build_html(
        stats=stats,
        initial_cash=initial_cash,
        max_positions=max_positions,
        position_pct=position_pct,
        stop_loss_pct=stop_loss_pct,
        trend_breaks=trend_breaks,
        stop_losses=stop_losses,
        eq_dates=eq_dates, eq_vals=eq_vals, dd=dd,
        pos_dates=pos_dates, pos_counts=pos_counts,
        cash_vals=cash_vals, inv_vals=inv_vals,
        pnl_tickers=pnl_tickers, pnl_values=pnl_values,
        stock_table=stock_table, trades_data=trades_data,
    )

    output_path.write_text(html, encoding="utf-8")
    return str(output_path)


def _build_html(
    stats, initial_cash, max_positions, position_pct, stop_loss_pct,
    trend_breaks, stop_losses,
    eq_dates, eq_vals, dd,
    pos_dates, pos_counts, cash_vals, inv_vals,
    pnl_tickers, pnl_values,
    stock_table, trades_data,
):
    ret_cls = "pos" if stats["Total Return [%]"] >= 0 else "neg"
    ann_cls = "pos" if stats["Annualized Return [%]"] >= 0 else "neg"
    cash_str = f"${initial_cash:,.0f}"
    pos_pct_str = f"{position_pct*100:.0f}%"
    stop_str = f"{stop_loss_pct*100:.0f}%"

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>EXPMA &amp; KDJ Strategy Report</title>
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Mono:wght@300;400;500&family=Instrument+Serif:ital@0;1&display=swap');
*{{margin:0;padding:0;box-sizing:border-box}}
:root{{
  --bg:#0a0a0b;--s1:#111113;--s2:#18181b;--s3:#1e1e22;
  --bd:#27272a;--bds:#1c1c1f;
  --t:#fafafa;--td:#a1a1aa;--tm:#71717a;
  --ac:#c8f542;--acd:#a5cc30;
  --red:#ef4444;--grn:#22c55e;--blu:#3b82f6;--org:#f59e0b;
}}
body{{background:var(--bg);color:var(--t);font-family:'DM Mono',monospace;font-size:13px;line-height:1.6;-webkit-font-smoothing:antialiased}}
.wrap{{max-width:1320px;margin:0 auto;padding:40px 32px 80px}}

.noise{{position:fixed;inset:0;pointer-events:none;z-index:9999;opacity:.03;
  background-image:url("data:image/svg+xml,%3Csvg viewBox='0 0 256 256' xmlns='http://www.w3.org/2000/svg'%3E%3Cfilter id='n'%3E%3CfeTurbulence type='fractalNoise' baseFrequency='0.9' numOctaves='4' stitchTiles='stitch'/%3E%3C/filter%3E%3Crect width='100%25' height='100%25' filter='url(%23n)'/%3E%3C/svg%3E")}}

header{{padding:48px 0 40px;border-bottom:1px solid var(--bd);margin-bottom:48px;position:relative}}
header::before{{content:'';position:absolute;top:0;left:0;right:0;height:1px;background:linear-gradient(90deg,transparent,var(--ac),transparent)}}
.htop{{display:flex;justify-content:space-between;align-items:flex-start;margin-bottom:20px;flex-wrap:wrap;gap:16px}}
.logo{{font-family:'Instrument Serif',serif;font-size:42px;font-weight:400;letter-spacing:-1px}}
.logo span{{color:var(--ac);font-style:italic}}
.sub{{font-size:14px;color:var(--td);margin-top:4px}}
.pills{{display:flex;gap:8px;flex-wrap:wrap}}
.pill{{background:var(--s2);border:1px solid var(--bd);padding:5px 12px;border-radius:4px;font-size:11px;color:var(--td);white-space:nowrap}}
.pill b{{color:var(--t);font-weight:500}}

.kpis{{display:grid;grid-template-columns:repeat(auto-fit,minmax(150px,1fr));gap:1px;background:var(--bds);border:1px solid var(--bd);border-radius:8px;overflow:hidden;margin-bottom:48px}}
.kpi{{background:var(--s1);padding:20px 24px}}
.kpi-l{{font-size:10px;text-transform:uppercase;letter-spacing:1.5px;color:var(--tm);margin-bottom:8px}}
.kpi-v{{font-size:22px;font-weight:500;letter-spacing:-.5px}}
.pos{{color:var(--grn)}}.neg{{color:var(--red)}}.acc{{color:var(--ac)}}.org{{color:var(--org)}}

.sec{{margin-bottom:48px}}
.sec-t{{font-family:'Instrument Serif',serif;font-size:24px;font-weight:400;margin-bottom:24px;padding-bottom:12px;border-bottom:1px solid var(--bd);display:flex;align-items:center;gap:12px}}
.dot{{width:6px;height:6px;background:var(--ac);border-radius:50%;flex-shrink:0}}

.cc{{background:var(--s1);border:1px solid var(--bd);border-radius:8px;padding:24px;margin-bottom:16px;position:relative;overflow:hidden}}
.cc::after{{content:'';position:absolute;bottom:0;left:0;right:0;height:1px;background:linear-gradient(90deg,transparent,var(--acd),transparent);opacity:.3}}
.cl{{font-size:10px;text-transform:uppercase;letter-spacing:1.5px;color:var(--tm);margin-bottom:16px}}
canvas{{width:100%!important;height:280px!important}}
.crow{{display:grid;grid-template-columns:1fr 1fr;gap:16px}}
@media(max-width:900px){{.crow{{grid-template-columns:1fr}}}}

.tw{{background:var(--s1);border:1px solid var(--bd);border-radius:8px;overflow:hidden}}
.tc{{display:flex;justify-content:space-between;align-items:center;padding:12px 16px;border-bottom:1px solid var(--bd)}}
.tc input{{background:var(--s2);border:1px solid var(--bd);color:var(--t);padding:6px 12px;border-radius:4px;font-family:'DM Mono',monospace;font-size:12px;outline:none;width:200px}}
.tc input:focus{{border-color:var(--ac)}}
.tc .cnt{{font-size:11px;color:var(--tm)}}
table{{width:100%;border-collapse:collapse}}
th{{font-size:10px;text-transform:uppercase;letter-spacing:1.2px;color:var(--tm);font-weight:400;text-align:left;padding:10px 16px;border-bottom:1px solid var(--bd);position:sticky;top:0;background:var(--s1);cursor:pointer;user-select:none}}
th:hover{{color:var(--td)}}th.sorted{{color:var(--ac)}}
td{{padding:8px 16px;border-bottom:1px solid var(--bds);font-size:12px;color:var(--td)}}
tr:hover td{{background:var(--s2);color:var(--t)}}
td.tk{{font-weight:500;color:var(--t)}}
td.pos{{color:var(--grn)}}td.neg{{color:var(--red)}}
td.buy{{color:var(--grn)}}td.sell{{color:var(--red)}}
td.reason-sl{{color:var(--red);font-weight:500}}
td.reason-tb{{color:var(--org)}}
td.reason-ao{{color:var(--blu)}}
.tscr{{max-height:520px;overflow-y:auto}}
.tscr::-webkit-scrollbar{{width:6px}}
.tscr::-webkit-scrollbar-track{{background:var(--s1)}}
.tscr::-webkit-scrollbar-thumb{{background:var(--bd);border-radius:3px}}

.sgrid{{display:grid;grid-template-columns:1fr 1fr;gap:16px}}
@media(max-width:900px){{.sgrid{{grid-template-columns:1fr}}}}
.scard{{background:var(--s1);border:1px solid var(--bd);border-radius:8px;padding:24px}}
.scard h3{{font-family:'Instrument Serif',serif;font-size:16px;font-weight:400;margin-bottom:16px;display:flex;align-items:center;gap:8px}}
.scard h3 .n{{background:var(--ac);color:var(--bg);width:20px;height:20px;border-radius:50%;display:flex;align-items:center;justify-content:center;font-family:'DM Mono',monospace;font-size:10px;font-weight:500;flex-shrink:0}}
.scard p{{color:var(--td);font-size:12px;line-height:1.7;margin-bottom:10px}}
.scard code{{background:var(--s3);padding:2px 6px;border-radius:3px;font-size:11px;color:var(--ac)}}
.scard ul{{list-style:none;padding:0}}
.scard li{{padding:4px 0;font-size:12px;color:var(--td);display:flex;align-items:baseline;gap:8px}}
.scard li::before{{content:'>';color:var(--ac);font-weight:500;flex-shrink:0}}

footer{{margin-top:64px;padding-top:24px;border-top:1px solid var(--bd);text-align:center;font-size:11px;color:var(--tm)}}

@keyframes fu{{from{{opacity:0;transform:translateY(12px)}}to{{opacity:1;transform:translateY(0)}}}}
.kpi,.cc,.scard,.tw{{animation:fu .5s ease-out both}}
.kpi:nth-child(2){{animation-delay:.05s}}.kpi:nth-child(3){{animation-delay:.1s}}
.kpi:nth-child(4){{animation-delay:.15s}}.kpi:nth-child(5){{animation-delay:.2s}}
.kpi:nth-child(6){{animation-delay:.25s}}.kpi:nth-child(7){{animation-delay:.3s}}
.kpi:nth-child(8){{animation-delay:.35s}}.kpi:nth-child(9){{animation-delay:.4s}}
.kpi:nth-child(10){{animation-delay:.45s}}.kpi:nth-child(11){{animation-delay:.5s}}
</style>
</head>
<body>
<div class="noise"></div>
<div class="wrap">

<header>
  <div class="htop">
    <div>
      <div class="logo">EXPMA &amp; <span>KDJ</span></div>
      <div class="sub">Hourly Technical Strategy — S&amp;P 500 Backtest</div>
    </div>
    <div class="pills">
      <div class="pill"><b>{cash_str}</b> capital</div>
      <div class="pill"><b>2021–2026</b></div>
      <div class="pill"><b>502</b> stocks</div>
      <div class="pill"><b>Hourly</b> signal</div>
      <div class="pill"><b>{stop_str}</b> hard stop</div>
      <div class="pill"><b>{pos_pct_str}</b> per position</div>
      <div class="pill"><b>{max_positions}</b> max positions</div>
    </div>
  </div>
</header>

<div class="kpis">
  <div class="kpi"><div class="kpi-l">Total Return</div><div class="kpi-v {ret_cls}">{stats['Total Return [%]']:+.2f}%</div></div>
  <div class="kpi"><div class="kpi-l">Annual Return</div><div class="kpi-v {ann_cls}">{stats['Annualized Return [%]']:+.2f}%</div></div>
  <div class="kpi"><div class="kpi-l">Sharpe Ratio</div><div class="kpi-v">{stats['Sharpe Ratio']:.3f}</div></div>
  <div class="kpi"><div class="kpi-l">Sortino Ratio</div><div class="kpi-v">{stats['Sortino Ratio']:.3f}</div></div>
  <div class="kpi"><div class="kpi-l">Calmar Ratio</div><div class="kpi-v">{stats.get('Calmar Ratio', 0):.3f}</div></div>
  <div class="kpi"><div class="kpi-l">Profit Factor</div><div class="kpi-v">{stats.get('Profit Factor', 0):.3f}</div></div>
  <div class="kpi"><div class="kpi-l">Max Drawdown</div><div class="kpi-v neg">{stats['Max Drawdown [%]']:.2f}%</div></div>
  <div class="kpi"><div class="kpi-l">Final Equity</div><div class="kpi-v acc">${stats['Equity Final [$]']:,.2f}</div></div>
  <div class="kpi"><div class="kpi-l">Total Trades</div><div class="kpi-v">{stats['# Trades']}</div></div>
  <div class="kpi"><div class="kpi-l">Commissions</div><div class="kpi-v">${stats['Total Commission [$]']:,.2f}</div></div>
  <div class="kpi"><div class="kpi-l">Trend Break Exits</div><div class="kpi-v org">{trend_breaks}</div></div>
  <div class="kpi"><div class="kpi-l">Stop-Loss Exits</div><div class="kpi-v neg">{stop_losses}</div></div>
  <div class="kpi"><div class="kpi-l">Duration</div><div class="kpi-v">{stats['Duration [days]']}d</div></div>
</div>

<div class="sec">
  <div class="sec-t"><span class="dot"></span>Equity Curve</div>
  <div class="cc"><canvas id="eqC"></canvas></div>
</div>

<div class="sec">
  <div class="sec-t"><span class="dot"></span>Risk &amp; Exposure</div>
  <div class="crow">
    <div class="cc"><div class="cl">Drawdown</div><canvas id="ddC"></canvas></div>
    <div class="cc"><div class="cl">Active Positions</div><canvas id="posC"></canvas></div>
  </div>
  <div class="cc" style="margin-top:16px"><div class="cl">Cash vs Invested</div><canvas id="allocC"></canvas></div>
</div>

<div class="sec">
  <div class="sec-t"><span class="dot"></span>Stock Contributions (Top/Bottom 30)</div>
  <div class="cc"><canvas id="pnlC" style="height:480px!important"></canvas></div>
  <div class="tw" style="margin-top:16px">
    <div class="tc"><input type="text" id="sF" placeholder="Filter ticker..."><span class="cnt" id="sN">{len(stock_table)} stocks</span></div>
    <div class="tscr"><table id="sT"><thead><tr>
      <th data-s="ticker">Ticker</th><th data-s="pnl">P&amp;L</th><th data-s="total_cost">Cost</th>
      <th data-s="total_proceeds">Proceeds</th><th data-s="unrealized">Unrealized</th>
      <th data-s="open">Open</th><th data-s="buys">Buys</th><th data-s="sells">Sells</th>
    </tr></thead><tbody></tbody></table></div>
  </div>
</div>

<div class="sec">
  <div class="sec-t"><span class="dot"></span>Trade Log</div>
  <div class="tw">
    <div class="tc"><input type="text" id="tF" placeholder="Filter ticker, date, or reason..."><span class="cnt" id="tN">{len(trades_data)} trades</span></div>
    <div class="tscr"><table id="tT"><thead><tr>
      <th data-s="date">Date</th><th data-s="ticker">Ticker</th><th data-s="side">Side</th>
      <th data-s="reason">Reason</th><th data-s="shares">Shares</th><th data-s="price">Price</th>
      <th data-s="value">Value</th><th data-s="commission">Comm.</th>
      <th data-s="cash_impact">+/-</th><th data-s="cash_after">Cash</th>
    </tr></thead><tbody></tbody></table></div>
  </div>
</div>

<div class="sec">
  <div class="sec-t"><span class="dot"></span>Strategy Details</div>
  <div class="sgrid">
    <div class="scard"><h3><span class="n">1</span>Indicators</h3>
      <p>Three technical indicators computed on <b>1-hour</b> OHLC bars:</p>
      <ul>
        <li><b>EXPMA(12)</b> — 12-period exponential moving average of close price</li>
        <li><b>EXPMA(50)</b> — 50-period exponential moving average of close price</li>
        <li><b>KDJ(9)</b> — RSV from 9-bar high/low range, smoothed K/D, J = 3K - 2D</li>
      </ul>
      <p style="margin-top:12px">Formulas:</p>
      <p><code>EXPMA(N) = close.ewm(span=N)</code></p>
      <p><code>RSV = (C - Low9) / (High9 - Low9) * 100</code></p>
      <p><code>K = 2/3 * K_prev + 1/3 * RSV</code></p>
      <p><code>J = 3K - 2D</code></p>
    </div>
    <div class="scard"><h3><span class="n">2</span>Entry Rules</h3>
      <p><b>Initial buy (50% of allocation)</b> — all 3 conditions must be true:</p>
      <ul>
        <li><b>Trend</b>: EXPMA(12) &gt; EXPMA(50) — bullish alignment</li>
        <li><b>Oversold</b>: J value on previous bar &lt; 0</li>
        <li><b>Upturn</b>: J value turning up (J_n &gt; J_n-1)</li>
      </ul>
      <p style="margin-top:12px"><b>Add-on buy (remaining 50%)</b> — on the next bar:</p>
      <ul>
        <li>Close_n &gt; Close_n-1 (price rising) → buy remaining capital</li>
        <li>If price not rising → skip add-on, stay at 50%</li>
      </ul>
      <p style="margin-top:12px">Commission: <code>0.1%</code> per trade.</p>
    </div>
    <div class="scard"><h3><span class="n">3</span>Exit Rules</h3>
      <p>Two exit triggers — <b>either</b> condition causes full liquidation:</p>
      <ul>
        <li><b>Trend break</b>: EXPMA(12) &lt; EXPMA(50) on previous bar → sell at current bar open</li>
        <li><b>Hard stop</b>: Price &le; avg cost &times; {1-stop_loss_pct:.2f} → sell at current bar open</li>
      </ul>
      <p style="margin-top:12px">After exit, the stock can trigger a new buy signal immediately if conditions are met again.</p>
    </div>
    <div class="scard"><h3><span class="n">4</span>Position Sizing</h3>
      <p>Shared capital pool across all S&amp;P 500 stocks:</p>
      <ul>
        <li><b>{pos_pct_str}</b> of portfolio value per new position</li>
        <li>Max <b>{max_positions}</b> concurrent positions</li>
        <li>50% on initial buy, 50% add-on next bar</li>
        <li>Capital freed on sell → available for new positions</li>
      </ul>
      <p style="margin-top:12px">Signals scanned across all 502 stocks every hourly bar. First-come first-served when multiple signals fire simultaneously.</p>
    </div>
  </div>
</div>

<footer>EXPMA &amp; KDJ Hourly Strategy — AlphaLab — Generated from backtest data</footer>
</div>

<script src="https://cdn.jsdelivr.net/npm/chart.js@4/dist/chart.umd.min.js"></script>
<script>
const D={json.dumps({"ed":eq_dates,"ev":eq_vals,"dd":dd,"pd":pos_dates,"pc":pos_counts,"cv":cash_vals,"iv":inv_vals,"pt":pnl_tickers,"pv":pnl_values,"st":stock_table,"td":trades_data})};
Chart.defaults.color='#71717a';Chart.defaults.borderColor='#27272a';
Chart.defaults.font.family="'DM Mono',monospace";Chart.defaults.font.size=10;
const G={{display:true,color:'rgba(39,39,42,0.5)',drawBorder:false}};
new Chart(document.getElementById('eqC'),{{type:'line',data:{{labels:D.ed,datasets:[{{data:D.ev,borderColor:'#c8f542',backgroundColor:'rgba(200,245,66,0.05)',borderWidth:1.5,fill:true,pointRadius:0,tension:.1}}]}},options:{{responsive:true,maintainAspectRatio:false,plugins:{{legend:{{display:false}},tooltip:{{callbacks:{{label:c=>'$'+c.parsed.y.toLocaleString()}}}}}},scales:{{x:{{grid:G,ticks:{{maxTicksLimit:8}}}},y:{{grid:G,ticks:{{callback:v=>'$'+(v/1000).toFixed(0)+'k'}}}}}}}}}});
new Chart(document.getElementById('ddC'),{{type:'line',data:{{labels:D.ed,datasets:[{{data:D.dd,borderColor:'#ef4444',backgroundColor:'rgba(239,68,68,0.08)',borderWidth:1,fill:true,pointRadius:0,tension:.1}}]}},options:{{responsive:true,maintainAspectRatio:false,plugins:{{legend:{{display:false}}}},scales:{{x:{{grid:G,ticks:{{maxTicksLimit:5}}}},y:{{grid:G,ticks:{{callback:v=>v.toFixed(0)+'%'}}}}}}}}}});
new Chart(document.getElementById('posC'),{{type:'bar',data:{{labels:D.pd,datasets:[{{data:D.pc,backgroundColor:'rgba(59,130,246,0.5)',borderColor:'#3b82f6',borderWidth:1,borderRadius:2}}]}},options:{{responsive:true,maintainAspectRatio:false,plugins:{{legend:{{display:false}}}},scales:{{x:{{grid:G,ticks:{{maxTicksLimit:5}}}},y:{{grid:G,beginAtZero:true}}}}}}}});
new Chart(document.getElementById('allocC'),{{type:'line',data:{{labels:D.pd,datasets:[{{label:'Invested',data:D.iv,borderColor:'#c8f542',backgroundColor:'rgba(200,245,66,0.1)',borderWidth:1.5,fill:true,pointRadius:0}},{{label:'Cash',data:D.cv,borderColor:'#3b82f6',backgroundColor:'rgba(59,130,246,0.08)',borderWidth:1.5,fill:true,pointRadius:0}}]}},options:{{responsive:true,maintainAspectRatio:false,plugins:{{legend:{{position:'top',align:'end',labels:{{boxWidth:12,padding:16}}}}}},scales:{{x:{{grid:G,ticks:{{maxTicksLimit:8}},stacked:true}},y:{{grid:G,stacked:true,ticks:{{callback:v=>'$'+(v/1000).toFixed(0)+'k'}}}}}}}}}});
const pc=D.pv.map(v=>v>=0?'rgba(34,197,94,0.6)':'rgba(239,68,68,0.6)');
const pb=D.pv.map(v=>v>=0?'#22c55e':'#ef4444');
new Chart(document.getElementById('pnlC'),{{type:'bar',data:{{labels:D.pt,datasets:[{{data:D.pv,backgroundColor:pc,borderColor:pb,borderWidth:1,borderRadius:2}}]}},options:{{responsive:true,maintainAspectRatio:false,indexAxis:'y',plugins:{{legend:{{display:false}},tooltip:{{callbacks:{{label:c=>'$'+c.parsed.x.toLocaleString()}}}}}},scales:{{x:{{grid:G,ticks:{{callback:v=>'$'+(v/1000).toFixed(0)+'k'}}}},y:{{grid:{{display:false}},ticks:{{font:{{size:9}}}}}}}}}}}});

function rS(d){{const b=document.querySelector('#sT tbody');b.innerHTML=d.map(r=>`<tr><td class="tk">${{r.ticker}}</td><td class="${{r.pnl>=0?'pos':'neg'}}">${{r.pnl>=0?'+':''}}$${{r.pnl.toLocaleString(undefined,{{minimumFractionDigits:2}})}}</td><td>$${{r.total_cost.toLocaleString(undefined,{{minimumFractionDigits:2}})}}</td><td>$${{r.total_proceeds.toLocaleString(undefined,{{minimumFractionDigits:2}})}}</td><td>${{r.unrealized>0?'$'+r.unrealized.toLocaleString(undefined,{{minimumFractionDigits:2}}):'—'}}</td><td>${{r.open>0?r.open:'—'}}</td><td>${{r.buys}}</td><td>${{r.sells}}</td></tr>`).join('');document.getElementById('sN').textContent=d.length+' stocks'}}
function rc(reason){{if(reason==='STOP_LOSS')return'reason-sl';if(reason==='TREND_BREAK')return'reason-tb';if(reason==='ADDON')return'reason-ao';return''}}
function rT(d){{const b=document.querySelector('#tT tbody');b.innerHTML=d.map(r=>`<tr><td>${{r.date}}</td><td class="tk">${{r.ticker}}</td><td class="${{r.side==='BUY'?'buy':'sell'}}">${{r.side}}</td><td class="${{rc(r.reason)}}">${{r.reason}}</td><td>${{r.shares.toLocaleString()}}</td><td>$${{r.price.toFixed(2)}}</td><td>$${{r.value.toLocaleString(undefined,{{minimumFractionDigits:2}})}}</td><td>$${{r.commission.toFixed(2)}}</td><td class="${{r.cash_impact>=0?'pos':'neg'}}">${{r.cash_impact>=0?'+':''}}$${{r.cash_impact.toLocaleString(undefined,{{minimumFractionDigits:2}})}}</td><td>$${{r.cash_after.toLocaleString(undefined,{{minimumFractionDigits:2}})}}</td></tr>`).join('');document.getElementById('tN').textContent=d.length+' trades'}}
rS(D.st);rT(D.td);
document.getElementById('sF').addEventListener('input',e=>{{const q=e.target.value.toUpperCase();rS(D.st.filter(r=>r.ticker.includes(q)))}});
document.getElementById('tF').addEventListener('input',e=>{{const q=e.target.value.toUpperCase();rT(D.td.filter(r=>r.ticker.includes(q)||r.date.includes(q)||r.reason.includes(q)))}});
document.querySelectorAll('th[data-s]').forEach(th=>{{let a=true;th.addEventListener('click',()=>{{const k=th.dataset.s;const isS=th.closest('table').id==='sT';const d=isS?[...D.st]:[...D.td];d.sort((x,y)=>{{const va=x[k],vb=y[k];if(typeof va==='string')return a?va.localeCompare(vb):vb.localeCompare(va);return a?va-vb:vb-va}});th.closest('thead').querySelectorAll('th').forEach(t=>t.classList.remove('sorted'));th.classList.add('sorted');a=!a;isS?rS(d):rT(d)}});}});
</script>
</body>
</html>"""
