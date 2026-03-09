"""Portfolio backtest HTML report generator."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


def generate_html_report(
    result: dict[str, Any],
    config_summary: dict[str, Any],
    output_path: str | Path = "reports/portfolio_report.html",
) -> str:
    """Generate a self-contained HTML report from portfolio backtest results."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    stats = result["stats"]
    equity_curve = result["equity_curve"]
    daily = result["daily"]
    trades_df = result["trades"]
    per_stock_pnl = result["per_stock_pnl"]

    def _safe(v):
        """Convert NaN/inf to None for JSON serialization."""
        if isinstance(v, float) and (np.isnan(v) or np.isinf(v)):
            return None
        return round(v, 2)

    # Prepare chart data
    eq_dates = [d.strftime("%Y-%m-%d") for d in equity_curve.index]
    eq_vals = [_safe(v) for v in equity_curve.values]

    cummax = equity_curve.cummax()
    dd = [_safe(v) for v in ((equity_curve - cummax) / cummax * 100).tolist()]

    # Sample weekly for lighter charts
    step = 5
    pos_dates = [d.strftime("%Y-%m-%d") for d in daily.index[::step]]
    pos_counts = [int(v) if not np.isnan(v) else 0 for v in daily["num_positions"].values[::step]]
    cash_vals = [_safe(v) for v in daily["cash"].values[::step]]
    inv_vals = [_safe(v) for v in daily["position_value"].values[::step]]

    pnl_tickers = per_stock_pnl["ticker"].tolist() if not per_stock_pnl.empty else []
    pnl_values = per_stock_pnl["total_pnl"].tolist() if not per_stock_pnl.empty else []

    trades_data = []
    if not trades_df.empty:
        for _, t in trades_df.iterrows():
            trades_data.append({
                "date": str(t["date"])[:10], "ticker": t["ticker"],
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

    # Build HTML
    html = _build_html(
        stats, config_summary,
        eq_dates, eq_vals, dd,
        pos_dates, pos_counts, cash_vals, inv_vals,
        pnl_tickers, pnl_values,
        stock_table, trades_data,
    )

    output_path.write_text(html, encoding="utf-8")
    return str(output_path)


def _build_html(
    stats, config_summary,
    eq_dates, eq_vals, dd,
    pos_dates, pos_counts, cash_vals, inv_vals,
    pnl_tickers, pnl_values,
    stock_table, trades_data,
):
    factors_str = config_summary.get(
        "factors", "graham_pe, price_to_book, graham_number, current_ratio, dividend_yield"
    )
    factor_tags = "".join(
        f'<span class="ftag">{f.strip()}</span>' for f in factors_str.split(",")
    )

    ret_cls = "pos" if stats["Total Return [%]"] >= 0 else "neg"
    ann_cls = "pos" if stats["Annualized Return [%]"] >= 0 else "neg"

    stop_loss_exits = stats.get("Stop-Loss Exits", 0)
    expired_exits = stats.get("Holding Expired Exits", 0)
    avg_positions = stats.get("Avg Positions", 0)
    trailing_stop_pct = config_summary.get("trailing_stop_pct", "20%")
    holding_days = config_summary.get("holding_period_days", "252")
    max_pos = config_summary.get("max_positions", "15")
    max_wt = config_summary.get("max_position_weight", "25%")

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>AlphaLab Portfolio Report</title>
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
td.reason-exp{{color:var(--org)}}
td.reason-si{{color:var(--blu)}}
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
.ftag{{display:inline-block;background:var(--s3);border:1px solid var(--bd);padding:3px 8px;border-radius:3px;font-size:11px;color:var(--td);margin:2px 4px 2px 0}}

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
    <div class="logo">Alpha<span>Lab</span></div>
    <div class="pills">
      <div class="pill"><b>${config_summary.get('initial_cash','100,000')}</b> capital</div>
      <div class="pill"><b>{config_summary.get('period','2019–2024')}</b></div>
      <div class="pill"><b>{config_summary.get('universe','100')}</b> stocks</div>
      <div class="pill"><b>Daily</b> signal check</div>
      <div class="pill"><b>{trailing_stop_pct}</b> stop-loss</div>
      <div class="pill"><b>{holding_days}d</b> holding period</div>
    </div>
  </div>
</header>

<div class="kpis">
  <div class="kpi"><div class="kpi-l">Total Return</div><div class="kpi-v {ret_cls}">{stats['Total Return [%]']:+.2f}%</div></div>
  <div class="kpi"><div class="kpi-l">Annual Return</div><div class="kpi-v {ann_cls}">{stats['Annualized Return [%]']:+.2f}%</div></div>
  <div class="kpi"><div class="kpi-l">Sharpe Ratio</div><div class="kpi-v">{stats['Sharpe Ratio']:.3f}</div></div>
  <div class="kpi"><div class="kpi-l">Sortino Ratio</div><div class="kpi-v">{stats['Sortino Ratio']:.3f}</div></div>
  <div class="kpi"><div class="kpi-l">Max Drawdown</div><div class="kpi-v neg">{stats['Max Drawdown [%]']:.2f}%</div></div>
  <div class="kpi"><div class="kpi-l">Final Equity</div><div class="kpi-v acc">${stats['Equity Final [$]']:,.2f}</div></div>
  <div class="kpi"><div class="kpi-l">Total Trades</div><div class="kpi-v">{stats['# Trades']}</div></div>
  <div class="kpi"><div class="kpi-l">Commissions</div><div class="kpi-v">${stats['Total Commission [$]']:,.2f}</div></div>
  <div class="kpi"><div class="kpi-l">Avg Positions</div><div class="kpi-v">{avg_positions}</div></div>
  <div class="kpi"><div class="kpi-l">Stop-Loss Exits</div><div class="kpi-v org">{stop_loss_exits}</div></div>
  <div class="kpi"><div class="kpi-l">Holding Expired</div><div class="kpi-v">{expired_exits}</div></div>
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
  <div class="sec-t"><span class="dot"></span>Stock Contributions</div>
  <div class="cc"><canvas id="pnlC" style="height:360px!important"></canvas></div>
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
    <div class="scard"><h3><span class="n">1</span>Factor Scoring</h3>
      <p>Each stock scored daily by 5 Graham value factors, each in <code>[0,1]</code> where 1.0 = most bullish:</p>
      <div style="margin:12px 0">{factor_tags}</div>
      <ul>
        <li><b>Graham PE</b> — PE &lt; 15 is cheap. sigmoid(PE,15) + percentile rank</li>
        <li><b>Price-to-Book</b> — P/B &lt; 1.5 is undervalued. sigmoid + percentile</li>
        <li><b>Graham Number</b> — Price vs sqrt(22.5 x EPS x BVPS). Below = undervalued</li>
        <li><b>Current Ratio</b> — CR &gt; 2.0 = financially safe. Higher is better</li>
        <li><b>Dividend Yield</b> — DY &gt; 3% signals cheap stock. Higher is better</li>
      </ul>
      <p style="margin-top:12px">Formula: <code>score = 0.4 x sigmoid(raw, threshold) + 0.6 x percentile_rank</code></p>
    </div>
    <div class="scard"><h3><span class="n">2</span>Signal-Weighted Sizing</h3>
      <p>5 factor scores combined via <b>equal-weight average</b> → combined score [0,1].</p>
      <p>Combined score &gt; <code>0.6</code> → buy signal with strength = <code>(score - 0.6) / 0.4</code></p>
      <p style="margin-top:12px"><b>Position weight</b> proportional to signal strength:</p>
      <ul>
        <li>Score 0.6 → minimal weight (0.05 floor)</li>
        <li>Score 0.8 → medium weight</li>
        <li>Score 1.0 → maximum weight</li>
      </ul>
      <p style="margin-top:12px">Max <code>{max_wt}</code> per stock, max <code>{max_pos}</code> positions. 5% cash reserve.</p>
    </div>
    <div class="scard"><h3><span class="n">3</span>Entry Rules</h3>
      <p>Signal checked <b>every trading day</b> (not monthly):</p>
      <ul>
        <li>New stock crosses combined &gt; 0.6 → buy that day</li>
        <li><b>Scale-in</b>: buy 50% of target on Day 1</li>
        <li>If signal still &gt; 0.6 after 5 days → buy remaining 50%</li>
        <li>If signal drops before scale-in → stay at 50%, mark scaled-in</li>
        <li>Stocks ranked by signal strength; top candidates get priority</li>
      </ul>
      <p style="margin-top:12px">Commission: <code>{config_summary.get('commission','0.1%')}</code> per trade.</p>
    </div>
    <div class="scard"><h3><span class="n">4</span>Exit Rules</h3>
      <p>Two exit triggers, checked <b>daily</b>:</p>
      <ul>
        <li><b>Fixed holding period</b>: auto-sell after <code>{holding_days}</code> trading days (~1 year)</li>
        <li><b>Trailing stop-loss</b>: sell if price drops <code>{trailing_stop_pct}</code> from peak since entry</li>
        <li><b>No cooldown</b>: after stop-loss, can re-buy immediately if signal still good</li>
        <li>Re-buy resets holding period and trailing high to new entry price</li>
        <li>On expiry: if signal still &gt; 0.6, can re-enter same day (续仓)</li>
      </ul>
      <p style="margin-top:12px"><b>No signal-based exit</b>: if signal drops below 0.6 mid-holding, position is kept until expiry or stop-loss.</p>
    </div>
  </div>
</div>

<footer>AlphaLab — LLM-Powered Factor Investing — Generated from backtest data</footer>
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
function rc(reason){{if(reason==='STOP_LOSS')return'reason-sl';if(reason==='EXPIRED')return'reason-exp';if(reason==='SCALE_IN')return'reason-si';return''}}
function rT(d){{const b=document.querySelector('#tT tbody');b.innerHTML=d.map(r=>`<tr><td>${{r.date}}</td><td class="tk">${{r.ticker}}</td><td class="${{r.side==='BUY'?'buy':'sell'}}">${{r.side}}</td><td class="${{rc(r.reason)}}">${{r.reason}}</td><td>${{r.shares.toLocaleString()}}</td><td>$${{r.price.toFixed(2)}}</td><td>$${{r.value.toLocaleString(undefined,{{minimumFractionDigits:2}})}}</td><td>$${{r.commission.toFixed(2)}}</td><td class="${{r.cash_impact>=0?'pos':'neg'}}">${{r.cash_impact>=0?'+':''}}$${{r.cash_impact.toLocaleString(undefined,{{minimumFractionDigits:2}})}}</td><td>$${{r.cash_after.toLocaleString(undefined,{{minimumFractionDigits:2}})}}</td></tr>`).join('');document.getElementById('tN').textContent=d.length+' trades'}}
rS(D.st);rT(D.td);
document.getElementById('sF').addEventListener('input',e=>{{const q=e.target.value.toUpperCase();rS(D.st.filter(r=>r.ticker.includes(q)))}});
document.getElementById('tF').addEventListener('input',e=>{{const q=e.target.value.toUpperCase();rT(D.td.filter(r=>r.ticker.includes(q)||r.date.includes(q)||r.reason.includes(q)))}});
document.querySelectorAll('th[data-s]').forEach(th=>{{let a=true;th.addEventListener('click',()=>{{const k=th.dataset.s;const isS=th.closest('table').id==='sT';const d=isS?[...D.st]:[...D.td];d.sort((x,y)=>{{const va=x[k],vb=y[k];if(typeof va==='string')return a?va.localeCompare(vb):vb.localeCompare(va);return a?va-vb:vb-va}});th.closest('thead').querySelectorAll('th').forEach(t=>t.classList.remove('sorted'));th.classList.add('sorted');a=!a;isS?rS(d):rT(d)}});}});
</script>
</body>
</html>"""
