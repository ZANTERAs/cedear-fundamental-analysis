#!/usr/bin/env python3
"""
Fundamental Analysis & Intrinsic Value Calculator
Usage: python fundamental_analysis.py TICKER [--growth RATE] [--wacc RATE] [--years N] [--pdf] [--peers]

Models: DCF, Graham Number, EV/EBITDA multiples, DDM (Gordon Growth)
Data:   Yahoo Finance via yfinance — works for US stocks and CEDEAR underlyings
"""

import argparse
import math
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import date
from typing import Optional

import yfinance as yf
from rich import box
from rich.console import Console
from rich.panel import Panel
from rich.rule import Rule
from rich.table import Table

console = Console()

# ── Constants ─────────────────────────────────────────────────────────────────
RISK_FREE_RATE = 0.045
EQUITY_RISK_PREMIUM = 0.055
TERMINAL_GROWTH = 0.025
DEFAULT_WACC = 0.09
TAX_RATE = 0.21
EQUITY_WEIGHT = 0.70

SECTOR_MULTIPLES = {
    "Technology": 20.0,
    "Consumer Cyclical": 12.0,
    "Healthcare": 14.0,
    "Financial Services": 10.0,
    "Communication Services": 13.0,
    "Industrials": 12.0,
    "Consumer Defensive": 11.0,
    "Energy": 7.0,
    "Basic Materials": 8.0,
    "Real Estate": 15.0,
    "Utilities": 10.0,
}

# Exchanges considered primary US listings (exclude OTC/Pink Sheets)
PRIMARY_EXCHANGES = {"NYQ", "NMS", "NGM", "PCX", "BTS", "NYSEArca"}


# ── Formatting helpers ────────────────────────────────────────────────────────

def safe_get(d: dict, *keys, default=None):
    for k in keys:
        v = d.get(k)
        if v is not None and v != "N/A":
            return v
    return default


def fmt_currency(val, decimals=2) -> str:
    if val is None:
        return "N/A"
    try:
        val = float(val)
    except (TypeError, ValueError):
        return "N/A"
    if abs(val) >= 1e12:
        return f"${val/1e12:.{decimals}f}T"
    if abs(val) >= 1e9:
        return f"${val/1e9:.{decimals}f}B"
    if abs(val) >= 1e6:
        return f"${val/1e6:.{decimals}f}M"
    return f"${val:.{decimals}f}"


def fmt_pct(val, decimals=1) -> str:
    if val is None:
        return "N/A"
    return f"{float(val)*100:.{decimals}f}%"


def fmt_ratio(val, decimals=2) -> str:
    if val is None:
        return "N/A"
    v = float(val)
    if v < 0 or v > 999:
        return "N/M"
    if v >= 100:  # decimals are noise on large multiples; keep the cell narrow
        return f"{v:.0f}x"
    return f"{v:.{decimals}f}x"


def color_upside(pct: Optional[float]) -> str:
    if pct is None:
        return "N/A"
    s = f"{pct*100:+.1f}%"
    if pct > 0.10:
        return f"[bold green]{s}[/bold green]"
    if pct > 0:
        return f"[green]{s}[/green]"
    if pct < -0.10:
        return f"[bold red]{s}[/bold red]"
    return f"[red]{s}[/red]"


def plain_upside(pct: Optional[float]) -> str:
    if pct is None:
        return "--"
    return f"{pct*100:+.1f}%"


# ── Valuation models ──────────────────────────────────────────────────────────

def dcf_valuation(
    info: dict, cashflow_df, growth_rate: float, wacc: float, years: int
) -> Optional[float]:
    try:
        inputs = _extract_fcf_inputs(info, cashflow_df)
        if inputs is None:
            return None
        fcf, shares, net_debt = inputs
        pv_fcfs = sum(
            fcf * (1 + growth_rate) ** i / (1 + wacc) ** i
            for i in range(1, years + 1)
        )
        terminal_fcf = fcf * (1 + growth_rate) ** years * (1 + TERMINAL_GROWTH)
        tv = terminal_fcf / (wacc - TERMINAL_GROWTH)
        pv_tv = tv / (1 + wacc) ** years
        ev = pv_fcfs + pv_tv
        return max((ev - net_debt) / shares, 0)
    except Exception:
        return None


def _extract_fcf_inputs(info: dict, cashflow_df):
    """Shared FCF extraction for DCF / Monte Carlo. Returns (fcf, shares, net_debt) or None."""
    shares = safe_get(info, "sharesOutstanding")
    if not shares:
        return None
    op_cf = None
    capex = 0.0
    if cashflow_df is not None and not cashflow_df.empty:
        for key in ["Operating Cash Flow", "Total Cash From Operating Activities"]:
            if key in cashflow_df.index:
                op_cf = float(cashflow_df.loc[key].iloc[0])
                break
        for key in ["Capital Expenditure", "Capital Expenditures"]:
            if key in cashflow_df.index:
                capex = float(cashflow_df.loc[key].iloc[0])
                break
    if op_cf is None:
        return None
    fcf = op_cf + capex
    if fcf <= 0:
        return None
    total_debt = float(safe_get(info, "totalDebt", default=0) or 0)
    cash = float(safe_get(info, "totalCash", default=0) or 0)
    return fcf, float(shares), total_debt - cash


def monte_carlo_dcf(
    info: dict, cashflow_df, base_growth: float, base_wacc: float, years: int,
    n_sims: int = 10000, growth_std: float = 0.03, wacc_std: float = 0.01,
    tg_std: float = 0.005, seed: int = 42,
) -> Optional[dict]:
    """
    Vectorized Monte Carlo DCF. Samples growth / WACC / terminal-growth from
    normal distributions and returns a per-share intrinsic-value distribution.
    """
    import numpy as np

    inputs = _extract_fcf_inputs(info, cashflow_df)
    if inputs is None:
        return None
    fcf, shares, net_debt = inputs

    rng = np.random.default_rng(seed)
    g  = np.clip(rng.normal(base_growth, growth_std, n_sims), -0.50, 1.00)
    tg = np.clip(rng.normal(TERMINAL_GROWTH, tg_std, n_sims), -0.01, 0.06)
    w  = rng.normal(base_wacc, wacc_std, n_sims)
    w  = np.maximum(w, tg + 0.01)  # WACC must exceed terminal growth

    t  = np.arange(1, years + 1).reshape(-1, 1)   # (years, 1)
    g_ = g.reshape(1, -1)                          # (1, n)
    w_ = w.reshape(1, -1)

    pv_explicit  = (fcf * (1 + g_) ** t / (1 + w_) ** t).sum(axis=0)  # (n,)
    terminal_fcf = fcf * (1 + g) ** years * (1 + tg)
    tv           = terminal_fcf / (w - tg)
    pv_tv        = tv / (1 + w) ** years
    iv           = np.maximum((pv_explicit + pv_tv - net_debt) / shares, 0.0)

    return {
        "iv":     iv,
        "mean":   float(np.mean(iv)),
        "median": float(np.median(iv)),
        "std":    float(np.std(iv)),
        "p10":    float(np.percentile(iv, 10)),
        "p25":    float(np.percentile(iv, 25)),
        "p50":    float(np.percentile(iv, 50)),
        "p75":    float(np.percentile(iv, 75)),
        "p90":    float(np.percentile(iv, 90)),
        "n_sims": n_sims,
        "growth_std": growth_std,
        "wacc_std":   wacc_std,
        "tg_std":     tg_std,
    }


def graham_number(info: dict) -> Optional[float]:
    eps  = safe_get(info, "trailingEps", "epsTrailingTwelveMonths")
    bvps = safe_get(info, "bookValue")
    if eps and bvps and float(eps) > 0 and float(bvps) > 0:
        return math.sqrt(22.5 * float(eps) * float(bvps))
    return None


def ddm_valuation(info: dict) -> Optional[float]:
    div_rate = safe_get(info, "dividendRate")
    if not div_rate or float(div_rate) == 0:
        return None
    beta = float(safe_get(info, "beta", default=1.0) or 1.0)
    ke   = RISK_FREE_RATE + beta * EQUITY_RISK_PREMIUM
    g_div = 0.04
    d1    = float(div_rate) * (1 + g_div)
    if ke <= g_div:
        return None
    return d1 / (ke - g_div)


def ev_ebitda_valuation(info: dict, financials_df) -> Optional[float]:
    try:
        sector = safe_get(info, "sector", default="Technology")
        target_multiple = SECTOR_MULTIPLES.get(sector, 12.0)
        ebitda = safe_get(info, "ebitda")
        if ebitda is None and financials_df is not None and not financials_df.empty:
            for key in ["EBITDA", "Normalized EBITDA"]:
                if key in financials_df.index:
                    ebitda = float(financials_df.loc[key].iloc[0])
                    break
        if not ebitda or float(ebitda) <= 0:
            return None
        shares     = safe_get(info, "sharesOutstanding")
        if not shares:
            return None
        total_debt = float(safe_get(info, "totalDebt", default=0) or 0)
        cash       = float(safe_get(info, "totalCash", default=0) or 0)
        return max((float(ebitda) * target_multiple - total_debt + cash) / shares, 0)
    except Exception:
        return None


# ── Peer comparison ───────────────────────────────────────────────────────────

def fetch_peer_tickers(sector_key: str, industry_key: str, exclude: str, max_peers: int = 8) -> list:
    """
    Use yfinance's Industry / Sector API to get same-industry US peers,
    ranked by market weight. Falls back to the broader sector if the
    industry list is unavailable or has fewer than 4 names.

    sector_key / industry_key are Yahoo slugs (info["sectorKey"] / info["industryKey"]),
    e.g. "technology" / "software-application".
    """
    def top_symbols(getter, key) -> list:
        if not key:
            return []
        try:
            tc = getter(key).top_companies
            if tc is None or tc.empty:
                return []
            return [s for s in tc.index.tolist() if s.upper() != exclude.upper()]
        except Exception:
            return []

    tickers = top_symbols(yf.Industry, industry_key)
    if len(tickers) < 4:
        tickers = top_symbols(yf.Sector, sector_key)

    return tickers[:max_peers]


def fetch_peers_info(tickers: list) -> dict:
    """Fetch yfinance info for a list of tickers in parallel."""
    def _fetch(t):
        try:
            return t, yf.Ticker(t).info
        except Exception:
            return t, {}

    results = {}
    with ThreadPoolExecutor(max_workers=min(len(tickers), 10)) as ex:
        for ticker, info in ex.map(lambda t: _fetch(t), tickers):
            results[ticker] = info
    return results


def build_peer_rows(peers_info: dict, target_ticker: str, target_info: dict) -> list:
    """
    Build a list of dicts with comparison metrics for target + each peer.
    Target is always first.
    """
    def metrics(ticker, info):
        rev_growth = safe_get(info, "revenueGrowth")
        earn_growth = safe_get(info, "earningsGrowth")
        return {
            "ticker":       ticker.upper(),
            "name":         (safe_get(info, "shortName", default=ticker) or ticker)[:22],
            "market_cap":   safe_get(info, "marketCap"),
            "price":        safe_get(info, "currentPrice", "regularMarketPrice"),
            "pe_ttm":       safe_get(info, "trailingPE"),
            "pe_fwd":       safe_get(info, "forwardPE"),
            "ev_ebitda":    safe_get(info, "enterpriseToEbitda"),
            "roe":          safe_get(info, "returnOnEquity"),
            "net_margin":   safe_get(info, "profitMargins"),
            "op_margin":    safe_get(info, "operatingMargins"),
            "rev_growth":   rev_growth,
            "earn_growth":  earn_growth,
            "debt_equity":  safe_get(info, "debtToEquity"),
            "current_ratio":safe_get(info, "currentRatio"),
            "is_target":    ticker.upper() == target_ticker.upper(),
        }

    rows = [metrics(target_ticker, target_info)]
    for t, info in peers_info.items():
        if info:
            rows.append(metrics(t, info))
    return rows


# Direction: True = higher is better (margins, ROE), False = lower is better (multiples, D/E)
_METRIC_DIRECTION = {
    "pe_ttm":        False,
    "pe_fwd":        False,
    "ev_ebitda":     False,
    "roe":           True,
    "net_margin":    True,
    "op_margin":     True,
    "rev_growth":    True,
    "earn_growth":   True,
    "debt_equity":   False,
    "current_ratio": True,
}


# Ratio metrics that render as "N/M" when negative or absurdly large (see fmt_ratio)
_NM_RATIO_METRICS = {"pe_ttm", "pe_fwd", "ev_ebitda", "current_ratio"}


def _is_colorable(metric: str, val) -> bool:
    """A value is colorable only if it is a real, displayable number (not N/M)."""
    if val is None:
        return False
    v = float(val)
    if metric in _NM_RATIO_METRICS and (v < 0 or v > 999):
        return False
    return True


# Gradient buckets keyed on a value's "goodness" percentile within its column
# (1.0 = best in column, 0.0 = worst). Order matters: checked high -> low.
_GRADIENT_BUCKETS = [
    (0.75, "green"),
    (0.50, "yellow"),
    (0.25, "orange"),
    (0.00, "red"),
]


def _gradient_color(goodness: float) -> str:
    for thr, name in _GRADIENT_BUCKETS:
        if goodness >= thr:
            return name
    return "red"


# Rich foreground colors for each gradient bucket (hex reads on dark + light terms)
_TERM_GRADIENT = {
    "green":  "#22c55e",
    "yellow": "#d4a017",
    "orange": "#f97316",
    "red":    "#ef4444",
}


def compute_peer_colors(rows: list) -> dict:
    """
    Returns {(row_idx, metric_key): 'green'|'yellow'|'orange'|'red'} for every
    numeric cell. Each cell is colored on a gradient by its rank within the
    column: goodness = share of values it is at least as good as (direction-aware).
    N/M cells (negative/extreme multiples) are left uncolored and excluded.
    """
    result = {}
    for metric, higher_better in _METRIC_DIRECTION.items():
        valid = [
            (i, float(r[metric]))
            for i, r in enumerate(rows)
            if _is_colorable(metric, r.get(metric))
        ]
        if len(valid) < 2:
            continue
        vals = [v for _, v in valid]
        n = len(vals)
        for idx, val in valid:
            if higher_better:
                goodness = sum(1 for u in vals if val >= u) / n
            else:
                goodness = sum(1 for u in vals if val <= u) / n
            result[(idx, metric)] = _gradient_color(goodness)
    return result


def render_peer_comparison_terminal(rows: list, cell_colors: dict):
    console.print(Rule("[bold yellow]Peer Comparison[/bold yellow]", style="yellow"))

    tbl = Table(box=box.ROUNDED, show_header=True, header_style="bold yellow", expand=False)
    tbl.add_column("Ticker",     style="bold",    min_width=7,  no_wrap=True)
    tbl.add_column("Company",     min_width=14, max_width=22, no_wrap=True, overflow="crop")
    tbl.add_column("Mkt Cap",    justify="right",  min_width=8,  no_wrap=True)
    tbl.add_column("P/E",        justify="right",  min_width=6,  no_wrap=True)
    tbl.add_column("Fwd P/E",    justify="right",  min_width=7,  no_wrap=True)
    tbl.add_column("EV/EBITDA",  justify="right",  min_width=9,  no_wrap=True)
    tbl.add_column("ROE",        justify="right",  min_width=7,  no_wrap=True)
    tbl.add_column("Net Mgn",    justify="right",  min_width=7,  no_wrap=True)
    tbl.add_column("Op Mgn",     justify="right",  min_width=7,  no_wrap=True)
    tbl.add_column("Rev Gr",     justify="right",  min_width=7,  no_wrap=True)
    tbl.add_column("EPS Gr",     justify="right",  min_width=7,  no_wrap=True)
    tbl.add_column("D/E",        justify="right",  min_width=6,  no_wrap=True)
    tbl.add_column("Curr Ratio", justify="right",  min_width=9,  no_wrap=True)

    # Crop cleanly (ASCII) instead of inserting a Unicode ellipsis when the
    # table is wider than the console — avoids garbled output on cp1252.
    for col in tbl.columns:
        col.overflow = "crop"

    # Metric keys in column order (None = no color applied)
    _COL_METRICS = [
        None, None, None,
        "pe_ttm", "pe_fwd", "ev_ebitda",
        "roe", "net_margin", "op_margin",
        "rev_growth", "earn_growth",
        "debt_equity", "current_ratio",
    ]

    for row_idx, r in enumerate(rows):
        de = r["debt_equity"]
        raw_vals = [
            r["ticker"],
            r["name"],
            fmt_currency(r["market_cap"], 1),
            fmt_ratio(r["pe_ttm"]),
            fmt_ratio(r["pe_fwd"]),
            fmt_ratio(r["ev_ebitda"]),
            fmt_pct(r["roe"]),
            fmt_pct(r["net_margin"]),
            fmt_pct(r["op_margin"]),
            fmt_pct(r["rev_growth"]),
            fmt_pct(r["earn_growth"]),
            f"{de:.1f}" if de else "N/A",
            fmt_ratio(r["current_ratio"]),
        ]

        cells = []
        for col_idx, metric in enumerate(_COL_METRICS):
            val = raw_vals[col_idx]
            if metric and (row_idx, metric) in cell_colors:
                c = _TERM_GRADIENT.get(cell_colors[(row_idx, metric)], "white")
                # Target row: bold color; peer rows: plain color
                tag = f"bold {c}" if r["is_target"] else c
                cells.append(f"[{tag}]{val}[/{tag}]")
            elif r["is_target"]:
                cells.append(f"[bold cyan]{val}[/bold cyan]")
            else:
                cells.append(val)

        tbl.add_row(*cells)

    console.print(tbl)
    console.print()


# ── Monte Carlo renderer ──────────────────────────────────────────────────────

def render_monte_carlo_terminal(mc: dict, price: Optional[float]):
    import numpy as np

    console.print(Rule("[bold green]Monte Carlo DCF[/bold green]", style="green"))
    iv = mc["iv"]
    prob_under = float(np.mean(iv > price)) if price else None

    tbl = Table(box=box.ROUNDED, show_header=True, header_style="bold green")
    tbl.add_column("Statistic", style="bold", min_width=20)
    tbl.add_column("Intrinsic Value", justify="right", min_width=16)
    tbl.add_row("Simulations", f"{mc['n_sims']:,}")
    tbl.add_row("Mean", f"${mc['mean']:.2f}")
    tbl.add_row("Std Dev", f"${mc['std']:.2f}")
    tbl.add_section()
    tbl.add_row("P10  (pessimistic)", f"${mc['p10']:.2f}")
    tbl.add_row("P25", f"${mc['p25']:.2f}")
    tbl.add_row("P50  (median)", f"${mc['p50']:.2f}")
    tbl.add_row("P75", f"${mc['p75']:.2f}")
    tbl.add_row("P90  (optimistic)", f"${mc['p90']:.2f}")
    if price:
        tbl.add_section()
        tbl.add_row("Current Price", f"${price:.2f}")
        color = "bold green" if prob_under >= 0.5 else "bold red"
        tbl.add_row("P(undervalued)", f"[{color}]{prob_under*100:.1f}%[/{color}]")
    console.print(tbl)
    console.print()

    counts, edges = np.histogram(iv, bins=20)
    max_count = max(int(counts.max()), 1)
    bar_width = 42
    console.print("[bold]Distribution of Intrinsic Value per Share[/bold]")
    for i in range(len(counts)):
        lo, hi = float(edges[i]), float(edges[i + 1])
        bar_len = int(round(counts[i] / max_count * bar_width))
        bar = "#" * bar_len
        line = f"  {lo:8.2f} - {hi:<8.2f} | {bar}"
        if price is not None and lo <= price < hi:
            console.print(f"[yellow]{line}  <= current price[/yellow]")
        else:
            console.print(line)
    console.print()
    console.print(
        f"[dim]Sampled: growth ~ N(base, {mc['growth_std']*100:.1f}%) | "
        f"WACC ~ N(base, {mc['wacc_std']*100:.1f}%) | "
        f"terminal g ~ N({TERMINAL_GROWTH*100:.1f}%, {mc['tg_std']*100:.1f}%)[/dim]\n"
    )


def _mc_histogram_png(mc: dict, price: Optional[float], path: str):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    iv = mc["iv"]
    fig, ax = plt.subplots(figsize=(7.0, 3.0), dpi=150)
    ax.hist(iv, bins=45, color="#2563eb", edgecolor="white", linewidth=0.3, alpha=0.85)
    ax.axvline(mc["median"], color="#15803d", linestyle="--", linewidth=1.4,
               label=f"Median ${mc['median']:.2f}")
    ax.axvline(mc["p10"], color="#9ca3af", linestyle=":", linewidth=1.0,
               label=f"P10 ${mc['p10']:.2f}")
    ax.axvline(mc["p90"], color="#9ca3af", linestyle=":", linewidth=1.0,
               label=f"P90 ${mc['p90']:.2f}")
    if price:
        ax.axvline(price, color="#dc2626", linestyle="-", linewidth=1.6,
                   label=f"Price ${price:.2f}")
    ax.set_xlabel("Intrinsic Value per Share ($)", fontsize=8)
    ax.set_ylabel("Frequency", fontsize=8)
    ax.tick_params(labelsize=7)
    ax.legend(fontsize=7, loc="upper right", framealpha=0.9)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


# ── Historical valuation ──────────────────────────────────────────────────────

def _series_stats(dates, vals, current):
    """Summarize a multiple series: mean / std / min / max / percentile / z-score of current."""
    import numpy as np
    arr = np.asarray(vals, dtype=float)
    if arr.size < 6:
        return None
    mean = float(arr.mean())
    std  = float(arr.std())
    cur  = float(current) if current is not None else float(arr[-1])
    pct  = float((arr <= cur).mean())
    z    = (cur - mean) / std if std > 0 else 0.0
    return {
        "dates": list(dates), "vals": [float(v) for v in arr],
        "current": cur, "mean": mean, "std": std,
        "min": float(arr.min()), "max": float(arr.max()),
        "pct": pct, "z": z,
    }


def historical_valuation(t, info: dict, years: int = 5) -> Optional[dict]:
    """
    Build monthly P/E and P/S series over the past N years vs their own history.
    TTM EPS / revenue-per-share are stepped from annual filings (forward-filled),
    with the most recent stub overridden by the current TTM figures.
    """
    import pandas as pd

    try:
        hist = t.history(period=f"{years}y", interval="1mo")
    except Exception:
        return None
    if hist is None or hist.empty or "Close" not in hist.columns:
        return None
    closes = hist["Close"].dropna()
    if len(closes) < 12:
        return None
    idx = closes.index
    if getattr(idx, "tz", None) is not None:
        idx = idx.tz_localize(None)

    try:
        inc = t.income_stmt
    except Exception:
        inc = None

    shares       = safe_get(info, "sharesOutstanding")
    trailing_eps = safe_get(info, "trailingEps")
    cur_pe       = safe_get(info, "trailingPE")
    cur_ps       = safe_get(info, "priceToSalesTrailingTwelveMonths")

    eps_points, rps_points = [], []
    if inc is not None and not inc.empty:
        eps_row = None
        for k in ["Diluted EPS", "Basic EPS"]:
            if k in inc.index:
                eps_row = inc.loc[k]
                break
        rev_row = inc.loc["Total Revenue"] if "Total Revenue" in inc.index else None
        for col in inc.columns:
            d = pd.Timestamp(col)
            if getattr(d, "tz", None) is not None:
                d = d.tz_localize(None)
            if eps_row is not None and pd.notna(eps_row.get(col)):
                eps_points.append((d, float(eps_row[col])))
            if rev_row is not None and shares and pd.notna(rev_row.get(col)):
                rps_points.append((d, float(rev_row[col]) / float(shares)))
    eps_points.sort()
    rps_points.sort()

    def step_value(points, d, latest_override=None):
        if not points:
            return None
        val = points[0][1]
        for pd_date, v in points:
            if pd_date <= d:
                val = v
        if latest_override is not None and d > points[-1][0]:
            val = latest_override
        return val

    pe_dates, pe_vals, ps_dates, ps_vals = [], [], [], []
    for d, price in zip(idx, closes.values):
        d = pd.Timestamp(d)
        eps = step_value(eps_points, d, latest_override=trailing_eps)
        if eps and eps > 0:
            pe_dates.append(d)
            pe_vals.append(price / eps)
        rps = step_value(rps_points, d)
        if rps and rps > 0:
            ps_dates.append(d)
            ps_vals.append(price / rps)

    pe = _series_stats(pe_dates, pe_vals, cur_pe if cur_pe else (pe_vals[-1] if pe_vals else None))
    ps = _series_stats(ps_dates, ps_vals, cur_ps if cur_ps else (ps_vals[-1] if ps_vals else None))
    if pe is None and ps is None:
        return None
    return {"years": years, "pe": pe, "ps": ps}


def _hv_verdict(z: float) -> tuple:
    """Return (short_label, rich_color) for a z-score of the current multiple vs history."""
    if z >= 1.0:
        return "expensive", "red"
    if z >= 0.4:
        return "above avg", "yellow"
    if z <= -1.0:
        return "cheap", "green"
    if z <= -0.4:
        return "below avg", "green"
    return "average", "cyan"


def render_historical_valuation_terminal(hv: dict):
    console.print(Rule("[bold blue]Historical Valuation[/bold blue]", style="blue"))

    tbl = Table(box=box.ROUNDED, show_header=True, header_style="bold blue")
    tbl.add_column("Multiple", style="bold", min_width=10)
    tbl.add_column("Current", justify="right", min_width=9)
    tbl.add_column(f"{hv['years']}Y Avg", justify="right", min_width=9)
    tbl.add_column(f"{hv['years']}Y Low", justify="right", min_width=9)
    tbl.add_column(f"{hv['years']}Y High", justify="right", min_width=9)
    tbl.add_column("Pctile", justify="right", min_width=7)
    tbl.add_column("vs Avg", justify="left", min_width=16, no_wrap=True)
    for col in tbl.columns:
        col.overflow = "crop"

    def add(label, s):
        if not s:
            return
        verdict, color = _hv_verdict(s["z"])
        tbl.add_row(
            label,
            f"{s['current']:.1f}x",
            f"{s['mean']:.1f}x",
            f"{s['min']:.1f}x",
            f"{s['max']:.1f}x",
            f"{s['pct']*100:.0f}%",
            f"[{color}]{s['z']:+.1f} sd  {verdict}[/{color}]",
        )

    add("P/E (TTM)", hv["pe"])
    add("P/S (TTM)", hv["ps"])
    console.print(tbl)
    console.print()
    console.print(
        "[dim]Multiples computed from monthly prices and trailing fundamentals "
        "(annual filings, forward-filled). Percentile = share of history at or below "
        "the current level.[/dim]\n"
    )


def _hist_valuation_png(hv: dict, path: str):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.dates as mdates

    panels = [(k, hv[k], title) for k, title in (("pe", "P/E (TTM)"), ("ps", "P/S (TTM)")) if hv.get(k)]
    n = len(panels)
    fig, axes = plt.subplots(n, 1, figsize=(7.0, 2.0 * n + 0.4), dpi=150, sharex=True)
    if n == 1:
        axes = [axes]

    for ax, (key, s, title) in zip(axes, panels):
        dates, vals = s["dates"], s["vals"]
        ax.plot(dates, vals, color="#2563eb", linewidth=1.3, label=title)
        ax.axhline(s["mean"], color="#374151", linestyle="--", linewidth=1.0,
                   label=f"Avg {s['mean']:.1f}x")
        ax.fill_between(dates, s["mean"] - s["std"], s["mean"] + s["std"],
                        color="#93c5fd", alpha=0.25, label="+/-1 sigma")
        ax.scatter([dates[-1]], [s["current"]], color="#dc2626", zorder=5, s=22,
                   label=f"Current {s['current']:.1f}x")
        ax.set_ylabel(title, fontsize=8)
        ax.tick_params(labelsize=7)
        ax.legend(fontsize=6.5, loc="upper left", framealpha=0.9, ncol=2)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    axes[-1].xaxis.set_major_locator(mdates.YearLocator())
    axes[-1].xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


# ── Terminal renderer ─────────────────────────────────────────────────────────

def render_terminal(data: dict):
    financials  = data["financials"]
    cashflow    = data["cashflow"]
    name        = data["name"]
    ticker      = data["ticker"]
    sector      = data["sector"]
    industry    = data["industry"]
    country     = data["country"]
    exchange    = data["exchange"]
    market_cap  = data["market_cap"]
    price       = data["current_price"]
    h52         = data["h52"]
    l52         = data["l52"]
    dcf_val     = data["dcf_val"]
    graham      = data["graham"]
    ddm_val     = data["ddm_val"]
    ev_val      = data["ev_val"]
    composite   = data["composite"]
    growth_rate = data["growth_rate"]
    wacc        = data["wacc"]
    years       = data["years"]

    def upside(iv):
        if iv and price:
            return (iv - price) / price
        return None

    console.print()
    console.print(Rule(f"[bold white]{name}  ({ticker.upper()})[/bold white]", style="bold blue"))

    ov = Table(box=box.SIMPLE, show_header=False, padding=(0, 2))
    ov.add_column("", style="bold dim", min_width=14)
    ov.add_column("", min_width=22)
    ov.add_column("", style="bold dim", min_width=14)
    ov.add_column("")
    ov.add_row("Sector",     sector,   "Exchange", exchange)
    ov.add_row("Industry",   industry, "Country",  country)
    ov.add_row("Market Cap", fmt_currency(market_cap), "Price", f"${price:.2f}" if price else "N/A")
    ov.add_row("52W High",   f"${h52:.2f}" if h52 else "N/A", "52W Low", f"${l52:.2f}" if l52 else "N/A")
    console.print(ov)

    console.print(Rule("[bold cyan]Financials[/bold cyan]", style="cyan"))

    r = data["ratios"]
    ratios = Table(title="Key Ratios & Metrics", box=box.ROUNDED, show_header=True, header_style="bold cyan")
    ratios.add_column("Metric", style="bold")
    ratios.add_column("Value", justify="right")
    ratios.add_column("Metric", style="bold")
    ratios.add_column("Value", justify="right")
    de = r["de"]
    ratios.add_row("P/E (TTM)", fmt_ratio(r["pe"]),        "ROE",          fmt_pct(r["roe"]))
    ratios.add_row("P/E (Fwd)", fmt_ratio(r["fpe"]),       "ROA",          fmt_pct(r["roa"]))
    ratios.add_row("P/B",       fmt_ratio(r["pb"]),         "Gross Margin", fmt_pct(r["gross_m"]))
    ratios.add_row("P/S",       fmt_ratio(r["ps"]),         "Net Margin",   fmt_pct(r["net_m"]))
    ratios.add_row("EV/EBITDA", fmt_ratio(r["ev_ebitda"]),  "Op. Margin",   fmt_pct(r["op_m"]))
    ratios.add_row("PEG Ratio", fmt_ratio(r["peg"]),        "Debt/Equity",  f"{de:.2f}" if de else "N/A")
    ratios.add_row("Beta",      f"{r['beta']:.2f}" if r["beta"] else "N/A", "Current Ratio", fmt_ratio(r["curr"]))
    console.print(ratios)
    console.print()

    if financials is not None and not financials.empty:
        hist = Table(title="Financial History (Annual)", box=box.ROUNDED, show_header=True, header_style="bold cyan")
        hist.add_column("Metric", style="bold", min_width=22)
        n_cols  = min(4, financials.shape[1])
        yr_cols = list(financials.columns[:n_cols])
        for col in yr_cols:
            hist.add_column(str(col.year), justify="right")

        def fin_row(label, *keys):
            for k in keys:
                if k in financials.index:
                    hist.add_row(label, *[fmt_currency(financials.loc[k, c]) for c in yr_cols])
                    return
            hist.add_row(label, *["N/A"] * len(yr_cols))

        fin_row("Revenue",      "Total Revenue")
        fin_row("Gross Profit", "Gross Profit")
        fin_row("EBITDA",       "EBITDA", "Normalized EBITDA")
        fin_row("Net Income",   "Net Income", "Net Income Common Stockholders")

        if cashflow is not None and not cashflow.empty:
            cf_cols = [c for c in yr_cols if c in cashflow.columns]
            def cf_row(label, *keys):
                for k in keys:
                    if k in cashflow.index:
                        hist.add_row(label, *[
                            fmt_currency(cashflow.loc[k, c]) if c in cf_cols else "N/A"
                            for c in yr_cols
                        ])
                        return
                hist.add_row(label, *["N/A"] * len(yr_cols))
            cf_row("Operating CF", "Operating Cash Flow", "Total Cash From Operating Activities")
            cf_row("CapEx",        "Capital Expenditure", "Capital Expenditures")
        console.print(hist)
    console.print()

    console.print(Rule("[bold magenta]Intrinsic Value Models[/bold magenta]", style="magenta"))
    sector_multiple = SECTOR_MULTIPLES.get(sector, 12.0)
    vt = Table(box=box.ROUNDED, show_header=True, header_style="bold magenta")
    vt.add_column("Model", style="bold", min_width=38)
    vt.add_column("Intrinsic Value", justify="right", min_width=16)
    vt.add_column("vs. Current", justify="right", min_width=12)

    vt.add_row(
        f"DCF  (g={growth_rate*100:.1f}%  WACC={wacc*100:.1f}%  {years}yr)",
        f"${dcf_val:.2f}" if dcf_val else "[dim]N/A (negative FCF)[/dim]",
        color_upside(upside(dcf_val)) if dcf_val else "--",
    )
    vt.add_row(
        "Graham Number  (sqrt(22.5 * EPS * BVPS))",
        f"${graham:.2f}" if graham else "[dim]N/A[/dim]",
        color_upside(upside(graham)) if graham else "--",
    )
    vt.add_row(
        f"EV/EBITDA  ({sector_multiple:.1f}x  {sector} median)",
        f"${ev_val:.2f}" if ev_val else "[dim]N/A[/dim]",
        color_upside(upside(ev_val)) if ev_val else "--",
    )
    vt.add_row(
        "DDM  (Gordon Growth  g=4%)",
        f"${ddm_val:.2f}" if ddm_val else "[dim]N/A (no dividend)[/dim]",
        color_upside(upside(ddm_val)) if ddm_val else "--",
    )
    vt.add_section()
    vt.add_row(
        "[bold]Composite Average[/bold]",
        f"[bold]${composite:.2f}[/bold]" if composite else "[dim]N/A[/dim]",
        color_upside(upside(composite)) if composite else "--",
    )
    console.print(vt)
    console.print()

    if composite and price:
        pct = upside(composite)
        if pct > 0.20:
            verdict = "[bold green]UNDERVALUED[/bold green]   Composite IV suggests significant upside"
        elif pct > 0.05:
            verdict = "[green]SLIGHTLY UNDERVALUED[/green]   Modest upside vs composite IV"
        elif pct < -0.20:
            verdict = "[bold red]OVERVALUED[/bold red]   Trading well above composite IV"
        elif pct < -0.05:
            verdict = "[red]SLIGHTLY OVERVALUED[/red]   Premium vs composite IV"
        else:
            verdict = "[yellow]FAIRLY VALUED[/yellow]   Near composite IV"
        console.print(Panel(
            f"Composite IV: [bold]${composite:.2f}[/bold]   "
            f"Current: [bold]${price:.2f}[/bold]   "
            f"Upside: {color_upside(pct)}\n\n{verdict}",
            title="[bold]Verdict[/bold]", border_style="blue", padding=(1, 3),
        ))

    console.print(
        f"\n[dim]Data: Yahoo Finance | DCF terminal g={TERMINAL_GROWTH*100:.1f}% | "
        f"Rf={RISK_FREE_RATE*100:.1f}%  ERP={EQUITY_RISK_PREMIUM*100:.1f}%\n"
        f"Not financial advice.[/dim]\n"
    )


# ── PDF renderer ──────────────────────────────────────────────────────────────

def build_pdf(path: str, data: dict, peer_rows: Optional[list] = None,
              cell_colors: Optional[dict] = None, mc: Optional[dict] = None,
              hv: Optional[dict] = None):
    import os
    import tempfile

    from reportlab.lib import colors
    from reportlab.lib.pagesizes import letter
    from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
    from reportlab.lib.units import inch
    from reportlab.platypus import (
        HRFlowable, Image, Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle,
    )

    C_NAVY    = colors.HexColor("#1e3a5f")
    C_BLUE    = colors.HexColor("#2563eb")
    C_LBLUE   = colors.HexColor("#dbeafe")
    C_LGRAY   = colors.HexColor("#f3f4f6")
    C_DGRAY   = colors.HexColor("#374151")
    C_GREEN   = colors.HexColor("#15803d")
    C_LGREEN  = colors.HexColor("#dcfce7")
    C_RED     = colors.HexColor("#dc2626")
    C_LRED    = colors.HexColor("#fee2e2")
    C_YELLOW  = colors.HexColor("#92400e")
    C_LYELLOW = colors.HexColor("#fef9c3")
    C_WHITE   = colors.white
    C_BORDER  = colors.HexColor("#cbd5e1")
    C_PURPLE  = colors.HexColor("#4c1d95")
    C_AMBER   = colors.HexColor("#b45309")

    W = letter[0] - 1.2 * inch

    base = getSampleStyleSheet()

    def sty(name, parent="Normal", **kw):
        return ParagraphStyle(name, parent=base[parent], **kw)

    S_TITLE   = sty("Title",   fontSize=20, textColor=C_NAVY, spaceAfter=2,  leading=24, fontName="Helvetica-Bold")
    S_TICKER  = sty("Ticker",  fontSize=13, textColor=C_BLUE, spaceAfter=6,  leading=16, fontName="Helvetica")
    S_SEC     = sty("Sec",     fontSize=11, textColor=C_WHITE, spaceBefore=6, spaceAfter=3, leading=14,
                    fontName="Helvetica-Bold", backColor=C_NAVY, leftIndent=6, rightIndent=6, borderPad=4)
    S_SMALL   = sty("Small",   fontSize=7.5, textColor=colors.HexColor("#6b7280"), leading=10)
    S_VERDICT = sty("Verdict", fontSize=10, textColor=C_DGRAY, leading=14, fontName="Helvetica-Bold")

    def base_ts(hdr_bg=C_NAVY, hdr_fg=C_WHITE):
        return TableStyle([
            ("BACKGROUND",    (0, 0), (-1, 0),  hdr_bg),
            ("TEXTCOLOR",     (0, 0), (-1, 0),  hdr_fg),
            ("FONTNAME",      (0, 0), (-1, 0),  "Helvetica-Bold"),
            ("FONTSIZE",      (0, 0), (-1, 0),  8),
            ("FONTNAME",      (0, 1), (-1, -1), "Helvetica"),
            ("FONTSIZE",      (0, 1), (-1, -1), 8),
            ("ROWBACKGROUNDS",(0, 1), (-1, -1), [C_WHITE, C_LGRAY]),
            ("GRID",          (0, 0), (-1, -1), 0.4, C_BORDER),
            ("VALIGN",        (0, 0), (-1, -1), "MIDDLE"),
            ("TOPPADDING",    (0, 0), (-1, -1), 4),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
            ("LEFTPADDING",   (0, 0), (-1, -1), 5),
            ("RIGHTPADDING",  (0, 0), (-1, -1), 5),
        ])

    name        = data["name"]
    ticker      = data["ticker"].upper()
    sector      = data["sector"]
    industry    = data["industry"]
    country     = data["country"]
    exchange    = data["exchange"]
    market_cap  = data["market_cap"]
    price       = data["current_price"]
    h52         = data["h52"]
    l52         = data["l52"]
    r           = data["ratios"]
    financials  = data["financials"]
    cashflow    = data["cashflow"]
    dcf_val     = data["dcf_val"]
    graham      = data["graham"]
    ddm_val     = data["ddm_val"]
    ev_val      = data["ev_val"]
    composite   = data["composite"]
    growth_rate = data["growth_rate"]
    wacc        = data["wacc"]
    years       = data["years"]

    def upside(iv):
        if iv and price:
            return (iv - price) / price
        return None

    story = []

    # ── Title ──
    story.append(Paragraph(name, S_TITLE))
    story.append(Paragraph(
        f"{ticker}  &bull;  {sector}  &bull;  {industry}  &bull;  {exchange}",
        S_TICKER
    ))
    story.append(Paragraph(
        f"Fundamental Analysis Report  &bull;  {date.today().strftime('%B %d, %Y')}",
        S_SMALL
    ))
    story.append(HRFlowable(width="100%", thickness=2, color=C_NAVY, spaceAfter=10))

    # ── Overview ──
    story.append(Paragraph("Company Overview", S_SEC))
    story.append(Spacer(1, 4))
    price_str = f"${price:.2f}" if price else "N/A"
    ov_data = [
        ["Market Cap", fmt_currency(market_cap), "Current Price", price_str],
        ["52-Week High", f"${h52:.2f}" if h52 else "N/A", "52-Week Low", f"${l52:.2f}" if l52 else "N/A"],
        ["Sector",    sector,   "Industry", industry],
        ["Exchange",  exchange, "Country",  country],
    ]
    ov_table = Table(ov_data, colWidths=[W*0.18, W*0.32, W*0.18, W*0.32])
    ov_table.setStyle(TableStyle([
        ("FONTNAME",      (0, 0), (-1, -1), "Helvetica"),
        ("FONTSIZE",      (0, 0), (-1, -1), 8.5),
        ("FONTNAME",      (0, 0), (0, -1),  "Helvetica-Bold"),
        ("FONTNAME",      (2, 0), (2, -1),  "Helvetica-Bold"),
        ("TEXTCOLOR",     (0, 0), (0, -1),  C_NAVY),
        ("TEXTCOLOR",     (2, 0), (2, -1),  C_NAVY),
        ("ROWBACKGROUNDS",(0, 0), (-1, -1), [C_WHITE, C_LGRAY]),
        ("GRID",          (0, 0), (-1, -1), 0.4, C_BORDER),
        ("VALIGN",        (0, 0), (-1, -1), "MIDDLE"),
        ("TOPPADDING",    (0, 0), (-1, -1), 5),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
        ("LEFTPADDING",   (0, 0), (-1, -1), 6),
        ("RIGHTPADDING",  (0, 0), (-1, -1), 6),
    ]))
    story.append(ov_table)
    story.append(Spacer(1, 10))

    # ── Ratios ──
    story.append(Paragraph("Key Ratios &amp; Metrics", S_SEC))
    story.append(Spacer(1, 4))
    de_str = f"{r['de']:.2f}" if r["de"] else "N/A"
    beta_str = f"{r['beta']:.2f}" if r["beta"] else "N/A"
    ratio_data = [
        ["Metric", "Value", "Metric", "Value"],
        ["P/E (TTM)",   fmt_ratio(r["pe"]),        "ROE",           fmt_pct(r["roe"])],
        ["P/E (Fwd)",   fmt_ratio(r["fpe"]),       "ROA",           fmt_pct(r["roa"])],
        ["P/B",         fmt_ratio(r["pb"]),         "Gross Margin",  fmt_pct(r["gross_m"])],
        ["P/S",         fmt_ratio(r["ps"]),         "Net Margin",    fmt_pct(r["net_m"])],
        ["EV/EBITDA",   fmt_ratio(r["ev_ebitda"]),  "Op. Margin",    fmt_pct(r["op_m"])],
        ["PEG Ratio",   fmt_ratio(r["peg"]),        "Debt / Equity", de_str],
        ["Beta",        beta_str,                    "Current Ratio", fmt_ratio(r["curr"])],
    ]
    ratio_table = Table(ratio_data, colWidths=[W*0.22, W*0.28, W*0.22, W*0.28])
    rt = base_ts()
    rt.add("ALIGN", (1, 0), (1, -1), "RIGHT")
    rt.add("ALIGN", (3, 0), (3, -1), "RIGHT")
    ratio_table.setStyle(rt)
    story.append(ratio_table)
    story.append(Spacer(1, 10))

    # ── Financial History ──
    if financials is not None and not financials.empty:
        story.append(Paragraph("Financial History (Annual)", S_SEC))
        story.append(Spacer(1, 4))
        n_cols  = min(4, financials.shape[1])
        yr_cols = list(financials.columns[:n_cols])
        yr_labels = [str(c.year) for c in yr_cols]

        def get_fin_row(label, *keys):
            for k in keys:
                if k in financials.index:
                    return [label] + [fmt_currency(financials.loc[k, c]) for c in yr_cols]
            return [label] + ["N/A"] * len(yr_cols)

        fin_rows = [
            ["Metric"] + yr_labels,
            get_fin_row("Revenue",      "Total Revenue"),
            get_fin_row("Gross Profit", "Gross Profit"),
            get_fin_row("EBITDA",       "EBITDA", "Normalized EBITDA"),
            get_fin_row("Net Income",   "Net Income", "Net Income Common Stockholders"),
        ]
        if cashflow is not None and not cashflow.empty:
            cf_cols = [c for c in yr_cols if c in cashflow.columns]
            def get_cf_row(label, *keys):
                for k in keys:
                    if k in cashflow.index:
                        return [label] + [
                            fmt_currency(cashflow.loc[k, c]) if c in cf_cols else "N/A"
                            for c in yr_cols
                        ]
                return [label] + ["N/A"] * len(yr_cols)
            fin_rows.append(get_cf_row("Operating CF", "Operating Cash Flow", "Total Cash From Operating Activities"))
            fin_rows.append(get_cf_row("CapEx", "Capital Expenditure", "Capital Expenditures"))

        lw = W * 0.26
        dw = (W - lw) / len(yr_cols)
        fin_table = Table(fin_rows, colWidths=[lw] + [dw] * len(yr_cols))
        ft = base_ts()
        for i in range(1, len(fin_rows)):
            ft.add("ALIGN", (1, i), (-1, i), "RIGHT")
        fin_table.setStyle(ft)
        story.append(fin_table)
        story.append(Spacer(1, 10))

    # ── Intrinsic Value ──
    story.append(Paragraph("Intrinsic Value Models", S_SEC))
    story.append(Spacer(1, 4))
    sector_multiple = SECTOR_MULTIPLES.get(sector, 12.0)

    def iv_row(label, iv, note=""):
        iv_str  = f"${iv:.2f}" if iv else (note or "N/A")
        ups_str = plain_upside(upside(iv))
        return [label, iv_str, ups_str]

    iv_data = [
        ["Model", "Intrinsic Value", "vs. Current Price"],
        iv_row(f"DCF  (g={growth_rate*100:.1f}%  WACC={wacc*100:.2f}%  {years}yr)", dcf_val, "N/A (negative FCF)"),
        iv_row("Graham Number  sqrt(22.5 x EPS x BVPS)", graham),
        iv_row(f"EV/EBITDA  ({sector_multiple:.1f}x  {sector} median)", ev_val),
        iv_row("DDM  Gordon Growth  g=4%", ddm_val, "N/A (no dividend)"),
        ["Composite Average",
         f"${composite:.2f}" if composite else "N/A",
         plain_upside(upside(composite))],
    ]
    iv_col_w = [W*0.52, W*0.24, W*0.24]
    iv_table = Table(iv_data, colWidths=iv_col_w)
    ivt = base_ts(hdr_bg=C_PURPLE)
    ivt.add("ALIGN",      (1, 0), (-1, -1), "RIGHT")
    ivt.add("FONTNAME",   (0, 5), (-1, 5),  "Helvetica-Bold")
    ivt.add("BACKGROUND", (0, 5), (-1, 5),  C_LBLUE)
    ivt.add("LINEABOVE",  (0, 5), (-1, 5),  1.0, C_NAVY)
    for row_i, iv in enumerate([dcf_val, graham, ev_val, ddm_val], start=1):
        u = upside(iv)
        if u is not None:
            ivt.add("TEXTCOLOR", (2, row_i), (2, row_i), C_GREEN if u > 0 else C_RED)
    if composite and price:
        u = upside(composite)
        if u is not None:
            ivt.add("TEXTCOLOR", (2, 5), (2, 5), C_GREEN if u > 0 else C_RED)
    iv_table.setStyle(ivt)
    story.append(iv_table)
    story.append(Spacer(1, 10))

    # ── Verdict ──
    if composite and price:
        pct = upside(composite)
        if pct > 0.20:
            vtext, vsub, vbg, vtc = "UNDERVALUED", "Composite IV suggests significant upside.", C_LGREEN, C_GREEN
        elif pct > 0.05:
            vtext, vsub, vbg, vtc = "SLIGHTLY UNDERVALUED", "Modest upside vs composite IV.", C_LGREEN, C_GREEN
        elif pct < -0.20:
            vtext, vsub, vbg, vtc = "OVERVALUED", "Trading well above composite intrinsic value.", C_LRED, C_RED
        elif pct < -0.05:
            vtext, vsub, vbg, vtc = "SLIGHTLY OVERVALUED", "Trading at a premium vs composite IV.", C_LRED, C_RED
        else:
            vtext, vsub, vbg, vtc = "FAIRLY VALUED", "Current price is near composite IV.", C_LYELLOW, C_YELLOW

        verdict_table = Table([[Paragraph(
            f'<font color="{vtc.hexval()}"><b>{vtext}</b></font><br/>'
            f'<font size="8">Composite IV: <b>${composite:.2f}</b>  |  '
            f'Current: <b>${price:.2f}</b>  |  '
            f'Upside: <b>{plain_upside(pct)}</b></font><br/>'
            f'<font size="8">{vsub}</font>',
            S_VERDICT
        )]], colWidths=[W])
        verdict_table.setStyle(TableStyle([
            ("BACKGROUND",    (0, 0), (-1, -1), vbg),
            ("BOX",           (0, 0), (-1, -1), 1.5, vtc),
            ("TOPPADDING",    (0, 0), (-1, -1), 10),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 10),
            ("LEFTPADDING",   (0, 0), (-1, -1), 12),
            ("RIGHTPADDING",  (0, 0), (-1, -1), 12),
        ]))
        story.append(verdict_table)
        story.append(Spacer(1, 10))

    # ── Monte Carlo DCF ──
    mc_png = None
    if mc:
        import numpy as np
        story.append(Paragraph("Monte Carlo DCF", S_SEC))
        story.append(Spacer(1, 4))

        prob_under = float(np.mean(mc["iv"] > price)) if price else None
        mc_left = [
            ["Statistic", "Value"],
            ["Simulations", f"{mc['n_sims']:,}"],
            ["Mean IV",   f"${mc['mean']:.2f}"],
            ["Median IV", f"${mc['median']:.2f}"],
            ["Std Dev",   f"${mc['std']:.2f}"],
        ]
        mc_right = [
            ["Percentile", "Value"],
            ["P10 (pessimistic)", f"${mc['p10']:.2f}"],
            ["P25", f"${mc['p25']:.2f}"],
            ["P75", f"${mc['p75']:.2f}"],
            ["P90 (optimistic)", f"${mc['p90']:.2f}"],
        ]
        cw_half = [W * 0.25, W * 0.25]
        mc_lt = Table(mc_left, colWidths=cw_half)
        mc_rt = Table(mc_right, colWidths=cw_half)
        for tt in (mc_lt, mc_rt):
            st = base_ts(hdr_bg=C_GREEN)
            st.add("ALIGN", (1, 0), (1, -1), "RIGHT")
            tt.setStyle(st)
        mc_pair = Table([[mc_lt, mc_rt]], colWidths=[W * 0.5, W * 0.5])
        mc_pair.setStyle(TableStyle([
            ("VALIGN", (0, 0), (-1, -1), "TOP"),
            ("LEFTPADDING", (0, 0), (-1, -1), 0),
            ("RIGHTPADDING", (0, 0), (-1, -1), 0),
        ]))
        story.append(mc_pair)
        story.append(Spacer(1, 6))

        if price and prob_under is not None:
            pu_color = C_GREEN if prob_under >= 0.5 else C_RED
            story.append(Paragraph(
                f'Probability the stock is undervalued (IV &gt; current '
                f'${price:.2f}): <font color="{pu_color.hexval()}"><b>'
                f'{prob_under*100:.1f}%</b></font>',
                S_VERDICT
            ))
            story.append(Spacer(1, 6))

        try:
            mc_png = os.path.join(tempfile.gettempdir(), f"_mc_{ticker}_{os.getpid()}.png")
            _mc_histogram_png(mc, price, mc_png)
            story.append(Image(mc_png, width=W, height=W * 0.43))
        except Exception:
            mc_png = None
        story.append(Spacer(1, 6))
        story.append(Paragraph(
            f"Monte Carlo over {mc['n_sims']:,} simulations. Sampled: FCF growth ~ "
            f"Normal(base, {mc['growth_std']*100:.1f}%), WACC ~ Normal(base, "
            f"{mc['wacc_std']*100:.1f}%), terminal growth ~ Normal("
            f"{TERMINAL_GROWTH*100:.1f}%, {mc['tg_std']*100:.1f}%).",
            S_SMALL
        ))
        story.append(Spacer(1, 10))

    # ── Historical Valuation ──
    hv_png = None
    if hv and (hv.get("pe") or hv.get("ps")):
        story.append(Paragraph("Historical Valuation", S_SEC))
        story.append(Spacer(1, 4))

        hv_header = ["Multiple", "Current", f"{hv['years']}Y Avg",
                     f"{hv['years']}Y Low", f"{hv['years']}Y High", "Pctile", "vs Avg"]
        hv_data = [hv_header]
        hv_color_rows = []  # (row_idx, reportlab_color)
        for label, s in (("P/E (TTM)", hv.get("pe")), ("P/S (TTM)", hv.get("ps"))):
            if not s:
                continue
            verdict, rc = _hv_verdict(s["z"])
            hv_data.append([
                label, f"{s['current']:.1f}x", f"{s['mean']:.1f}x",
                f"{s['min']:.1f}x", f"{s['max']:.1f}x", f"{s['pct']*100:.0f}%",
                f"{s['z']:+.1f} sigma ({verdict})",
            ])
            tc = C_RED if rc in ("red",) else (C_GREEN if rc == "green" else
                 (C_AMBER if rc == "yellow" else C_BLUE))
            hv_color_rows.append((len(hv_data) - 1, tc))

        hv_cw = [W*0.13, W*0.11, W*0.11, W*0.11, W*0.11, W*0.10, W*0.33]
        hv_tbl = Table(hv_data, colWidths=hv_cw)
        hvt = base_ts(hdr_bg=C_BLUE)
        hvt.add("ALIGN", (1, 0), (5, -1), "RIGHT")
        for ri, tc in hv_color_rows:
            hvt.add("TEXTCOLOR", (6, ri), (6, ri), tc)
            hvt.add("FONTNAME", (6, ri), (6, ri), "Helvetica-Bold")
        hv_tbl.setStyle(hvt)
        story.append(hv_tbl)
        story.append(Spacer(1, 6))

        try:
            hv_png = os.path.join(tempfile.gettempdir(), f"_hv_{ticker}_{os.getpid()}.png")
            _hist_valuation_png(hv, hv_png)
            n_panels = sum(1 for k in ("pe", "ps") if hv.get(k))
            story.append(Image(hv_png, width=W, height=W * (0.30 * n_panels + 0.06)))
        except Exception:
            hv_png = None
        story.append(Spacer(1, 6))
        story.append(Paragraph(
            f"Multiples over the past {hv['years']} years from monthly prices and "
            "trailing fundamentals (annual filings, forward-filled). Band = average "
            "+/-1 standard deviation. Percentile = share of history at or below today's level.",
            S_SMALL
        ))
        story.append(Spacer(1, 10))

    # ── Peer Comparison ──
    if peer_rows:
        story.append(Paragraph("Peer Comparison", S_SEC))
        story.append(Spacer(1, 4))

        peer_header = [
            "Ticker", "Company", "Mkt Cap",
            "P/E", "Fwd P/E", "EV/EBITDA",
            "ROE", "Net Mgn", "Op Mgn",
            "Rev Gr", "EPS Gr",
            "D/E", "Curr Ratio",
        ]
        peer_table_data = [peer_header]
        target_row_idx = None

        for i, row in enumerate(peer_rows, start=1):
            if row["is_target"]:
                target_row_idx = i
            de = row["debt_equity"]
            peer_table_data.append([
                row["ticker"],
                row["name"],
                fmt_currency(row["market_cap"], 1),
                fmt_ratio(row["pe_ttm"]),
                fmt_ratio(row["pe_fwd"]),
                fmt_ratio(row["ev_ebitda"]),
                fmt_pct(row["roe"]),
                fmt_pct(row["net_margin"]),
                fmt_pct(row["op_margin"]),
                fmt_pct(row["rev_growth"]),
                fmt_pct(row["earn_growth"]),
                f"{de:.1f}" if de else "N/A",
                fmt_ratio(row["current_ratio"]),
            ])

        # Column widths — narrow to fit on page (landscape-ish)
        cw = [
            W*0.07, W*0.14, W*0.08,       # ticker, name, mktcap
            W*0.06, W*0.07, W*0.08,        # P/E, fwdPE, EV/EBITDA
            W*0.06, W*0.07, W*0.07,        # ROE, NetMgn, OpMgn
            W*0.07, W*0.07,                # RevGr, EPSGr
            W*0.06, W*0.08,                # D/E, CurrRatio
        ]
        peer_tbl = Table(peer_table_data, colWidths=cw, repeatRows=1)
        pt = base_ts(hdr_bg=colors.HexColor("#78350f"), hdr_fg=C_WHITE)
        pt.add("FONTSIZE", (0, 0), (-1, -1), 7)
        for col in range(2, 13):
            pt.add("ALIGN", (col, 0), (col, -1), "RIGHT")
        # Target row highlight
        if target_row_idx is not None:
            pt.add("BACKGROUND", (0, target_row_idx), (-1, target_row_idx), C_LBLUE)
            pt.add("FONTNAME",   (0, target_row_idx), (-1, target_row_idx), "Helvetica-Bold")
            pt.add("TEXTCOLOR",  (0, target_row_idx), (-1, target_row_idx), C_NAVY)

        # Per-cell green / red backgrounds for metric columns
        _METRIC_TO_COL = {
            "pe_ttm": 3, "pe_fwd": 4, "ev_ebitda": 5,
            "roe": 6, "net_margin": 7, "op_margin": 8,
            "rev_growth": 9, "earn_growth": 10,
            "debt_equity": 11, "current_ratio": 12,
        }
        # Gradient cell backgrounds (green -> yellow -> orange -> red), light tints
        _CELL_GRADIENT = {
            "green":  colors.HexColor("#bbf7d0"),
            "yellow": colors.HexColor("#fef08a"),
            "orange": colors.HexColor("#fed7aa"),
            "red":    colors.HexColor("#fecaca"),
        }

        if cell_colors:
            for (row_idx, metric), color in cell_colors.items():
                col_idx = _METRIC_TO_COL.get(metric)
                if col_idx is None:
                    continue
                pdf_row = row_idx + 1  # +1 for header
                bg = _CELL_GRADIENT.get(color, colors.HexColor("#fecaca"))
                pt.add("BACKGROUND", (col_idx, pdf_row), (col_idx, pdf_row), bg)

        peer_tbl.setStyle(pt)
        story.append(peer_tbl)
        story.append(Spacer(1, 6))
        story.append(Paragraph(
            "Peers from Yahoo Finance same-industry top companies (ranked by market weight).  "
            "Target highlighted in blue. Cell shading ranks each metric within the column: "
            "green (best) - yellow - orange - red (worst).",
            S_SMALL
        ))
        story.append(Spacer(1, 10))

    # ── Footer ──
    story.append(HRFlowable(width="100%", thickness=0.5, color=C_BORDER, spaceBefore=6, spaceAfter=4))
    story.append(Paragraph(
        f"<b>DCF assumptions:</b> FCF growth {growth_rate*100:.1f}%/yr  |  "
        f"WACC {wacc*100:.2f}%  |  Terminal growth {TERMINAL_GROWTH*100:.1f}%  |  "
        f"Projection: {years} years  |  Rf={RISK_FREE_RATE*100:.1f}%  |  ERP={EQUITY_RISK_PREMIUM*100:.1f}%",
        S_SMALL
    ))
    story.append(Spacer(1, 3))
    story.append(Paragraph(
        "Data: Yahoo Finance via yfinance. Not financial advice.",
        S_SMALL
    ))

    doc = SimpleDocTemplate(
        path, pagesize=letter,
        leftMargin=0.6*inch, rightMargin=0.6*inch,
        topMargin=0.6*inch,  bottomMargin=0.6*inch,
    )
    doc.build(story)

    for tmp in (mc_png, hv_png):
        if tmp:
            try:
                os.remove(tmp)
            except OSError:
                pass


# ── Main analysis ─────────────────────────────────────────────────────────────

def analyze(
    ticker: str,
    growth_rate: float,
    wacc: float,
    years: int,
    pdf_path: Optional[str] = None,
    include_peers: bool = False,
    include_mc: bool = False,
    n_sims: int = 10000,
    include_hist: bool = False,
):
    console.print(f"\n[bold blue]Fetching data for [yellow]{ticker.upper()}[/yellow]...[/bold blue]")

    t = yf.Ticker(ticker)
    try:
        info = t.info
    except Exception as e:
        console.print(f"[red]Error fetching data: {e}[/red]")
        sys.exit(1)

    if not info or "shortName" not in info:
        console.print(f"[red]Ticker '{ticker}' not found or no data available.[/red]")
        sys.exit(1)

    financials = t.financials
    cashflow   = t.cashflow

    name       = safe_get(info, "shortName", "longName", default=ticker.upper())
    sector     = safe_get(info, "sector",   default="N/A")
    industry   = safe_get(info, "industry", default="N/A")
    country    = safe_get(info, "country",  default="N/A")
    exchange   = safe_get(info, "exchange", default="N/A")
    market_cap = safe_get(info, "marketCap")
    price      = safe_get(info, "currentPrice", "regularMarketPrice")
    h52        = safe_get(info, "fiftyTwoWeekHigh")
    l52        = safe_get(info, "fiftyTwoWeekLow")

    ratios = {
        "pe":        safe_get(info, "trailingPE"),
        "fpe":       safe_get(info, "forwardPE"),
        "pb":        safe_get(info, "priceToBook"),
        "ps":        safe_get(info, "priceToSalesTrailingTwelveMonths"),
        "ev_ebitda": safe_get(info, "enterpriseToEbitda"),
        "roe":       safe_get(info, "returnOnEquity"),
        "roa":       safe_get(info, "returnOnAssets"),
        "de":        safe_get(info, "debtToEquity"),
        "curr":      safe_get(info, "currentRatio"),
        "gross_m":   safe_get(info, "grossMargins"),
        "net_m":     safe_get(info, "profitMargins"),
        "op_m":      safe_get(info, "operatingMargins"),
        "beta":      safe_get(info, "beta"),
        "peg":       safe_get(info, "pegRatio"),
    }

    dcf_val   = dcf_valuation(info, cashflow, growth_rate, wacc, years)
    graham    = graham_number(info)
    ddm_val   = ddm_valuation(info)
    ev_val    = ev_ebitda_valuation(info, financials)
    models    = [v for v in [dcf_val, graham, ddm_val, ev_val] if v is not None]
    composite = sum(models) / len(models) if models else None

    data = dict(
        info=info, financials=financials, cashflow=cashflow,
        ticker=ticker, name=name, sector=sector, industry=industry,
        country=country, exchange=exchange, market_cap=market_cap,
        current_price=price, h52=h52, l52=l52, ratios=ratios,
        dcf_val=dcf_val, graham=graham, ddm_val=ddm_val,
        ev_val=ev_val, composite=composite,
        growth_rate=growth_rate, wacc=wacc, years=years,
    )

    render_terminal(data)

    # ── Monte Carlo DCF ──
    mc = None
    if include_mc:
        console.print(f"[bold blue]Running Monte Carlo DCF ({n_sims:,} simulations)...[/bold blue]")
        mc = monte_carlo_dcf(info, cashflow, growth_rate, wacc, years, n_sims=n_sims)
        if mc:
            render_monte_carlo_terminal(mc, price)
        else:
            console.print("[yellow]Monte Carlo skipped (negative or unavailable FCF).[/yellow]")

    # ── Historical valuation ──
    hv = None
    if include_hist:
        console.print("[bold blue]Building historical valuation (5Y)...[/bold blue]")
        hv = historical_valuation(t, info)
        if hv:
            render_historical_valuation_terminal(hv)
        else:
            console.print("[yellow]Historical valuation skipped (insufficient data).[/yellow]")

    # ── Peer comparison ──
    peer_rows   = None
    cell_colors = None
    if include_peers and sector != "N/A":
        console.print(f"[bold blue]Fetching peers for {industry} / {sector}...[/bold blue]")
        peer_tickers = fetch_peer_tickers(
            safe_get(info, "sectorKey"), safe_get(info, "industryKey"), ticker
        )
        if peer_tickers:
            console.print(f"[dim]Fetching data for {len(peer_tickers)} peers in parallel...[/dim]")
            peers_info  = fetch_peers_info(peer_tickers)
            peer_rows   = build_peer_rows(peers_info, ticker, info)
            cell_colors = compute_peer_colors(peer_rows)
            render_peer_comparison_terminal(peer_rows, cell_colors)
        else:
            console.print("[yellow]No peers found via screener.[/yellow]")

    if pdf_path:
        console.print(f"[bold blue]Generating PDF...[/bold blue]")
        build_pdf(pdf_path, data, peer_rows=peer_rows, cell_colors=cell_colors, mc=mc, hv=hv)
        console.print(f"[bold green]PDF saved:[/bold green] {pdf_path}")


# ── CLI ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Fundamental Analysis & Intrinsic Value — CEDEARs / US Stocks",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python fundamental_analysis.py AAPL
  python fundamental_analysis.py HWM --peers
  python fundamental_analysis.py HWM --mc
  python fundamental_analysis.py HWM --mc --peers --hist --pdf
  python fundamental_analysis.py MSFT --growth 0.12 --wacc 0.10 --mc --sims 50000 --hist --pdf
        """,
    )
    parser.add_argument("ticker", help="Stock ticker (e.g. AAPL, MSFT, HWM)")
    parser.add_argument("--growth", type=float, default=0.08,
                        help="FCF annual growth rate for DCF (default: 0.08 = 8%%)")
    parser.add_argument("--wacc",   type=float, default=None,
                        help="WACC override (default: auto-computed from beta)")
    parser.add_argument("--years",  type=int,   default=10,
                        help="DCF projection horizon in years (default: 10)")
    parser.add_argument("--pdf",    action="store_true",
                        help="Export a PDF report")
    parser.add_argument("--peers",  action="store_true",
                        help="Add same-industry peer comparison table")
    parser.add_argument("--mc",     action="store_true",
                        help="Run a Monte Carlo DCF (probabilistic valuation)")
    parser.add_argument("--sims",   type=int,   default=10000,
                        help="Number of Monte Carlo simulations (default: 10000)")
    parser.add_argument("--hist",   action="store_true",
                        help="Add historical valuation (5Y P/E & P/S vs own average)")
    parser.add_argument("--out",    type=str,   default=None,
                        help="PDF output path (default: TICKER_date.pdf)")

    args = parser.parse_args()

    wacc = args.wacc
    if wacc is None:
        try:
            beta = float(yf.Ticker(args.ticker).info.get("beta") or 1.0)
            ke = RISK_FREE_RATE + beta * EQUITY_RISK_PREMIUM
            kd_after_tax = RISK_FREE_RATE * (1 - TAX_RATE)
            wacc = ke * EQUITY_WEIGHT + kd_after_tax * (1 - EQUITY_WEIGHT)
            console.print(f"[dim]Auto-computed WACC: {wacc*100:.2f}%  (beta={beta:.2f}  Ke={ke*100:.2f}%)[/dim]")
        except Exception:
            wacc = DEFAULT_WACC

    pdf_path = None
    if args.pdf:
        pdf_path = args.out or f"{args.ticker.upper()}_{date.today().isoformat()}.pdf"

    analyze(
        args.ticker,
        growth_rate=args.growth,
        wacc=wacc,
        years=args.years,
        pdf_path=pdf_path,
        include_peers=args.peers,
        include_mc=args.mc,
        n_sims=args.sims,
        include_hist=args.hist,
    )


if __name__ == "__main__":
    main()
