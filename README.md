# CEDEAR Fundamental Analysis

A command-line tool for **fundamental analysis and intrinsic-value estimation** of US stocks
and the underlyings of Argentine **CEDEARs** (BYMA). It pulls data from Yahoo Finance and
combines several valuation models, peer comparison, a Monte Carlo DCF, and a historical
valuation view — with rich terminal output and an optional polished PDF report.

> For a CEDEAR, analyze its **underlying US ticker** (e.g. `AAPL`, `MSFT`, `MELI`, `VIST`).

---

## Features

- **Valuation models**
  - **DCF** (FCFF-based, auto-computes WACC from beta)
  - **Graham Number** — `sqrt(22.5 * EPS * BVPS)`
  - **EV/EBITDA** using sector-median multiples
  - **DDM** (Gordon Growth, for dividend payers)
  - **Composite** average of all applicable models, with an under/over-valued verdict
- **Monte Carlo DCF** (`--mc`) — probabilistic valuation sampling growth, WACC, and terminal
  growth; reports P10–P90 and the probability the stock is undervalued.
- **Peer comparison** (`--peers`) — same-industry companies from Yahoo Finance, with a
  **green → yellow → orange → red gradient** ranking each metric within the column.
- **Historical valuation** (`--hist`) — 5-year P/E and P/S versus the stock's own average
  (percentile + z-score, "cheap/expensive" verdict).
- **PDF report** (`--pdf`) — a formatted report with all of the above, including charts.

---

## Installation

```bash
git clone https://github.com/ZANTERA/cedear-fundamental-analysis.git
cd cedear-fundamental-analysis
pip install -r requirements.txt
```

Requires Python 3.10+.

---

## Usage

```bash
python fundamental_analysis.py TICKER [options]
```

### Options

| Flag | Description | Default |
|------|-------------|---------|
| `--growth RATE` | FCF annual growth rate for the DCF | `0.08` (8%) |
| `--wacc RATE` | WACC override | auto from beta |
| `--years N` | DCF projection horizon (years) | `10` |
| `--peers` | Add the same-industry peer comparison table | off |
| `--mc` | Run a Monte Carlo DCF | off |
| `--sims N` | Number of Monte Carlo simulations | `10000` |
| `--hist` | Add 5Y historical P/E & P/S vs own average | off |
| `--pdf` | Export a PDF report | off |
| `--out PATH` | PDF output path | `TICKER_<date>.pdf` |

### Examples

```bash
# Quick valuation
python fundamental_analysis.py AAPL

# Full report with everything, exported to PDF
python fundamental_analysis.py MELI --mc --peers --hist --pdf

# Custom DCF assumptions and more simulations
python fundamental_analysis.py MSFT --growth 0.12 --wacc 0.10 --mc --sims 50000 --hist --pdf
```

---

## How the valuation works

- **WACC** is auto-computed from CAPM: `Ke = Rf + beta * ERP`, blended with an after-tax cost
  of debt at a 70/30 equity/debt weighting. Override with `--wacc` when beta is unreliable
  (e.g. names with a negative reported beta).
- **DCF** projects free cash flow (`Operating CF + CapEx`) at the chosen growth rate, discounts
  it, and adds a Gordon-growth terminal value, then nets out debt. Companies with **negative
  FCF** (heavy-capex growth names) are skipped rather than producing a misleading number.
- **Defaults:** risk-free rate `4.5%`, equity risk premium `5.5%`, terminal growth `2.5%`.

---

## Disclaimer

This tool is for **educational and informational purposes only**. It is **not financial
advice**. Data comes from Yahoo Finance via [`yfinance`](https://github.com/ranaroussi/yfinance)
and may be delayed, incomplete, or inaccurate. Always do your own research before investing.
