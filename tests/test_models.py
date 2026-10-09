"""Unit tests for DDM cost of equity and composite filtering (no network)."""
from datetime import date

import pandas as pd
import pytest

import fundamental_analysis as fa

AS_OF = date(2026, 10, 2)


def divs(*payments, tz=None):
    """Dividend history like yf.Ticker(t).dividends: (ex-date, amount) pairs."""
    return pd.Series([a for _, a in payments], index=pd.DatetimeIndex([d for d, _ in payments], tz=tz))


FOUR = divs(("2025-12-01", 1.0), ("2026-03-01", 1.0), ("2026-06-01", 1.0), ("2026-09-01", 1.0))


def test_ddm_uses_ke_backed_out_of_wacc():
    info = {"beta": 0.06}                              # raw beta: Ke = 4.83%, only 0.83pp over g
    assert fa.ddm_valuation(info, dividends=FOUR, as_of=AS_OF) is None   # without wacc: spread < 3pp -> N/A
    ke = fa.cost_of_equity(info, wacc=0.07)
    assert ke * fa.EQUITY_WEIGHT + fa.RISK_FREE_RATE * (1 - fa.TAX_RATE) * (1 - fa.EQUITY_WEIGHT) == pytest.approx(0.07)
    assert fa.ddm_valuation(info, 0.07, FOUR, AS_OF) == pytest.approx(4.0 * 1.04 / (ke - 0.04))


def test_ddm_not_applicable_below_min_spread():
    wacc_at_edge = (0.04 + 0.03) * fa.EQUITY_WEIGHT + fa.RISK_FREE_RATE * (1 - fa.TAX_RATE) * (1 - fa.EQUITY_WEIGHT)
    assert fa.ddm_valuation({}, wacc_at_edge + 1e-6, FOUR, AS_OF) is not None
    assert fa.ddm_valuation({}, wacc_at_edge - 1e-6, FOUR, AS_OF) is None


def test_composite_drops_none_zero_and_negative():
    assert fa.composite_value([100.0, None, 0.0, -5.0, 50.0]) == 75.0
    assert fa.composite_value([None, 0.0]) is None
    assert fa.composite_value([float("nan"), 40.0]) == 40.0   # e.g. DCF with NaN capex


def test_ddm_uses_paid_dividends_not_info_fields():
    # BP ADR: Yahoo's info gives the forward rate and a trailing per ordinary share; the history is per ADR.
    info = {"dividendRate": 2.02, "trailingAnnualDividendRate": 0.336, "currency": "USD", "financialCurrency": "USD"}
    bp = divs(("2025-11-07", 0.499), ("2026-02-13", 0.499), ("2026-05-08", 0.499), ("2026-08-14", 0.52))
    ke = fa.cost_of_equity(info, wacc=0.07)
    assert fa.ddm_valuation(info, 0.07, bp, AS_OF) == pytest.approx(2.017 * 1.04 / (ke - 0.04))
    assert fa.ddm_valuation(info, wacc=0.07) is None                       # no history -> no DDM


def test_trailing_dividend_window():
    h = divs(("2025-10-02", 9.0), ("2025-10-03", 1.0), ("2026-10-02", 2.0), ("2026-10-03", 9.0))
    assert fa.trailing_dividend(h, AS_OF) == 3.0                           # (as_of - 365d, as_of]
    assert fa.trailing_dividend(divs(("2026-09-01", 0.5), tz="America/New_York"), AS_OF) == 0.5
    assert fa.trailing_dividend(divs(("2024-01-01", 1.0)), AS_OF) is None   # stopped paying
    assert fa.trailing_dividend(divs(), AS_OF) is None and fa.trailing_dividend(None) is None
