"""Unit tests for DDM cost of equity and composite filtering (no network)."""
import pytest

import fundamental_analysis as fa


def test_ddm_uses_ke_backed_out_of_wacc():
    info = {"dividendRate": 4.0, "beta": 0.06}        # raw beta: Ke = 4.83%, only 0.83pp over g
    assert fa.ddm_valuation(info) is None              # without wacc: spread < 3pp -> not applicable
    ke = fa.cost_of_equity(info, wacc=0.07)
    assert ke * fa.EQUITY_WEIGHT + fa.RISK_FREE_RATE * (1 - fa.TAX_RATE) * (1 - fa.EQUITY_WEIGHT) == pytest.approx(0.07)
    assert fa.ddm_valuation(info, wacc=0.07) == pytest.approx(4.0 * 1.04 / (ke - 0.04))


def test_ddm_not_applicable_below_min_spread():
    info = {"dividendRate": 4.0}
    wacc_at_edge = (0.04 + 0.03) * fa.EQUITY_WEIGHT + fa.RISK_FREE_RATE * (1 - fa.TAX_RATE) * (1 - fa.EQUITY_WEIGHT)
    assert fa.ddm_valuation(info, wacc=wacc_at_edge + 1e-6) is not None
    assert fa.ddm_valuation(info, wacc=wacc_at_edge - 1e-6) is None


def test_composite_drops_none_zero_and_negative():
    assert fa.composite_value([100.0, None, 0.0, -5.0, 50.0]) == 75.0
    assert fa.composite_value([None, 0.0]) is None
    assert fa.composite_value([float("nan"), 40.0]) == 40.0   # e.g. DCF with NaN capex
