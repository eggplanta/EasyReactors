import math

import numpy as np
import pytest

from easyreactors.kinetics.analysis import (arrhenius, differential, finite_differences, fit_expression, integral,
                                            integral_scan, integrated_conc, nonlinear, polynomial_derivative,
                                            power_law)

# Fogler 7-1 to 7-3: trityl + methanol, batch at 25 °C
T7 = [0, 50, 100, 150, 200, 250, 300]
C7 = np.array([50, 38, 30.6, 25.6, 22.2, 19.5, 17.4]) * 1e-3


def test_fogler_7_2_finite_differences_table():
    book = [2.86, 1.94, 1.24, 0.84, 0.61, 0.48, 0.36]
    assert (-finite_differences(T7, C7) * 1e4).tolist() == pytest.approx(book, abs=5e-3)


def test_fogler_7_2_differential_method():
    fit = differential(T7, C7)
    # Book (Excel): α = 1.99, k' = 0.13 dm3/(mol·min)
    assert fit.order == pytest.approx(1.99, abs=0.01)
    assert fit.k == pytest.approx(0.123, abs=0.001)
    assert differential(T7, C7, order=2).k == pytest.approx(0.125, abs=0.001)


def test_fogler_7_1_integral_method():
    fits = {f.order: f for f in integral_scan(T7, C7)}
    assert max(fits.values(), key=lambda f: f.r2).order == 2
    # Book reads k' = 0.12 from its plot; regression with α = 2 gives 0.125 (Example 7-3)
    assert fits[2].k == pytest.approx(0.125, abs=0.001)
    assert fits[2].k * 0.5 / 0.5 ** 2 == pytest.approx(0.25, abs=0.002)


def test_fogler_7_3_nonlinear_regression():
    fit = nonlinear(T7, C7)
    assert fit.order == pytest.approx(2.04, abs=0.005)
    assert fit.k == pytest.approx(0.147, abs=0.0005)
    assert nonlinear(T7, C7, order=2).k == pytest.approx(0.125, abs=0.0005)


def test_exact_data_recover_order_and_k():
    t = np.linspace(0, 60, 31)
    for n, k in ((0, 0.01), (1, 0.05), (1.5, 0.08), (2, 0.4)):
        C = integrated_conc(t, 1.0, k, n)
        t_ok, C_ok = t[C > 0.05], C[C > 0.05]
        assert integral(t_ok, C_ok, n).k == pytest.approx(k, rel=1e-9)
        fit = nonlinear(t_ok, C_ok)
        assert (fit.order, fit.k) == pytest.approx((n, k), rel=1e-6, abs=1e-8)
        assert differential(t_ok, C_ok).order == pytest.approx(n, abs=0.02)


def test_polynomial_derivative_exact_for_polynomials():
    t = np.linspace(0, 5, 8)
    assert polynomial_derivative(t, 2 + 3 * t - t ** 2, 2) == pytest.approx(3 - 2 * t)


def test_fogler_3_1_arrhenius():
    T = [313.0, 319.0, 323.0, 328.0, 333.0]
    k = [0.00043, 0.00103, 0.00180, 0.00355, 0.00717]
    fit = arrhenius(T, k)
    # Book regresses 1/T rounded to 3 digits (0.00320, 0.00314, ...), giving E/R = 14017 K and ln A = 37.12
    slope, intercept = np.polyfit([0.00320, 0.00314, 0.00310, 0.00305, 0.00300], np.log(k), 1)
    assert (-slope, intercept) == pytest.approx((14017, 37.12), abs=1)
    assert fit.Ea / 8.314462618 == pytest.approx(14612, abs=1)
    assert fit.r2 > 0.999
    exact = arrhenius([300, 350, 400], [math.exp(30 - 10000 / T) for T in (300, 350, 400)])
    assert (exact.Ea, exact.A) == pytest.approx((10000 * 8.314462618, math.exp(30)), rel=1e-9)


# Fogler 7-4: methanation in a differential reactor
P_CO = [1.0, 1.8, 4.08, 1.0, 1.0, 1.0]
P_H2 = [1.0, 1.0, 1.0, 0.1, 0.5, 4.0]
RATE = np.array([5.2, 13.2, 30.0, 4.95, 7.42, 5.25]) * 1e-3


def test_fogler_7_4_power_law_in_co():
    fit = power_law({"p_CO": P_CO[:3]}, RATE[:3])
    assert fit.params["order_p_CO"] == pytest.approx(1.22, abs=0.015)


def test_fogler_7_4_nonlinear_rate_law():
    cols = {"p_CO": P_CO, "p_H2": P_H2}
    first = fit_expression("a*p_CO*p_H2^b1/(1 + b*p_H2^b2)", cols, RATE, {"a": 0.02, "b1": 0.5, "b": 2, "b2": 1})
    assert first.params["a"] == pytest.approx(0.025, abs=0.001)
    assert first.params["b1"] == pytest.approx(0.61, abs=0.01)
    assert first.params["b"] == pytest.approx(2.49, abs=0.01)
    second = fit_expression("a*p_CO*p_H2^0.5/(1 + b*p_H2)", cols, RATE, {"a": 0.02, "b": 2})
    assert second.params["a"] == pytest.approx(0.018, abs=0.0005)
    assert second.params["b"] == pytest.approx(1.49, abs=0.005)
    with pytest.raises(ValueError, match="C_A or p_A"):
        fit_expression("a*CO", {"CO": P_CO}, RATE, {"a": 1})
