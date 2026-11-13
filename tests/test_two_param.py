import math

import numpy as np
import pytest

from easyreactors.core import PowerLaw, Reaction, ReactionSet
from easyreactors.nonideal import (RTD, E_cstr, bypass_dead, fit_bypass_dead, fit_interchange,
                                   fit_pfr_bypass, fit_pfr_cstr, interchange, pfr_bypass, pfr_cstr)
from easyreactors.reactors import Stream
from easyreactors.units import to_si


def first_order(k):
    return ReactionSet([Reaction("A -> B", PowerLaw(k, {"A": 1}))])


def liq(v0="1 m^3/min"):
    return Stream.liquid(v0=v0, T="300 K", C={"A": 1000.0})


def test_aula11_bypass_dead_fit_and_conversion():
    fit = fit_bypass_dead([4, 8, 10, 14, 16, 18], [1000, 1333, 1500, 1666, 1750, 1800], 2000, 10, "min")
    assert fit.beta == pytest.approx(0.189, abs=0.002)
    assert fit.alpha == pytest.approx(0.701, abs=0.002)
    rs = ReactionSet([Reaction("A + B -> C + D", PowerLaw(to_si("0.28 m^3/(kmol*min)", "m^3/(mol*s)"), {"A": 1, "B": 1}))])
    feed = Stream.liquid(v0="0.1 m^3/min", T="300 K", C={"A": "2 mol/L", "B": "2 mol/L"})
    lecture = bypass_dead(rs, feed, "1 m^3", 0.701, 0.189)
    assert lecture.units[0].outlet.conc()["A"] / 1000 == pytest.approx(0.726, abs=0.001)
    assert lecture.conversion("A") == pytest.approx(1 - 0.967 / 2, abs=0.001)
    assert bypass_dead(rs, feed, "1 m^3", fit.alpha, fit.beta).conversion("A") == pytest.approx(0.517, abs=0.001)


def test_bypass_dead_first_order_analytic():
    a, b, k, tau = 0.7, 0.2, 0.1 / 60, 10 * 60
    X = bypass_dead(first_order(k), liq(), "10 m^3", a, b).conversion("A")
    assert X == pytest.approx(1 - (b + (1 - b) ** 2 / (1 - b + a * k * tau)), rel=1e-9)


def test_aula11_interchange_fit_and_conversion():
    t = [0, 20, 40, 60, 80, 120, 160, 200, 240]
    C = [2000, 1050, 520, 280, 160, 61, 29, 16.4, 10]
    fit = fit_interchange(t, C, 40, "min")
    assert fit.alpha == pytest.approx(0.8, abs=0.01)
    assert fit.beta == pytest.approx(0.1, abs=0.01)
    X = interchange(first_order(0.03 / 60), liq("1 m^3/min"), "40 m^3", 0.8, 0.1).conversion("A")
    assert X == pytest.approx(0.51, abs=0.005)


def test_interchange_without_exchange_is_cstr():
    k = 0.03 / 60
    X = interchange(first_order(k), liq(), "40 m^3", 0.6, 1e-9).conversion("A")
    assert X == pytest.approx(k * 0.6 * 2400 / (1 + k * 0.6 * 2400), rel=1e-6)


def test_pfr_cstr_fit_recovers_parameters():
    tp, ts = 4 * 60, 6 * 60
    t = np.linspace(0, 150 * 60, 30001)
    E = np.where(t >= tp, E_cstr(np.maximum(t - tp, 0), ts), 0.0)
    fit = fit_pfr_cstr(RTD(t, E), 12 * 60)
    assert fit.alpha == pytest.approx(4 / 12, rel=1e-3)
    assert fit.beta == pytest.approx(6 / 12, rel=1e-3)


def test_pfr_cstr_first_order_analytic():
    k = 0.1 / 60
    res = pfr_cstr(first_order(k), liq(), "10 m^3", 0.4, 0.6)
    assert res.conversion("A") == pytest.approx(1 - math.exp(-k * 240) / (1 + k * 360), rel=1e-6)
    rev = pfr_cstr(first_order(k), liq(), "10 m^3", 0.4, 0.6, order="cstr_first")
    assert rev.conversion("A") == pytest.approx(res.conversion("A"), rel=1e-6)


def test_pfr_bypass_fit_recovers_parameters():
    t = np.linspace(0, 60, 6001)
    g = lambda mu: np.exp(-((t - mu) / 0.5) ** 2 / 2)
    E = 0.25 * g(1.5) / np.trapezoid(g(1.5), t) + 0.75 * g(30) / np.trapezoid(g(30), t)
    fit = fit_pfr_bypass(RTD(t, E), tau=40.0, t_split=10.0)
    assert fit.beta == pytest.approx(0.25, rel=1e-3)
    assert fit.alpha == pytest.approx(30 * 0.75 / 40, rel=1e-3)


def test_pfr_bypass_first_order_analytic():
    k, a, b = 0.1 / 60, 0.5, 0.3
    X = pfr_bypass(first_order(k), liq(), "10 m^3", a, b).conversion("A")
    assert X == pytest.approx((1 - b) * (1 - math.exp(-k * a * 600 / (1 - b))), rel=1e-6)


def test_fogler_18_2_book_parameters_reproduce_intermediates():
    rs = ReactionSet([Reaction("A + B -> C + D", PowerLaw(to_si("0.28 m^3/(kmol*min)", "m^3/(mol*s)"), {"A": 1, "B": 1}))])
    feed = Stream.liquid(v0="0.1 m^3/min", T="300 K", C={"A": "2 mol/L", "B": "2 mol/L"})
    res = bypass_dead(rs, feed, "1 m^3", 0.7, 0.2)
    k, CA0 = 0.28, 2.0
    CAs = lambda ts: (math.sqrt(1 + 4 * ts * k * CA0) - 1) / (2 * ts * k)
    assert CAs(8.7) == pytest.approx(0.724, abs=5e-4)  # book rounds ts = 0.7/0.08 = 8.75 to 8.7
    assert res.units[0].outlet.conc()["A"] / 1000 == pytest.approx(CAs(8.75), rel=1e-9)
    assert res.outlet.conc()["A"] / 1000 == pytest.approx(0.2 * 2 + 0.8 * CAs(8.75), rel=1e-9)
    assert res.outlet.conc()["A"] / 1000 == pytest.approx(0.979, abs=0.002)
    ideal = bypass_dead(rs, feed, "1 m^3", 1.0, 0.0)
    assert ideal.outlet.conc()["A"] / 1000 == pytest.approx(0.685, abs=0.001)


def test_fogler_18_3_model_vs_ideal():
    k = 0.03 / 60
    X = interchange(first_order(k), liq("1 m^3/min"), "40 m^3", 0.8, 0.1).conversion("A")
    X_cstr = interchange(first_order(k), liq("1 m^3/min"), "40 m^3", 0.999999, 1e-9).conversion("A")
    assert X == pytest.approx(0.51, abs=0.005)
    assert X_cstr == pytest.approx(0.55, abs=0.005)
