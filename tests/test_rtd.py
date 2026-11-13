import math

import numpy as np
import pytest
from scipy.integrate import quad

from easyreactors.core import PowerLaw, Reaction, ReactionSet
from easyreactors.nonideal import (RTD, E_tanks, dispersion, dispersion_first_order, max_mixedness,
                                   segregation, tis_first_order)
from easyreactors.reactors import Batch, Stream

T16 = [0, 0.5, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 12, 14]
C16 = [0, 0.6, 1.4, 5, 8, 10, 8, 6, 4, 3, 2.2, 1.5, 0.6, 0]
T17 = [0, 5, 10, 15, 20, 30, 40, 50, 70, 100, 150, 200]
C17 = [112, 95.8, 82.2, 70.6, 60.9, 45.6, 34.5, 26.3, 15.7, 7.67, 2.55, 0.9]


def dimerization():
    return ReactionSet([Reaction("2A -> B", PowerLaw(0.01 / 1000 / 60, {"A": 2}))])


def test_fogler_16_2_moments():
    r = RTD.from_pulse(T16, C16, "min")
    assert r.tm / 60 == pytest.approx(5.15, rel=0.02)
    assert r.variance / 3600 == pytest.approx(6.1, rel=0.02)
    assert r.F[-1] == pytest.approx(1.0)


def test_fogler_17_4_segregation():
    r = RTD.from_pulse(T17, C17, "min")
    assert segregation(r, dimerization(), {"A": "8 mol/L"}, "320 K").conversion("A") == pytest.approx(0.61, rel=0.02)


def test_fogler_17_5_max_mixedness():
    r = RTD.from_pulse(T17, C17, "min")
    assert max_mixedness(r, dimerization(), {"A": "8 mol/L"}, "320 K").conversion("A") == pytest.approx(0.563, rel=0.02)


def test_fogler_18_1_dispersion_and_tis():
    r = RTD.from_pulse(T16, C16, "min")
    Da = 0.25 * r.tm / 60
    assert dispersion_first_order(Da, r.peclet()) == pytest.approx(0.68, rel=0.02)
    assert tis_first_order(Da, r.n_tanks) == pytest.approx(0.677, rel=0.02)
    rs = ReactionSet([Reaction("A -> B", PowerLaw(0.25 / 60, {"A": 1}))])
    feed = Stream.liquid(v0="1 L/min", T="300 K", C={"A": "1 mol/L"})
    bvp = dispersion(rs, feed, r.tm * feed.v, r.peclet()).conversion("A")
    assert bvp == pytest.approx(dispersion_first_order(Da, r.peclet()), rel=1e-6)


def test_segregation_first_order_equals_cstr_for_exponential_rtd():
    tau, k = 10.0, 0.1
    t = np.linspace(0, 200, 4001)
    r = RTD(t, np.exp(-t / tau) / tau)
    rs = ReactionSet([Reaction("A -> B", PowerLaw(k, {"A": 1}))])
    X = segregation(r, rs, {"A": 1.0}, 300).conversion("A")
    assert X == pytest.approx(k * tau / (1 + k * tau), rel=1e-4)


def test_e_tanks_normalized():
    area = quad(lambda t: E_tanks(t, 10.0, 3), 0, np.inf)[0]
    mean = quad(lambda t: t * E_tanks(t, 10.0, 3), 0, np.inf)[0]
    assert area == pytest.approx(1.0) and mean == pytest.approx(10.0)


def test_higher_order_methods_more_accurate():
    tau = 10.0
    t = np.linspace(0, 100, 21)
    E = np.exp(-t / tau) / tau
    err = {m: abs(RTD(t, E, m).tm - tau) for m in ("trapezoid", "simpson", "pchip", "spline")}
    assert err["simpson"] < err["trapezoid"]
    assert err["pchip"] < err["trapezoid"]
    assert err["spline"] < err["trapezoid"]


def test_invalid_method():
    with pytest.raises(ValueError):
        RTD([0, 1], [1, 1], "euler")


def test_zero_order_stops_at_zero():
    rs = ReactionSet([Reaction("A -> B", PowerLaw(0.1, {}))])
    res = Batch(rs, V=1.0).simulate(100.0, 300, C0={"A": 1.0})
    assert res.profile("A")[-1] == pytest.approx(0.0, abs=1e-6)
    assert res.profile("B")[-1] == pytest.approx(1.0, rel=1e-6)


def test_zero_order_mm_equals_segregation():
    from easyreactors.units import to_si
    k = to_si("0.025 mol/(L*min)", "mol/(m^3*s)")
    rs = ReactionSet([Reaction("A -> B", PowerLaw(k, {}))])
    for m in ("trapezoid", "simpson", "pchip", "spline"):
        r = RTD.from_E([0, 5, 10, 15, 20, 25, 30], [0, 0.03, 0.05, 0.05, 0.04, 0.02, 0.01], "min", m)
        Xs = segregation(r, rs, {"A": "1.5 mol/L"}, 300).conversion("A")
        Xm = max_mixedness(r, rs, {"A": "1.5 mol/L"}, 300).conversion("A")
        assert Xm == pytest.approx(Xs, rel=0.005)


def _laminar_rtd(tau):
    from easyreactors.nonideal import E_laminar
    t = tau / 2 * np.geomspace(1, 2000, 20001)
    return RTD(t, E_laminar(t, tau))


def test_fogler_17_1_laminar_first_order_exact():
    from scipy.special import exp1
    tau, k = 100.0, 0.02
    Da = tau * k
    exact = 1 - ((1 - Da / 2) * math.exp(-Da / 2) + (Da / 2) ** 2 * exp1(Da / 2))
    rs = ReactionSet([Reaction("A -> B", PowerLaw(k, {"A": 1}))])
    X = segregation(_laminar_rtd(tau), rs, {"A": 1.0}, 300).conversion("A")
    assert X == pytest.approx(exact, rel=1e-4)


def test_fogler_17_3_laminar_second_order():
    from easyreactors.reactors import PFR
    from easyreactors.units import to_si
    k = to_si("4.93e-3 L/(mol*s)", "m^3/(mol*s)")
    rs = ReactionSet([Reaction("A + B -> C + D", PowerLaw(k, {"A": 1, "B": 1}))])
    C0 = {"A": "0.75 mol/L", "B": "0.75 mol/L"}
    feed = Stream.liquid(v0="0.1 L/s", T="323 K", C=C0)
    assert PFR(rs, V="100 L").simulate(feed).conversion("A") == pytest.approx(0.787, abs=0.001)
    Da = 4.93e-3 * 0.75 * 1000
    X = segregation(_laminar_rtd(1000.0), rs, C0, 323).conversion("A")
    assert X == pytest.approx(Da * (1 - Da / 2 * math.log(1 + 2 / Da)), rel=1e-3)
    assert X == pytest.approx(0.742, abs=0.001)
