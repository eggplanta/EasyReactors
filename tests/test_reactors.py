import math

import numpy as np
import pytest

from easyreactors.core import Arrhenius, PowerLaw, Reaction, ReactionSet, Reversible
from easyreactors.reactors import CSTR, PFR, Batch, Stream
from easyreactors.units import to_si

H = 3600.0


def eo_hydrolysis():
    return ReactionSet([Reaction("EO + W -> EG", PowerLaw(0.311 / 60, {"EO": 1}), ref="EO")])


def eo_feed(v0):
    return Stream.liquid(v0=v0, T="300 K", C={"EO": "8.05 mol/L", "W": "27.8 mol/L"})


def series(k1=0.5 / H, k2=0.2 / H):
    return ReactionSet([Reaction("A -> B", PowerLaw(k1, {"A": 1})),
                        Reaction("B -> C", PowerLaw(k2, {"B": 1}))])


def test_fogler_5_2a_cstr_size():
    res = CSTR(eo_hydrolysis()).size(eo_feed("7.24 L/s"), X=0.8)
    assert res.end == pytest.approx(7.24e-3 * 0.8 / (0.311 / 60 * 0.2), rel=1e-6)
    assert res.end == pytest.approx(5.538, rel=0.015)


def test_fogler_5_2b_cstr_simulate():
    res = CSTR(eo_hydrolysis(), V="800 gallon").simulate(eo_feed("3.62 L/s"))
    assert res.conversion("EO") == pytest.approx(0.81, abs=0.005)


def test_fogler_5_3_pfr_gas():
    k = Arrhenius.from_ref(0.072, to_si("82 kcal/mol", "J/mol"), 1000)
    rs = ReactionSet([Reaction("C2H6 -> C2H4 + H2", PowerLaw(k, {"C2H6": 1}))])
    feed = Stream.gas(F={"C2H6": "0.425 lbmol/s"}, T="1100 K", P="6 atm")
    res = PFR(rs).size(feed, X=0.8)
    assert res.end == pytest.approx(2.28, rel=0.01)


def test_fogler_8_3_batch_series_optimum():
    res = Batch(series(), V="1 m^3").simulate("10 h", "300 K", C0={"A": "2 mol/L"})
    t, NB = res.optimum("B")
    assert t / H == pytest.approx(math.log(0.2 / 0.5) / (0.2 - 0.5), rel=1e-4)
    assert NB / 1.0 / 1000 == pytest.approx(2 * (0.5 / 0.2) ** (0.2 / (0.2 - 0.5)), rel=1e-5)


def test_fogler_8_4_cstr_series_optimum():
    feed = Stream.liquid(v0="1 m^3/s", T="300 K", C={"A": "2 mol/L"})
    V, _ = CSTR(series()).optimum(feed, "B", V_bounds=(1, 1e6))
    assert V / H == pytest.approx(1 / math.sqrt(0.5 * 0.2), rel=1e-4)


def test_pfr_first_order_analytic():
    rs = ReactionSet([Reaction("A -> B", PowerLaw(0.1, {"A": 1}))])
    res = PFR(rs, V="2 m^3").simulate(Stream.liquid(v0="1 m^3/s", T="300 K", C={"A": "1 mol/L"}))
    assert res.conversion("A") == pytest.approx(1 - math.exp(-0.2), rel=1e-6)


def test_batch_second_order_analytic():
    rs = ReactionSet([Reaction("2A -> B", PowerLaw(1e-3, {"A": 2}))])
    res = Batch(rs, V="1 m^3").simulate("100 s", "300 K", C0={"A": 10.0})
    assert res.profile("A")[-1] == pytest.approx(10 / (1 + 1e-3 * 10 * 100), rel=1e-6)


def test_batch_gas_constant_pressure():
    rs = ReactionSet([Reaction("A -> 2B", PowerLaw(0.01, {"A": 1}))])
    res = Batch(rs, V="1 m^3", constant_pressure=True).simulate("100 s", "300 K", N0={"A": 10.0})
    assert res.profile("A")[-1] == pytest.approx(10 * math.exp(-1), rel=1e-6)


def test_time_to_and_productivity():
    rs = ReactionSet([Reaction("A -> B", PowerLaw(0.01, {"A": 1}))])
    b = Batch(rs, V="1 m^3", t_dead="100 s")
    res = b.time_to(0.5, "300 K", N0={"A": 10.0})
    assert res.end == pytest.approx(math.log(2) / 0.01, rel=1e-6)
    assert b.productivity(res, "B") == pytest.approx(5 / (res.end + 100), rel=1e-6)


def test_cstr_mass_balance_multiple():
    rs = series(0.01, 0.005)
    res = CSTR(rs, V="10 m^3").simulate(Stream.liquid(v0="0.1 m^3/s", T="300 K", C={"A": 1000.0}))
    assert res.n[-1].sum() == pytest.approx(100.0, rel=1e-9)


def test_unreachable_conversion():
    rs = ReactionSet([Reaction("A <=> B", Reversible(0.1, {"A": 1}, {"B": 1}, K=1.0))])
    feed = Stream.liquid(v0="1 m^3/s", T="300 K", C={"A": 1.0})
    with pytest.raises(ValueError):
        PFR(rs).size(feed, X=0.6, V_max=1e3)
