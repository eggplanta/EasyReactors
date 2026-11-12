import math

import pytest

from easyreactors.core import PowerLaw, Reaction, ReactionSet
from easyreactors.network import Bypass, Parallel, Recycle, Series, SideFeed, size_network, size_split
from easyreactors.reactors import CSTR, PFR, Stream


def first_order(k=0.1):
    return ReactionSet([Reaction("A -> B", PowerLaw(k, {"A": 1}))])


def liq(v0="1 m^3/s", CA=1.0, T="300 K"):
    return Stream.liquid(v0=v0, T=T, C={"A": CA})


def eo_net(blocks):
    return blocks, Stream.liquid(v0="7.24 L/s", T="300 K", C={"EO": "8.05 mol/L", "W": "27.8 mol/L"})


def eo_rs():
    return ReactionSet([Reaction("EO + W -> EG", PowerLaw(0.311 / 60, {"EO": 1}), ref="EO")])


def test_fogler_5_2b_parallel():
    rs = eo_rs()
    _, feed = eo_net(None)
    res = Parallel([CSTR(rs, V="800 gallon"), CSTR(rs, V="800 gallon")]).simulate(feed)
    assert res.conversion("EO") == pytest.approx(0.81, abs=0.005)


def test_fogler_5_2c_series():
    rs = eo_rs()
    _, feed = eo_net(None)
    res = Series([CSTR(rs, V="800 gallon"), CSTR(rs, V="800 gallon")]).simulate(feed)
    assert res.units[0].result.conversion("EO") == pytest.approx(0.684, abs=0.003)
    assert res.conversion("EO") == pytest.approx(0.90, abs=0.005)


def test_series_cstr_analytic():
    res = Series([CSTR(first_order(), V=10.0) for _ in range(3)]).simulate(liq())
    assert res.conversion("A") == pytest.approx(1 - 1 / (1 + 0.1 * 10) ** 3, rel=1e-9)


def test_parallel_unequal_split_mass_balance():
    res = Parallel([PFR(first_order(), V=5.0), Bypass()], split=[0.7, 0.3]).simulate(liq())
    assert res.conversion("A") == pytest.approx(0.7 * (1 - math.exp(-0.1 * 5 / 0.7)), rel=1e-6)
    assert sum(res.outlet.F.values()) == pytest.approx(1.0, rel=1e-9)


def test_side_feed_conversion_basis():
    side = Stream.liquid(v0="1 m^3/s", T="300 K", C={"A": 1.0})
    res = Series([CSTR(first_order(), V=10.0), SideFeed(side), CSTR(first_order(), V=20.0)]).simulate(liq())
    assert res.fed["A"] == pytest.approx(2.0)
    FA1 = 1.0 / 2
    FA2 = (FA1 + 1.0) / (1 + 0.1 * 20 / 2)
    assert res.outlet.F["A"] == pytest.approx(FA2, rel=1e-9)


def test_recycle_pfr_levenspiel():
    R, k, X = 2.0, 0.1, 0.8
    V = (R + 1) * math.log((1 + R * (1 - X)) / ((R + 1) * (1 - X))) / k
    res = Recycle(PFR(first_order(k), V=V), R=R).simulate(liq())
    assert res.conversion("A") == pytest.approx(X, rel=1e-6)


def test_recycle_infinite_tends_to_cstr():
    res = Recycle(PFR(first_order(), V=10.0), R=200.0).simulate(liq())
    assert res.conversion("A") == pytest.approx(0.5, abs=0.002)


def test_size_one_unknown_volume():
    net = Series([CSTR(first_order(), V=10.0), PFR(first_order())])
    res = size_network(net, liq(), X=0.9)
    assert res.units[1].V == pytest.approx(math.log(0.5 / 0.1) / 0.1, rel=1e-6)


def test_size_equal_pfrs_second_order():
    rs = ReactionSet([Reaction("2A -> B", PowerLaw(1e-3, {"A": 2}))])
    res = size_network(Series([PFR(rs), PFR(rs)]), liq(CA=1000.0), X=0.8)
    V_total = 1 / (1e-3 * 1000) * 0.8 / 0.2
    assert res.units[0].V == pytest.approx(V_total / 2, rel=1e-6)
    assert res.units[1].V == pytest.approx(V_total / 2, rel=1e-6)


def test_size_split_equal_conversion():
    par = Parallel([CSTR(first_order(), V=10.0), CSTR(first_order(), V=30.0)])
    res = size_split(par, par, liq(), equal_conversion=True)
    assert res.ctx.nodes[id(par)].split[0] == pytest.approx(0.25, rel=1e-6)


def test_size_split_target_x():
    par = Parallel([PFR(first_order(), V=20.0), Bypass()])
    res = size_split(par, par, liq(), X=0.5)
    assert res.conversion("A") == pytest.approx(0.5, rel=1e-8)


def test_mix_different_temperature_raises():
    side = Stream.liquid(v0="1 m^3/s", T="350 K", C={"A": 1.0})
    with pytest.raises(ValueError):
        Series([CSTR(first_order(), V=1.0), SideFeed(side)]).simulate(liq())


def test_draw_contains_units():
    net = Series([CSTR(first_order(), V=0.35), Parallel([PFR(first_order(), V=0.2), Bypass()], split=[0.6, 0.4])])
    txt = net.simulate(liq()).draw("A")
    assert "CSTR-1" in txt and "PFR-1" in txt and "0.60" in txt and "X=0" in txt
