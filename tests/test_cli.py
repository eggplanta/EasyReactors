import math

import numpy as np
import pytest

from easyreactors.cli.inputs import mapping, numbers, quantity, table_file
from easyreactors.cli.kinetics import K_unit, k_unit
from easyreactors.cli.network import build, parse, to_text
from easyreactors.cli.rtd import model_conversions, two_param_conversion
from easyreactors.core import PowerLaw, Reaction, ReactionSet
from easyreactors.network import size_network
from easyreactors.nonideal import RTD, max_mixedness, segregation
from easyreactors.reactors import Stream
from easyreactors.units import to_si

T17 = [0, 5, 10, 15, 20, 30, 40, 50, 70, 100, 150, 200]
C17 = [112, 95.8, 82.2, 70.6, 60.9, 45.6, 34.5, 26.3, 15.7, 7.67, 2.55, 0.9]


# Inputs

def test_quantity_units_and_defaults():
    assert quantity("0.015 dm3/(mol*min)", k_unit(2)) == pytest.approx(0.015e-3 / 60)
    assert quantity("0.015 dm3 mol-1 min-1", k_unit(2)) == pytest.approx(0.015e-3 / 60)
    assert quantity("0.25 1/min", k_unit(1)) == pytest.approx(0.25 / 60)
    assert quantity("2", "mol/m^3", "mol/L") == pytest.approx(2000)
    assert quantity("25 degC", "K", "K") == pytest.approx(298.15)
    assert quantity("800 gallon", "m^3", "L") == pytest.approx(to_si("800 gallon", "m^3"))
    with pytest.raises(ValueError, match="Wrong units"):
        quantity("5 kg", "m^3", "L")
    with pytest.raises(ValueError):
        quantity("abc", "m^3")


def test_rate_constant_units_match_to_si():
    assert quantity("0.025 mol/(L*min)", k_unit(0)) == pytest.approx(to_si("0.025 mol/L/min", "mol/m^3/s"))
    assert quantity("2 mol/L", K_unit({"A": 1}, {"B": 2})) == pytest.approx(2000)
    assert quantity("2 L/mol", K_unit({"A": 2}, {"B": 1})) == pytest.approx(0.002)


def test_simple_arrows():
    from easyreactors.core import parse_equation
    assert parse_equation("2A + B > C") == ({"A": -2, "B": -1, "C": 1}, False)
    assert parse_equation("A => B")[1] is False
    assert parse_equation("A <> B")[1] is True
    assert parse_equation("A <=> B")[1] is True


def test_numbers_and_mapping():
    assert numbers("0 5, 10  15").tolist() == [0, 5, 10, 15]
    assert numbers("95,8; 82,2\t70,6").tolist() == [95.8, 82.2, 70.6]
    assert mapping("A=2 mol/L, B: 0") == {"A": "2 mol/L", "B": "0"}
    with pytest.raises(ValueError):
        mapping("A 2")


def test_table_file_columns_and_rows(tmp_path):
    cols = tmp_path / "cols.csv"
    cols.write_text("t,C\n" + "\n".join(f"{t},{c}" for t, c in zip(T17, C17)))
    rows = tmp_path / "rows.txt"
    rows.write_text(" ".join(map(str, T17)) + "\n" + " ".join(map(str, C17)))
    for f in (cols, rows):
        t, C = table_file(f)
        assert t.tolist() == T17 and C.tolist() == C17


# Network expressions

def eo():
    rs = ReactionSet([Reaction("EO + W -> EG", PowerLaw(0.311 / 60, {"EO": 1}), ref="EO")])
    return rs, Stream.liquid(v0="7.24 L/s", T="300 K", C={"EO": "8.05 mol/L", "W": "27.8 mol/L"})


def test_parse_roundtrip_and_errors():
    for s in ["CSTR(800 gallon) > PFR(?)", "[CSTR(1 m3) | PFR(2 m3) > bypass]{0.3, 0.7}",
              "recycle(PFR(?), 2) > feed", "[CSTR(?) | PFR(?)]{?}"]:
        assert to_text(parse(s)) == s
    for bad in ["CSTR(1 m3) >", "[CSTR]", "foo", "CSTR(5 kg)", "[CSTR | PFR]{0.5}"]:
        with pytest.raises(ValueError):
            parse(bad)


def test_fogler_5_2c_series_from_expression():
    rs, feed = eo()
    res = build(parse("CSTR(800 gallon) > CSTR(800 gallon)"), rs).net.simulate(feed)
    assert res.conversion("EO") == pytest.approx(0.90, abs=0.005)


def test_size_equal_volumes_from_expression():
    rs, feed = eo()
    built = build(parse("CSTR(?) > CSTR(?)"), rs)
    res = size_network(built.net, feed, 0.9, "EO")
    assert res.units[0].V == pytest.approx(to_si("800 gallon", "m^3"), rel=0.02)


def test_split_and_volume_unknown_rejected():
    rs, _ = eo()
    with pytest.raises(ValueError, match="not both"):
        build(parse("[CSTR(?) | PFR(1 m3)]{?}"), rs)


# RTD model table, CR2 list Ex.1

def test_cr2_ex1_model_table():
    r = RTD.from_pulse(T17, C17, "min")
    rs = ReactionSet([Reaction("2A -> B", PowerLaw(quantity("0.015 dm3/(mol*min)", k_unit(2)), {"A": 2}))])
    C0 = {"A": 4000.0}
    rows = dict(model_conversions(r, rs, C0, 300.0, tau=40 * 60))
    Da = 0.015 * 4 * 40
    assert rows["Ideal PFR (τ = 40 min)"] == pytest.approx(Da / (1 + Da), rel=1e-6)
    assert rows["Ideal CSTR (τ = 40 min)"] == pytest.approx((1 + 2 * Da - math.sqrt(1 + 4 * Da)) / (2 * Da), rel=1e-6)
    assert rows["Segregation"] == pytest.approx(segregation(r, rs, C0, 300.0).conversion("A"))
    assert rows["Maximum mixedness"] == pytest.approx(max_mixedness(r, rs, C0, 300.0).conversion("A"))
    assert rows["Segregation"] > rows["Maximum mixedness"]


def test_bypass_dead_conversion_first_order_analytic():
    k, tau, a, b = 0.1, 10.0, 0.7, 0.2
    rs = ReactionSet([Reaction("A -> B", PowerLaw(k, {"A": 1}))])
    rows = dict(two_param_conversion("CSTR with bypass and dead volume", rs, {"A": 1000.0}, 300.0, tau, a, b))
    Da = k * tau * a / (1 - b)
    assert rows["Model"] == pytest.approx((1 - b) * Da / (1 + Da), rel=1e-6)
    assert rows["Ideal CSTR (τ)"] == pytest.approx(k * tau / (1 + k * tau), rel=1e-6)
