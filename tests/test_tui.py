import math

import numpy as np
import pytest

from easyreactors.tui.compute import (ReactionSpec, build_feed, build_reactions, model_conversions,
                                     run_batch, run_flow_reactor, two_param_conversion)
from easyreactors.tui.netexpr import build, parse, to_text
from easyreactors.tui.parsing import K_unit, k_unit, mapping, numbers, quantity, table_file
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


def test_human_units_parse_to_the_expected_si_units():
    from easyreactors.tui.parsing import human_K, human_k
    for n in (0, 0.5, 1, 2, 3):
        for basis in ("C", "P"):
            for per_mass in (False, True):
                assert quantity(f"1 {human_k(n, basis, per_mass)}", k_unit(n, basis, per_mass)) > 0
    assert human_K({"A": 1}, {"B": 2}) == "mol/L"
    assert quantity("1 " + human_K({"A": 2}, {"B": 1}).replace("^", "**"), K_unit({"A": 2}, {"B": 1})) > 0


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


# Form layer, Fogler examples

def test_fogler_5_3_pfr_from_form():
    rs = build_reactions([ReactionSpec("C2H6 > C2H4 + H2", k="0.072 1/s", Ea="82 kcal/mol", T_ref="1000")])
    feed = build_feed("Gas", "1100", "6", "", "C2H6=0.425 lbmol/s")
    rep = run_flow_reactor("PFR", "Size", rs, feed, X="0.8")
    assert rep.summary[0].startswith("PFR: V = 2.288 m3")


def test_fogler_8_3_and_8_4_maxima_from_form():
    specs = [ReactionSpec("A > B", k="0.5 1/h"), ReactionSpec("B > C", k="0.2 1/h")]
    tau = 1 / math.sqrt(0.5 * 0.2) * 60
    rep = run_batch("Simulate", build_reactions(specs), "1000", "300", "A=2", t="10 h")
    assert f"at t = {math.log(0.2 / 0.5) / (0.2 - 0.5) * 60:.4g} min" in rep.summary[1]
    feed = build_feed("Liquid", "300", "", "1 m3/s", "A=2")
    rep = run_flow_reactor("CSTR", "Maximize", build_reactions(specs), feed, species="B")
    assert f"τ = {tau:.4g} min" in rep.summary[0]


def test_reversible_and_missing_fields():
    rs = build_reactions([ReactionSpec("A <> B", k="0.1 1/s", K="2")])
    assert rs.reactions[0].reversible
    with pytest.raises(ValueError, match="K of"):
        build_reactions([ReactionSpec("A <> B", k="0.1 1/s")])
    with pytest.raises(ValueError, match="required"):
        build_reactions([ReactionSpec("A > B")])
    with pytest.raises(ValueError, match=r"Reaction 2 \(2A > D\): k has the wrong units for total order 1: use "
                                         r"something like 1/min"):
        build_reactions([ReactionSpec("A > B", k="0.1 1/s"), ReactionSpec("2A > D", orders="A=1", k="0.05 L/(mol*min)")])


# App smoke test

def test_app_examples_run():
    import asyncio
    from textual.widgets import DataTable, TabbedContent
    from easyreactors.tui.app import EasyReactors

    async def run():
        app = EasyReactors()
        out = {}
        async with app.run_test(size=(160, 50)) as pilot:
            for tab in ("tab-reactor", "tab-network", "tab-trans", "tab-rtd", "tab-two", "tab-diff", "tab-kin"):
                app.query_one(TabbedContent).active = tab
                await pilot.pause()
                await app.action_example()
                await pilot.pause()
                await app.action_run()
                await app.workers.wait_for_complete()
                await pilot.pause(0.2)
                tables = list(app.pane().results.query(DataTable))
                out[tab] = [str(c) for c in tables[0].get_row_at(0)] if tables else None
        return out

    out = asyncio.run(run())
    assert out["tab-reactor"][0] == "EO"
    assert out["tab-network"][1] == "3.02 m3"
    assert out["tab-trans"][0] == "A"
    assert out["tab-rtd"][1] == "37.25"
    assert out["tab-two"][1] == "0.7017"
    assert out["tab-diff"] == ["Thiele modulus φ (at Cs)", "18.01"]
    assert out["tab-kin"][:2] == ["0", "0.05"]


# Diffusion forms, Fogler 15-3 and 14-2

def test_diffusion_forms_fogler():
    from easyreactors.tui.compute import (CORRELATIONS, effective_De, run_criteria, run_mass_transfer,
                                         run_packed_bed, run_pellet)
    assert quantity("4.42e-10*530 m3/(g*s)", "m^3/kg/s") == pytest.approx(2.3426e-4)
    rs = build_reactions([ReactionSpec("NO > P", k="4.42e-10*530 m3/(g*s)", catalytic=True)])
    De = effective_De("1.82e-8")
    pellet = dict(run_pellet(rs, "NO=2e-4", "1173", "3 mm", De, "2.8 g/cm3").tables[0][2])
    assert pellet["Thiele modulus φ (at Cs)"] == pytest.approx(18, rel=1e-3)
    mt = dict(run_mass_transfer(CORRELATIONS[0], "150", "", "4.94e-5", "0.69e-4*2.06", "0.4",
                                cyl_D="0.25 cm", cyl_L="0.5 cm", L="0.05").tables[0][2])
    assert mt["kc [m/s]"] == pytest.approx(6.15, rel=2e-3)
    assert mt["X if mass-transfer limited"] == pytest.approx(0.87, abs=0.002)
    crit = run_criteria("2.872e-6", "2.8 g/cm3", "1.4 g/cm3", "3 mm", De, "", "2.078e-4", "1", "6e-5")
    assert dict((r[0], r[1]) for r in crit.tables[0][2])["Mears MR"] == pytest.approx(0.97, abs=0.01)
    feed = build_feed("Liquid", "1173", "", "1e-6 m3/s", "NO=2.078e-4")
    bed = run_packed_bed(rs, feed, "Size", "0.998", "3 mm", De, "2.8 g/cm3", "1.4 g/cm3", "6e-5", "500")
    assert bed.summary[0].startswith("W = 0.4586 kg")


def test_quantity_accepts_numeric_products():
    assert quantity("1/3 h", "s") == pytest.approx(1200)
    assert quantity("5 1/min", "1/s") == pytest.approx(5 / 60)


# Energy, PBR and transient forms (Fogler 11-3, 12-6, 5-7, 6-3)

def test_energy_forms_fogler():
    from easyreactors.tui.compute import (DropSpec, EnergySpec, build_heat, run_flow_reactor, run_semibatch,
                                         run_startup)
    specs = [ReactionSpec("A <> B", model="Reversible", orders="A=1", reverse_orders="B=1", k="31.1 1/h",
                          Ea="65.7 kJ/mol", T_ref="360", K="3.03", dH="-6900 J/mol", K_T="333")]
    feed = build_feed("Liquid", "330", "", "262.9", "A=9.3, I=1.033")
    heat = build_heat(EnergySpec("Adiabatic", "A=141, B=141, I=161"), specs, False)
    rep = run_flow_reactor("PFR", "Size", build_reactions(specs), feed, X="0.4", heat=heat)
    assert rep.summary[0].startswith("PFR: V = 1.149 m3")
    assert any("Adiabatic equilibrium" in line for line in rep.summary)
    specs = [ReactionSpec("A > B", k="3.3 1/min", Ea="9900 cal/mol", T_ref="300", dH="-55000 J/mol"),
             ReactionSpec("B > C", k="4.58 1/min", Ea="27000 cal/mol", T_ref="500", dH="-71500 J/mol")]
    heat = build_heat(EnergySpec("Heat exchange (constant Ta)", "A=200, B=200, C=200", "40000 J/(min*K)", "330"),
                      specs, True)
    rep = run_flow_reactor("CSTR", "Simulate", build_reactions(specs),
                           build_feed("Liquid", "283", "", "1000", "A=0.3"), V="10", heat=heat)
    assert "5 steady states" in rep.summary[0]
    with pytest.raises(ValueError, match="ΔHrx of"):
        build_heat(EnergySpec("Adiabatic", "A=1"), [ReactionSpec("A > B", k="1 1/s")], False)
    specs = [ReactionSpec("A + 0.5B > C", ref="A", orders="A=1/3, B=2/3", k="0.00392 mol/(atm*kg*s)", basis="P",
                          catalytic=True)]
    feed = build_feed("Gas", "260 degC", "10", "", "A=0.1362 mol/s, B=0.0681 mol/s, I=0.256 mol/s")
    rep = run_flow_reactor("PBR", "Size", build_reactions(specs), feed, X="0.6", drop=DropSpec("α given", "0.0367"))
    assert rep.summary[0] == "PBR: W = 20.58 kg"
    rep = run_semibatch(build_reactions([ReactionSpec("A + B > C + D", ref="A", k="2.2 L/(mol*s)")]), "5", "300",
                        "A=0.05", "0.05 L/s", "B=0.025", "500 s")
    assert rep.summary[0].startswith("Semibatch: t = 8.333 min, V = 5 L → 30 L")
    rep = run_startup(build_reactions([ReactionSpec("A > B", k="0.02 1/min")]), "500",
                      build_feed("Liquid", "300", "", "4", "A=3"), "A=0.5", "300")
    assert f"t_s = {125 / 3.5 * math.log(100):.4g} min" in rep.summary[1]


def test_ergun_from_bed_data_fogler_5_4():
    from easyreactors.tui.compute import DropSpec, build_drop
    from easyreactors.units import to_si
    v0 = to_si("252.8 ft^3/h", "m^3/s")
    feed = Stream(F={"A": 1.0}, T=533.15, P=to_si("10 atm", "Pa"), phase="gas")
    feed.F = {"A": v0 * feed.P / (8.314462618 * feed.T)}
    alpha, rho_b, notes = build_drop(DropSpec("Ergun (bed data)", Ac="0.01414 ft^2", Dp="0.0208 ft", porosity="0.45",
                                              rho_c="120 lb/ft^3", mu="0.0673 lb/(ft*h)", rho0="0.413 lb/ft^3"), feed)
    assert alpha == pytest.approx(0.0367, abs=3e-4)
    assert rho_b == pytest.approx(to_si("120 lb/ft^3", "kg/m^3") * 0.55)


# Kinetics forms (Fogler 7-1 to 7-4, 3-1)

def test_kinetics_forms_fogler():
    from easyreactors.tui.compute import KIN_METHODS, run_arrhenius, run_batch_kinetics, run_rate_kinetics
    from easyreactors.tui.parsing import named_table, table_text
    data = "t  0 50 100 150 200 250 300\nC  0.05 0.038 0.0306 0.0256 0.0222 0.0195 0.0174"
    assert table_text(data)[0].tolist() == [0, 50, 100, 150, 200, 250, 300]
    rep = run_batch_kinetics(data, "min", "mol/L", KIN_METHODS[2])
    assert rep.summary[0].startswith("Order n = 2, k = 0.1253 L/(mol*min)")
    assert quantity("0.1253 L/(mol*min)", k_unit(2)) > 0
    rep = run_batch_kinetics(data, "min", "mol/L", KIN_METHODS[3])
    assert rep.summary[0].startswith("Order n = 2.045, k = 0.1467")
    rep = run_batch_kinetics(data, "min", "mol/L", KIN_METHODS[0], order="2")
    assert rep.summary[0].startswith("Order n = 2, k = 0.1253")
    rates = "p_CO p_H2 r\n1.0 1.0 5.2e-3\n1.8 1.0 13.2e-3\n4.08 1.0 30.0e-3\n1.0 0.1 4.95e-3\n1.0 0.5 7.42e-3\n1.0 4.0 5.25e-3"
    assert list(named_table(rates)) == ["p_CO", "p_H2", "r"]
    rep = run_rate_kinetics(rates, "a*p_CO*p_H2^0.5/(1 + b*p_H2)", "a=0.02, b=2")
    params = dict(rep.tables[0][2])
    assert params["a"] == pytest.approx(0.018, abs=5e-4) and params["b"] == pytest.approx(1.49, abs=0.005)
    with pytest.raises(ValueError, match="Line 2"):
        named_table("p_A r\n1 2 3")
    rep = run_arrhenius("T 313 319 323 328 333\nk 0.00043 0.00103 0.00180 0.00355 0.00717", "1/s")
    assert rep.summary[0].startswith("Ea = 121.5 kJ/mol (Ea/R = 14612 K)")
