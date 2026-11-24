import json
import math

import pytest

from easyreactors.flowsheet import Flowsheet, SpecError, simulate

# Flowsheet engine: specs checked against Fogler and analytical results


def first_order(k="0.1 1/s"):
    return {"r1": {"equation": "A -> B", "rate": {"model": "power", "k": k, "orders": {"A": 1}}}}


def liquid_feed(v0="1 m^3/s", CA="1 mol/m^3", T="300 K"):
    return {"type": "feed", "v0": v0, "T": T, "C": {"A": CA}}


def eo():
    reactions = {"r1": {"equation": "EO + W -> EG", "ref": "EO",
                        "rate": {"model": "power", "k": "0.311 1/min", "orders": {"EO": 1}}}}
    feed = {"type": "feed", "v0": "7.24 L/s", "T": "300 K", "C": {"EO": "8.05 mol/L", "W": "27.8 mol/L"}}
    return reactions, feed


def test_fogler_5_2c_two_cstrs_in_series():
    reactions, feed = eo()
    r = simulate({"reactions": reactions,
                  "units": {"F": feed, "R1": {"type": "cstr", "V": "800 gallon"},
                            "R2": {"type": "cstr", "V": "800 gallon"}, "P": {"type": "product"}},
                  "streams": {"s1": ["F.out", "R1.in"], "s2": ["R1.out", "R2.in"], "s3": ["R2.out", "P.in"]}})
    assert r["units"]["R1"]["X"]["EO"] == pytest.approx(0.684, abs=0.003)
    assert r["overall"]["X"]["EO"] == pytest.approx(0.90, abs=0.005)


def test_fogler_5_2b_parallel_cstrs_with_splitter_and_mixer():
    reactions, feed = eo()
    r = simulate({"reactions": reactions,
                  "units": {"F": feed, "S": {"type": "splitter", "fractions": {"a": 0.5, "b": None}},
                            "R1": {"type": "cstr", "V": "800 gallon"}, "R2": {"type": "cstr", "V": "800 gallon"},
                            "M": {"type": "mixer"}, "P": {"type": "product"}},
                  "streams": {"s1": ["F.out", "S.in"], "s2": ["S.a", "R1.in"], "s3": ["S.b", "R2.in"],
                              "s4": ["R1.out", "M.in1"], "s5": ["R2.out", "M.in2"], "s6": ["M.out", "P.in"]}})
    tau_k = 0.311 / 60 * 800 * 3.785411784e-3 / (7.24e-3 / 2)
    assert r["overall"]["X"]["EO"] == pytest.approx(tau_k / (1 + tau_k), rel=1e-9)
    assert r["overall"]["X"]["EO"] == pytest.approx(0.81, abs=0.005)


def recycle_spec(R, V):
    return {"reactions": first_order(),
            "units": {"F": liquid_feed(), "M": {"type": "mixer"}, "R1": {"type": "pfr", "V": V},
                      "S": {"type": "splitter", "fractions": {"rec": R / (1 + R), "out": None}},
                      "P": {"type": "product"}},
            "streams": {"s1": ["F.out", "M.in1"], "s2": ["M.out", "R1.in"], "s3": ["R1.out", "S.in"],
                        "s4": ["S.rec", "M.in2"], "s5": ["S.out", "P.in"]}}


def test_recycle_pfr_levenspiel():
    R, k, X = 2.0, 0.1, 0.8
    V = (R + 1) * math.log((1 + R * (1 - X)) / ((R + 1) * (1 - X))) / k
    r = simulate(recycle_spec(R, V))
    assert r["tears"] == ["s4"]
    assert r["overall"]["X"]["A"] == pytest.approx(X, rel=1e-8)
    assert r["streams"]["s4"]["v"] == pytest.approx(R, rel=1e-9)


def glycol_spec():
    btu = "Btu/lbmol/delta_degF"
    return {
        "thermo": {"Cp": {"A": f"35 {btu}", "B": f"18 {btu}", "C": f"46 {btu}", "M": f"19.5 {btu}"},
                   "T_ref": "68 degF"},
        "reactions": {"r1": {"equation": "A + B -> C", "ref": "A", "dH": "-36400 Btu/lbmol",
                             "rate": {"model": "power", "orders": {"A": 1},
                                      "k": {"A": "16.96e12 1/h", "Ea": "32400 Btu/lbmol"}}}},
        "units": {"F": {"type": "feed", "v0": "326.3 ft^3/h", "T": "75 degF",
                        "F": {"A": "43.04 lbmol/h", "B": "802.8 lbmol/h", "M": "71.87 lbmol/h"}},
                  "R": {"type": "cstr", "V": "300 gallon", "energy": "adiabatic"}, "P": {"type": "product"}},
        "streams": {"s1": ["F.out", "R.in"], "s2": ["R.out", "P.in"]},
    }


def test_fogler_12_3_adiabatic_cstr():
    r = simulate(glycol_spec())
    T = r["streams"]["s2"]["T"]
    assert T * 1.8 == pytest.approx(613, abs=1)
    xmb = (lambda a: a / (1 + a))(2.084e12 * math.exp(-16306 / (T * 1.8)))
    assert r["units"]["R"]["X"]["A"] == pytest.approx(xmb, abs=0.003)
    assert len(r["units"]["R"]["steady_states"]) == 1


def test_recycle_around_adiabatic_cstr_changes_nothing():
    plain = simulate(glycol_spec())
    spec = glycol_spec()
    spec["units"].update(M={"type": "mixer"}, S={"type": "splitter", "fractions": {"rec": 0.5, "out": None}})
    spec["streams"] = {"s1": ["F.out", "M.in1"], "s2": ["M.out", "R.in"], "s3": ["R.out", "S.in"],
                       "s4": ["S.rec", "M.in2"], "s5": ["S.out", "P.in"]}
    r = simulate(spec)
    assert r["streams"]["s5"]["T"] == pytest.approx(plain["streams"]["s2"]["T"], rel=1e-8)
    assert r["overall"]["X"]["A"] == pytest.approx(plain["overall"]["X"]["A"], rel=1e-7)


def test_fogler_5_7_pbr_with_pressure_drop():
    k = 0.00392 / 101325
    spec = {"reactions": {"r1": {"equation": "A + 0.5B -> C", "ref": "A", "catalytic": True,
                                 "rate": {"model": "power", "k": k, "orders": {"A": 1 / 3, "B": 2 / 3},
                                          "basis": "P"}}},
            "units": {"F": {"type": "feed", "phase": "gas", "T": "260 degC", "P": "10 atm",
                            "F": {"A": 0.1362, "B": 0.0681, "I": 0.256}},
                      "R": {"type": "pbr", "W": "16 kg", "alpha": "0.0367 1/kg"}, "P": {"type": "product"}},
            "streams": {"s1": ["F.out", "R.in"], "s2": ["R.out", "P.in"]}}
    r = simulate(spec)
    assert r["units"]["R"]["X"]["A"] == pytest.approx(0.53, abs=0.01)
    assert r["units"]["R"]["profile"]["p"][-1] == pytest.approx(r["streams"]["s2"]["P"] / 1013250, rel=1e-9)


def test_mixer_at_different_temperatures_and_heater_duty():
    spec = {"thermo": {"Cp": {"A": "100 J/mol/K", "B": "50 J/mol/K"}},
            "units": {"F1": {"type": "feed", "v0": 1, "T": 300, "F": {"A": 2.0}},
                      "F2": {"type": "feed", "v0": 1, "T": 400, "F": {"B": 1.0}},
                      "M": {"type": "mixer"}, "H": {"type": "heater", "T": "350 K"}, "P": {"type": "product"}},
            "streams": {"s1": ["F1.out", "M.a"], "s2": ["F2.out", "M.b"], "s3": ["M.out", "H.in"],
                        "s4": ["H.out", "P.in"]}}
    r = simulate(spec)
    T_mix = (200 * 300 + 50 * 400) / 250
    assert r["streams"]["s3"]["T"] == pytest.approx(T_mix)
    assert r["streams"]["s3"]["v"] == pytest.approx(2.0)
    assert r["units"]["H"]["Q"] == pytest.approx(250 * (350 - T_mix))


def test_cstr_multiplicity_reported_through_spec():
    spec = {"reactions": {"r1": {"equation": "A -> B", "rate": {"model": "expression",
                                                              "expr": "k*C_A/(1 + K*C_A)^2",
                                                              "params": {"k": "0.365 1/s", "K": "1 L/mol"}}}},
            "units": {"F": liquid_feed("1 L/s", "10 mol/L"), "R": {"type": "cstr", "V": "100 L"},
                      "P": {"type": "product"}},
            "streams": {"s1": ["F.out", "R.in"], "s2": ["R.out", "P.in"]}}
    r = simulate(spec)
    ss = r["units"]["R"]["steady_states"]
    assert [s["stable"] for s in ss] == [True, False, True]
    assert r["units"]["R"]["selected"] == 2
    assert r["overall"]["X"]["A"] == pytest.approx(ss[2]["X"]["A"])
    assert "3 steady states" in r["warnings"][0]


def test_json_in_and_out():
    r = simulate(json.dumps(recycle_spec(1.0, 5.0)))
    assert json.loads(json.dumps(r))["converged"] is True


def base_spec():
    return {"reactions": first_order(),
            "units": {"F": liquid_feed(), "R": {"type": "cstr", "V": "10 m^3"}, "P": {"type": "product"}},
            "streams": {"s1": ["F.out", "R.in"], "s2": ["R.out", "P.in"]}}


@pytest.mark.parametrize("edit, path", [
    (lambda s: s["units"]["R"].update(type="reactorx"), "units.R.type"),
    (lambda s: s["units"]["R"].update(V="10 kg"), "units.R.V"),
    (lambda s: s["units"]["R"].pop("V"), "units.R.V"),
    (lambda s: s["units"]["R"].update(reactions=["r9"]), "units.R.reactions"),
    (lambda s: s["units"]["R"].update(energy="adiabatic"), "units.R.energy"),
    (lambda s: s["streams"].pop("s2"), "units.R"),
    (lambda s: s["streams"].update(s2=["R.out", "X.in"]), "streams.s2"),
    (lambda s: s["streams"].update(s2=["R.out", "R.in"]), "streams.s2"),
    (lambda s: s["reactions"]["r1"]["rate"].update(model="magic"), "reactions.r1.rate.model"),
    (lambda s: s["units"]["F"].update(colour="red"), "units.F"),
])
def test_spec_errors_point_to_the_field(edit, path):
    spec = base_spec()
    edit(spec)
    with pytest.raises(SpecError) as e:
        Flowsheet.from_spec(spec).simulate()
    assert e.value.path == path


def test_splitter_fractions_must_close():
    spec = recycle_spec(1.0, 5.0)
    spec["units"]["S"]["fractions"] = {"rec": 0.5, "out": 0.6}
    with pytest.raises(SpecError, match="add up to 1"):
        Flowsheet.from_spec(spec)
