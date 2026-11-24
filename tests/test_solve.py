import math

import pytest
from scipy.optimize import brentq

from easyreactors.flowsheet import SpecError, simulate, solve

# Generic solver: inverse problems and optimization, checked against Fogler and analytical results

K1, K2 = 0.5, 0.2


def series_rx():
    return {"r1": {"equation": "A -> B", "rate": {"model": "power", "k": f"{K1} 1/h", "orders": {"A": 1}}},
            "r2": {"equation": "B -> C", "rate": {"model": "power", "k": f"{K2} 1/h", "orders": {"B": 1}}}}


def one_reactor(kind, V, reactions, feed=None, **extra):
    return {"reactions": reactions,
            "units": {"F": feed or {"type": "feed", "v0": "1 L/h", "T": 300, "C": {"A": "2 mol/L"}},
                      "R": {"type": kind, "V": V}, "P": {"type": "product"}},
            "streams": {"s1": ["F.out", "R.in"], "s2": ["R.out", "P.in"]}, **extra}


def V_of(r, name="units.R.V"):
    return r["variables"][name]["value"]


def test_fogler_8_4_cstr_optimum_space_time():
    r = solve(one_reactor("cstr", {"vary": True, "min": "0.01 L", "max": "100 L"}, series_rx(),
                          objective={"maximize": "streams.s2.C.B"}))
    assert r["status"] == "optimized"
    assert V_of(r) == pytest.approx(1 / math.sqrt(K1 * K2), rel=1e-6)
    C = r["result"]["streams"]["s2"]["C"]
    assert V_of(r) == pytest.approx(3.16, abs=0.005)
    assert C["B"] / 1000 == pytest.approx(0.75, abs=0.005)
    # Book rounds CA = 2/(1 + 3.16·0.5) = 0.775 up to 0.78
    assert 2 / (1 + 3.16 * K1) == pytest.approx(0.775, abs=5e-4)
    assert C["A"] / 1000 == pytest.approx(0.775, abs=5e-4)
    assert r["result"]["overall"]["X"]["A"] == pytest.approx(0.61, abs=0.005)


def test_fogler_8_3_pfr_optimum_space_time():
    r = solve(one_reactor("pfr", {"vary": True, "min": "0.01 L", "max": "100 L"}, series_rx(),
                          objective={"maximize": "streams.s2.C.B"}))
    assert V_of(r) == pytest.approx(math.log(K1 / K2) / (K1 - K2), rel=1e-5)
    assert V_of(r) == pytest.approx(3.05, abs=0.005)
    C = r["result"]["streams"]["s2"]["C"]
    # Book's own formula (E8-3) with its numbers gives CB = 1.086 and CA = 0.434; it prints 1.07 and 0.44
    a, b = K1 / (K2 - K1), K2 / (K2 - K1)
    assert 2 * K1 / (K2 - K1) * ((K1 / K2) ** a - (K1 / K2) ** b) == pytest.approx(1.086, abs=5e-4)
    assert C["B"] / 1000 == pytest.approx(1.086, abs=5e-4)
    assert C["A"] / 1000 == pytest.approx(2 * (K1 / K2) ** a, rel=1e-5)


def test_fogler_5_2a_size_cstr_for_conversion():
    rx = {"r1": {"equation": "EO + W -> EG", "ref": "EO",
                 "rate": {"model": "power", "k": "0.311 1/min", "orders": {"EO": 1}}}}
    feed = {"type": "feed", "v0": "7.24 L/s", "T": 300, "C": {"EO": "8.05 mol/L", "W": "27.8 mol/L"}}
    r = solve(one_reactor("cstr", {"unknown": True, "guess": "1 m^3"}, rx, feed,
                          specs=[{"var": "units.R.X.EO", "value": 0.8}]))
    assert r["status"] == "solved"
    assert V_of(r) == pytest.approx(7.24e-3 * 0.8 / (0.311 / 60 * 0.2), rel=1e-8)
    assert V_of(r) == pytest.approx(5.538, rel=0.015)


def test_find_rate_constant_from_conversion():
    rx = {"r1": {"equation": "A -> B", "rate": {"model": "power", "orders": {"A": 1},
                                                "k": {"unknown": True, "guess": "1 1/h"}}}}
    r = solve(one_reactor("cstr", "2 L", rx, specs=[{"var": "overall.X.A", "value": 0.6}]))
    k = r["variables"]["reactions.r1.rate.k"]
    assert k["unit"] == "1 / hour"
    assert k["value"] == pytest.approx(0.6 / (2 * 0.4), rel=1e-8)


def test_target_with_units():
    r = solve(one_reactor("cstr", {"unknown": True, "min": "0.1 L", "max": "100 L"}, series_rx(),
                          specs=[{"var": "streams.s2.C.A", "value": "1 mol/L"}]))
    assert V_of(r) == pytest.approx(1 / K1, rel=1e-8)


def test_two_unknown_volumes():
    rx = {"r1": {"equation": "A -> B", "rate": {"model": "power", "k": "0.1 1/s", "orders": {"A": 1}}}}
    spec = {"reactions": rx,
            "units": {"F": {"type": "feed", "v0": 1, "T": 300, "C": {"A": 1}},
                      "R1": {"type": "cstr", "V": {"unknown": True, "guess": 3}},
                      "R2": {"type": "cstr", "V": {"unknown": True, "guess": 30}}, "P": {"type": "product"}},
            "streams": {"s1": ["F.out", "R1.in"], "s2": ["R1.out", "R2.in"], "s3": ["R2.out", "P.in"]},
            "specs": [{"var": "units.R1.X.A", "value": 0.5}, {"var": "overall.X.A", "value": 0.75}]}
    r = solve(spec)
    assert V_of(r, "units.R1.V") == pytest.approx(10, rel=1e-7)
    assert V_of(r, "units.R2.V") == pytest.approx(10, rel=1e-7)


def test_two_solutions_on_each_side_of_the_maximum():
    target = 0.9

    def CB(tau):
        return 2 * K1 / (K2 - K1) * (math.exp(-K1 * tau) - math.exp(-K2 * tau))

    exact = [brentq(lambda t: CB(t) - target, 0.01, 3.05), brentq(lambda t: CB(t) - target, 3.06, 30)]
    r = solve(one_reactor("pfr", {"unknown": True, "min": "0.01 L", "max": "30 L"}, series_rx(),
                          specs=[{"var": "streams.s2.C.B", "value": f"{target} mol/L"}]))
    sols = [s["units.R.V"]["value"] for s in r["solutions"]]
    assert sols == pytest.approx(exact, rel=1e-6)
    assert V_of(r) == pytest.approx(exact[0], rel=1e-6)


def test_constrained_maximum_sits_on_the_constraint():
    r = solve(one_reactor("pfr", {"vary": True, "min": "0.01 L", "max": "30 L"}, series_rx(),
                          objective={"maximize": "streams.s2.F.B"},
                          constraints=[{"var": "overall.X.A", "max": 0.5}]))
    assert V_of(r) == pytest.approx(math.log(2) / K1, rel=1e-5)
    assert r["constraints"][0]["value"] == pytest.approx(0.5, abs=1e-6)


def test_minimum_total_volume_of_two_cstrs_is_equal_split():
    k, X = 0.1, 0.9
    rx = {"r1": {"equation": "A -> B", "rate": {"model": "power", "k": k, "orders": {"A": 1}}}}
    spec = {"reactions": rx,
            "units": {"F": {"type": "feed", "v0": 1, "T": 300, "C": {"A": 1}},
                      "R1": {"type": "cstr", "V": {"vary": True, "min": 1, "max": 200}},
                      "R2": {"type": "cstr", "V": {"unknown": True, "min": 1, "max": 200}},
                      "P": {"type": "product"}},
            "streams": {"s1": ["F.out", "R1.in"], "s2": ["R1.out", "R2.in"], "s3": ["R2.out", "P.in"]},
            "specs": [{"var": "overall.X.A", "value": X}],
            "objective": {"minimize": "units.R1.V + units.R2.V"}}
    r = solve(spec)
    V = (math.sqrt(1 / (1 - X)) - 1) / k
    assert V_of(r, "units.R1.V") == pytest.approx(V, rel=1e-4)
    assert V_of(r, "units.R2.V") == pytest.approx(V, rel=1e-4)
    assert r["result"]["overall"]["X"]["A"] == pytest.approx(X, rel=1e-6)


def test_unknown_bypass_fraction():
    k, V, X = 0.1, 20.0, 0.5
    rx = {"r1": {"equation": "A -> B", "rate": {"model": "power", "k": k, "orders": {"A": 1}}}}
    spec = {"reactions": rx,
            "units": {"F": {"type": "feed", "v0": 1, "T": 300, "C": {"A": 1}},
                      "S": {"type": "splitter", "fractions": {"r": None, "b": {"unknown": True, "min": 0, "max": 0.9}}},
                      "R": {"type": "pfr", "V": V}, "M": {"type": "mixer"}, "P": {"type": "product"}},
            "streams": {"s1": ["F.out", "S.in"], "s2": ["S.r", "R.in"], "s3": ["S.b", "M.b"],
                        "s4": ["R.out", "M.a"], "s5": ["M.out", "P.in"]},
            "specs": [{"var": "overall.X.A", "value": X}]}
    r = solve(spec)
    beta = brentq(lambda b: (1 - b) * (1 - math.exp(-k * V / (1 - b))) - X, 0, 0.9)
    assert r["variables"]["units.S.fractions.b"]["value"] == pytest.approx(beta, rel=1e-7)


# Errors

def test_degrees_of_freedom_mismatch():
    with pytest.raises(SpecError, match="need exactly 1 spec"):
        solve(one_reactor("cstr", {"unknown": True, "guess": 1}, series_rx()))


def test_vary_needs_objective_and_bounds():
    with pytest.raises(SpecError, match="need an objective"):
        solve(one_reactor("cstr", {"vary": True, "min": 1, "max": 2}, series_rx()))
    with pytest.raises(SpecError, match="min and max"):
        solve(one_reactor("cstr", {"vary": True, "min": 1}, series_rx(), objective={"maximize": "overall.X.A"}))


def test_unreachable_spec_reports_range():
    with pytest.raises(SpecError, match="spans"):
        solve(one_reactor("cstr", {"unknown": True, "min": "0.1 L", "max": "10 L"}, series_rx(),
                          specs=[{"var": "overall.X.A", "value": 0.999}]))


def test_bad_result_reference():
    with pytest.raises(SpecError, match="not found"):
        solve(one_reactor("cstr", {"unknown": True, "guess": "1 L"}, series_rx(),
                          specs=[{"var": "streams.s9.F.A", "value": 1}]))


def test_simulate_rejects_variables():
    with pytest.raises(SpecError, match="use solve"):
        simulate(one_reactor("cstr", {"unknown": True, "guess": "1 L"}, series_rx()))


def test_solve_without_variables_just_simulates():
    r = solve(one_reactor("cstr", "2 L", series_rx()))
    assert r["status"] == "simulated"
    assert r["result"]["overall"]["X"]["A"] == pytest.approx(0.5)
