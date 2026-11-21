import math

import pytest

from easyreactors.core import Arrhenius, PowerLaw, Reaction, ReactionSet, Reversible, VantHoff
from easyreactors.core.rates import R
from easyreactors.energy.equilibrium import adiabatic_equilibrium, equilibrium_conversion
from easyreactors.energy.heat import Heat, lookup
from easyreactors.reactors import CSTR, PFR, Batch, Stream
from easyreactors.reactors.pfr import PBR
from easyreactors.reactors.pressure import ergun_alpha, ergun_beta0
from easyreactors.reactors.transient import CSTRStartup, Semibatch
from easyreactors.units import to_si

CAL = 4.184


# Pressure drop, Fogler 5-4 and 5-7

def test_fogler_5_4_ergun_parameters():
    beta0 = ergun_beta0("7383.3 lb/ft^2/h", "0.413 lb/ft^3", "0.0673 lb/ft/h", "0.0208 ft", 0.45)
    assert to_si(f"{beta0} Pa/m", "atm/ft") == pytest.approx(0.0775, abs=5e-5)
    assert ergun_alpha(beta0, "0.01414 ft^2", "120 lb/ft^3", 0.45, "10 atm") == pytest.approx(0.0367, abs=2e-4)


def ethylene_oxide():
    rate = PowerLaw(quantity_k := 0.00392 / 101325, {"A": 1 / 3, "B": 2 / 3}, basis="P")
    rs = ReactionSet([Reaction("A + 0.5B -> C", rate, ref="A", catalytic=True)], inerts=["I"])
    # Book rounds FB0 = 0.1362/2 to 0.068
    feed = Stream.gas(F={"A": 0.1362, "B": 0.0681, "I": 0.256}, T="260 degC", P="10 atm")
    return rs, feed, quantity_k


def test_fogler_5_7_pbr_with_pressure_drop():
    rs, feed, k = ethylene_oxide()
    no_drop = PBR(rs).size(feed, 0.6)
    yA0 = 0.1362 / sum(feed.F.values())
    eps = yA0 * -0.5
    k_prime = k * yA0 * feed.P * 0.5 ** (2 / 3)
    exact = 0.1362 / k_prime * ((1 + eps) * math.log(1 / 0.4) - eps * 0.6)
    assert no_drop.end == pytest.approx(exact, rel=1e-6)
    # Book rounds PA0 to 3 atm and ε to -0.15, giving 16 kg
    assert 0.1362 / 0.0074 * (0.85 * math.log(2.5) + 0.15 * 0.6) == pytest.approx(16.0, abs=0.05)
    assert no_drop.end == pytest.approx(16.2, abs=0.05)
    res = PBR(rs, alpha=0.0367).size(feed, 0.6)
    # Book reads ≈ 20 kg and ΔP ≈ 5 atm from its plot
    assert res.end == pytest.approx(20.6, abs=0.1)
    assert (1 - res.extras["p"][-1]) * 10 == pytest.approx(4.7, abs=0.1)
    assert PBR(rs, alpha=0.0367).simulate(feed, 16).conversion("A") == pytest.approx(0.53, abs=0.01)


def test_ergun_first_order_analytic():
    k, v0 = 1e-4, 1e-3
    rs = ReactionSet([Reaction("A -> B", PowerLaw(k, {"A": 1}), catalytic=True)])
    feed = Stream.gas(F={"A": 1.0}, T=500, P="5 atm")
    v0 = 1.0 * R * 500 / feed.P
    W, alpha = 20.0, 0.02
    res = PBR(rs, alpha=alpha).simulate(feed, W)
    integral = 2 / (3 * alpha) * (1 - (1 - alpha * W) ** 1.5)
    assert res.conversion("A") == pytest.approx(1 - math.exp(-k / v0 * integral), rel=1e-6)
    assert res.extras["p"][-1] == pytest.approx(math.sqrt(1 - alpha * W), rel=1e-6)


# Adiabatic and constant Ta, Fogler 11-3 and 12-1

def butane(dH=-6900, CA0=9.3, FT=163, T0=330):
    k = Arrhenius.from_ref(31.1 / 3600, 65700, 360)
    rs = ReactionSet([Reaction("A <> B", Reversible(k, {"A": 1}, {"B": 1}, VantHoff(3.03, dH, 333)))],
                     inerts=["I"])
    FA, FI = 0.9 * FT / 3.6, 0.1 * FT / 3.6
    return rs, Stream.liquid(v0=FA / (CA0 * 1000), T=T0, F={"A": FA, "I": FI})


CP_BUTANE = {"A": 141, "B": 141, "I": 161}


def test_fogler_11_3_adiabatic_pfr_and_cstr():
    rs, feed = butane()
    heat = Heat([-6900], CP_BUTANE)
    assert PFR(rs, heat=heat).size(feed, 0.4).end == pytest.approx(1.15, abs=0.005)
    res = PFR(rs, heat=heat).size(feed, 0.7)
    assert res.extras["T"][-1] == pytest.approx(330 + 6900 / 159 * 0.7, abs=0.2)
    # Book integrates its rounded table by Simpson's rule (2.6 m3); our ODE gives 2.49 m3
    book = 0.075 * (3.74 + 3 * 2.78 + 3 * 2.50 + 3.88) + 0.05 / 3 * (3.88 + 4 * 5.99 + 23.29)
    assert book == pytest.approx(2.6, abs=0.02)
    assert res.end == pytest.approx(2.49, abs=0.01)
    T = 338.6
    k = rs.reactions[0].rate.kf(T) * 3600
    assert (T, k, rs.reactions[0].rate.K(T)) == pytest.approx((338.7, 7.76, 2.9), rel=5e-3)
    cstr = CSTR(rs, heat=heat).size(feed, 0.4)
    assert cstr.end == pytest.approx(1.0, abs=0.01)
    assert cstr.outlet.T == pytest.approx(347.3, abs=0.1)


def test_fogler_12_1_constant_ta_and_adiabatic():
    rs, feed = butane(-34500, 1.86, 16.3, 305)
    cool = PFR(rs, heat=Heat([-34500], CP_BUTANE, Ua="5000 kJ/(m^3*h*K)", Ta=315)).simulate(feed, 5)
    assert cool.conversion("A") == pytest.approx(0.7632, abs=2e-4)
    assert cool.extras["T"][-1] == pytest.approx(317.47, abs=0.01)
    heat = Heat([-34500], CP_BUTANE)
    adia = PFR(rs, heat=heat).simulate(feed, 5)
    assert adia.conversion("A") == pytest.approx(0.365, abs=1e-3)
    assert adia.extras["T"][-1] == pytest.approx(384, abs=0.5)
    heat.bind(rs, ["A", "B", "I"])
    Xe, Te = adiabatic_equilibrium(rs.reactions[0], heat, ["A", "B", "I"], feed.F,
                                   lambda F, T: feed.conc(F), feed.T)
    assert (Xe, Te) == pytest.approx((adia.conversion("A"), adia.extras["T"][-1]), rel=1e-4)
    K = rs.reactions[0].rate.K(Te)
    assert equilibrium_conversion(rs.reactions[0], feed.F, lambda F, T: feed.conc(F), Te) == pytest.approx(K / (1 + K))


# Gas phase with ΔCp, Fogler 12-2

def acetone():
    rs = ReactionSet([Reaction("A -> B + C", PowerLaw(Arrhenius.from_ref(3.58, 34222 * R, 1035), {"A": 1}))])
    return rs, Stream.gas(F={"A": 0.0376}, T=1035, P="162 kPa")


def test_fogler_12_2_endothermic_gas():
    rs, feed = acetone()
    assert feed.conc()["A"] == pytest.approx(18.8, rel=2e-3)
    heat = Heat([80770], {"A": 163, "B": 83, "C": 71}, T_ref=298, Ua=16500, Ta=1250)
    res = PFR(rs, heat=heat).simulate(feed, "1 L")
    assert res.conversion("A") == pytest.approx(0.95, abs=0.005)
    adia = PFR(rs, heat=Heat([80770], {"A": 163, "B": 83, "C": 71}, T_ref=298)).simulate(feed, "5 L")
    X, T = adia.conversion("A"), adia.extras["T"][-1]
    dH = 80770 - 9 * (T - 298)
    assert 163 * (T - 1035) + X * dH == pytest.approx(0, abs=1e-3)


# Multiple reactions, Fogler 12-5

def test_fogler_12_5_parallel_pfr():
    k1 = Arrhenius.from_ref(10, 4000 * R, 300)
    k2 = Arrhenius.from_ref(0.09e-3, 9000 * R, 300)
    rs = ReactionSet([Reaction("A -> B", PowerLaw(k1, {"A": 1})), Reaction("2A -> C", PowerLaw(k2, {"A": 2}))])
    feed = Stream.gas(F={"A": 100}, T=423.15, P=100 * R * 423.15)
    # Polymath uses Ua = 4000 with V in dm3, i.e. 4000 J/(dm3·s·K)
    heat = Heat([-20000, -60000], {"A": 90, "B": 90, "C": 180}, Ua="4000 J/(L*s*K)", Ta=373)
    res = PFR(rs, heat=heat).simulate(feed, "1 L")
    assert res.outlet.F["B"] == pytest.approx(55, abs=0.5)
    assert res.outlet.F["C"] == pytest.approx(22.5, abs=0.2)


# Nonisothermal CSTR, Fogler 12-3, 12-4 and 12-6

LBMOL_H = 453.59237 / 3600
BTU = to_si("1 Btu/lb/delta_degF", "J/kg/K") / 1000


def glycol(E_R=None):
    E = to_si("32400 Btu/lb", "J/kg") / 1000 if E_R is None else E_R / 1.8 * R
    rs = ReactionSet([Reaction("A + B -> C", PowerLaw(Arrhenius(16.96e12 / 3600, E), {"A": 1}), ref="A")],
                     inerts=["M"])
    Cp = {"A": 35 * BTU, "B": 18 * BTU, "C": 46 * BTU, "M": 19.5 * BTU}
    dH = to_si("-36400 Btu/lb", "J/kg") / 1000
    return rs, Cp, dH


def glycol_feed(T0):
    return Stream.liquid(v0="326.3 ft^3/h", T=T0,
                         F={"A": 43.04 * LBMOL_H, "B": 802.8 * LBMOL_H, "M": 71.87 * LBMOL_H})


def test_fogler_12_3_adiabatic_cstr():
    rs, Cp, dH = glycol()
    ss = CSTR(rs, heat=Heat([dH], Cp, T_ref="68 degF")).steady_states(glycol_feed("75 degF"), "300 gallon")
    assert len(ss.states) == 1
    s = ss.states[0]
    assert s.outlet.T * 1.8 == pytest.approx(613, abs=1)
    # Book reads X = 0.83 off its plot; its own X_MB at 613 °R is 0.85
    xmb = (lambda a: a / (1 + a))(2.084e12 * math.exp(-16306 / (s.outlet.T * 1.8)))
    assert s.conversion("A") == pytest.approx(xmb, abs=0.003)


def test_fogler_12_4_cstr_with_cooling_coil():
    UA = to_si("4000 Btu/h/delta_degF", "W/K")
    # Curves are nearly tangent: book's E/R = 16306 °R (R = 1.987) and T0 = 535 °R matter
    rs, Cp, dH = glycol(E_R=16306)
    heat = Heat([dH], Cp, T_ref=528 / 1.8, Ua=UA, Ta=545 / 1.8)
    s = CSTR(rs, heat=heat).simulate(glycol_feed(535 / 1.8), "300 gallon")
    assert s.outlet.T * 1.8 == pytest.approx(563.7, abs=0.1)
    assert s.conversion("A") == pytest.approx(0.364, abs=0.001)


def test_fogler_12_6_five_steady_states():
    k1 = Arrhenius.from_ref(3.3 / 60, 9900 * CAL, 300)
    k2 = Arrhenius.from_ref(4.58 / 60, 27000 * CAL, 500)
    rs = ReactionSet([Reaction("A -> B", PowerLaw(k1, {"A": 1})), Reaction("B -> C", PowerLaw(k2, {"B": 1}))])
    feed = Stream.liquid(v0="1000 L/min", T=283, C={"A": "0.3 mol/L"})
    heat = Heat([-55000, -71500], {"A": 200, "B": 200, "C": 200}, Ua="40000 J/(min*K)", Ta=330)
    ss = CSTR(rs, heat=heat).steady_states(feed, "10 L")
    book = [(310, 0.285, 0.015), (363, 0.189, 0.111), (449, 0.033, 0.265), (558, 0.004, 0.163),
            (677, 0.001, 0.005)]
    assert len(ss.states) == 5
    assert ss.stable == [True, False, True, False, True]
    for s, (T, CA, CB) in zip(ss.states, book):
        C = s.outlet.conc()
        assert s.outlet.T == pytest.approx(T, abs=1.5)
        assert C["A"] / 1000 == pytest.approx(CA, abs=0.002)
        assert C["B"] / 1000 == pytest.approx(CB, abs=0.003)


# Batch, Fogler 13-1

def test_fogler_13_1_adiabatic_batch():
    k = Arrhenius.from_ref(2.73e-4, 9059 * R, 297)
    rs = ReactionSet([Reaction("A + B -> C", PowerLaw(k, {"A": 1}), ref="A")], inerts=["M"])
    # ΔCp = 0 as in the book (it neglects the ΔCp term)
    heat = Heat([-20202 * CAL], {"A": 35 * CAL, "B": 18 * CAL, "C": 53 * CAL, "M": 19.5 * CAL})
    N0 = {"A": 54.8, "B": 555, "M": 98.8}
    res = Batch(rs, "18 L", heat=heat).simulate(2500, 286, N0=N0)
    cps = 35 + 18 * 555 / 54.8 + 19.5 * 98.8 / 54.8
    assert cps == pytest.approx(252.5, abs=0.1)
    X = 1 - res.profile("A") / 54.8
    assert max(abs(res.extras["T"] - (286 + 20202 / cps * X))) < 1e-6
    assert Batch(rs, "18 L", heat=heat).time_to(0.515, 286, N0=N0).extras["T"][-1] == pytest.approx(327.2, abs=0.1)


# Semibatch and startup, UFU CR1 Aulas 5 and 6

def test_cstr_startup_first_order_analytic():
    rs = ReactionSet([Reaction("A -> B", PowerLaw(0.02 / 60, {"A": 1}))])
    feed = Stream.liquid(v0="4 L/min", T=300, C={"A": "3 mol/L"})
    st = CSTRStartup(rs, "500 L").simulate(feed, "30 min", {"A": "0.5 mol/L"})
    tau, k = 125, 0.02
    assert st.t / 60 == pytest.approx(tau / (tau * k + 1) * math.log(100), rel=1e-6)
    assert st.t / 60 == pytest.approx(164, abs=1)
    CAEE = 3 / (tau * k + 1)
    CA = st.result.profile("A")[-1] / 0.5 / 1000
    assert CA == pytest.approx(CAEE + (0.5 - CAEE) * math.exp(-(tau * k + 1) / tau * 30), rel=1e-7)


def test_semibatch_first_order_analytic():
    k, V0, v0, CA0, CAi, t = 0.05 / 60, 0.1, 5e-3 / 60, 2000.0, 500.0, 600.0
    rs = ReactionSet([Reaction("A -> B", PowerLaw(k, {"A": 1}))])
    res = Semibatch(rs, V0, 300).simulate(t, {"A": CAi}, v0, {"A": CA0})
    tau0 = V0 / v0
    exact = CA0 / ((tau0 + t) * k) + (CAi - CA0 / (tau0 * k)) * tau0 / (tau0 + t) * math.exp(-k * t)
    assert res.profile("A")[-1] / res.extras["V"][-1] == pytest.approx(exact, rel=1e-7)
    assert res.extras["V"][-1] == pytest.approx(V0 + v0 * t)


# Thermodynamic data lookup

def test_lookup_ketene_cracking():
    # Database Hf of ketene is -48.6 kJ/mol (Fogler 12-2 uses -61.09), so ΔHrx differs by 12 kJ/mol
    Hf = {s: lookup(n).Hf for s, n in {"A": "acetone", "B": "ketene", "C": "methane"}.items()}
    assert Hf["A"] == pytest.approx(-216.67e3, rel=0.005) and Hf["C"] == pytest.approx(-74.81e3, rel=0.005)
    liquid = lookup("n-butane", 330, "liquid")
    assert 130 < liquid.Cp < 160
    assert liquid.Hf < lookup("n-butane").Hf
    with pytest.raises(ValueError):
        lookup("not-a-chemical-xyz")
