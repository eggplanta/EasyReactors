import warnings

import numpy as np
import pytest

from easyreactors.core import Reaction, ReactionSet
from easyreactors.reactors import CSTR, Stream

# Isothermal CSTR multiplicity, checked against analytical roots

V0 = 1e-3
CA0 = 10_000.0
K = 1e-3
TAU_K = 36.5
TAU = 100.0
k1 = TAU_K / TAU


def inhibited(C, T):
    return k1 * C["A"] / (1 + K * C["A"]) ** 2


def feed():
    return Stream.liquid(V0, 300.0, C={"A": CA0})


def cubic_roots():
    # (CA0 - C)(1 + K C)^2 - tau k C = 0
    p = np.polymul([-1, CA0], np.polymul([K, 1], [K, 1])) - np.array([0, 0, TAU * k1, 0])
    r = np.roots(p)
    return np.sort(r[np.isreal(r)].real)


def by_conversion(ss, key="A"):
    order = np.argsort([s.conversion(key) for s in ss.states])
    return [ss.states[i] for i in order], [ss.stable[i] for i in order]


def test_substrate_inhibition_three_states():
    ss = CSTR(ReactionSet([Reaction("A -> B", inhibited)])).steady_states(feed(), TAU * V0)
    states, stable = by_conversion(ss)
    CA = np.sort([s.outlet.conc()["A"] for s in states])
    np.testing.assert_allclose(CA, cubic_roots(), rtol=1e-8)
    assert stable == [True, False, True]


def test_simulate_warns_and_returns_highest_stable():
    reactor = CSTR(ReactionSet([Reaction("A -> B", inhibited)]))
    with pytest.warns(UserWarning, match="3 steady states"):
        res = reactor.simulate(feed(), TAU * V0)
    assert res.outlet.conc()["A"] == pytest.approx(cubic_roots()[0], rel=1e-8)


def test_autocatalytic_washout_and_ignited():
    k = 1e-4
    rs = ReactionSet([Reaction("A + B -> 2B", lambda C, T: k * C["A"] * C["B"], ref="A")])
    tau = 5.0
    ss = CSTR(rs).steady_states(Stream.liquid(V0, 300.0, C={"A": 5000.0}), tau * V0)
    states, stable = by_conversion(ss)
    assert len(states) == 2
    assert states[0].conversion("A") == pytest.approx(0.0, abs=1e-12)
    assert states[1].outlet.conc()["A"] == pytest.approx(1 / (k * tau), rel=1e-8)
    assert stable == [False, True]


def test_series_reactions_multistart():
    k2 = 0.02
    rs = ReactionSet([Reaction("A -> B", inhibited), Reaction("B -> C", lambda C, T: k2 * C["B"])])
    ss = CSTR(rs).steady_states(feed(), TAU * V0)
    states, stable = by_conversion(ss)
    CA = np.sort([s.outlet.conc()["A"] for s in states])
    np.testing.assert_allclose(CA, cubic_roots(), rtol=1e-6)
    for s in states:
        C = s.outlet.conc()
        CB = TAU * inhibited(C, 300.0) / (1 + k2 * TAU)
        assert C["B"] == pytest.approx(CB, rel=1e-6)
    assert stable == [True, False, True]


def test_monotonic_kinetics_single_state_no_warning():
    k = 0.01
    rs = ReactionSet([Reaction("A -> B", lambda C, T: k * C["A"])])
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        res = CSTR(rs).simulate(feed(), TAU * V0)
    assert res.conversion("A") == pytest.approx(k * TAU / (1 + k * TAU), rel=1e-10)
