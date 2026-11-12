import math

import pytest

from easyreactors.core import (
    AmbiguousReference, Arrhenius, Expression, PowerLaw, Reaction, ReactionSet,
    Reversible, gas, parse_equation,
)
from easyreactors.core.rates import R
from easyreactors.units import to_si


def test_to_si():
    assert to_si("10 L/min", "m^3/s") == pytest.approx(1e-2 / 60)
    assert to_si("2 atm", "Pa") == pytest.approx(202650)
    assert to_si(5.0, "Pa") == 5.0


def test_parse_equation():
    nu, rev = parse_equation("A + 2B -> C")
    assert nu == {"A": -1, "B": -2, "C": 1} and not rev
    nu, rev = parse_equation("CO + 1/2 O2 <=> CO2")
    assert nu["O2"] == -0.5 and rev


def test_arrhenius_ref():
    k = Arrhenius.from_ref(0.1, 50e3, 300)
    assert k(300) == pytest.approx(0.1)
    assert k(310) == pytest.approx(0.1 * math.exp(50e3 / R * (1 / 300 - 1 / 310)))


def test_reversible_equilibrium_zero():
    r = Reversible(1.0, {"A": 1}, {"B": 1}, K=2.0)
    assert r({"A": 1.0, "B": 2.0}, 300) == pytest.approx(0.0)


def test_expression_reserved_names():
    e = Expression("k*C_A/(1+E*C_A)", k=2.0, E=1.0)
    assert e({"A": 1.0}, 300) == pytest.approx(1.0)


def test_expression_blocks_code():
    with pytest.raises(ValueError):
        Expression('__import__("os")')


def test_species_rates_ref():
    rx = Reaction("A + 2B -> C", PowerLaw(1.0, {}), ref="B")
    assert rx.species_rates({}, 300) == {"A": -0.5, "B": -1.0, "C": 0.5}


def test_limiting_reactant():
    assert Reaction("A + 2B -> C", PowerLaw(1.0, {})).resolve_ref({"A": 1, "B": 1}) == "B"
    assert Reaction("A + 2B -> C", PowerLaw(1.0, {})).resolve_ref({"A": 1, "B": 3}) == "A"


def test_ambiguous_reference():
    rs = ReactionSet([Reaction("A + B -> C", PowerLaw(1.0, {})),
                      Reaction("C + A -> D", PowerLaw(1.0, {})),
                      Reaction("C -> E", PowerLaw(1.0, {}))])
    amb = rs.resolve_refs({"A": 1, "B": 2})
    assert [r.equation for r in amb] == ["C + A -> D"]
    with pytest.raises(AmbiguousReference):
        amb[0].species_rates({}, 300)


def test_net_rates_series():
    rs = ReactionSet([Reaction("A -> B", PowerLaw(2.0, {"A": 1})),
                      Reaction("B -> C", PowerLaw(1.0, {"B": 1}))], inerts=["I"])
    r = rs.net_rates({"A": 1.0, "B": 1.0}, 300)
    assert rs.species == ["A", "B", "C", "I"]
    assert list(r) == pytest.approx([-2.0, 1.0, 1.0, 0.0])


def test_gas_concentration():
    C = gas({"A": 1.0, "I": 1.0}, 300, 101325)
    assert C["A"] == pytest.approx(0.5 * 101325 / (R * 300))
