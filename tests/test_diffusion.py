import math

import pytest

from easyreactors.core import PowerLaw, Reaction
from easyreactors.diffusion.packed_bed import PackedBed, first_order_conversion, first_order_weight
from easyreactors.diffusion.pellet import (effectiveness, eta_asymptotic, eta_first_order, falsified, mears,
                                           overall_effectiveness, phi_for_eta, radius_for_eta, thiele,
                                           two_pellets, weisz_prater)
from easyreactors.diffusion.transport import (bed_area, colburn, cylinder_dp, effective_diffusivity, frossling,
                                              mt_limited_conversion, thoenes_kramers)
from easyreactors.reactors import Stream
from easyreactors.units import to_si


# Chapter 14: external mass transfer

def test_fogler_14_1_frossling():
    mt = frossling("0.1 m/s", "1 cm", "0.5 cSt", "1e-10 m^2/s")
    assert mt.Re == pytest.approx(2000) and mt.Sc == pytest.approx(5000)
    assert mt.Sh == pytest.approx(460.7, rel=5e-4)
    assert mt.kc * 1000 == pytest.approx(4.61e-3, rel=5e-4)


def hydrazine():
    dp, gamma = cylinder_dp("0.25 cm", "0.5 cm")
    return dp, gamma, 0.69e-4 * (450 / 298) ** 1.75


def test_fogler_14_2a_thoenes_kramers():
    dp, gamma, D = hydrazine()
    assert dp == pytest.approx(3.61e-3, abs=5e-6) and gamma == pytest.approx(1.20, abs=0.005)
    mt = thoenes_kramers(150, dp, 4.94e-5, D, 0.4, gamma)
    ac = bed_area(dp, 0.4)
    assert mt.Sh == pytest.approx(86.66, rel=1e-3)
    assert mt.kc == pytest.approx(6.15, rel=1e-3)
    assert ac == pytest.approx(998, rel=1e-3)
    assert mt_limited_conversion(mt.kc, ac, 0.05, 150) == pytest.approx(0.87, abs=0.002)


def test_fogler_14_2b_colburn():
    _, _, D = hydrazine()
    dp = math.sqrt(0.0025 * 0.005 + 0.0025 ** 2 / 2)
    mt = colburn(150, dp, 4.94e-5, D, 0.4)
    assert mt.Sh == pytest.approx(212, rel=5e-3)
    assert mt.kc == pytest.approx(7.63, rel=1e-3)
    assert mt_limited_conversion(mt.kc, bed_area(dp, 0.4), 0.05, 150) == pytest.approx(0.90, abs=0.002)


def test_fogler_14_3_split_flow_lowers_conversion():
    k1 = thoenes_kramers(1.0, 1e-3, 1e-5, 1e-5, 0.4).kc
    k2 = thoenes_kramers(0.5, 1e-3, 1e-5, 1e-5, 0.4).kc
    ln2 = math.log(1 / (1 - 0.865)) * (k2 / k1) * 0.5 / 0.5
    assert 1 - math.exp(-ln2) == pytest.approx(0.76, abs=0.005)


def test_fogler_14_4_temperature_barely_helps():
    T1, T2 = 673.0, 773.0
    k1 = thoenes_kramers(1.0, 1e-3, 1e-5, 1e-5, 0.4).kc
    k2 = thoenes_kramers(T2 / T1, 1e-3, 1e-5 * (T2 / T1) ** 1.5, 1e-5 * (T2 / T1) ** 1.75, 0.4).kc
    factor = (k2 / k1) / (T2 / T1)
    assert factor == pytest.approx((T2 / T1) ** (5 / 12), rel=1e-12)
    assert factor == pytest.approx(1.059, abs=5e-4)
    assert 1 - math.exp(-factor * math.log(1 / 0.135)) == pytest.approx(0.88, abs=0.002)


# Chapter 15: internal diffusion

def test_fogler_15_1_effective_diffusivity():
    assert effective_diffusivity(1e-6, 0.4, 0.8, 3.0) == pytest.approx(0.1067e-6, rel=1e-3)


def test_fogler_15_2_two_pellets():
    tp = two_pellets(3e-5, 0.01, 15e-5, 0.001)
    assert tp.phi2 == pytest.approx(1.65, abs=0.01)
    assert tp.phi1 == pytest.approx(16.5, abs=0.1)
    assert tp.eta2 == pytest.approx(0.856, abs=0.001)
    # Book uses the large-φ limit 3/φ = 0.182; exact sphere formula gives 0.171
    assert 3 / tp.phi1 == pytest.approx(0.182, abs=0.001)
    assert tp.eta1 == pytest.approx(eta_first_order(tp.phi1))
    assert phi_for_eta(0.95) == pytest.approx(0.9, abs=0.01)
    assert radius_for_eta(0.95, tp.phi1, 0.01) == pytest.approx(5.5e-4, rel=0.01)


def fogler_15_3():
    k_prime = 4.42e-10 * 530e3
    Ac = math.pi * to_si("2 in", "m") ** 2 / 4
    mt = thoenes_kramers(1e-6 / Ac, 6e-3, 1.53e-8, 2.0e-8, 0.5)
    return k_prime, Ac, mt, bed_area(6e-3, 0.5)


def test_fogler_15_3_nitric_oxide_bed():
    k_prime, Ac, mt, ac = fogler_15_3()
    phi = thiele(k_prime * 2800, 3e-3, 1.82e-8)
    assert phi == pytest.approx(18, rel=1e-3)
    assert Ac == pytest.approx(2.03e-3, rel=2e-3)
    assert mt.Re == pytest.approx(386.7, rel=2e-3)
    assert mt.Sh == pytest.approx(18.0, rel=1e-3)
    assert mt.kc == pytest.approx(6e-5, rel=1e-3)
    assert ac == pytest.approx(500)
    # Book text says Sa = 500 m2/g, but k'1 = 2.34e-7 and φ = 18 only hold with 530 m2/g.
    # Book rounds η to 3/φ = 0.167; with that value every result matches.
    eta_book = 3 / 18
    assert eta_book * 18 ** 2 == pytest.approx(54)
    Omega = overall_effectiveness(eta_book, k_prime * 1400, mt.kc, ac)
    assert Omega == pytest.approx(0.059, abs=5e-4)
    assert mears(Omega * k_prime, 1400, 3e-3, 1, mt.kc, 1.0) == pytest.approx(0.97, abs=0.01)
    W = first_order_weight(0.998, 1e-6, Omega, k_prime)
    assert W * 1000 == pytest.approx(450, rel=0.005)
    assert W / (Ac * 1400) == pytest.approx(0.16, abs=0.005)
    assert weisz_prater(eta_book * k_prime * 1.0, 2800, 3e-3, 1.82e-8, 1.0) == pytest.approx(54, rel=2e-3)


def test_fogler_15_3_numerical_bed_matches_closed_form():
    k_prime, _, mt, ac = fogler_15_3()
    eta = eta_first_order(thiele(k_prime * 2800, 3e-3, 1.82e-8))
    Omega = overall_effectiveness(eta, k_prime * 1400, mt.kc, ac)
    rx = Reaction("NO -> P", PowerLaw(k_prime, {"NO": 1}), catalytic=True)
    feed = Stream.liquid(v0=1e-6, T=1173.0, C={"NO": 0.02 * 101.3e3 / (8.314462618 * 1173)})
    bed = PackedBed(rx, 3e-3, 1.82e-8, 2800, 1400, mt.kc, ac)
    res = bed.size(feed, 0.998)
    assert res.end == pytest.approx(first_order_weight(0.998, 1e-6, Omega, k_prime), rel=1e-6)
    assert res.Omega[0] == pytest.approx(Omega, rel=1e-6)
    assert first_order_conversion(res.end, 1e-6, Omega, k_prime) == pytest.approx(0.998, rel=1e-9)


# Analytical limits and the numerical pellet

@pytest.mark.parametrize("geometry", ["sphere", "cylinder", "slab"])
@pytest.mark.parametrize("phi", [0.1, 1.65, 18.0, 60.0])
def test_numerical_eta_first_order_exact(geometry, phi):
    rx = Reaction("A -> B", PowerLaw(phi ** 2, {"A": 1}))
    e = effectiveness(rx, {"A": 1.0}, 300, 1.0, 1.0, geometry=geometry)
    assert e.eta == pytest.approx(eta_first_order(phi, geometry), rel=1e-7)


def test_eta_limits():
    assert eta_first_order(1e-4) == pytest.approx(1.0, abs=1e-8)
    assert eta_first_order(1e3) == pytest.approx(3 / 1e3, rel=2e-3)


def test_numerical_eta_second_order_tends_to_asymptote():
    gaps = []
    for phi in (30.0, 300.0):
        rx = Reaction("A -> B", PowerLaw(phi ** 2, {"A": 2}))
        e = effectiveness(rx, {"A": 1.0}, 300, 1.0, 1.0)
        gaps.append(abs(e.eta / eta_asymptotic(e.phi, 2) - 1))
    assert gaps[1] < gaps[0] / 5 and gaps[1] < 0.004


def test_falsified_kinetics():
    assert falsified(2, 100e3) == (1.5, 50e3)
