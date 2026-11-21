import math
from dataclasses import dataclass
from typing import Literal

import numpy as np
from scipy.integrate import solve_bvp
from scipy.interpolate import PchipInterpolator
from scipy.optimize import brentq
from scipy.special import i0e, i1e

from ..core import Reaction
from ..units import Value, to_si

Geometry = Literal["sphere", "cylinder", "slab"]
SHAPE = {"slab": 0, "cylinder": 1, "sphere": 2}


# Analytical relations

def thiele(k: float, R: Value, De: Value, n: float = 1.0, Cs: float = 1.0) -> float:
    return to_si(R, "m") * math.sqrt(k * Cs ** (n - 1) / to_si(De, "m^2/s"))


def eta_first_order(phi: float, geometry: Geometry = "sphere") -> float:
    if phi < 1e-6:
        return 1.0
    if geometry == "sphere" and phi < 1e-2:
        return 1 - phi ** 2 / 15 + 2 * phi ** 4 / 315
    if geometry == "sphere":
        return 3 / phi ** 2 * (phi / math.tanh(phi) - 1)
    if geometry == "slab":
        return math.tanh(phi) / phi
    return 2 * i1e(phi) / (phi * i0e(phi))


def eta_asymptotic(phi_n: float, n: float = 1.0) -> float:
    return math.sqrt(2 / (n + 1)) * 3 / phi_n


def phi_for_eta(eta: float, geometry: Geometry = "sphere") -> float:
    if not 0 < eta < 1:
        raise ValueError("η must be between 0 and 1")
    return brentq(lambda p: eta_first_order(p, geometry) - eta, 1e-8, 1e8, xtol=1e-14, rtol=1e-12)


def overall_effectiveness(eta: float, k1: float, kc: Value, ac: Value) -> float:
    return eta / (1 + eta * k1 / (to_si(kc, "m/s") * to_si(ac, "1/m")))


def weisz_prater(r_obs: float, rho_c: Value, R: Value, De: Value, Cs: Value) -> float:
    return (r_obs * to_si(rho_c, "kg/m^3") * to_si(R, "m") ** 2
            / (to_si(De, "m^2/s") * to_si(Cs, "mol/m^3")))


def mears(r_obs: float, rho_b: Value, R: Value, n: float, kc: Value, Cb: Value) -> float:
    return (r_obs * to_si(rho_b, "kg/m^3") * to_si(R, "m") * n
            / (to_si(kc, "m/s") * to_si(Cb, "mol/m^3")))


def falsified(n: float, E: float) -> tuple[float, float]:
    return (n + 1) / 2, E / 2


@dataclass
class TwoPellets:
    phi1: float
    phi2: float
    eta1: float
    eta2: float


def two_pellets(r1: float, R1: Value, r2: float, R2: Value) -> TwoPellets:
    R1, R2 = to_si(R1, "m"), to_si(R2, "m")
    a = R1 / R2

    def f(p2):
        return (eta_first_order(a * p2) * (a * p2) ** 2) / (eta_first_order(p2) * p2 ** 2) - r1 * R1 ** 2 / (r2 * R2 ** 2)

    p2 = brentq(f, 1e-6, 1e6, xtol=1e-14, rtol=1e-12)
    return TwoPellets(a * p2, p2, eta_first_order(a * p2), eta_first_order(p2))


def radius_for_eta(eta: float, phi_ref: float, R_ref: Value, geometry: Geometry = "sphere") -> float:
    return to_si(R_ref, "m") * phi_for_eta(eta, geometry) / phi_ref


# Numerical effectiveness factor for any rate law

@dataclass
class Effectiveness:
    eta: float
    phi: float
    xi: np.ndarray
    C: np.ndarray


def effectiveness(reaction: Reaction, Cs: dict[str, Value], T: Value, R: Value, De: Value,
                  rho_c: Value | None = None, geometry: Geometry = "sphere",
                  key: str | None = None) -> Effectiveness:
    Cs = {s: to_si(c, "mol/m^3") for s, c in Cs.items()}
    T, R, De = to_si(T, "K"), to_si(R, "m"), to_si(De, "m^2/s")
    reaction.resolve_ref(Cs)
    key = key or reaction.ref
    rho = 1.0 if rho_c is None else to_si(rho_c, "kg/m^3")
    nu = reaction.nu
    a = SHAPE[geometry]
    C_A = Cs[key]

    def conc(CA):
        return {s: Cs.get(s, 0.0) + nu[s] / nu[key] * (CA - C_A) for s in set(Cs) | set(nu)}

    def consumption(CA):
        return -rho * reaction.species_rates(conc(CA), T)[key]

    r_s = consumption(C_A)
    if r_s <= 0:
        raise ValueError("Rate at the surface must be positive for the key species")
    phi = R * math.sqrt(r_s / (De * C_A))
    scale = R ** 2 / (De * C_A)

    psi = np.unique(np.concatenate([[0.0], np.geomspace(1e-9, 1, 1000), np.linspace(0, 1, 1001)]))
    table = PchipInterpolator(psi, [consumption(c * C_A) for c in psi])

    def fun(x, y):
        return np.vstack([y[1], scale * table(np.clip(y[0], 0.0, 1.0))])

    def bc(ya, yb):
        return np.array([ya[1], yb[0] - 1])

    xi = np.linspace(0, 1, 101)
    p = max(phi, 1e-3)
    guess = np.cosh(p * xi) / math.cosh(p) if p < 50 else np.exp(p * (xi - 1))
    y0 = np.vstack([guess, np.gradient(guess, xi)])
    S = np.array([[0, 0], [0, -a]]) if a else None
    sol = solve_bvp(fun, bc, xi, y0, S=S, tol=1e-8, max_nodes=200000)
    if not sol.success:
        raise RuntimeError(f"Pellet BVP did not converge: {sol.message}")
    eta = (a + 1) * sol.y[1, -1] / (scale * r_s)
    return Effectiveness(float(eta), phi, sol.x, sol.y[0] * C_A)
