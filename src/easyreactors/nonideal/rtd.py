import math
from dataclasses import dataclass
from typing import Literal

import numpy as np
from scipy.integrate import (cumulative_simpson, cumulative_trapezoid, simpson, solve_bvp,
                             solve_ivp, trapezoid)
from scipy.interpolate import CubicSpline, PchipInterpolator, make_interp_spline
from scipy.optimize import brentq

from ..core import ReactionSet
from ..reactors import CSTR, PFR, Batch, Stream
from ..reactors._common import rates, species_list
from ..units import to_si

Method = Literal["trapezoid", "simpson", "pchip", "spline"]


class RTD:
    def __init__(self, t, E, method: Method = "trapezoid", n_fine: int = 4001):
        t, E = np.asarray(t, float), np.asarray(E, float)
        self.t_data, self.E_data, self.method = t, E, method
        if method in ("pchip", "spline"):
            f = PchipInterpolator(t, E) if method == "pchip" else CubicSpline(t, E, bc_type="natural")
            t = np.linspace(t[0], t[-1], n_fine)
            E = np.clip(f(t), 0.0, None)
            self._rule = "trapezoid"
        elif method in ("trapezoid", "simpson"):
            self._rule = method
        else:
            raise ValueError(f"Unknown integration method: {method}")
        self.t = t
        self.E = E / self.integrate(E)
        self.F = self.cumulative(self.E)
        tf = np.linspace(self.t[0], self.t[-1], n_fine)
        if method == "simpson" and len(self.t) >= 3:
            Ef = np.clip(make_interp_spline(self.t, self.E, k=2)(tf), 0.0, None)
        else:
            Ef = np.interp(tf, self.t, self.E)
        Ef /= trapezoid(Ef, tf)
        Ff = cumulative_trapezoid(Ef, tf, initial=0.0)
        self._fine = (tf, Ef / np.clip(1 - Ff, 1e-9, None))

    @classmethod
    def from_pulse(cls, t, C, t_unit: str = "s", method: Method = "trapezoid") -> "RTD":
        return cls(np.asarray(t, float) * to_si(f"1 {t_unit}", "s"), C, method)

    @classmethod
    def from_E(cls, t, E, t_unit: str = "s", method: Method = "trapezoid") -> "RTD":
        return cls(np.asarray(t, float) * to_si(f"1 {t_unit}", "s"), E, method)

    @classmethod
    def from_step(cls, t, C, C0: float, t_unit: str = "s", method: Method = "trapezoid") -> "RTD":
        t = np.asarray(t, float) * to_si(f"1 {t_unit}", "s")
        return cls(t, np.gradient(np.asarray(C, float) / C0, t), method)

    def integrate(self, y) -> float:
        return float(simpson(y, x=self.t) if self._rule == "simpson" else trapezoid(y, self.t))

    def cumulative(self, y) -> np.ndarray:
        if self._rule == "simpson":
            return cumulative_simpson(y, x=self.t, initial=0.0)
        return cumulative_trapezoid(y, self.t, initial=0.0)

    @property
    def tm(self) -> float:
        return self.integrate(self.t * self.E)

    @property
    def variance(self) -> float:
        return self.integrate((self.t - self.tm) ** 2 * self.E)

    @property
    def skewness(self) -> float:
        return self.integrate((self.t - self.tm) ** 3 * self.E) / self.variance ** 1.5

    @property
    def theta(self) -> np.ndarray:
        return self.t / self.tm

    @property
    def E_theta(self) -> np.ndarray:
        return self.E * self.tm

    @property
    def intensity(self) -> np.ndarray:
        return self.E / np.clip(1 - self.F, 1e-9, None)

    @property
    def n_tanks(self) -> float:
        return self.tm ** 2 / self.variance

    def peclet(self, bc: Literal["closed", "open"] = "closed") -> float:
        s2 = self.variance / self.tm ** 2
        if bc == "closed":
            f = lambda Pe: 2 / Pe - 2 / Pe ** 2 * (1 - math.exp(-Pe)) - s2
        else:
            f = lambda Pe: 2 / Pe + 8 / Pe ** 2 - s2
        return brentq(f, 1e-6, 1e6)


def E_cstr(t, tau):
    return np.exp(-np.asarray(t, float) / tau) / tau


def E_tanks(t, tau, n):
    t = np.asarray(t, float)
    return t ** (n - 1) / (math.gamma(n) * (tau / n) ** n) * np.exp(-n * t / tau)


def E_laminar(t, tau):
    t = np.asarray(t, float)
    return np.where(t >= tau / 2, tau ** 2 / (2 * np.maximum(t, 1e-300) ** 3), 0.0)


@dataclass
class ModelResult:
    species: list[str]
    C0: dict[str, float]
    C: dict[str, float]

    def conversion(self, s: str) -> float:
        return (self.C0[s] - self.C[s]) / self.C0[s]


def _C0(C0: dict) -> dict[str, float]:
    return {s: to_si(c, "mol/m^3") for s, c in C0.items()}


def segregation(rtd: RTD, reactions: ReactionSet, C0: dict, T) -> ModelResult:
    C0 = _C0(C0)
    res = Batch(reactions, V=1.0).simulate(rtd.t[-1], T, C0=C0)
    C_t = res.interp(rtd.t).T
    C = {s: rtd.integrate(C_t[:, i] * rtd.E) for i, s in enumerate(res.species)}
    return ModelResult(res.species, {s: C0.get(s, 0.0) for s in res.species}, C)


def max_mixedness(rtd: RTD, reactions: ReactionSet, C0: dict, T) -> ModelResult:
    C0 = _C0(C0)
    T = to_si(T, "K")
    reactions.resolve_refs(C0)
    species = species_list(reactions, C0)
    c0 = np.array([C0.get(s, 0.0) for s in species])

    def rhs(lam, C):
        Lam = np.interp(lam, *rtd._fine)
        return -rates(reactions, species, dict(zip(species, C)), T) + (C - c0) * Lam

    sol = solve_ivp(rhs, (rtd.t[-1], 0.0), c0, method="LSODA", rtol=1e-8, atol=1e-12 * c0.sum())
    if not sol.success:
        raise RuntimeError(sol.message)
    return ModelResult(species, dict(zip(species, c0)), dict(zip(species, sol.y[:, -1])))


def tanks_in_series(reactions: ReactionSet, feed: Stream, V, n: int):
    from ..network import Series
    V = to_si(V, "m^3")
    return Series([CSTR(reactions, V=V / n) for _ in range(n)]).simulate(feed)


def tis_first_order(Da: float, n: float) -> float:
    return 1 - 1 / (1 + Da / n) ** n


def dispersion_first_order(Da: float, Pe: float) -> float:
    q = math.sqrt(1 + 4 * Da / Pe)
    num = 4 * q * math.exp(Pe / 2)
    den = (1 + q) ** 2 * math.exp(Pe * q / 2) - (1 - q) ** 2 * math.exp(-Pe * q / 2)
    return 1 - num / den


def dispersion(reactions: ReactionSet, feed: Stream, V, Pe: float, n_points: int = 101) -> ModelResult:
    V = to_si(V, "m^3")
    v = feed.v
    tau = V / v
    guess = PFR(reactions, V=V).simulate(feed, n_points=n_points)
    species = guess.species
    c0 = np.array([feed.F.get(s, 0.0) / v for s in species])
    z = np.linspace(0, 1, n_points)
    Cg = guess.n.T / v
    y0 = np.vstack([Cg, np.gradient(Cg, z, axis=1)])
    m = len(species)

    def fun(z, y):
        C, dC = y[:m], y[m:]
        r = np.array([rates(reactions, species, dict(zip(species, C[:, k])), feed.T)
                      for k in range(C.shape[1])]).T
        return np.vstack([dC, Pe * (dC - tau * r)])

    def bc(ya, yb):
        return np.concatenate([ya[:m] - ya[m:] / Pe - c0, yb[m:]])

    sol = solve_bvp(fun, bc, z, y0, tol=1e-8, max_nodes=100000)
    if not sol.success:
        raise RuntimeError(f"Dispersion BVP did not converge: {sol.message}")
    return ModelResult(species, dict(zip(species, c0)), dict(zip(species, sol.y[:m, -1])))
