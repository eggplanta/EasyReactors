from dataclasses import dataclass
from typing import Literal

import numpy as np
from scipy.linalg import expm
from scipy.optimize import curve_fit
from scipy.stats import linregress

from ..core import ReactionSet
from ..network import Bypass, Parallel, Series
from ..reactors import CSTR, PFR, Stream
from ..reactors._common import check_refs, rates, solve, species_list
from ..units import to_si
from .rtd import RTD, ModelResult


@dataclass
class Fit:
    alpha: float
    beta: float
    r2: float | None = None


def _seconds(t, t_unit: str) -> np.ndarray:
    return np.asarray(t, float) * to_si(f"1 {t_unit}", "s")


def _tau(tau, t_unit: str) -> float:
    return to_si(tau, "s") if isinstance(tau, str) else float(tau) * to_si(f"1 {t_unit}", "s")


# CSTR with bypass and dead volume

def bypass_dead_step(t, alpha: float, beta: float, tau: float) -> np.ndarray:
    return 1 - (1 - beta) * np.exp(-(1 - beta) * np.asarray(t, float) / (alpha * tau))


def fit_bypass_dead(t, C, C0: float, tau, t_unit: str = "s") -> Fit:
    t, tau = _seconds(t, t_unit), _tau(tau, t_unit)
    fit = linregress(t / tau, np.log(C0 / (C0 - np.asarray(C, float))))
    beta = 1 - np.exp(-fit.intercept)
    return Fit((1 - beta) / fit.slope, beta, fit.rvalue ** 2)


def bypass_dead(reactions: ReactionSet, feed: Stream, V, alpha: float, beta: float):
    V = to_si(V, "m^3")
    return Parallel([CSTR(reactions, V=alpha * V), Bypass()], split=[1 - beta, beta]).simulate(feed)


# Two CSTRs with interchange

def interchange_pulse(t, alpha: float, beta: float, tau: float) -> np.ndarray:
    M = np.array([[-(1 + beta) / alpha, beta / alpha],
                  [beta / (1 - alpha), -beta / (1 - alpha)]]) / tau
    return np.array([expm(M * ti)[0, 0] for ti in np.atleast_1d(np.asarray(t, float))])


def fit_interchange(t, C, tau, t_unit: str = "s",
                    loss: Literal["relative", "absolute"] = "relative") -> Fit:
    t, tau = _seconds(t, t_unit), _tau(tau, t_unit)
    y = np.asarray(C, float) / C[0]
    if loss == "relative":
        f, target = (lambda t, a, b: np.log(interchange_pulse(t, a, b, tau))), np.log(y)
    else:
        f, target = (lambda t, a, b: interchange_pulse(t, a, b, tau)), y
    (a, b), _ = curve_fit(f, t, target, p0=[0.5, 0.5], bounds=([1e-3, 1e-6], [0.999, 1e3]))
    ss_res = np.sum((f(t, a, b) - target) ** 2)
    return Fit(a, b, 1 - ss_res / np.sum((target - target.mean()) ** 2))


def interchange(reactions: ReactionSet, feed: Stream, V, alpha: float, beta: float) -> ModelResult:
    V, v0 = to_si(V, "m^3"), feed.v
    check_refs(reactions, feed.F)
    species = species_list(reactions, feed.F)
    c0 = np.array([feed.F.get(s, 0.0) / v0 for s in species])
    V1, V2, v1, m = alpha * V, (1 - alpha) * V, beta * v0, len(species)

    def r(C):
        return rates(reactions, species, dict(zip(species, C)), feed.T)

    def res(x):
        C1, C2 = x[:m], x[m:]
        return np.concatenate([v0 * c0 + v1 * C2 - (v0 + v1) * C1 + r(C1) * V1,
                               v1 * C1 - v1 * C2 + r(C2) * V2]) / (v0 * c0.sum())

    x = solve(res, np.concatenate([c0, c0]), "Interchange model")
    return ModelResult(species, dict(zip(species, c0)), dict(zip(species, x[:m])))


# PFR and CSTR in series

def fit_pfr_cstr(rtd: RTD, tau) -> Fit:
    tau = to_si(tau, "s")
    tau_s = np.sqrt(rtd.variance)
    return Fit((rtd.tm - tau_s) / tau, tau_s / tau)


def pfr_cstr(reactions: ReactionSet, feed: Stream, V, alpha: float, beta: float,
             order: Literal["pfr_first", "cstr_first"] = "pfr_first"):
    V = to_si(V, "m^3")
    units = [PFR(reactions, V=alpha * V), CSTR(reactions, V=beta * V)]
    return Series(units if order == "pfr_first" else units[::-1]).simulate(feed)


# PFR with bypass

def fit_pfr_bypass(rtd: RTD, tau, t_split) -> Fit:
    tau, t_split = to_si(tau, "s"), to_si(t_split, "s")
    late = rtd.t >= t_split
    beta = rtd.integrate(np.where(late, 0.0, rtd.E))
    t_p = rtd.integrate(np.where(late, rtd.t * rtd.E, 0.0)) / (1 - beta)
    return Fit(t_p * (1 - beta) / tau, beta)


def pfr_bypass(reactions: ReactionSet, feed: Stream, V, alpha: float, beta: float):
    V = to_si(V, "m^3")
    return Parallel([PFR(reactions, V=alpha * V), Bypass()], split=[1 - beta, beta]).simulate(feed)
