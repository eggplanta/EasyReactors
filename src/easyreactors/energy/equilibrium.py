from typing import Callable

from scipy.optimize import brentq

from ..core import Reaction
from .heat import Heat

Conc = Callable[[dict[str, float], float], dict[str, float]]


def _moles(rx: Reaction, F0: dict[str, float], key: str, xi: float) -> dict[str, float]:
    out = dict(F0)
    for s, v in rx.nu.items():
        out[s] = F0.get(s, 0.0) + v / -rx.nu[key] * xi
    return out


def _bounds(rx: Reaction, F0: dict[str, float], key: str) -> tuple[float, float]:
    hi = min(F0.get(s, 0.0) / (abs(v) / abs(rx.nu[key])) for s, v in rx.nu.items() if v < 0)
    lo = -min(F0.get(s, 0.0) / (v / abs(rx.nu[key])) for s, v in rx.nu.items() if v > 0)
    return lo, hi


def equilibrium_conversion(rx: Reaction, F0: dict[str, float], conc: Conc, T: float,
                           key: str | None = None) -> float:
    key = key or rx.ref
    lo, hi = _bounds(rx, F0, key)
    f = lambda xi: rx.rate(conc(_moles(rx, F0, key, xi), T), T)
    span = hi - lo
    a, b = lo + 1e-12 * span, hi - 1e-12 * span
    if f(a) * f(b) > 0:
        return (b if f(b) > 0 else a) / F0[key]
    return brentq(f, a, b, xtol=1e-14 * span, rtol=1e-12) / F0[key]


def adiabatic_temperature(heat: Heat, species: list[str], F0: dict[str, float], key: str, X: float,
                          T0: float, i: int = 0) -> float:
    cp0 = sum(heat.Cp[s] * F0.get(s, 0.0) for s in species)
    n = F0[key] * X
    return (cp0 * T0 - n * (heat.dH[i] - heat.dCp[i] * heat.T_ref)) / (cp0 + n * heat.dCp[i])


def adiabatic_equilibrium(rx: Reaction, heat: Heat, species: list[str], F0: dict[str, float], conc: Conc,
                          T0: float, key: str | None = None) -> tuple[float, float]:
    key = key or rx.ref
    T = lambda X: adiabatic_temperature(heat, species, F0, key, X, T0)
    g = lambda X: equilibrium_conversion(rx, F0, conc, T(X), key) - X
    X = brentq(g, 0.0, 1.0 - 1e-12, xtol=1e-13) if g(1.0 - 1e-12) < 0 else 1.0
    return X, T(X)
