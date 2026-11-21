import math
from dataclasses import dataclass

import numpy as np
from scipy.integrate import solve_ivp
from scipy.interpolate import PchipInterpolator
from scipy.optimize import brentq

from ..core import Reaction
from ..reactors import Stream
from ..units import Value, to_si
from .pellet import Geometry, effectiveness


def first_order_weight(X: float, v0: Value, Omega: float, k_prime: float) -> float:
    return to_si(v0, "m^3/s") / (Omega * k_prime) * math.log(1 / (1 - X))


def first_order_conversion(W: Value, v0: Value, Omega: float, k_prime: float) -> float:
    return 1 - math.exp(-Omega * k_prime * to_si(W, "kg") / to_si(v0, "m^3/s"))


@dataclass
class BedResult:
    W: np.ndarray
    X: np.ndarray
    Cb: np.ndarray
    Cs: np.ndarray
    eta: np.ndarray
    Omega: np.ndarray
    key: str = ""

    @property
    def end(self) -> float:
        return float(self.W[-1])


@dataclass
class PackedBed:
    reaction: Reaction
    R: Value
    De: Value
    rho_c: Value
    rho_b: Value
    kc: Value
    ac: Value
    geometry: Geometry = "sphere"
    n_table: int = 40

    def _setup(self, feed: Stream):
        C0 = feed.conc()
        self.reaction.resolve_ref(C0)
        key, T = self.reaction.ref, feed.T
        nu = self.reaction.nu
        CA0 = C0[key]
        rho_b, kc_ac = to_si(self.rho_b, "kg/m^3"), to_si(self.kc, "m/s") * to_si(self.ac, "1/m")

        def conc(CA):
            return {s: max(C0.get(s, 0.0) + nu[s] / nu[key] * (CA - CA0), 0.0) for s in set(C0) | set(nu)}

        def intrinsic(CA):
            return -self.reaction.species_rates(conc(CA), T)[key]

        grid = CA0 * np.logspace(-6, 0, self.n_table)
        etas = [effectiveness(self.reaction, conc(c), T, self.R, self.De, self.rho_c, self.geometry, key).eta
                for c in grid]
        eta = PchipInterpolator(np.log(grid), etas, extrapolate=True)

        def eta_at(Cs):
            return float(np.clip(eta(math.log(max(Cs, grid[0]))), 0.0, 1.0))

        def observed(Cb):
            if Cb <= 0:
                return 0.0, 0.0, 1.0
            f = lambda Cs: kc_ac * (Cb - Cs) - rho_b * eta_at(Cs) * intrinsic(Cs)
            Cs = brentq(f, 0.0, Cb, xtol=1e-14 * CA0) if f(0.0) > 0 else 0.0
            return eta_at(Cs) * intrinsic(Cs), Cs, eta_at(Cs)

        return key, CA0, observed, intrinsic

    def _result(self, sol, v0, CA0, key, observed, intrinsic) -> BedResult:
        Cb = sol.y[0] / v0
        obs = [observed(c) for c in Cb]
        r_b = np.array([intrinsic(c) for c in Cb])
        r_o = np.array([o[0] for o in obs])
        Omega = np.divide(r_o, r_b, out=np.ones_like(r_o), where=r_b > 0)
        return BedResult(sol.t, 1 - Cb / CA0, Cb, np.array([o[1] for o in obs]),
                         np.array([o[2] for o in obs]), Omega, key)

    def simulate(self, feed: Stream, W: Value, n_points: int = 100, _setup=None) -> BedResult:
        key, CA0, observed, intrinsic = _setup or self._setup(feed)
        v0, W = feed.v, to_si(W, "kg")
        sol = solve_ivp(lambda w, F: [-observed(F[0] / v0)[0]], (0, W), [CA0 * v0], method="LSODA",
                        t_eval=np.linspace(0, W, n_points), rtol=1e-8, atol=1e-14 * CA0 * v0)
        return self._result(sol, v0, CA0, key, observed, intrinsic)

    def size(self, feed: Stream, X: float, W_max: Value = 1e6) -> BedResult:
        setup = self._setup(feed)
        _, CA0, observed, _ = setup
        v0 = feed.v
        hit = lambda w, F: F[0] - CA0 * v0 * (1 - X)
        hit.terminal = True
        sol = solve_ivp(lambda w, F: [-observed(F[0] / v0)[0]], (0, to_si(W_max, "kg")), [CA0 * v0],
                        method="LSODA", events=hit, rtol=1e-8, atol=1e-14 * CA0 * v0)
        if not sol.t_events[0].size:
            raise ValueError(f"X = {X} not reachable with W <= {W_max} kg")
        return self.simulate(feed, sol.t_events[0][0], _setup=setup)
