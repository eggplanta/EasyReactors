from dataclasses import dataclass, replace

import numpy as np
from scipy.optimize import brentq, minimize_scalar

from ..core import ReactionSet
from ..energy.heat import Heat
from ..units import Value, to_si
from ._common import flow_setup, key_species, solve
from .result import Result
from .stream import Stream


@dataclass
class SteadyStates:
    states: list[Result]
    stable: list[bool]
    T: np.ndarray
    G: np.ndarray
    R: np.ndarray


@dataclass
class CSTR:
    reactions: ReactionSet
    V: Value | None = None
    rho_b: Value | None = None
    heat: Heat | None = None

    def _result(self, feed, species, F0, F, V, rate, T=None) -> Result:
        outlet = feed.with_F(dict(zip(species, F)))
        extras = {}
        if T is not None:
            outlet = replace(outlet, T=float(T))
            extras["T"] = np.array([T])
        return Result(species, np.array([V]), "V", F[None, :], rate(F)[None, :],
                      dict(zip(species, F0)), outlet, extras=extras)

    def _iso(self):
        return replace(self, heat=None)

    def simulate(self, feed: Stream, V: Value | None = None) -> Result:
        V = to_si(self.V if V is None else V, "m^3")
        if self.heat is not None:
            return self.steady_states(feed, V).states[0]
        species, F0, rate = flow_setup(self.reactions, feed, self.rho_b)
        scale = F0.sum()
        F = solve(lambda F: (F0 - F + rate(F) * V) / scale, F0, "CSTR")
        if (F < -1e-9 * scale).any():
            raise RuntimeError("CSTR converged to negative flows")
        return self._result(feed, species, F0, F, V, rate)

    def size(self, feed: Stream, X: float, key: str | None = None) -> Result:
        if self.heat is not None:
            return self._size_energy(feed, X, key)
        species, F0, rate = flow_setup(self.reactions, feed, self.rho_b)
        i = species.index(key_species(self.reactions, key))
        Fi, scale = F0[i] * (1 - X), F0.sum()

        def full(z):
            return np.concatenate([z[:i], [Fi], z[i:-1]])

        def res(z):
            return (F0 - full(z) + rate(full(z)) * np.exp(z[-1])) / scale

        guess = F0.copy()
        guess[i] = Fi
        r_i = rate(guess)[i]
        V_guess = (F0[i] - Fi) / -r_i if r_i < 0 else 1.0
        z = solve(res, np.append(np.delete(guess, i), np.log(V_guess)), "CSTR sizing")
        return self._result(feed, species, F0, full(z), float(np.exp(z[-1])), rate)

    def optimum(self, feed: Stream, species: str, V_bounds=(1e-6, 1e4)) -> tuple[float, float]:
        def neg(logV):
            return -self.simulate(feed, np.exp(logV)).profile(species)[-1]

        opt = minimize_scalar(neg, bounds=np.log(V_bounds), method="bounded")
        return float(np.exp(opt.x)), float(-opt.fun)

    # Energy balance

    def _balance(self, feed: Stream, res: Result, T: float) -> tuple[float, float]:
        h = self.heat
        C = res.outlet.conc()
        rho = None if self.rho_b is None else to_si(self.rho_b, "kg/m^3")
        G = res.end * h.generation(self.reactions, C, T, rho)
        R = sum(h.Cp[s] * f for s, f in res.n0.items()) * (T - feed.T) + h.removal(T)
        return G, R

    def T_range(self, feed: Stream) -> tuple[float, float]:
        h = self.heat
        T0 = feed.T
        lo, hi = (min(T0, h.Ta), max(T0, h.Ta)) if h.exchange else (T0, T0)
        cp0 = sum(h.Cp.get(s, 0.0) * f for s, f in feed.F.items())
        span = min(max(sum(abs(x) for x in h.dH) * sum(feed.F.values()) / cp0, 50.0), 3000.0)
        exo, endo = any(x < 0 for x in h.dH), any(x > 0 for x in h.dH)
        return max(lo - (span if endo else 1.0), 20.0), hi + (span if exo else 1.0)

    def _scan(self, feed, at, T_range, n):
        species, _, _ = flow_setup(self.reactions, feed, self.rho_b)
        self.heat.bind(self.reactions, species)
        lo, hi = T_range or self.T_range(feed)
        Ts = np.linspace(lo, hi, n)
        G, R = np.full(n, np.nan), np.full(n, np.nan)
        for j, T in enumerate(Ts):
            try:
                G[j], R[j] = self._balance(feed, at(T), T)
            except (RuntimeError, ValueError):
                pass
        roots = []
        f = lambda T: np.subtract(*self._balance(feed, at(T), T))
        for j in range(n - 1):
            a, b = G[j] - R[j], G[j + 1] - R[j + 1]
            if np.isfinite(a) and np.isfinite(b) and (a * b < 0 or a == 0):
                T = Ts[j] if a == 0 else brentq(f, Ts[j], Ts[j + 1], xtol=1e-9)
                roots.append((T, a > b))
        return Ts, G, R, roots

    def steady_states(self, feed: Stream, V: Value | None = None, T_range: tuple[float, float] | None = None,
                      n: int = 800) -> SteadyStates:
        V = to_si(self.V if V is None else V, "m^3")
        iso = self._iso()
        at = lambda T: iso.simulate(replace(feed, T=float(T)), V)
        Ts, G, R, roots = self._scan(feed, at, T_range, n)
        if not roots:
            raise RuntimeError("No steady state found in the temperature range")
        states = [self._at(feed, at(T), T) for T, _ in roots]
        return SteadyStates(states, [s for _, s in roots], Ts, G, R)

    def _at(self, feed: Stream, res: Result, T: float) -> Result:
        res.outlet = replace(res.outlet, T=float(T))
        res.extras["T"] = np.array([T])
        return res

    def _size_energy(self, feed: Stream, X: float, key: str | None, T_range=None, n: int = 400) -> Result:
        iso = self._iso()
        at = lambda T: iso.size(replace(feed, T=float(T)), X, key)
        _, _, _, roots = self._scan(feed, at, T_range, n)
        if not roots:
            raise RuntimeError(f"No temperature satisfies both balances for X = {X}")
        T = roots[0][0]
        return self._at(feed, at(T), T)
