from dataclasses import dataclass

import numpy as np
from scipy.integrate import solve_ivp

from ..core import ReactionSet
from ..energy.heat import Heat
from ..units import Value, to_si
from ._common import check_refs, key_species, rates, species_list
from .result import Result


@dataclass
class Batch:
    reactions: ReactionSet
    V: Value
    constant_pressure: bool = False
    t_dead: Value = 0.0
    heat: Heat | None = None

    def _setup(self, T, N0, C0):
        V0, T0 = to_si(self.V, "m^3"), to_si(T, "K")
        if C0 is not None:
            N0 = {s: to_si(c, "mol/m^3") * V0 for s, c in C0.items()}
        else:
            N0 = {s: to_si(n, "mol") for s, n in N0.items()}
        check_refs(self.reactions, N0)
        species = species_list(self.reactions, N0)
        n0 = np.array([N0.get(s, 0.0) for s in species])
        n = len(species)
        h = self.heat
        if h is not None:
            h.bind(self.reactions, species)
            n0 = np.append(n0, T0)

        def state(y):
            return (y[:n], y[n]) if h is not None else (y, T0)

        def volume(N, T):
            return V0 * N.sum() / n0[:n].sum() * T / T0 if self.constant_pressure else V0

        def rate(y):
            N, T = state(y)
            return rates(self.reactions, species, dict(zip(species, N / volume(N, T))), T)

        def rhs(_, y):
            N, T = state(y)
            V = volume(N, T)
            dN = rate(y) * V
            if h is None:
                return dN
            C = dict(zip(species, N / V))
            dT = (V * h.generation(self.reactions, C, T) - h.removal(T)) / h.capacity(species, N)
            return np.append(dN, dT)

        return species, n0, rate, rhs, n

    def _result(self, sol, species, n0, rate, n) -> Result:
        Y = sol.y.T
        extras = {"T": Y[:, n]} if self.heat is not None else {}
        return Result(species, sol.t, "t", Y[:, :n], np.array([rate(y) for y in Y]),
                      dict(zip(species, n0[:n])), None, lambda t: sol.sol(t)[:n], extras)

    def _atol(self, n0, n):
        return np.concatenate([np.full(n, 1e-12 * n0[:n].sum()), np.full(len(n0) - n, 1e-9)])

    def simulate(self, t: Value, T: Value, N0=None, C0=None, n_points: int = 200) -> Result:
        t = to_si(t, "s")
        species, n0, rate, rhs, n = self._setup(T, N0, C0)
        sol = solve_ivp(rhs, (0, t), n0, method="LSODA", t_eval=np.linspace(0, t, n_points),
                        dense_output=True, rtol=1e-8, atol=self._atol(n0, n))
        if not sol.success:
            raise RuntimeError(sol.message)
        return self._result(sol, species, n0, rate, n)

    def time_to(self, X: float, T: Value, key: str | None = None, N0=None, C0=None,
                t_max: Value = 1e7) -> Result:
        species, n0, rate, rhs, n = self._setup(T, N0, C0)
        i = species.index(key_species(self.reactions, key))
        hit = lambda _, y: y[i] - n0[i] * (1 - X)
        hit.terminal = True
        sol = solve_ivp(rhs, (0, to_si(t_max, "s")), n0, method="LSODA", events=hit,
                        rtol=1e-8, atol=self._atol(n0, n))
        if not sol.t_events[0].size:
            raise ValueError(f"X = {X} not reachable for {species[i]} (equilibrium or t_max)")
        return self.simulate(sol.t_events[0][0], T, N0, C0)

    def productivity(self, res: Result, species: str) -> float:
        produced = res.profile(species)[-1] - res.n0[species]
        return produced / (res.end + to_si(self.t_dead, "s"))
