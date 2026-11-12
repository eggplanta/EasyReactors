from dataclasses import dataclass

import numpy as np
from scipy.integrate import solve_ivp

from ..core import ReactionSet
from ..units import Value, to_si
from ._common import check_refs, key_species, rates, species_list
from .result import Result


@dataclass
class Batch:
    reactions: ReactionSet
    V: Value
    constant_pressure: bool = False
    t_dead: Value = 0.0

    def _setup(self, T, N0, C0):
        V0, T = to_si(self.V, "m^3"), to_si(T, "K")
        if C0 is not None:
            N0 = {s: to_si(c, "mol/m^3") * V0 for s, c in C0.items()}
        else:
            N0 = {s: to_si(n, "mol") for s, n in N0.items()}
        check_refs(self.reactions, N0)
        species = species_list(self.reactions, N0)
        n0 = np.array([N0.get(s, 0.0) for s in species])

        def volume(N):
            return V0 * N.sum() / n0.sum() if self.constant_pressure else V0

        def rate(N):
            return rates(self.reactions, species, dict(zip(species, N / volume(N))), T)

        return species, n0, rate, volume

    def simulate(self, t: Value, T: Value, N0=None, C0=None, n_points: int = 200) -> Result:
        t = to_si(t, "s")
        species, n0, rate, volume = self._setup(T, N0, C0)
        sol = solve_ivp(lambda _, N: rate(N) * volume(N), (0, t), n0, method="LSODA",
                        t_eval=np.linspace(0, t, n_points), dense_output=True,
                        rtol=1e-8, atol=1e-12 * n0.sum())
        if not sol.success:
            raise RuntimeError(sol.message)
        N = sol.y.T
        return Result(species, sol.t, "t", N, np.array([rate(x) for x in N]),
                      dict(zip(species, n0)), None, sol.sol)

    def time_to(self, X: float, T: Value, key: str | None = None, N0=None, C0=None,
                t_max: Value = 1e7) -> Result:
        species, n0, rate, volume = self._setup(T, N0, C0)
        i = species.index(key_species(self.reactions, key))
        hit = lambda _, N: N[i] - n0[i] * (1 - X)
        hit.terminal = True
        sol = solve_ivp(lambda _, N: rate(N) * volume(N), (0, to_si(t_max, "s")), n0,
                        method="LSODA", events=hit, rtol=1e-8, atol=1e-12 * n0.sum())
        if not sol.t_events[0].size:
            raise ValueError(f"X = {X} not reachable for {species[i]} (equilibrium or t_max)")
        return self.simulate(sol.t_events[0][0], T, N0, C0)

    def productivity(self, res: Result, species: str) -> float:
        produced = res.profile(species)[-1] - res.n0[species]
        return produced / (res.end + to_si(self.t_dead, "s"))
