from dataclasses import dataclass

import numpy as np
from scipy.integrate import solve_ivp

from ..core import ReactionSet
from ..units import Value, to_si
from ._common import flow_setup, key_species
from .result import Result
from .stream import Stream


@dataclass
class PFR:
    reactions: ReactionSet
    V: Value | None = None
    rho_b: Value | None = None

    def _setup(self, feed: Stream):
        species, F0, rate = flow_setup(self.reactions, feed, self.rho_b)
        return species, F0, lambda V, F: rate(F)

    def simulate(self, feed: Stream, V: Value | None = None, n_points: int = 200) -> Result:
        V = to_si(self.V if V is None else V, "m^3")
        species, F0, rhs = self._setup(feed)
        sol = solve_ivp(rhs, (0, V), F0, method="LSODA", t_eval=np.linspace(0, V, n_points),
                        dense_output=True, rtol=1e-8, atol=1e-12 * F0.sum())
        if not sol.success:
            raise RuntimeError(sol.message)
        F = sol.y.T
        r = np.array([rhs(v, f) for v, f in zip(sol.t, F)])
        return Result(species, sol.t, "V", F, r, dict(zip(species, F0)),
                      feed.with_F(dict(zip(species, F[-1]))), sol.sol)

    def size(self, feed: Stream, X: float, key: str | None = None, V_max: Value = 1e6) -> Result:
        species, F0, rhs = self._setup(feed)
        i = species.index(key_species(self.reactions, key))
        hit = lambda V, F: F[i] - F0[i] * (1 - X)
        hit.terminal = True
        sol = solve_ivp(rhs, (0, to_si(V_max, "m^3")), F0, method="LSODA", events=hit,
                        rtol=1e-8, atol=1e-12 * F0.sum())
        if not sol.t_events[0].size:
            raise ValueError(f"X = {X} not reachable for {species[i]} (equilibrium or V_max)")
        return self.simulate(feed, sol.t_events[0][0])
