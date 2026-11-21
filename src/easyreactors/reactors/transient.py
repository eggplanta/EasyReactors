from dataclasses import dataclass, replace

import numpy as np
from scipy.integrate import solve_ivp

from ..core import ReactionSet
from ..units import Value, to_si
from ._common import check_refs, key_species, rates, species_list
from .cstr import CSTR
from .result import Result
from .stream import Stream


def _conc(C: dict[str, Value] | None) -> dict[str, float]:
    return {s: to_si(c, "mol/m^3") for s, c in (C or {}).items()}


@dataclass
class Semibatch:
    reactions: ReactionSet
    V0: Value
    T: Value

    def simulate(self, t: Value, C0: dict[str, Value], v0: Value, C_feed: dict[str, Value],
                 n_points: int = 200) -> Result:
        V0, T, t, v0 = to_si(self.V0, "m^3"), to_si(self.T, "K"), to_si(t, "s"), to_si(v0, "m^3/s")
        C0, Cf = _conc(C0), _conc(C_feed)
        N0 = {s: c * V0 for s, c in C0.items()}
        check_refs(self.reactions, {s: N0.get(s, 0.0) + Cf.get(s, 0.0) for s in set(N0) | set(Cf)})
        species = species_list(self.reactions, N0 | {s: 0.0 for s in Cf})
        n0 = np.array([N0.get(s, 0.0) for s in species])
        fed = np.array([Cf.get(s, 0.0) * v0 for s in species])
        volume = lambda x: V0 + v0 * x

        def rate(x, N):
            return rates(self.reactions, species, dict(zip(species, N / volume(x))), T)

        sol = solve_ivp(lambda x, N: rate(x, N) * volume(x) + fed, (0, t), n0, method="LSODA",
                        t_eval=np.linspace(0, t, n_points), dense_output=True, rtol=1e-8,
                        atol=1e-12 * max(n0.sum(), fed.sum() * t))
        if not sol.success:
            raise RuntimeError(sol.message)
        N = sol.y.T
        return Result(species, sol.t, "t", N, np.array([rate(x, y) for x, y in zip(sol.t, N)]),
                      dict(zip(species, n0)), None, sol.sol, {"V": volume(sol.t)})


@dataclass
class Startup:
    t: float
    steady: Result
    result: Result


@dataclass
class CSTRStartup:
    reactions: ReactionSet
    V: Value

    def simulate(self, feed: Stream, t: Value, C_init: dict[str, Value] | None = None, key: str | None = None,
                 fraction: float = 0.99, n_points: int = 300) -> Startup:
        if feed.phase != "liquid":
            raise ValueError("CSTR startup is implemented for liquid-phase (constant v0) only")
        V, t = to_si(self.V, "m^3"), to_si(t, "s")
        steady = CSTR(self.reactions).simulate(feed, V)
        species = steady.species
        Ci = _conc(C_init)
        n0 = np.array([Ci.get(s, 0.0) * V for s in species])
        F0 = np.array([feed.F.get(s, 0.0) for s in species])
        v0 = feed.v0

        def rate(N):
            return rates(self.reactions, species, dict(zip(species, N / V)), feed.T)

        rhs = lambda _, N: F0 - v0 * N / V + rate(N) * V
        sol = solve_ivp(rhs, (0, t), n0, method="LSODA", t_eval=np.linspace(0, t, n_points), dense_output=True,
                        rtol=1e-9, atol=1e-13 * max(F0.sum() * V / v0, n0.sum(), 1e-30))
        if not sol.success:
            raise RuntimeError(sol.message)
        k = species.index(key_species(self.reactions, key))
        N_ss = steady.outlet.F[species[k]] * V / v0
        gap0 = abs(n0[k] - N_ss)
        reached = lambda x, N: abs(N[k] - N_ss) - (1 - fraction) * gap0
        reached.terminal = True
        ev = solve_ivp(rhs, (0, max(t, 1e3 * V / v0)), n0, method="LSODA", events=reached, rtol=1e-10,
                       atol=1e-13 * max(F0.sum() * V / v0, 1e-30))
        ts = float(ev.t_events[0][0]) if gap0 > 0 and ev.t_events[0].size else float("nan")
        N = sol.y.T
        res = Result(species, sol.t, "t", N, np.array([rate(y) for y in N]), dict(zip(species, n0)), None,
                     sol.sol)
        return Startup(ts, steady, res)
