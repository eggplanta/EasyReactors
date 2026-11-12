from dataclasses import dataclass

import numpy as np
from scipy.optimize import minimize_scalar

from ..core import ReactionSet
from ..units import Value, to_si
from ._common import flow_setup, key_species, solve
from .result import Result
from .stream import Stream


@dataclass
class CSTR:
    reactions: ReactionSet
    V: Value | None = None
    rho_b: Value | None = None

    def _result(self, feed, species, F0, F, V, rate) -> Result:
        return Result(species, np.array([V]), "V", F[None, :], rate(F)[None, :],
                      dict(zip(species, F0)), feed.with_F(dict(zip(species, F))))

    def simulate(self, feed: Stream, V: Value | None = None) -> Result:
        V = to_si(self.V if V is None else V, "m^3")
        species, F0, rate = flow_setup(self.reactions, feed, self.rho_b)
        scale = F0.sum()
        F = solve(lambda F: (F0 - F + rate(F) * V) / scale, F0, "CSTR")
        if (F < -1e-9 * scale).any():
            raise RuntimeError("CSTR converged to negative flows")
        return self._result(feed, species, F0, F, V, rate)

    def size(self, feed: Stream, X: float, key: str | None = None) -> Result:
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
