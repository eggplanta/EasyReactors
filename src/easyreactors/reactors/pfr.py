from dataclasses import dataclass

import numpy as np
from scipy.integrate import solve_ivp

from ..core import ReactionSet
from ..energy.heat import Heat
from ..units import Value, to_si
from ._common import key_species
from ._flow import FlowModel
from .result import Result
from .stream import Stream


@dataclass
class PFR:
    reactions: ReactionSet
    V: Value | None = None
    rho_b: Value | None = None
    heat: Heat | None = None

    axis = "V"
    unit = "m^3"

    def _model(self, feed: Stream) -> FlowModel:
        rho = None if self.rho_b is None else to_si(self.rho_b, "kg/m^3")
        return FlowModel(self.reactions, feed, self.heat, rho_b=rho)

    def _size_attr(self):
        return self.V

    def _integrate(self, m: FlowModel, end: float, **kw):
        y0 = m.y0
        return solve_ivp(m.rhs, (0, end), y0, method="LSODA", rtol=1e-8,
                         atol=np.concatenate([np.full(m.n, 1e-12 * m.F0.sum()), np.full(len(y0) - m.n, 1e-9)]),
                         **kw)

    def simulate(self, feed: Stream, V: Value | None = None, n_points: int = 200) -> Result:
        end = to_si(self._size_attr() if V is None else V, self.unit)
        m = self._model(feed)
        sol = self._integrate(m, end, t_eval=np.linspace(0, end, n_points), dense_output=True)
        if not sol.success:
            raise RuntimeError(sol.message)
        Y = sol.y.T
        F = Y[:, :m.n]
        r = np.array([m.rhs(x, y)[:m.n] for x, y in zip(sol.t, Y)])
        interp = (lambda x: sol.sol(x)[:m.n]) if len(Y[0]) > m.n else sol.sol
        return Result(m.species, sol.t, self.axis, F, r, dict(zip(m.species, m.F0)), m.outlet(Y[-1]),
                      interp, m.extras(Y))

    def size(self, feed: Stream, X: float, key: str | None = None, V_max: Value = 1e6) -> Result:
        m = self._model(feed)
        i = m.key_index(key_species(self.reactions, key))
        hit = lambda x, y: y[i] - m.F0[i] * (1 - X)
        hit.terminal = True
        sol = self._integrate(m, to_si(V_max, self.unit), events=hit)
        if not sol.t_events[0].size:
            raise ValueError(f"X = {X} not reachable for {m.species[i]} (equilibrium, pressure or size limit)")
        return self.simulate(feed, sol.t_events[0][0])


@dataclass
class PBR(PFR):
    W: Value | None = None
    alpha: Value = 0.0

    axis = "W"
    unit = "kg"

    def _model(self, feed: Stream) -> FlowModel:
        rho = None if self.rho_b is None else to_si(self.rho_b, "kg/m^3")
        return FlowModel(self.reactions, feed, self.heat, to_si(self.alpha, "1/kg"), rho, per_mass=True)

    def _size_attr(self):
        return self.W
