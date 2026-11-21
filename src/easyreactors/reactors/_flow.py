from dataclasses import dataclass, replace

import numpy as np

from ..core import ReactionSet
from ..core.concentration import gas, liquid
from ..energy.heat import Heat
from ._common import check_refs, species_list
from .stream import Stream


@dataclass
class FlowModel:
    rxns: ReactionSet
    feed: Stream
    heat: Heat | None = None
    alpha: float = 0.0
    rho_b: float | None = None
    per_mass: bool = False

    def __post_init__(self):
        check_refs(self.rxns, self.feed.F)
        self.species = species_list(self.rxns, self.feed.F)
        self.n = len(self.species)
        self.F0 = np.array([self.feed.F.get(s, 0.0) for s in self.species])
        self.energy = self.heat is not None
        self.drop = self.alpha > 0
        if self.drop and self.feed.phase != "gas":
            raise ValueError("Pressure drop (Ergun) applies to gas-phase feeds")
        if self.energy:
            self.heat.bind(self.rxns, self.species)
        homogeneous = any(not rx.catalytic for rx in self.rxns.reactions)
        if self.per_mass and self.rho_b is None and (homogeneous or (self.energy and self.heat.exchange)):
            raise ValueError("Bulk density ρb is needed for homogeneous reactions or heat exchange in a PBR")
        self.rho = self.rho_b if self.rho_b is not None or not self.per_mass else 1.0

    @property
    def y0(self) -> np.ndarray:
        extra = ([self.feed.T] if self.energy else []) + ([1.0] if self.drop else [])
        return np.concatenate([self.F0, extra])

    def unpack(self, y):
        F = y[:self.n]
        T = y[self.n] if self.energy else self.feed.T
        p = y[-1] if self.drop else 1.0
        return F, T, p

    def conc(self, F, T, p) -> dict[str, float]:
        Fd = dict(zip(self.species, F))
        if self.feed.phase == "liquid":
            return liquid(Fd, self.feed.v0)
        return gas(Fd, T, self.feed.P * p)

    def rhs(self, _x, y) -> np.ndarray:
        F, T, p = self.unpack(y)
        C = self.conc(F, T, p)
        r = self.rxns.net_rates(C, T, self.rho)
        r = np.concatenate([r, np.zeros(self.n - len(r))])
        scale = self.rho if self.per_mass else 1.0
        out = [r / scale]
        if self.energy:
            q = self.heat.generation(self.rxns, C, T, self.rho) - self.heat.removal(T)
            out.append([q / scale / self.heat.capacity(self.species, F)])
        if self.drop:
            out.append([-self.alpha / (2 * p) * (T / self.feed.T) * (F.sum() / self.F0.sum())])
        return np.concatenate(out)

    def extras(self, Y: np.ndarray) -> dict[str, np.ndarray]:
        out = {}
        if self.energy:
            out["T"] = Y[:, self.n]
        if self.drop:
            out["p"] = Y[:, -1]
        return out

    def outlet(self, y) -> Stream:
        F, T, p = self.unpack(y)
        return replace(self.feed, F=dict(zip(self.species, F)), T=float(T), P=self.feed.P * float(p))

    def key_index(self, key: str) -> int:
        return self.species.index(key)
