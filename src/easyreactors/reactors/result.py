from dataclasses import dataclass, field
from typing import Callable

import numpy as np
from scipy.optimize import minimize_scalar

from .stream import Stream


@dataclass
class Result:
    species: list[str]
    axis: np.ndarray
    axis_name: str
    n: np.ndarray
    rates: np.ndarray
    n0: dict[str, float]
    outlet: Stream | None = None
    interp: Callable | None = None
    extras: dict[str, np.ndarray] = field(default_factory=dict)

    @property
    def end(self) -> float:
        return float(self.axis[-1])

    def profile(self, s: str) -> np.ndarray:
        return self.n[:, self.species.index(s)]

    def conversion(self, s: str) -> float:
        return (self.n0[s] - self.profile(s)[-1]) / self.n0[s]

    def selectivity(self, desired: str, undesired: str, instantaneous: bool = False):
        if instantaneous:
            i, j = self.species.index(desired), self.species.index(undesired)
            return self.rates[:, i] / self.rates[:, j]
        return self.profile(desired)[-1] / self.profile(undesired)[-1]

    def optimum(self, s: str) -> tuple[float, float]:
        j, k = self.species.index(s), int(np.argmax(self.profile(s)))
        if self.interp is None or k in (0, len(self.axis) - 1):
            return float(self.axis[k]), float(self.n[k, j])
        opt = minimize_scalar(lambda x: -self.interp(x)[j], method="bounded",
                              bounds=(self.axis[k - 1], self.axis[k + 1]))
        return float(opt.x), float(-opt.fun)
