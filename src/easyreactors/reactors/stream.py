from dataclasses import dataclass, replace
from typing import Literal

from ..core.concentration import gas, gas_volumetric_flow, liquid
from ..units import to_si


@dataclass
class Stream:
    F: dict[str, float]
    T: float
    P: float = 101325.0
    phase: Literal["liquid", "gas"] = "liquid"
    v0: float | None = None

    @classmethod
    def liquid(cls, v0, T, F=None, C=None, P="1 atm") -> "Stream":
        v0 = to_si(v0, "m^3/s")
        if C is not None:
            F = {s: to_si(c, "mol/m^3") * v0 for s, c in C.items()}
        else:
            F = {s: to_si(f, "mol/s") for s, f in F.items()}
        return cls(F, to_si(T, "K"), to_si(P, "Pa"), "liquid", v0)

    @classmethod
    def gas(cls, F, T, P) -> "Stream":
        return cls({s: to_si(f, "mol/s") for s, f in F.items()},
                   to_si(T, "K"), to_si(P, "Pa"), "gas")

    @property
    def v(self) -> float:
        return self.v0 if self.phase == "liquid" else gas_volumetric_flow(self.F, self.T, self.P)

    def conc(self, F=None) -> dict[str, float]:
        F = self.F if F is None else F
        return liquid(F, self.v0) if self.phase == "liquid" else gas(F, self.T, self.P)

    def with_F(self, F) -> "Stream":
        return replace(self, F=dict(F))
