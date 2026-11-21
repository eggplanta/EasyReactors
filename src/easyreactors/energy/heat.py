from dataclasses import dataclass, field

import numpy as np

from ..core import Reaction, ReactionSet
from ..units import Value, to_si


@dataclass
class Heat:
    dH: list[Value]
    Cp: dict[str, Value]
    T_ref: Value = 298.15
    Ua: Value = 0.0
    Ta: Value | None = None
    dCp: list[float] = field(init=False, default_factory=list)

    def __post_init__(self):
        self.dH = [to_si(h, "J/mol") for h in self.dH]
        self.Cp = {s: to_si(c, "J/mol/K") for s, c in self.Cp.items()}
        self.T_ref = to_si(self.T_ref, "K")
        self.Ua = _ua(self.Ua)
        self.Ta = None if self.Ta is None else to_si(self.Ta, "K")

    @property
    def exchange(self) -> bool:
        return self.Ua > 0

    def bind(self, rxns: ReactionSet, species: list[str]) -> None:
        if len(self.dH) != len(rxns.reactions):
            raise ValueError(f"Give ΔHrx for each of the {len(rxns.reactions)} reactions")
        missing = [s for s in species if s not in self.Cp]
        if missing:
            raise ValueError("Cp missing for: " + ", ".join(missing))
        if self.exchange and self.Ta is None:
            raise ValueError("Heat exchange needs the coolant temperature Ta")
        self.dCp = [delta_cp(rx, self.Cp) for rx in rxns.reactions]

    def dH_at(self, i: int, T: float) -> float:
        return self.dH[i] + self.dCp[i] * (T - self.T_ref)

    def generation(self, rxns: ReactionSet, C: dict[str, float], T: float, rho_b: float | None = None) -> float:
        q = 0.0
        for i, rx in enumerate(rxns.reactions):
            factor = (rho_b or 1.0) if rx.catalytic else 1.0
            q += rx.species_rates(C, T)[rx.ref] * factor * self.dH_at(i, T)
        return q

    def capacity(self, species: list[str], n: np.ndarray) -> float:
        return float(sum(self.Cp[s] * x for s, x in zip(species, n)))

    def removal(self, T: float) -> float:
        return self.Ua * (T - self.Ta) if self.exchange else 0.0


def _ua(Ua: Value) -> float:
    if isinstance(Ua, str):
        from ..units import Q_
        q = Q_(Ua)
        for unit in ("W/K", "W/m^3/K"):
            if q.check(unit):
                return float(q.to(unit).magnitude)
        raise ValueError(f"Wrong units for Ua: '{Ua}'")
    return float(Ua)


def delta_cp(rx: Reaction, Cp: dict[str, float]) -> float:
    return sum(v * Cp[s] for s, v in rx.nu.items()) / abs(rx.nu[rx.ref])


@dataclass
class Properties:
    name: str
    Cp: float
    Hf: float | None
    phase: str


def lookup(name: str, T: Value = 298.15, phase: str = "gas") -> Properties:
    from thermo import Chemical
    try:
        c = Chemical(name.strip(), T=to_si(T, "K"))
        ref = Chemical(name.strip(), T=298.15)
    except Exception as e:
        raise ValueError(f"Species not found in the database: '{name}'") from e
    if phase == "gas":
        Cp, Hf = c.Cpgm, ref.Hfgm
    else:
        Cp = c.Cplm
        Hf = None if ref.Hfgm is None or ref.Hvapm is None else ref.Hfgm - ref.Hvapm
    if Cp is None:
        raise ValueError(f"No {phase} heat capacity data for '{name}'")
    return Properties(c.name, float(Cp), None if Hf is None else float(Hf), phase)


def dH_from_formation(rx: Reaction, Hf: dict[str, float]) -> float:
    missing = [s for s in rx.nu if s not in Hf]
    if missing:
        raise ValueError("Heat of formation missing for: " + ", ".join(missing))
    return sum(v * Hf[s] for s, v in rx.nu.items()) / abs(rx.nu[rx.ref])
