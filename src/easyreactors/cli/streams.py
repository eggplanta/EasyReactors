from ..core import ReactionSet
from ..reactors import Stream
from ..units import to_si
from .inputs import ask, ask_quantity, mapping, quantity, select
from .kinetics import confirm_refs


def amounts(text: str, si_unit: str, default_unit: str) -> dict[str, float]:
    return {s: quantity(v, si_unit, default_unit) for s, v in mapping(text).items()}


def _default(rs: ReactionSet) -> str:
    return ", ".join(f"{s}=" for s in dict.fromkeys(s for rx in rs.reactions for s in rx.reactants))


def ask_concentrations(rs: ReactionSet, label: str = "Feed concentrations") -> dict[str, float]:
    C = ask(f"{label} (e.g. A=2, B=0.5 mol/L) [mol/L]",
            lambda s: amounts(s, "mol/m^3", "mol/L"), _default(rs))
    confirm_refs(rs, C)
    return C


def ask_feed(rs: ReactionSet) -> Stream:
    phase = select("Feed phase", ["Liquid", "Gas"])
    T = ask_quantity("Temperature", "K", "K")
    if phase == "Liquid":
        v0 = ask_quantity("Volumetric flow v0", "m^3/s", "L/min")
        return Stream.liquid(v0=v0, T=T, C=ask_concentrations(rs))
    P = ask_quantity("Pressure", "Pa", "atm")
    F = ask("Molar flows (e.g. A=10, I=5 mol/min) [mol/min]",
            lambda s: amounts(s, "mol/s", "mol/min"), _default(rs))
    confirm_refs(rs, F)
    return Stream.gas(F=F, T=T, P=P)


def ask_side_feed(like: Stream) -> Stream:
    if like.phase == "liquid":
        v0 = ask_quantity("Side feed flow", "m^3/s", "L/min")
        C = ask("Side feed concentrations [mol/L]", lambda s: amounts(s, "mol/m^3", "mol/L"))
        return Stream.liquid(v0=v0, T=like.T, C=C, P=like.P)
    F = ask("Side feed molar flows [mol/min]", lambda s: amounts(s, "mol/s", "mol/min"))
    return Stream.gas(F=F, T=like.T, P=like.P)


# Display units

def per_min(x: float) -> float:
    return x * 60


def mol_L(x: float) -> float:
    return x / 1000


def liters(x: float) -> float:
    return x * 1000


def minutes(x: float) -> float:
    return x / 60


def to_unit(x: float, si_unit: str, unit: str) -> float:
    return x / to_si(f"1 {unit}", si_unit)
