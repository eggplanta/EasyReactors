import numpy as np

from ..core import AmbiguousReference, ReactionSet


def check_refs(rxns: ReactionSet, amounts: dict[str, float]) -> None:
    amb = rxns.resolve_refs(amounts)
    if amb:
        raise AmbiguousReference("Set reference species for: " + ", ".join(r.equation for r in amb))


def species_list(rxns: ReactionSet, amounts: dict[str, float]) -> list[str]:
    return rxns.species + [s for s in amounts if s not in rxns.index]


def rates(rxns: ReactionSet, species: list[str], C: dict[str, float], T: float,
          rho_b: float | None = None) -> np.ndarray:
    r = rxns.net_rates(C, T, rho_b)
    return np.concatenate([r, np.zeros(len(species) - len(r))])


def key_species(rxns: ReactionSet, key: str | None) -> str:
    return key or rxns.reactions[0].ref


def flow_setup(rxns: ReactionSet, feed, rho_b=None):
    from ..units import to_si
    check_refs(rxns, feed.F)
    species = species_list(rxns, feed.F)
    F0 = np.array([feed.F.get(s, 0.0) for s in species])
    rho = None if rho_b is None else to_si(rho_b, "kg/m^3")

    def rate(F):
        return rates(rxns, species, feed.conc(dict(zip(species, F))), feed.T, rho)

    return species, F0, rate
