import numpy as np

from .reaction import AmbiguousReference, Reaction


class ReactionSet:
    def __init__(self, reactions: list[Reaction], inerts: list[str] = ()):
        self.reactions = reactions
        self.species = list(dict.fromkeys([s for r in reactions for s in r.nu] + list(inerts)))
        self.index = {s: i for i, s in enumerate(self.species)}
        self.nu = np.array([[r.nu.get(s, 0.0) for s in self.species] for r in reactions])

    def resolve_refs(self, feed: dict[str, float]) -> list[Reaction]:
        ambiguous = []
        for rx in self.reactions:
            try:
                rx.resolve_ref(feed)
            except AmbiguousReference:
                ambiguous.append(rx)
        return ambiguous

    def net_rates(self, C: dict[str, float], T: float, rho_b: float | None = None) -> np.ndarray:
        out = np.zeros(len(self.species))
        for rx in self.reactions:
            if rx.catalytic and rho_b is None:
                raise ValueError(f"Catalytic reaction needs bulk density rho_b: {rx.equation}")
            factor = rho_b if rx.catalytic else 1.0
            for s, v in rx.species_rates(C, T).items():
                out[self.index[s]] += v * factor
        return out
