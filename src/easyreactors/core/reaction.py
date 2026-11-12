import re
from dataclasses import dataclass, field
from fractions import Fraction
from typing import Callable

_ARROWS = re.compile(r"\s*(<=>|<->|->|=)\s*")
_TERM = re.compile(r"^(\d+(?:[./]\d+)?)?\s*([A-Za-z(].*)$")


class AmbiguousReference(ValueError):
    pass


def parse_equation(eq: str) -> tuple[dict[str, float], bool]:
    parts = _ARROWS.split(eq.strip())
    if len(parts) != 3:
        raise ValueError(f"Invalid equation: {eq}")
    left, arrow, right = parts
    nu: dict[str, float] = {}
    for side, sign in ((left, -1), (right, 1)):
        for term in side.split("+"):
            m = _TERM.match(term.strip())
            if not m:
                raise ValueError(f"Invalid term '{term}' in: {eq}")
            coef = float(Fraction(m[1])) if m[1] else 1.0
            nu[m[2]] = nu.get(m[2], 0.0) + sign * coef
    return nu, arrow != "->"


@dataclass
class Reaction:
    equation: str
    rate: Callable
    ref: str | None = None
    catalytic: bool = False
    name: str | None = None
    nu: dict[str, float] = field(init=False)
    reversible: bool = field(init=False)

    def __post_init__(self):
        self.nu, self.reversible = parse_equation(self.equation)
        if self.ref is None and len(self.reactants) == 1:
            self.ref = self.reactants[0]
        if self.ref is not None and self.ref not in self.nu:
            raise ValueError(f"Reference species '{self.ref}' not in {self.equation}")

    @property
    def reactants(self) -> list[str]:
        return [s for s, v in self.nu.items() if v < 0]

    @property
    def delta(self) -> float:
        return sum(self.nu.values()) / abs(self.nu[self.ref])

    def resolve_ref(self, feed: dict[str, float]) -> str:
        if self.ref is None:
            if any(feed.get(s, 0.0) <= 0 for s in self.reactants):
                raise AmbiguousReference(f"Cannot infer reference species for: {self.equation}")
            self.ref = min(self.reactants, key=lambda s: feed[s] / abs(self.nu[s]))
        return self.ref

    def species_rates(self, C, T) -> dict[str, float]:
        if self.ref is None:
            raise AmbiguousReference(f"Reference species not set for: {self.equation}")
        r = self.rate(C, T) / abs(self.nu[self.ref])
        return {s: v * r for s, v in self.nu.items()}
