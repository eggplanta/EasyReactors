from dataclasses import dataclass


@dataclass(frozen=True)
class Species:
    name: str
    cas: str | None = None
    mw: float | None = None
