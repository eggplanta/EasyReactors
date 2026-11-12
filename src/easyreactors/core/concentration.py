from .rates import R


def liquid(F: dict[str, float], v0: float) -> dict[str, float]:
    return {s: f / v0 for s, f in F.items()}


def gas(F: dict[str, float], T: float, P: float) -> dict[str, float]:
    FT = sum(F.values())
    return {s: f / FT * P / (R * T) for s, f in F.items()}


def gas_volumetric_flow(F: dict[str, float], T: float, P: float) -> float:
    return sum(F.values()) * R * T / P
