import math
from dataclasses import dataclass

from ..units import Value, to_si


@dataclass
class MassTransfer:
    Re: float
    Sc: float
    Sh: float
    kc: float


def _si(U, d, nu, D):
    return to_si(U, "m/s"), to_si(d, "m"), to_si(nu, "m^2/s"), to_si(D, "m^2/s")


def frossling(U: Value, dp: Value, nu: Value, DAB: Value) -> MassTransfer:
    U, dp, nu, D = _si(U, dp, nu, DAB)
    Re, Sc = U * dp / nu, nu / D
    Sh = 2 + 0.6 * Re ** 0.5 * Sc ** (1 / 3)
    return MassTransfer(Re, Sc, Sh, Sh * D / dp)


def thoenes_kramers(U: Value, dp: Value, nu: Value, DAB: Value, porosity: float,
                    gamma: float = 1.0) -> MassTransfer:
    U, dp, nu, D = _si(U, dp, nu, DAB)
    Re = U * dp / ((1 - porosity) * gamma * nu)
    Sc = nu / D
    Sh = Re ** 0.5 * Sc ** (1 / 3)
    return MassTransfer(Re, Sc, Sh, Sh * D * (1 - porosity) * gamma / (dp * porosity))


def colburn(U: Value, dp: Value, nu: Value, DAB: Value, porosity: float) -> MassTransfer:
    U, dp, nu, D = _si(U, dp, nu, DAB)
    Re, Sc = U * dp / nu, nu / D
    JD = (0.765 / Re ** 0.82 + 0.365 / Re ** 0.386) / porosity
    Sh = Sc ** (1 / 3) * Re * JD
    return MassTransfer(Re, Sc, Sh, Sh * D / dp)


def bed_area(dp: Value, porosity: float) -> float:
    return 6 * (1 - porosity) / to_si(dp, "m")


def cylinder_dp(D: Value, L: Value) -> tuple[float, float]:
    D, L = to_si(D, "m"), to_si(L, "m")
    dp = (1.5 * D ** 2 * L) ** (1 / 3)
    gamma = (math.pi * D * L + math.pi * D ** 2 / 2) / (math.pi * dp ** 2)
    return dp, gamma


def effective_diffusivity(DAB: Value, porosity: float, constriction: float, tortuosity: float) -> float:
    return to_si(DAB, "m^2/s") * porosity * constriction / tortuosity


def mt_limited_conversion(kc: Value, ac: Value, L: Value, U: Value) -> float:
    return 1 - math.exp(-to_si(kc, "m/s") * to_si(ac, "1/m") * to_si(L, "m") / to_si(U, "m/s"))
