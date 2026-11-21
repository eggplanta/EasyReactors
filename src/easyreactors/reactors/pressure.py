from ..units import Value, to_si


def ergun_beta0(G: Value, rho0: Value, mu: Value, Dp: Value, porosity: float) -> float:
    G, rho0, mu, Dp = to_si(G, "kg/m^2/s"), to_si(rho0, "kg/m^3"), to_si(mu, "Pa*s"), to_si(Dp, "m")
    phi = porosity
    return G * (1 - phi) / (rho0 * Dp * phi ** 3) * (150 * (1 - phi) * mu / Dp + 1.75 * G)


def ergun_alpha(beta0: float, Ac: Value, rho_c: Value, porosity: float, P0: Value) -> float:
    return 2 * beta0 / (to_si(Ac, "m^2") * to_si(rho_c, "kg/m^3") * (1 - porosity) * to_si(P0, "Pa"))
