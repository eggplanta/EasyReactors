from ..core import Arrhenius, Expression, MichaelisMenten, PowerLaw, Reaction, ReactionSet, Reversible
from ..core.reaction import parse_equation
from .inputs import ask, ask_quantity, confirm, mapping, select
from .output import info

INFER = "Limiting reactant (inferred from feed)"


def k_unit(order: float, basis: str = "C") -> str:
    drive = "mol/m^3" if basis == "C" else "Pa"
    return f"mol/m^3/s/({drive})**({order:g})"


def K_unit(orders_f: dict, orders_r: dict, basis: str = "C") -> str:
    drive = "mol/m^3" if basis == "C" else "Pa"
    return f"({drive})**({sum(orders_r.values()) - sum(orders_f.values()):g})"


def orders(text: str) -> dict[str, float]:
    return {s: float(v) for s, v in mapping(text).items()}


def _default_orders(nu: dict, sign: int) -> str:
    return ", ".join(f"{s}={abs(v):g}" for s, v in nu.items() if v * sign > 0)


def _param(name: str, si_unit: str):
    k = ask_quantity(f"{name} (e.g. 0.015 dm3/(mol*min))", si_unit)
    if not confirm(f"Temperature dependent {name} (Arrhenius / van 't Hoff)?", default=False):
        return k
    E = ask_quantity("Activation energy (or ΔH for K)", "J/mol", "J/mol")
    T_ref = ask_quantity(f"Temperature at which {name} was given", "K", "K")
    return Arrhenius.from_ref(k, E, T_ref)


def _rate(eq: str, nu: dict, reversible: bool, ref: str | None):
    models = ["Power law", "Custom expression"]
    if reversible:
        models = ["Reversible power law (K)", "Custom expression"]
    else:
        models.insert(1, "Michaelis-Menten")
    model = select(f"Rate law for {eq}", models)
    if model == "Custom expression":
        return _expression()
    if model == "Michaelis-Menten":
        substrate = ref or select("Substrate", [s for s, v in nu.items() if v < 0])
        vmax = ask_quantity("Vmax", "mol/m^3/s", "mol/L/min")
        Km = ask_quantity("Km", "mol/m^3", "mol/L")
        return MichaelisMenten(vmax, Km, substrate)
    basis = "C" if select("Rate in terms of", ["Concentrations (C)", "Partial pressures (p)"]).startswith("C") else "P"
    of = ask("Forward orders (e.g. A=1, B=2)", orders, _default_orders(nu, -1))
    if not reversible:
        return PowerLaw(_param("k", k_unit(sum(of.values()), basis)), of, basis)
    orr = ask("Reverse orders", orders, _default_orders(nu, 1))
    kf = _param("kf", k_unit(sum(of.values()), basis))
    K = _param("K (equilibrium)", K_unit(of, orr, basis))
    return Reversible(kf, of, orr, K, basis)


def _expression():
    info("Symbols: C_A (mol/m3), p_A (Pa), T (K); functions exp, log, sqrt. Parameters in SI.")
    expr = ask("-r = ", lambda s: Expression(s, **{n: 1.0 for n in _params(s)}))
    params = {n: ask(f"Value of {n} (SI)", float) for n in _params(expr)}
    return Expression(expr, **params)


def _params(expr: str) -> list[str]:
    import ast
    tree = ast.parse(expr.replace("^", "**"), mode="eval")
    names = {n.id for n in ast.walk(tree) if isinstance(n, ast.Name)}
    return sorted(n for n in names if not n.startswith(("C_", "p_")) and n != "T"
                  and n not in {"exp", "log", "sqrt", "Abs", "Max", "Min"})


def ask_reactions() -> ReactionSet:
    reactions = []
    while True:
        eq = ask("Reaction (e.g. 2A + B > C, reversible A <> B)", lambda s: parse_equation(s) and s.strip())
        nu, reversible = parse_equation(eq)
        reactants = [s for s, v in nu.items() if v < 0]
        ref = reactants[0] if len(reactants) == 1 else None
        if ref is None:
            choice = select("Rate law is written for -r of", reactants + [INFER])
            ref = None if choice == INFER else choice
        else:
            info(f"Rate law is -r{ref}")
        reactions.append(Reaction(eq, _rate(eq, nu, reversible, ref), ref=ref))
        if not confirm("Add another reaction?", default=False):
            return ReactionSet(reactions)


def confirm_refs(rs: ReactionSet, feed: dict[str, float]) -> None:
    inferred = [rx for rx in rs.reactions if rx.ref is None]
    ambiguous = rs.resolve_refs(feed)
    for rx in ambiguous:
        rx.ref = select(f"Cannot infer limiting reactant of {rx.equation}. Reference species",
                        rx.reactants)
    for rx in inferred:
        if rx in ambiguous:
            continue
        if rx.ref and not confirm(f"Reference (limiting) species for {rx.equation}: {rx.ref}. OK?"):
            rx.ref = select("Reference species", rx.reactants)
