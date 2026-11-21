from dataclasses import dataclass, field, replace

import numpy as np

from ..core import (Arrhenius, Expression, MichaelisMenten, PowerLaw, Reaction, ReactionSet, Reversible,
                    VantHoff, parse_equation)
from ..core.rates import R as R_GAS
from ..energy.equilibrium import adiabatic_equilibrium
from ..energy.heat import Heat, dH_from_formation, lookup
from ..reactors.pfr import PBR
from ..reactors.pressure import ergun_alpha, ergun_beta0
from ..reactors.transient import CSTRStartup, Semibatch
from ..network import size_network, size_split
from ..network.ascii import fmt_volume
from ..nonideal import (RTD, bypass_dead, dispersion, fit_bypass_dead, fit_interchange, fit_pfr_bypass,
                        fit_pfr_cstr, interchange, max_mixedness, pfr_bypass, pfr_cstr, segregation,
                        tanks_in_series)
from ..reactors import CSTR, PFR, Batch, Result, Stream
from ..units import to_si
from .netexpr import FeedSpec, build, parse, to_text, walk
from ..diffusion.packed_bed import PackedBed
from ..diffusion.pellet import (effectiveness, eta_first_order, mears, radius_for_eta, two_pellets,
                                weisz_prater)
from ..diffusion.transport import (bed_area, colburn, cylinder_dp, effective_diffusivity, frossling,
                                   mt_limited_conversion, thoenes_kramers)
from .parsing import K_unit, amounts, human_K, human_k, k_unit, mapping, orders, quantity, series

MODELS = ["Power law", "Reversible", "Michaelis-Menten", "Expression"]
RTD_KINDS = ["Pulse C(t)", "Step C(t)", "E(t)"]
METHODS = ["trapezoid", "simpson", "pchip", "spline"]
TWO_PARAM = {
    "CSTR + bypass + dead volume": "Step tracer data. α = V_active/V, β = v_bypass/v0",
    "Two CSTRs with interchange": "Pulse tracer data (decay). α = V_agitated/V, β = v_exchange/v0",
    "PFR + CSTR in series": "Pulse tracer data. α = V_PFR/V, β = V_CSTR/V, dead = 1 − α − β",
    "PFR + bypass": "Pulse tracer data with two peaks. α = V_PFR/V, β = v_bypass/v0",
}


@dataclass
class Report:
    summary: list[str] = field(default_factory=list)
    tables: list[tuple[str, list[str], list[list]]] = field(default_factory=list)
    plots: list[tuple[str, np.ndarray, dict, str, str]] = field(default_factory=list)
    drawing: str | None = None


# Kinetics

@dataclass
class ReactionSpec:
    equation: str
    model: str = "Power law"
    ref: str = ""
    orders: str = ""
    reverse_orders: str = ""
    k: str = ""
    K: str = ""
    Ea: str = ""
    T_ref: str = ""
    basis: str = "C"
    expr: str = ""
    params: str = ""
    catalytic: bool = False
    dH: str = ""
    K_T: str = ""


def _elementary(nu: dict, sign: int) -> dict[str, float]:
    return {s: abs(v) for s, v in nu.items() if v * sign > 0}


def _required(text: str, name: str) -> str:
    if not text.strip():
        raise ValueError(f"{name} is required")
    return text


def _k(spec: ReactionSpec, si_unit: str, order: float = 1.0):
    try:
        k = quantity(_required(spec.k, f"k of {spec.equation}"), si_unit)
    except ValueError as e:
        if not str(e).startswith("Wrong units"):
            raise
        raise ValueError(f"k has the wrong units for total order {order:g}: use something like "
                         f"{human_k(order, spec.basis, spec.catalytic)} (got '{spec.k}')") from None
    if not spec.Ea.strip():
        return k
    return Arrhenius.from_ref(k, quantity(spec.Ea, "J/mol", "J/mol"),
                              quantity(_required(spec.T_ref, "T_ref"), "K", "K"))


def build_reaction(spec: ReactionSpec) -> Reaction:
    eq = _required(spec.equation, "Reaction equation").strip()
    nu, reversible = parse_equation(eq)
    reactants = [s for s, v in nu.items() if v < 0]
    ref = spec.ref.strip() or (reactants[0] if len(reactants) == 1 else None)
    if spec.model == "Expression":
        params = {n: float(v) for n, v in mapping(spec.params).items()} if spec.params.strip() else {}
        rate = Expression(_required(spec.expr, "Rate expression"), **params)
    elif spec.model == "Michaelis-Menten":
        substrate = ref or reactants[0]
        rate = MichaelisMenten(quantity(_required(spec.k, "Vmax"), "mol/kg/s" if spec.catalytic else "mol/m^3/s",
                                        None if spec.catalytic else "mol/L/min"),
                               quantity(_required(spec.K, "Km"), "mol/m^3", "mol/L"), substrate)
    else:
        of = orders(spec.orders) if spec.orders.strip() else _elementary(nu, -1)
        if spec.model == "Reversible" or reversible:
            orr = orders(spec.reverse_orders) if spec.reverse_orders.strip() else _elementary(nu, 1)
            try:
                K = quantity(_required(spec.K, f"K of {eq}"), K_unit(of, orr, spec.basis))
            except ValueError as e:
                if not str(e).startswith("Wrong units"):
                    raise
                raise ValueError(f"K has the wrong units: use {human_K(of, orr, spec.basis)} (got '{spec.K}')") from None
            if spec.dH.strip() and spec.K_T.strip():
                K = VantHoff(K, quantity(spec.dH, "J/mol", "J/mol"), quantity(spec.K_T, "K", "K"))
            rate = Reversible(_k(spec, k_unit(sum(of.values()), spec.basis, spec.catalytic), sum(of.values())), of, orr, K,
                              spec.basis)
        else:
            rate = PowerLaw(_k(spec, k_unit(sum(of.values()), spec.basis, spec.catalytic), sum(of.values())), of,
                            spec.basis)
    return Reaction(eq, rate, ref=ref, catalytic=spec.catalytic)


def build_reactions(specs: list[ReactionSpec]) -> ReactionSet:
    if not specs:
        raise ValueError("Add at least one reaction")
    built = []
    for i, s in enumerate(specs, 1):
        try:
            built.append(build_reaction(s))
        except ValueError as e:
            raise ValueError(f"Reaction {i} ({s.equation.strip()}): {e}") from e
    return ReactionSet(built)


def resolve(rs: ReactionSet, feed: dict[str, float]) -> list[str]:
    inferred = [rx for rx in rs.reactions if rx.ref is None]
    ambiguous = rs.resolve_refs(feed)
    if ambiguous:
        raise ValueError("Set '-r of' (reference species) for: " + ", ".join(r.equation for r in ambiguous))
    return [f"Reference of {rx.equation}: {rx.ref} (limiting reactant)" for rx in inferred]


# Feeds

def build_feed(phase: str, T: str, P: str, v0: str, composition: str) -> Stream:
    T = quantity(_required(T, "Temperature"), "K", "K")
    if phase == "Liquid":
        C = amounts(_required(composition, "Feed concentrations"), "mol/m^3", "mol/L")
        return Stream.liquid(v0=quantity(_required(v0, "v0"), "m^3/s", "L/min"), T=T, C=C,
                             P=quantity(P or "1", "Pa", "atm"))
    F = amounts(_required(composition, "Feed molar flows"), "mol/s", "mol/min")
    return Stream.gas(F=F, T=T, P=quantity(_required(P, "Pressure"), "Pa", "atm"))


def side_feed(text: str, like: Stream) -> Stream:
    m = mapping(text)
    if like.phase == "liquid":
        v0 = quantity(_required(m.pop("v0", ""), "Side feed v0"), "m^3/s", "L/min")
        return Stream.liquid(v0=v0, T=like.T, C={s: quantity(c, "mol/m^3", "mol/L") for s, c in m.items()},
                             P=like.P)
    return Stream.gas(F={s: quantity(f, "mol/s", "mol/min") for s, f in m.items()}, T=like.T, P=like.P)


# Single reactors

def _stream_rows(feed: Stream, outlet: Stream, species: list[str]) -> list[list]:
    C = outlet.conc()
    rows = []
    for s in species:
        F0, F = feed.F.get(s, 0.0), outlet.F.get(s, 0.0)
        rows.append([s, F0 * 60, F * 60, C.get(s, 0.0) / 1000, (F0 - F) / F0 if F0 > 0 and F < F0 else None])
    return rows


STREAM_HEADERS = ["Species", "F0 [mol/min]", "F [mol/min]", "C [mol/L]", "X"]


def _maxima(res: Result) -> list[tuple[str, float, float]]:
    out = []
    for s in res.species:
        y = res.profile(s)
        k = int(np.argmax(y))
        if 0 < k < len(y) - 1 and y[k] > res.n0.get(s, 0.0):
            out.append((s, *res.optimum(s)))
    return out


def _key(rs: ReactionSet, key: str) -> str:
    key = key.strip() or rs.reactions[0].ref
    if key not in rs.species:
        raise ValueError(f"Unknown species '{key}'")
    return key


def _X(text: str) -> float:
    X = float(_required(text, "Target conversion X"))
    if not 0 < X < 1:
        raise ValueError("X must be between 0 and 1")
    return X


# Energy and pressure drop

ENERGY_MODES = ["Isothermal", "Adiabatic", "Heat exchange (constant Ta)"]
DROP_MODES = ["None", "α given", "Ergun (bed data)"]


@dataclass
class EnergySpec:
    mode: str = "Isothermal"
    Cp: str = ""
    Ua: str = ""
    Ta: str = ""
    T_ref: str = ""


@dataclass
class DropSpec:
    mode: str = "None"
    alpha: str = ""
    rho_b: str = ""
    Ac: str = ""
    Dp: str = ""
    porosity: str = ""
    rho_c: str = ""
    mu: str = ""
    rho0: str = ""


def build_heat(e: EnergySpec | None, specs: list[ReactionSpec], total_ua: bool) -> Heat | None:
    if e is None or e.mode == "Isothermal":
        return None
    dH = [quantity(_required(s.dH, f"ΔHrx of {s.equation}"), "J/mol", "J/mol") for s in specs]
    Cp = amounts(_required(e.Cp, "Heat capacities Cp"), "J/mol/K", "J/mol/K")
    T_ref = quantity(e.T_ref, "K", "K") if e.T_ref.strip() else 298.15
    if not e.mode.startswith("Heat"):
        return Heat(dH, Cp, T_ref)
    Ua = quantity(_required(e.Ua, "UA" if total_ua else "Ua"), "W/K" if total_ua else "W/m^3/K",
                  "W/K" if total_ua else "W/m^3/K")
    return Heat(dH, Cp, T_ref, Ua, quantity(_required(e.Ta, "Coolant Ta"), "K", "K"))


def build_drop(d: DropSpec | None, feed: Stream) -> tuple[float, float | None, list[str]]:
    if d is None:
        return 0.0, None, []
    rho_b = _opt(d.rho_b, "kg/m^3", "kg/m^3")
    if d.mode == "α given":
        return _q(d.alpha, "α", "1/kg", "1/kg"), rho_b, []
    if d.mode.startswith("Ergun"):
        Ac, phi = _q(d.Ac, "Ac", "m^2", "m^2"), float(_required(d.porosity, "Porosity φ"))
        rho0, rho_c = _q(d.rho0, "Gas density ρ0", "kg/m^3", "kg/m^3"), _q(d.rho_c, "ρc", "kg/m^3", "kg/m^3")
        G = rho0 * feed.v / Ac
        beta0 = ergun_beta0(G, rho0, _q(d.mu, "Viscosity μ", "Pa*s", "Pa*s"), _q(d.Dp, "Dp", "m", "mm"), phi)
        alpha = ergun_alpha(beta0, Ac, rho_c, phi, feed.P)
        notes = [f"Ergun: G = {G:.4g} kg/(m2·s), β0 = {beta0 / 1000:.4g} kPa/m, α = {alpha:.4g} 1/kg"]
        return alpha, rho_b if rho_b is not None else rho_c * (1 - phi), notes
    return 0.0, rho_b, []


def lookup_properties(names: str, T: str, specs: list[ReactionSpec], phase: str = "Gas") -> tuple[str, list[str]]:
    m = mapping(_required(names, "Species names (A=n-butane, ...)"))
    T = quantity(T, "K", "K") if T.strip() else 298.15
    props = {s: lookup(n, T, phase.lower()) for s, n in m.items()}
    Cp = ", ".join(f"{s}={p.Cp:.4g}" for s, p in props.items())
    Hf = {s: p.Hf for s, p in props.items() if p.Hf is not None}
    dH = []
    for spec in specs:
        try:
            rx = build_reaction(replace(spec, model="Expression", expr="0", params=""))
            dH.append(f"{dH_from_formation(rx, Hf) / 1000:.6g} kJ/mol" if rx.ref else "")
        except ValueError:
            dH.append("")
    return Cp, dH


def _gas_conc(feed: Stream, res: Result) -> np.ndarray:
    T = res.extras.get("T", np.full(len(res.axis), feed.T))
    p = res.extras.get("p", np.ones(len(res.axis)))
    if feed.phase == "liquid":
        return res.n / feed.v0
    return res.n / res.n.sum(axis=1)[:, None] * (feed.P * p / (R_GAS * T))[:, None]


def _equilibrium_note(rs: ReactionSet, heat: Heat, feed: Stream, res: Result) -> list[str]:
    rx = rs.reactions[0]
    if len(rs.reactions) > 1 or not rx.reversible or heat.exchange:
        return []
    conc = (lambda F, T: feed.conc(F)) if feed.phase == "liquid" else (
        lambda F, T: {s: f / sum(F.values()) * feed.P / (R_GAS * T) for s, f in F.items()})
    try:
        Xe, Te = adiabatic_equilibrium(rx, heat, res.species, feed.F, conc, feed.T)
    except ValueError:
        return []
    return [f"Adiabatic equilibrium: X_{rx.ref} = {Xe:.4g} at T = {Te:.4g} K"]


def run_flow_reactor(kind: str, mode: str, rs: ReactionSet, feed: Stream, V: str = "", X: str = "",
                     key: str = "", species: str = "", heat: Heat | None = None,
                     drop: DropSpec | None = None) -> Report:
    rep = Report(resolve(rs, feed.F))
    pbr = kind == "PBR"
    size_unit, size_si, size_name = ("kg", "kg", "W") if pbr else ("L", "m^3", "V")
    if pbr:
        alpha, rho_b, notes = build_drop(drop, feed)
        rep.summary += notes
        reactor = PBR(rs, rho_b=rho_b, heat=heat, alpha=alpha)
    else:
        reactor = (CSTR if kind == "CSTR" else PFR)(rs, heat=heat)
    fmt_size = (lambda w: f"{w:.4g} kg") if pbr else (lambda v: fmt_volume(v)[2:])
    if kind == "CSTR" and heat is not None and mode == "Simulate":
        return _cstr_states(rep, reactor, feed, quantity(_required(V, "Volume"), "m^3", "L"), _key(rs, key))
    if mode == "Maximize":
        if heat is not None and kind == "CSTR":
            raise ValueError("Maximize with energy balance: use PFR, or Simulate the CSTR")
        s = _required(species, "Species to maximize").strip()
        if kind == "CSTR":
            Vopt, _ = reactor.optimum(feed, s, (feed.v * 1e-2, feed.v * 1e8))
        else:
            end = quantity(V, size_si, size_unit) if V.strip() else (feed.v * 1e6 if not pbr else 1e6)
            Vopt, _ = reactor.simulate(feed, end, n_points=2000).optimum(s)
        res = reactor.simulate(feed, Vopt)
        rep.summary.append(f"Optimum for {s}: {size_name} = {fmt_size(Vopt)}")
    elif mode == "Size":
        res = reactor.size(feed, _X(X), _key(rs, key))
    else:
        res = reactor.simulate(feed, quantity(_required(V, "Weight W" if pbr else "Volume"), size_si, size_unit))
    head = f"{kind}: {size_name} = {fmt_size(res.end)}"
    rep.summary.insert(0, head if pbr else head + f", τ = {res.end / feed.v / 60:.4g} min")
    if heat is not None:
        rep.summary.append(f"Outlet T = {res.outlet.T:.4g} K (inlet {feed.T:.4g} K)")
        rep.summary += _equilibrium_note(rs, heat, feed, res)
    if "p" in res.extras:
        rep.summary.append(f"Outlet P = {res.outlet.P / 101325:.4g} atm (P/P0 = {res.extras['p'][-1]:.4g})")
    rep.tables.append(("Outlet", STREAM_HEADERS, _stream_rows(feed, res.outlet, res.species)))
    if kind != "CSTR":
        for s, Vm, Fm in _maxima(res):
            rep.summary.append(f"Maximum F_{s} = {Fm * 60:.4g} mol/min at {size_name} = {fmt_size(Vm)}")
        x = res.axis if pbr else res.axis * 1000
        xl = "W [kg]" if pbr else "V [L]"
        C = _gas_conc(feed, res)
        rep.plots.append(("Concentration profile", x, {s: C[:, i] / 1000 for i, s in enumerate(res.species)},
                          xl, "C [mol/L]"))
        if "T" in res.extras:
            rep.plots.append(("Temperature profile", x, {"T": res.extras["T"]}, xl, "T [K]"))
        if "p" in res.extras:
            rep.plots.append(("Pressure ratio", x, {"P/P0": res.extras["p"]}, xl, "P/P0"))
    return rep


def _cstr_states(rep: Report, reactor: CSTR, feed: Stream, V: float, key: str) -> Report:
    ss = reactor.steady_states(feed, V)
    n = len(ss.states)
    rep.summary.insert(0, f"CSTR: V = {fmt_volume(V)[2:]}, τ = {V / feed.v / 60:.4g} min, "
                          f"{n} steady state{'s' if n > 1 else ''}")
    species = ss.states[0].species
    rows = []
    for i, (s, stable) in enumerate(zip(ss.states, ss.stable), 1):
        C = s.outlet.conc()
        rows.append([str(i), s.outlet.T, s.conversion(key), "stable" if stable else "unstable",
                     *[C.get(x, 0.0) / 1000 for x in species]])
    rep.tables.append(("Steady states", ["#", "T [K]", f"X_{key}", "", *[f"C_{x} [mol/L]" for x in species]],
                       rows))
    rep.tables.append((f"Outlet, steady state 1 (T = {ss.states[0].outlet.T:.4g} K)", STREAM_HEADERS,
                       _stream_rows(feed, ss.states[0].outlet, species)))
    F0 = feed.F[key]
    ok = np.isfinite(ss.G)
    rep.plots.append(("Heat generated G(T) and removed R(T)", ss.T[ok],
                      {"G(T)": ss.G[ok] / F0 / 1000, "R(T)": ss.R[ok] / F0 / 1000}, "T [K]",
                      f"kJ per mol {key} fed"))
    if n > 1:
        rep.summary.append("Multiple steady states: start-up conditions decide which one is reached")
    return rep


def run_batch(mode: str, rs: ReactionSet, V: str, T: str, C0: str, t: str = "", X: str = "",
              key: str = "", heat: Heat | None = None) -> Report:
    V = quantity(_required(V, "Volume"), "m^3", "L")
    T = quantity(_required(T, "Temperature"), "K", "K")
    C0 = amounts(_required(C0, "Initial concentrations"), "mol/m^3", "mol/L")
    rep = Report(resolve(rs, C0))
    reactor = Batch(rs, V=V, heat=heat)
    if mode == "Size":
        res = reactor.time_to(_X(X), T, _key(rs, key), C0=C0)
    else:
        res = reactor.simulate(quantity(_required(t, "Time"), "s", "min"), T, C0=C0)
    rep.summary.append(f"Batch: t = {res.end / 60:.4g} min, V = {fmt_volume(V)[2:]}")
    if heat is not None:
        Tt = res.extras["T"]
        rep.summary.append(f"Final T = {Tt[-1]:.4g} K (initial {T:.4g} K, max {Tt.max():.4g} K)")
    rows = []
    for s in res.species:
        N0, N = res.n0[s], res.profile(s)[-1]
        rows.append([s, N0 / V / 1000, N / V / 1000, N0, N, (N0 - N) / N0 if N0 > 0 and N < N0 else None])
    rep.tables.append(("Final state", ["Species", "C0 [mol/L]", "C [mol/L]", "N0 [mol]", "N [mol]", "X"], rows))
    for s, tm, Nm in _maxima(res):
        rep.summary.append(f"Maximum C_{s} = {Nm / V / 1000:.4g} mol/L at t = {tm / 60:.4g} min")
    rep.plots.append(("Concentration vs time", res.axis / 60,
                      {s: res.profile(s) / V / 1000 for s in res.species}, "t [min]", "C [mol/L]"))
    if heat is not None:
        rep.plots.append(("Temperature vs time", res.axis / 60, {"T": res.extras["T"]}, "t [min]", "T [K]"))
    return rep


# Transient reactors

def run_semibatch(rs: ReactionSet, V0: str, T: str, C0: str, v0: str, C_feed: str, t: str) -> Report:
    V0 = _q(V0, "Initial volume V0", "m^3", "L")
    C0 = amounts(C0, "mol/m^3", "mol/L") if C0.strip() else {}
    Cf = amounts(_required(C_feed, "Feed concentrations"), "mol/m^3", "mol/L")
    rep = Report(resolve(rs, {s: C0.get(s, 0.0) + Cf.get(s, 0.0) for s in set(C0) | set(Cf)}))
    res = Semibatch(rs, V0, _q(T, "Temperature", "K", "K")).simulate(
        _q(t, "Time", "s", "min"), C0, _q(v0, "Feed v0", "m^3/s", "L/min"), Cf)
    Vt = res.extras["V"]
    rep.summary.insert(0, f"Semibatch: t = {res.end / 60:.4g} min, V = {fmt_volume(V0)[2:]} → {fmt_volume(Vt[-1])[2:]}")
    rows = []
    for s in res.species:
        N0, N = res.n0[s], res.profile(s)[-1]
        X = (N0 - N) / N0 if N0 > 0 and N < N0 and s not in Cf else None
        rows.append([s, N0 / V0 / 1000, N / Vt[-1] / 1000, N0, N, X])
    rep.tables.append(("Final state", ["Species", "C0 [mol/L]", "C [mol/L]", "N0 [mol]", "N [mol]", "X"], rows))
    C = res.n / Vt[:, None] / 1000
    for i, s in enumerate(res.species):
        k = int(np.argmax(C[:, i]))
        if 0 < k < len(C) - 1:
            rep.summary.append(f"Maximum C_{s} = {C[k, i]:.4g} mol/L at t = {res.axis[k] / 60:.4g} min")
    rep.plots.append(("Concentration vs time", res.axis / 60, {s: C[:, i] for i, s in enumerate(res.species)},
                      "t [min]", "C [mol/L]"))
    return rep


def run_startup(rs: ReactionSet, V: str, feed: Stream, C_init: str, t: str, key: str = "",
                fraction: str = "") -> Report:
    rep = Report(resolve(rs, feed.F))
    Ci = amounts(C_init, "mol/m^3", "mol/L") if C_init.strip() else {}
    f = float(fraction) if fraction.strip() else 0.99
    if not 0 < f < 1:
        raise ValueError("Fraction must be between 0 and 1")
    k = _key(rs, key)
    V = _q(V, "Volume", "m^3", "L")
    st = CSTRStartup(rs, V).simulate(feed, _q(t, "Time", "s", "min"), Ci, k, f)
    rep.summary.insert(0, f"CSTR start-up: V = {fmt_volume(V)[2:]}, τ = {V / feed.v0 / 60:.4g} min")
    if np.isfinite(st.t):
        rep.summary.append(f"t_s = {st.t / 60:.4g} min for C_{k} to cover {f:.0%} of the way to steady state")
    res = st.result
    Css = st.steady.outlet.conc()
    rows = [[s, res.n0[s] / V / 1000, res.profile(s)[-1] / V / 1000, Css.get(s, 0.0) / 1000] for s in res.species]
    rep.tables.append((f"At t = {res.end / 60:.4g} min", ["Species", "C(0) [mol/L]", "C(t) [mol/L]",
                                                           "C steady [mol/L]"], rows))
    rep.plots.append(("Concentration vs time", res.axis / 60,
                      {s: res.profile(s) / V / 1000 for s in res.species}, "t [min]", "C [mol/L]"))
    return rep


# Networks

def run_network(expr: str, rs: ReactionSet, feed: Stream, side_feeds: list[str], X: str = "",
                key: str = "", split_mode: str = "Target X") -> Report:
    spec = parse(_required(expr, "Network expression"))
    rep = Report(resolve(rs, feed.F))
    feeds = [x for x in walk(spec) if isinstance(x, FeedSpec)]
    if len(side_feeds) < len(feeds):
        raise ValueError(f"Network has {len(feeds)} side feed(s): one line per side feed is needed")
    for x, text in zip(feeds, side_feeds):
        x.stream = side_feed(text, feed)
    built = build(spec, rs)
    key = _key(rs, key)
    if any(r.V is None for r in built.net.reactors()):
        res = size_network(built.net, feed, _X(X), key)
        rep.summary.append("Reactors with unknown V were sized with the same volume")
    elif built.sized_split is not None:
        if split_mode == "Equal conversion":
            res = size_split(built.net, built.sized_split, feed, key=key, equal_conversion=True)
        else:
            res = size_split(built.net, built.sized_split, feed, _X(X), key)
        split = res.ctx.nodes[id(built.sized_split)].split
        rep.summary.append("Sized split: " + ", ".join(f"{f:.4g}" for f in split))
    else:
        res = built.net.simulate(feed)
    rep.summary.insert(0, f"Overall X_{key} = {res.conversion(key):.4g} (based on all {key} fed)")
    rep.drawing = res.draw(key)
    rows = []
    for u in res.units:
        Fi, Fo = u.inlet.F.get(key, 0.0), u.outlet.F.get(key, 0.0)
        rows.append([u.name, fmt_volume(u.V)[2:], Fi * 60, Fo * 60, (Fi - Fo) / Fi if Fi else None])
    rep.tables.append((f"Units: {to_text(spec)}", ["Unit", "V", f"F{key} in [mol/min]",
                                                   f"F{key} out [mol/min]", "X unit"], rows))
    fed = Stream(res.fed, feed.T, feed.P, feed.phase, None)
    rep.tables.append(("Outlet", STREAM_HEADERS, _stream_rows(fed, res.outlet, list(res.outlet.F))))
    return rep


# RTD

def build_rtd(kind: str, data: str, unit: str, method: str, C0: str = "") -> RTD:
    t, y = series(_required(data, "Tracer data"))
    if kind == RTD_KINDS[1]:
        return RTD.from_step(t, y, float(_required(C0, "Step C0")), unit, method)
    if kind == RTD_KINDS[2]:
        return RTD.from_E(t, y, unit, method)
    return RTD.from_pulse(t, y, unit, method)


def rtd_report(rtd: RTD, unit: str) -> Report:
    u = to_si(f"1 {unit}", "s")
    rows = [["Mean residence time tm", rtd.tm / u, unit],
            ["Variance σ²", rtd.variance / u ** 2, f"{unit}²"],
            ["Std deviation σ", np.sqrt(rtd.variance) / u, unit],
            ["Skewness s³", rtd.skewness, "-"],
            ["σ²/tm²", rtd.variance / rtd.tm ** 2, "-"],
            ["Tanks in series n", rtd.n_tanks, "-"]]
    for bc in ("closed", "open"):
        try:
            rows.append([f"Peclet ({bc})", rtd.peclet(bc), "-"])
        except ValueError:
            rows.append([f"Peclet ({bc})", "out of range", "-"])
    t = rtd.t_data
    E, F, lam = (np.interp(t, rtd.t, y) for y in (rtd.E, rtd.F, rtd.intensity))
    ef = [[ti / u, Ei * u, Fi, ti / rtd.tm, Ei * rtd.tm, li * u if Fi < 1 - 1e-9 else "∞"]
          for ti, Ei, Fi, li in zip(t, E, F, lam)]
    return Report(
        tables=[("RTD moments", ["Quantity", "Value", "Unit"], rows),
                ("E(t), F(t), θ, E(θ), Λ(t)", [f"t [{unit}]", f"E [1/{unit}]", "F", "θ", "E(θ)", f"Λ [1/{unit}]"], ef)],
        plots=[("E(t)", rtd.t / u, {"E": rtd.E * u}, f"t [{unit}]", f"E [1/{unit}]"),
               ("F(t)", rtd.t / u, {"F": rtd.F}, f"t [{unit}]", "F")])


def model_conversions(rtd: RTD, rs: ReactionSet, C0: dict[str, float], T: float,
                      tau: float | None = None) -> list[list]:
    key = rs.reactions[0].ref
    tau = tau or rtd.tm
    feed = Stream.liquid(v0=1.0, T=T, C=C0)
    n_int = max(1, round(rtd.n_tanks))
    cases = [
        (f"Ideal PFR (τ = {tau / 60:.4g} min)", lambda: PFR(rs, V=tau).simulate(feed).conversion(key)),
        (f"Ideal CSTR (τ = {tau / 60:.4g} min)", lambda: CSTR(rs, V=tau).simulate(feed).conversion(key)),
        ("Segregation", lambda: segregation(rtd, rs, C0, T).conversion(key)),
        ("Maximum mixedness", lambda: max_mixedness(rtd, rs, C0, T).conversion(key)),
        (f"Tanks in series (n = {rtd.n_tanks:.3g} → {n_int})",
         lambda: tanks_in_series(rs, feed, rtd.tm, n_int).conversion(key)),
    ]
    try:
        Pe = rtd.peclet("closed")
        cases.append((f"Dispersion, closed (Pe = {Pe:.3g})", lambda: dispersion(rs, feed, rtd.tm, Pe).conversion(key)))
    except ValueError:
        pass
    rows = []
    for name, f in cases:
        try:
            rows.append([name, f()])
        except Exception as e:
            rows.append([name, f"failed: {e}"])
    return rows


def run_rtd_models(rtd: RTD, rs: ReactionSet, C0: str, T: str, tau: str = "") -> Report:
    C0 = amounts(_required(C0, "Feed concentrations"), "mol/m^3", "mol/L")
    rep = Report(resolve(rs, C0))
    tau = quantity(tau, "s", "min") if tau.strip() else None
    rows = model_conversions(rtd, rs, C0, quantity(_required(T, "Temperature"), "K", "K"), tau)
    rep.tables.append((f"Conversion of {rs.reactions[0].ref}", ["Model", "X"], rows))
    rep.summary.append("Order > 1: segregation > max mixedness; order < 1: the opposite; order 1: equal")
    return rep


# Two-parameter models

def fit_two_param(model: str, tau: str, data: str, unit: str, C0: str = "", t_split: str = ""):
    tau_s = quantity(_required(tau, "τ"), "s", "min")
    t, y = series(_required(data, "Tracer data"))
    tau_u = tau_s / to_si(f"1 {unit}", "s")
    if model.startswith("CSTR"):
        return fit_bypass_dead(t, y, float(_required(C0, "Step C0")), tau_u, unit)
    if model.startswith("Two"):
        return fit_interchange(t, y, tau_u, unit)
    rtd = RTD.from_pulse(t, y, unit)
    if model.startswith("PFR + CSTR"):
        return fit_pfr_cstr(rtd, tau_s)
    return fit_pfr_bypass(rtd, tau_s, quantity(_required(t_split, "t split"), "s", unit))


def two_param_conversion(model: str, rs: ReactionSet, C0: dict, T: float, tau: float,
                         alpha: float, beta: float) -> list[list]:
    key = rs.reactions[0].ref
    feed = Stream.liquid(v0=1.0, T=T, C=C0)
    if model.startswith("CSTR"):
        rows = [["Model", bypass_dead(rs, feed, tau, alpha, beta).conversion(key)]]
    elif model.startswith("Two"):
        rows = [["Model", interchange(rs, feed, tau, alpha, beta).conversion(key)]]
    elif model.startswith("PFR + CSTR"):
        rows = [["Model (PFR first)", pfr_cstr(rs, feed, tau, alpha, beta, "pfr_first").conversion(key)],
                ["Model (CSTR first)", pfr_cstr(rs, feed, tau, alpha, beta, "cstr_first").conversion(key)]]
    else:
        rows = [["Model", pfr_bypass(rs, feed, tau, alpha, beta).conversion(key)]]
    rows += [["Ideal PFR (τ)", PFR(rs, V=tau).simulate(feed).conversion(key)],
             ["Ideal CSTR (τ)", CSTR(rs, V=tau).simulate(feed).conversion(key)]]
    return rows


def run_two_param(model: str, tau: str, alpha: float, beta: float, rs: ReactionSet | None,
                  C0: str, T: str, fit=None) -> Report:
    rep = Report([TWO_PARAM[model]])
    rows = [["α", alpha], ["β", beta]]
    if fit is not None and fit.r2 is not None:
        rows.append(["R²", fit.r2])
    if model.startswith("CSTR"):
        rows.append(["Dead volume 1 − α", 1 - alpha])
    if model.startswith("PFR + CSTR"):
        rows.append(["Dead volume 1 − α − β", 1 - alpha - beta])
    rep.tables.append(("Parameters" + (" (fitted)" if fit else ""), ["Parameter", "Value"], rows))
    if rs is not None:
        C0 = amounts(_required(C0, "Feed concentrations"), "mol/m^3", "mol/L")
        rep.summary += resolve(rs, C0)
        tau_s = quantity(_required(tau, "τ"), "s", "min")
        rep.tables.append((f"Conversion of {rs.reactions[0].ref} (τ = {tau_s / 60:.4g} min)", ["Case", "X"],
                           two_param_conversion(model, rs, C0, quantity(_required(T, "Temperature"), "K", "K"),
                                                tau_s, alpha, beta)))
    return rep


# Diffusion

GEOMETRIES = ["sphere", "cylinder", "slab"]
CORRELATIONS = ["Thoenes-Kramers (packed bed)", "Frössling (single sphere)", "Colburn JD (packed bed)"]


def _q(text: str, name: str, si: str, default: str | None = None) -> float:
    return quantity(_required(text, name), si, default)


def _opt(text: str, si: str, default: str | None = None) -> float | None:
    return quantity(text, si, default) if text.strip() else None


def effective_De(De: str, DAB: str = "", porosity: str = "", constriction: str = "", tortuosity: str = "") -> float:
    if De.strip():
        return quantity(De, "m^2/s", "m^2/s")
    return effective_diffusivity(_q(DAB, "DAB (or De)", "m^2/s", "m^2/s"), float(_required(porosity, "Pellet porosity")),
                                 float(_required(constriction, "Constriction σc")),
                                 float(_required(tortuosity, "Tortuosity τ")))


def run_pellet(rs: ReactionSet, Cs: str, T: str, R: str, De: float, rho_c: str = "",
               geometry: str = "sphere") -> Report:
    rx = rs.reactions[0]
    if len(rs.reactions) > 1:
        raise ValueError("Pellet calculation takes a single reaction")
    rho = _opt(rho_c, "kg/m^3", "kg/m^3")
    if rx.catalytic and rho is None:
        raise ValueError("Rate per catalyst mass needs the pellet density ρc")
    Cs = amounts(_required(Cs, "Surface concentrations"), "mol/m^3", "mol/L")
    rep = Report(resolve(rs, Cs))
    e = effectiveness(rx, Cs, _q(T, "Temperature", "K", "K"), _q(R, "Pellet radius", "m", "m"), De, rho, geometry)
    rows = [["Thiele modulus φ (at Cs)", e.phi], ["η numerical (exact)", e.eta],
            [f"η first-order formula ({geometry})", eta_first_order(e.phi, geometry)],
            ["η large-φ, 1st order: 3/φ" if geometry == "sphere" else "", 3 / e.phi if geometry == "sphere" else None],
            ["η·φ² = Weisz-Prater C_WP", e.eta * e.phi ** 2]]
    rep.tables.append(("Internal diffusion", ["Quantity", "Value"], [r for r in rows if r[0]]))
    verdict = "no internal diffusion limitation" if e.eta * e.phi ** 2 < 1 else "internal diffusion limits the rate"
    rep.summary.append(f"C_WP = {e.eta * e.phi ** 2:.4g}: {verdict}")
    R_m = _q(R, "Pellet radius", "m", "m")
    rep.plots.append((f"{rx.ref} inside the pellet", e.xi * R_m * 1000, {rx.ref: e.C / 1000},
                      "r [mm]", "C [mol/L]"))
    return rep


def run_criteria(r_obs: str, rho_c: str, rho_b: str, R: str, De: float | None, Cs: str, Cb: str, n: str,
                 kc: str) -> Report:
    r = _q(r_obs, "Observed rate -r'A", "mol/kg/s", "mol/kg/s")
    R_m = _q(R, "Pellet radius", "m", "m")
    rows, rep = [], Report()
    if De is not None and rho_c.strip() and Cs.strip():
        cwp = weisz_prater(r, _q(rho_c, "ρc", "kg/m^3", "kg/m^3"), R_m, De, _q(Cs, "Cs", "mol/m^3", "mol/L"))
        rows.append(["Weisz-Prater C_WP", cwp, "≪ 1: no internal limitation; ≫ 1: limited"])
        rep.summary.append(f"C_WP = {cwp:.4g}: " + ("internal diffusion limits" if cwp > 1 else "no internal limitation"))
    if rho_b.strip() and kc.strip() and Cb.strip():
        mr = mears(r, _q(rho_b, "ρb", "kg/m^3", "kg/m^3"), R_m, float(n or 1), _q(kc, "kc", "m/s", "m/s"),
                   _q(Cb, "Cb", "mol/m^3", "mol/L"))
        rows.append(["Mears MR", mr, "< 0.15: external transfer negligible"])
        rep.summary.append(f"MR = {mr:.4g}: " + ("external mass transfer limits" if mr >= 0.15 else "external transfer negligible"))
    if not rows:
        raise ValueError("Fill De, ρc and Cs (Weisz-Prater) and/or ρb, kc and Cb (Mears)")
    rep.tables.append(("Criteria", ["Criterion", "Value", "Meaning"], rows))
    return rep


def run_mass_transfer(correlation: str, U: str, dp: str, nu: str, DAB: str, porosity: str = "",
                      gamma: str = "", cyl_D: str = "", cyl_L: str = "", L: str = "") -> Report:
    rep = Report()
    if cyl_D.strip() or cyl_L.strip():
        dp_m, g = cylinder_dp(_q(cyl_D, "Cylinder D", "m", "m"), _q(cyl_L, "Cylinder L", "m", "m"))
        rep.summary.append(f"Cylinder: equivalent dp = {dp_m:.4g} m, shape factor γ = {g:.4g}")
    else:
        dp_m, g = _q(dp, "Particle diameter dp", "m", "m"), None
    gamma = float(gamma) if gamma.strip() else (g or 1.0)
    args = (_q(U, "Velocity U", "m/s", "m/s"), dp_m, _q(nu, "Kinematic viscosity ν", "m^2/s", "m^2/s"),
            _q(DAB, "DAB", "m^2/s", "m^2/s"))
    if correlation.startswith("Frössling"):
        mt = frossling(*args)
    else:
        eps = float(_required(porosity, "Bed porosity"))
        mt = thoenes_kramers(*args, eps, gamma) if correlation.startswith("Thoenes") else colburn(*args, eps)
    rows = [["Re" + ("′" if correlation.startswith("Thoenes") else ""), mt.Re], ["Sc", mt.Sc],
            ["Sh" + ("′" if correlation.startswith("Thoenes") else ""), mt.Sh], ["kc [m/s]", mt.kc]]
    if porosity.strip():
        ac = bed_area(dp_m, float(porosity))
        rows.append(["ac = 6(1−φ)/dp [m²/m³]", ac])
        if L.strip():
            X = mt_limited_conversion(mt.kc, ac, _q(L, "Bed length", "m", "m"), args[0])
            rows.append(["X if mass-transfer limited", X])
    rep.tables.append((correlation, ["Quantity", "Value"], rows))
    return rep


def run_two_pellets(r1: str, R1: str, r2: str, R2: str, eta_target: str = "") -> Report:
    tp = two_pellets(float(_required(r1, "Rate 1")), _q(R1, "R1", "m", "m"), float(_required(r2, "Rate 2")),
                     _q(R2, "R2", "m", "m"))
    rows = [["Pellet 1", tp.phi1, tp.eta1, 3 / tp.phi1], ["Pellet 2", tp.phi2, tp.eta2, 3 / tp.phi2]]
    rep = Report(tables=[("Two pellet sizes (1st order, sphere)", ["", "φ", "η exact", "3/φ (large φ)"], rows)])
    if eta_target.strip():
        Rn = radius_for_eta(float(eta_target), tp.phi1, _q(R1, "R1", "m", "m"))
        rep.summary.append(f"Radius for η = {float(eta_target):.3g}: R = {Rn:.4g} m ({Rn * 1000:.4g} mm)")
    return rep


def run_packed_bed(rs: ReactionSet, feed: Stream, mode: str, target: str, R: str, De: float, rho_c: str,
                   rho_b: str, kc: str, ac: str, geometry: str = "sphere") -> Report:
    if len(rs.reactions) > 1:
        raise ValueError("Packed bed with diffusion takes a single reaction")
    rx = rs.reactions[0]
    rep = Report(resolve(rs, feed.F))
    bed = PackedBed(rx, _q(R, "Pellet radius", "m", "m"), De, _q(rho_c, "ρc", "kg/m^3", "kg/m^3"),
                    _q(rho_b, "ρb", "kg/m^3", "kg/m^3"), _q(kc, "kc", "m/s", "m/s"), _q(ac, "ac", "1/m", "1/m"),
                    geometry)
    res = bed.size(feed, _X(target)) if mode == "Size" else bed.simulate(feed, _q(target, "W", "kg", "kg"))
    rep.summary.insert(0, f"W = {res.end:.4g} kg ({res.end * 1000:.4g} g), X_{res.key} = {res.X[-1]:.4g}")
    rep.summary.append("Constant v0 (liquid or dilute gas); equal kc and De for all species")
    idx = np.unique(np.linspace(0, len(res.W) - 1, 8).astype(int))
    rows = [[res.W[i] * 1000, res.X[i], res.Cb[i] / 1000, res.Cs[i] / 1000, res.eta[i], res.Omega[i]] for i in idx]
    rep.tables.append(("Along the bed", ["W [g]", "X", "Cb [mol/L]", "Cs [mol/L]", "η", "Ω"], rows))
    rep.plots.append(("Conversion along the bed", res.W * 1000, {"X": res.X}, "W [g]", "X"))
    rep.plots.append(("Effectiveness along the bed", res.W * 1000, {"η": res.eta, "Ω": res.Omega}, "W [g]", "-"))
    return rep
