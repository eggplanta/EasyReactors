from dataclasses import dataclass, field

import numpy as np

from ..core import (Arrhenius, Expression, MichaelisMenten, PowerLaw, Reaction, ReactionSet, Reversible,
                    parse_equation)
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
from .parsing import K_unit, amounts, k_unit, mapping, orders, quantity, series

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


def _elementary(nu: dict, sign: int) -> dict[str, float]:
    return {s: abs(v) for s, v in nu.items() if v * sign > 0}


def _required(text: str, name: str) -> str:
    if not text.strip():
        raise ValueError(f"{name} is required")
    return text


def _k(spec: ReactionSpec, si_unit: str):
    k = quantity(_required(spec.k, f"k of {spec.equation}"), si_unit)
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
            K = quantity(_required(spec.K, f"K of {eq}"), K_unit(of, orr, spec.basis))
            rate = Reversible(_k(spec, k_unit(sum(of.values()), spec.basis, spec.catalytic)), of, orr, K, spec.basis)
        else:
            rate = PowerLaw(_k(spec, k_unit(sum(of.values()), spec.basis, spec.catalytic)), of, spec.basis)
    return Reaction(eq, rate, ref=ref, catalytic=spec.catalytic)


def build_reactions(specs: list[ReactionSpec]) -> ReactionSet:
    if not specs:
        raise ValueError("Add at least one reaction")
    return ReactionSet([build_reaction(s) for s in specs])


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


def run_flow_reactor(kind: str, mode: str, rs: ReactionSet, feed: Stream, V: str = "", X: str = "",
                     key: str = "", species: str = "") -> Report:
    rep = Report(resolve(rs, feed.F))
    reactor = (CSTR if kind == "CSTR" else PFR)(rs)
    if mode == "Maximize":
        s = _required(species, "Species to maximize").strip()
        bounds = (feed.v * 1e-2, feed.v * 1e8)
        if kind == "CSTR":
            Vopt, _ = reactor.optimum(feed, s, bounds)
        else:
            V_max = quantity(V, "m^3", "L") if V.strip() else feed.v * 1e6
            Vopt, _ = reactor.simulate(feed, V_max, n_points=2000).optimum(s)
        res = reactor.simulate(feed, Vopt)
        rep.summary.append(f"Optimum for {s}: V = {fmt_volume(Vopt)[2:]}")
    elif mode == "Size":
        res = reactor.size(feed, _X(X), _key(rs, key))
    else:
        res = reactor.simulate(feed, quantity(_required(V, "Volume"), "m^3", "L"))
    rep.summary.insert(0, f"{kind}: V = {fmt_volume(res.end)[2:]}, τ = {res.end / feed.v / 60:.4g} min")
    rep.tables.append(("Outlet", STREAM_HEADERS, _stream_rows(feed, res.outlet, res.species)))
    if kind == "PFR":
        for s, Vm, Fm in _maxima(res):
            rep.summary.append(f"Maximum F_{s} = {Fm * 60:.4g} mol/min at V = {fmt_volume(Vm)[2:]}")
        C = np.array([[feed.conc(dict(zip(res.species, row)))[s] for s in res.species] for row in res.n])
        rep.plots.append(("Concentration profile", res.axis * 1000,
                          {s: C[:, i] / 1000 for i, s in enumerate(res.species)}, "V [L]", "C [mol/L]"))
    return rep


def run_batch(mode: str, rs: ReactionSet, V: str, T: str, C0: str, t: str = "", X: str = "",
              key: str = "") -> Report:
    V = quantity(_required(V, "Volume"), "m^3", "L")
    T = quantity(_required(T, "Temperature"), "K", "K")
    C0 = amounts(_required(C0, "Initial concentrations"), "mol/m^3", "mol/L")
    rep = Report(resolve(rs, C0))
    reactor = Batch(rs, V=V)
    if mode == "Size":
        res = reactor.time_to(_X(X), T, _key(rs, key), C0=C0)
    else:
        res = reactor.simulate(quantity(_required(t, "Time"), "s", "min"), T, C0=C0)
    rep.summary.append(f"Batch: t = {res.end / 60:.4g} min, V = {fmt_volume(V)[2:]}")
    rows = []
    for s in res.species:
        N0, N = res.n0[s], res.profile(s)[-1]
        rows.append([s, N0 / V / 1000, N / V / 1000, N0, N, (N0 - N) / N0 if N0 > 0 and N < N0 else None])
    rep.tables.append(("Final state", ["Species", "C0 [mol/L]", "C [mol/L]", "N0 [mol]", "N [mol]", "X"], rows))
    for s, tm, Nm in _maxima(res):
        rep.summary.append(f"Maximum C_{s} = {Nm / V / 1000:.4g} mol/L at t = {tm / 60:.4g} min")
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
