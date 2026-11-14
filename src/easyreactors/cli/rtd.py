import numpy as np

from ..core import ReactionSet
from ..nonideal import (RTD, bypass_dead, dispersion, fit_bypass_dead, fit_interchange, fit_pfr_bypass,
                        fit_pfr_cstr, interchange, max_mixedness, pfr_bypass, pfr_cstr, segregation,
                        tanks_in_series)
from ..reactors import CSTR, PFR, Stream
from ..units import to_si
from .inputs import Back, ask, ask_float, ask_quantity, numbers, quantity, select, table_file
from .kinetics import ask_reactions
from .output import error, fmt, header, info, plot, table
from .streams import ask_concentrations

METHODS = ["trapezoid", "simpson", "pchip", "spline"]


# Data input

def _time_unit(text: str) -> str:
    quantity(f"1 {text}", "s")
    return text


def ask_series(y_label: str = "C") -> tuple[np.ndarray, np.ndarray, str]:
    if select("Data input", ["Paste values", "File (CSV/TXT, columns or rows t and C)"]).startswith("Paste"):
        info("Values separated by spaces or commas (or ';' with decimal commas)")
        t = ask("t values", numbers)
        y = ask(f"{y_label} values", numbers)
    else:
        t, y = ask("File path", table_file)[:2]
    if len(t) != len(y):
        raise ValueError(f"t has {len(t)} values but {y_label} has {len(y)}")
    if np.any(np.diff(t) <= 0):
        raise ValueError("t must be strictly increasing")
    return t, y, ask("Time unit", _time_unit, "min")


def ask_rtd() -> tuple[RTD, str]:
    kind = select("Tracer data", ["Pulse: C(t)", "Step: C(t)", "E(t) directly"])
    t, y, unit = ask_series("E" if kind.startswith("E") else "C")
    method = select("Integration method", METHODS, "trapezoid")
    if kind.startswith("Step"):
        C0 = ask_float("Step concentration C0 (same units as C)", lo=0)
        return RTD.from_step(t, y, C0, unit, method), unit
    if kind.startswith("E"):
        return RTD.from_E(t, y, unit, method), unit
    return RTD.from_pulse(t, y, unit, method), unit


# Analysis

def moments_rows(rtd: RTD, unit: str) -> list[list]:
    u = to_si(f"1 {unit}", "s")
    rows = [["Mean residence time tm", rtd.tm / u, unit],
            ["Variance σ²", rtd.variance / u ** 2, f"{unit}²"],
            ["Std deviation σ", np.sqrt(rtd.variance) / u, unit],
            ["Skewness s³", rtd.skewness, "-"],
            ["σ²/tm²", rtd.variance / rtd.tm ** 2, "-"],
            ["Tanks in series n = tm²/σ²", rtd.n_tanks, "-"]]
    for bc in ("closed", "open"):
        try:
            rows.append([f"Peclet ({bc} vessel)", rtd.peclet(bc), "-"])
        except ValueError:
            rows.append([f"Peclet ({bc} vessel)", "out of range", "-"])
    return rows


def show_rtd(rtd: RTD, unit: str) -> None:
    u = to_si(f"1 {unit}", "s")
    header("RTD moments")
    table("", ["Quantity", "Value", "Unit"], moments_rows(rtd, unit))
    plot(rtd.t / u, {"E(t)": rtd.E * u}, f"t [{unit}]", f"E [1/{unit}]", "E(t)")
    plot(rtd.t / u, {"F(t)": rtd.F}, f"t [{unit}]", "F", "F(t)")


def ef_table(rtd: RTD, unit: str) -> None:
    u = to_si(f"1 {unit}", "s")
    t = rtd.t_data
    E, F = np.interp(t, rtd.t, rtd.E), np.interp(t, rtd.t, rtd.F)
    lam = np.interp(t, rtd.t, rtd.intensity)
    rows = [[ti / u, Ei * u, Fi, ti / rtd.tm, Ei * rtd.tm, li * u if Fi < 1 - 1e-9 else "∞"]
            for ti, Ei, Fi, li in zip(t, E, F, lam)]
    table("E(t), F(t), θ, E(θ), Λ(t)", [f"t [{unit}]", f"E [1/{unit}]", "F", "θ", "E(θ)", f"Λ [1/{unit}]"], rows)


def _key(rs: ReactionSet) -> str:
    return rs.reactions[0].ref


def model_conversions(rtd: RTD, rs: ReactionSet, C0: dict[str, float], T: float,
                      tau: float | None = None) -> list[list]:
    key = _key(rs)
    tau = tau or rtd.tm
    feed = Stream.liquid(v0=1.0, T=T, C=C0)
    n = rtd.n_tanks
    n_int = max(1, round(n))
    cases = [
        (f"Ideal PFR (τ = {fmt(tau / 60)} min)", lambda: PFR(rs, V=tau).simulate(feed).conversion(key)),
        (f"Ideal CSTR (τ = {fmt(tau / 60)} min)", lambda: CSTR(rs, V=tau).simulate(feed).conversion(key)),
        ("Segregation", lambda: segregation(rtd, rs, C0, T).conversion(key)),
        ("Maximum mixedness", lambda: max_mixedness(rtd, rs, C0, T).conversion(key)),
        (f"Tanks in series (n = {fmt(n, 3)} → {n_int})",
         lambda: tanks_in_series(rs, feed, rtd.tm, n_int).conversion(key)),
    ]
    try:
        Pe = rtd.peclet("closed")
        cases.append((f"Dispersion, closed (Pe = {fmt(Pe, 3)})",
                      lambda: dispersion(rs, feed, rtd.tm, Pe).conversion(key)))
    except ValueError:
        pass
    rows = []
    for name, f in cases:
        try:
            rows.append([name, f()])
        except Exception as e:
            rows.append([name, f"failed: {e}"])
    return rows


def _conditions(rs: ReactionSet) -> tuple[dict, float]:
    T = ask_quantity("Temperature", "K", "K", "300")
    return ask_concentrations(rs), T


def _models(rtd: RTD) -> None:
    rs = ask_reactions()
    C0, T = _conditions(rs)
    tau = ask("Space time τ = V/v0 for the ideal reactors (empty = tm) [min]",
              lambda s: None if not s.strip() else quantity(s, "s", "min"))
    rows = model_conversions(rtd, rs, C0, T, tau)
    table(f"Conversion of {_key(rs)}", ["Model", "X"], rows)
    info("Segregation and max mixedness bound X for a given RTD (order > 1: seg > mm; order < 1: mm > seg)")


def menu() -> None:
    rtd, unit = ask_rtd()
    show_rtd(rtd, unit)
    while True:
        c = select("RTD", ["E(t)/F(t) table", "Conversion with non-ideal models", "Plots", "Back"])
        if c == "Back":
            return
        try:
            {"E(t)/F(t) table": ef_table, "Plots": show_rtd}.get(c, lambda r, u: _models(r))(rtd, unit)
        except Back:
            continue
        except Exception as e:
            error(str(e))


# Two-parameter models

TWO_PARAM = {
    "CSTR with bypass and dead volume": "α = V_active/V, β = v_bypass/v0",
    "Two CSTRs with interchange": "α = V_highly-agitated/V, β = v_interchange/v0",
    "PFR + CSTR in series": "α = V_PFR/V, β = V_CSTR/V (dead = 1 - α - β)",
    "PFR with bypass": "α = V_PFR/V, β = v_bypass/v0",
}


def _fit(model: str, tau: float):
    if model.startswith("CSTR with"):
        info("Step tracer data at the outlet")
        t, C, unit = ask_series()
        C0 = ask_float("Step concentration C0", lo=0)
        return fit_bypass_dead(t, C, C0, tau / to_si(f"1 {unit}", "s"), unit)
    if model.startswith("Two"):
        info("Pulse tracer data at the outlet (decay curve)")
        t, C, unit = ask_series()
        return fit_interchange(t, C, tau / to_si(f"1 {unit}", "s"), unit)
    rtd, unit = ask_rtd()
    show_rtd(rtd, unit)
    if model.startswith("PFR +"):
        return fit_pfr_cstr(rtd, tau)
    t_split = ask_quantity("Time separating the bypass peak from the PFR peak", "s", unit)
    return fit_pfr_bypass(rtd, tau, t_split)


def two_param_conversion(model: str, rs: ReactionSet, C0: dict, T: float, tau: float,
                         alpha: float, beta: float) -> list[list]:
    key = _key(rs)
    feed = Stream.liquid(v0=1.0, T=T, C=C0)
    if model.startswith("CSTR with"):
        rows = [["Model", bypass_dead(rs, feed, tau, alpha, beta).conversion(key)]]
    elif model.startswith("Two"):
        rows = [["Model", interchange(rs, feed, tau, alpha, beta).conversion(key)]]
    elif model.startswith("PFR +"):
        rows = [["Model (PFR first)", pfr_cstr(rs, feed, tau, alpha, beta, "pfr_first").conversion(key)],
                ["Model (CSTR first)", pfr_cstr(rs, feed, tau, alpha, beta, "cstr_first").conversion(key)]]
    else:
        rows = [["Model", pfr_bypass(rs, feed, tau, alpha, beta).conversion(key)]]
    rows += [["Ideal PFR (τ)", PFR(rs, V=tau).simulate(feed).conversion(key)],
             ["Ideal CSTR (τ)", CSTR(rs, V=tau).simulate(feed).conversion(key)]]
    return rows


def two_param_menu() -> None:
    model = select("Model", list(TWO_PARAM))
    info(TWO_PARAM[model])
    tau = ask_quantity("Space time τ = V/v0", "s", "min")
    if select("Parameters", ["Fit from tracer data", "Enter α and β"]).startswith("Fit"):
        fit = _fit(model, tau)
        alpha, beta = fit.alpha, fit.beta
        rows = [["α", alpha], ["β", beta]] + ([["R²", fit.r2]] if fit.r2 is not None else [])
        if model.startswith("CSTR with"):
            rows.append(["Dead volume fraction 1 - α", 1 - alpha])
        if model.startswith("PFR +"):
            rows.append(["Dead volume fraction 1 - α - β", 1 - alpha - beta])
        table("Fitted parameters", ["Parameter", "Value"], rows)
    else:
        alpha, beta = ask_float("α", lo=0), ask_float("β", lo=0)
    rs = ask_reactions()
    C0, T = _conditions(rs)
    table(f"Conversion of {_key(rs)} (τ = {fmt(tau / 60)} min)", ["Case", "X"],
          two_param_conversion(model, rs, C0, T, tau, alpha, beta))
