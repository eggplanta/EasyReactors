import numpy as np

from ..reactors import CSTR, PFR, Batch, Result, Stream
from .inputs import ask_float, ask_quantity, select
from .kinetics import ask_reactions
from .output import fmt, header, info, plot, table
from .streams import ask_concentrations, ask_feed, liters, minutes, mol_L, per_min

SIMULATE, SIZE, MAXIMIZE = "Simulate (given V)", "Size (V for a target conversion)", "Volume that maximizes a species"


def reactants(rs) -> list[str]:
    return list(dict.fromkeys(s for rx in rs.reactions for s in rx.reactants))


def ask_X(rs) -> tuple[float, str]:
    species = reactants(rs)
    key = select("Conversion of", species, rs.reactions[0].ref) if len(species) > 1 else species[0]
    return ask_float(f"Target conversion X_{key} (0-1)", lo=0, hi=0.999999), key


def stream_table(title: str, feed: Stream, outlet: Stream, species: list[str]) -> None:
    C = outlet.conc()
    rows = []
    for s in species:
        F0, F = feed.F.get(s, 0.0), outlet.F.get(s, 0.0)
        X = (F0 - F) / F0 if F0 > 0 and F < F0 else None
        rows.append([s, per_min(F0), per_min(F), mol_L(C.get(s, 0.0)), X])
    table(title, ["Species", "F0 [mol/min]", "F [mol/min]", "C [mol/L]", "X"], rows)


def interior_maxima(res: Result, n0: dict) -> list[tuple[str, float, float]]:
    out = []
    for s in res.species:
        y = res.profile(s)
        k = int(np.argmax(y))
        if 0 < k < len(y) - 1 and y[k] > n0.get(s, 0.0):
            x, ymax = res.optimum(s)
            out.append((s, x, ymax))
    return out


def _flow_reactor(kind: str) -> None:
    rs = ask_reactions()
    feed = ask_feed(rs)
    modes = [SIMULATE, SIZE] + ([MAXIMIZE] if kind == "CSTR" else [])
    mode = select("Problem", modes)
    reactor = (CSTR if kind == "CSTR" else PFR)(rs)
    if mode == MAXIMIZE:
        s = select("Species to maximize", [s for s in rs.species if s not in feed.F])
        bounds = (feed.v * 1e-2, feed.v * 1e8)
        V, best = reactor.optimum(feed, s, bounds)
        if V > 0.999 * bounds[1]:
            raise ValueError(f"No interior maximum of {s} for tau up to 1e8 s")
        res = reactor.simulate(feed, V)
        info(f"Optimum V = {fmt(liters(V))} L, F_{s} = {fmt(per_min(best))} mol/min")
    elif mode == SIZE:
        X, key = ask_X(rs)
        res = reactor.size(feed, X, key)
    else:
        res = reactor.simulate(feed, ask_quantity("Volume V", "m^3", "L"))
    V = res.end
    header(f"{kind}: V = {fmt(liters(V))} L ({fmt(V)} m3), tau = {fmt(minutes(V / feed.v))} min")
    stream_table("Outlet", feed, res.outlet, res.species)
    if kind == "PFR":
        _pfr_plots(res, feed)


def _pfr_plots(res: Result, feed: Stream) -> None:
    for s, V, Fmax in interior_maxima(res, res.n0):
        info(f"Maximum F_{s} = {fmt(per_min(Fmax))} mol/min at V = {fmt(liters(V))} L")
    C = np.array([[feed.conc(dict(zip(res.species, row)))[s] for s in res.species] for row in res.n])
    plot(liters(res.axis), {s: mol_L(C[:, i]) for i, s in enumerate(res.species)},
         "V [L]", "C [mol/L]", "Concentration profile")


def _batch() -> None:
    rs = ask_reactions()
    V = ask_quantity("Reactor volume V", "m^3", "L")
    T = ask_quantity("Temperature", "K", "K")
    C0 = ask_concentrations(rs, "Initial concentrations")
    reactor = Batch(rs, V=V)
    if select("Problem", ["Simulate for a time t", "Time to reach a conversion"]).startswith("Sim"):
        res = reactor.simulate(ask_quantity("Reaction time t", "s", "min"), T, C0=C0)
    else:
        X, key = ask_X(rs)
        res = reactor.time_to(X, T, key, C0=C0)
    header(f"Batch: t = {fmt(minutes(res.end))} min, V = {fmt(liters(V))} L")
    rows = []
    for s in res.species:
        N0, N = res.n0[s], res.profile(s)[-1]
        rows.append([s, mol_L(N0 / V), mol_L(N / V), N0, N, (N0 - N) / N0 if N0 > 0 and N < N0 else None])
    table("Final state", ["Species", "C0 [mol/L]", "C [mol/L]", "N0 [mol]", "N [mol]", "X"], rows)
    for s, t, Nmax in interior_maxima(res, res.n0):
        info(f"Maximum C_{s} = {fmt(mol_L(Nmax / V))} mol/L at t = {fmt(minutes(t))} min")
    plot(minutes(res.axis), {s: mol_L(res.profile(s) / V) for s in res.species},
         "t [min]", "C [mol/L]", "Concentration vs time")


def menu() -> None:
    kind = select("Reactor", ["Batch", "CSTR", "PFR"])
    _batch() if kind == "Batch" else _flow_reactor(kind)
