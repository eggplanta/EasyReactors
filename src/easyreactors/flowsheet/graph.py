from dataclasses import dataclass, replace

import numpy as np
from scipy.optimize import root

from ..reactors import Stream
from .ops import OPS, Feed, Heater, Op, Product, Thermo, total
from .spec import SpecError, load, obj, quantity, reaction, sub

SI_UNITS = {"F": "mol/s", "C": "mol/m^3", "T": "K", "P": "Pa", "v": "m^3/s", "V": "m^3", "W": "kg",
            "tau": "s", "Q": "W"}


@dataclass
class Link:
    name: str
    src: str
    src_port: str
    dst: str
    dst_port: str


@dataclass
class Context:
    reactions: dict
    thermo: Thermo

    def reaction(self, name):
        return reaction(name, self.reactions[name], sub("reactions", name))


class Flowsheet:
    def __init__(self, ops: dict[str, Op], links: dict[str, Link]):
        self.ops = ops
        self.links = links
        self._check()
        self.order, self.tears = self._sequence()
        self.species = self._species()

    @classmethod
    def from_spec(cls, spec) -> "Flowsheet":
        d = load(spec)
        obj(d, "", {"species", "thermo", "reactions", "units", "streams"}, ("units", "streams"))
        thermo = _thermo(d.get("thermo", {}))
        rx = obj(d.get("reactions", {}), "reactions")
        for name, r in rx.items():
            reaction(name, r, sub("reactions", name))
            if obj(r, sub("reactions", name)).get("dH") is not None:
                thermo.dH[name] = quantity(r["dH"], "J/mol", sub(sub("reactions", name), "dH"))
        ctx = Context(rx, thermo)
        ops = {}
        for name, u in obj(d["units"], "units").items():
            path = sub("units", name)
            kind = obj(u, path, required=("type",))["type"]
            if kind not in OPS:
                raise SpecError(sub(path, "type"), f"unknown unit type '{kind}'; use one of {sorted(OPS)}")
            ops[name] = OPS[kind].build(name, u, path, ctx)
        links = {}
        for name, s in obj(d["streams"], "streams").items():
            path = sub("streams", name)
            if isinstance(s, dict):
                obj(s, path, {"from", "to"}, ("from", "to"))
                s = [s["from"], s["to"]]
            if not (isinstance(s, list) and len(s) == 2):
                raise SpecError(path, "expected [\"unit.port\", \"unit.port\"] or {\"from\": ..., \"to\": ...}")
            (a, ap), (b, bp) = (_port(x, ops, path) for x in s)
            links[name] = Link(name, a, ap, b, bp)
        fs = cls(ops, links)
        if "species" in d:
            known = set(d["species"])
            extra = sorted(set(fs.species) - known)
            if extra:
                raise SpecError("species", f"species used but not listed: {extra}")
        return fs

    # Topology

    def _check(self):
        used = {}
        for l in self.links.values():
            for key in ((l.src, "out", l.src_port), (l.dst, "in", l.dst_port)):
                if key in used:
                    raise SpecError(sub("streams", l.name), f"port {key[0]}.{key[2]} already used by "
                                                            f"stream '{used[key]}'")
                used[key] = l.name
        for name, op in self.ops.items():
            ins = {l.dst_port for l in self.links.values() if l.dst == name}
            outs = {l.src_port for l in self.links.values() if l.src == name}
            bad = outs - set(op.outlets)
            if bad:
                raise SpecError(op.path, f"no outlet port(s) {sorted(bad)}")
            op.check_ports(ins, outs)
            bad = ins - set(op.inlets)
            if bad:
                raise SpecError(op.path, f"no inlet port(s) {sorted(bad)}")
        if not any(isinstance(op, Feed) for op in self.ops.values()):
            raise SpecError("units", "flowsheet has no feed")
        phases = {op.stream.phase for op in self.ops.values() if isinstance(op, Feed)}
        if len(phases) > 1:
            raise SpecError("units", "feeds of different phases are not supported")

    def _sequence(self):
        out = {n: [l for l in self.links.values() if l.src == n] for n in self.ops}
        state, tears = {}, []

        def dfs(n):
            state[n] = 1
            for l in out[n]:
                if state.get(l.dst) == 1:
                    tears.append(l.name)
                elif l.dst not in state:
                    dfs(l.dst)
            state[n] = 2

        for n, op in self.ops.items():
            if isinstance(op, Feed) and n not in state:
                dfs(n)
        lost = [n for n in self.ops if n not in state]
        if lost:
            raise SpecError(sub("units", lost[0]), "not reachable from any feed")
        indeg = {n: 0 for n in self.ops}
        for l in self.links.values():
            if l.name not in tears:
                indeg[l.dst] += 1
        ready = [n for n in self.ops if indeg[n] == 0]
        order = []
        while ready:
            n = ready.pop(0)
            order.append(n)
            for l in out[n]:
                if l.name not in tears:
                    indeg[l.dst] -= 1
                    if indeg[l.dst] == 0:
                        ready.append(l.dst)
        return order, tears

    def _species(self):
        sp = []
        for op in self.ops.values():
            if isinstance(op, Feed):
                sp += list(op.stream.F)
            for rx in getattr(getattr(op, "rs", None), "reactions", []):
                sp += list(rx.nu)
        return list(dict.fromkeys(sp))

    # Simulation

    def _pass(self, tears: dict[str, Stream], final: bool = False):
        values = dict(tears)
        reports = {}
        for n in self.order:
            op = self.ops[n]
            ins = {l.dst_port: values[l.name] for l in self.links.values() if l.dst == n}
            outs, rep = op.run(ins, final)
            reports[n] = rep
            for l in self.links.values():
                if l.src == n:
                    values[l.name] = outs[l.src_port]
        return values, reports

    def _thermal(self) -> bool:
        feeds = [op.stream.T for op in self.ops.values() if isinstance(op, Feed)]
        return (max(feeds) - min(feeds) > 1e-9
                or any(isinstance(op, Heater) or getattr(op, "heat", None) is not None for op in self.ops.values()))

    def _template(self) -> Stream:
        feeds = [op.stream for op in self.ops.values() if isinstance(op, Feed)]
        T = sum(s.T * total(s) for s in feeds) / max(sum(total(s) for s in feeds), 1e-300)
        return replace(feeds[0], F={s: 0.0 for s in self.species}, T=T,
                       v0=0.0 if feeds[0].phase == "liquid" else None)

    def simulate(self, tol: float = 1e-9) -> dict:
        info = {"converged": True, "iterations": 0, "tears": list(self.tears)}
        tears = {}
        if self.tears:
            tears, info = self._solve_tears(tol)
        values, reports = self._pass(tears, final=True)
        return self._report(values, reports, info)

    def _solve_tears(self, tol):
        tmpl = self._template()
        liquid = tmpl.phase == "liquid"
        ns = len(self.species)
        thermal = self._thermal()
        width = ns + thermal + liquid
        feeds = [op.stream for op in self.ops.values() if isinstance(op, Feed)]
        Fs = max(sum(total(s) for s in feeds), 1e-300)
        vs = sum(s.v0 for s in feeds) if liquid else 1.0
        scale = np.tile(np.concatenate([np.full(ns, Fs), [100.0] if thermal else [], [vs] if liquid else []]),
                        len(self.tears))
        last = {t: tmpl for t in self.tears}

        def unpack(x):
            out = {}
            for i, t in enumerate(self.tears):
                y = x[i * width:(i + 1) * width]
                F = dict(zip(self.species, np.maximum(y[:ns], 0.0)))
                T = float(y[ns]) if thermal else tmpl.T
                out[t] = replace(last[t], F=F, T=T, v0=float(max(y[-1], 0.0)) if liquid else None)
            return out

        def pack(streams):
            return np.concatenate([np.concatenate([[streams[t].F.get(s, 0.0) for s in self.species],
                                                   [streams[t].T] if thermal else [],
                                                   [streams[t].v0] if liquid else []]) for t in self.tears])

        count = [0]

        def G(x):
            count[0] += 1
            values, _ = self._pass(unpack(x))
            for t in self.tears:
                last[t] = values[t]
            return pack(values)

        x = pack(last)
        for _ in range(5):
            x = G(x)
        sol = root(lambda z: (z - G(z)) / scale, x, method="hybr", tol=1e-12)
        ok = np.all(np.isfinite(sol.fun)) and np.abs(sol.fun).max() < tol
        x = sol.x
        if not ok:
            x, ok = _wegstein(G, x, scale, tol)
        if not ok:
            raise SpecError("streams", f"recycle did not converge (tear streams {self.tears})")
        return unpack(x), {"converged": True, "iterations": count[0], "tears": list(self.tears)}

    def _report(self, values, reports, info) -> dict:
        streams = {}
        for name, l in self.links.items():
            s = values[name]
            C = s.conc() if total(s) > 0 and (s.phase == "gas" or s.v0 > 0) else {k: 0.0 for k in s.F}
            Ft = total(s)
            streams[name] = {"from": f"{l.src}.{l.src_port}", "to": f"{l.dst}.{l.dst_port}", "phase": s.phase,
                             "T": s.T, "P": s.P, "v": s.v if Ft > 0 else 0.0, "F": dict(s.F), "C": C,
                             "y": {k: v / Ft for k, v in s.F.items()} if Ft > 0 else {}}
        units = {n: {"type": _type(op), **reports[n]} for n, op in self.ops.items()}
        fed, out = {}, {}
        for op in self.ops.values():
            if isinstance(op, Feed):
                for k, v in op.stream.F.items():
                    fed[k] = fed.get(k, 0.0) + v
        for l in self.links.values():
            if isinstance(self.ops[l.dst], Product):
                for k, v in values[l.name].F.items():
                    out[k] = out.get(k, 0.0) + v
        overall = {"fed": fed, "products": out,
                   "X": {k: 1 - out.get(k, 0.0) / v for k, v in fed.items() if v > 0}}
        warn = [f"{n}: {w}" for n, r in reports.items() for w in r.get("warnings", [])]
        return _plain({**info, "si_units": SI_UNITS, "streams": streams, "units": units, "overall": overall,
                       "warnings": warn})


def _wegstein(G, x, scale, tol, n=500):
    gx = G(x)
    x_prev, g_prev = x, gx
    x = gx
    for _ in range(n):
        gx = G(x)
        if np.abs((x - gx) / scale).max() < tol:
            return gx, True
        dx = x - x_prev
        with np.errstate(divide="ignore", invalid="ignore"):
            s = np.where(np.abs(dx) > 1e-300, (gx - g_prev) / dx, 0.0)
        q = np.clip(s / (s - 1), -5.0, 0.0)
        q = np.nan_to_num(q)
        x_prev, g_prev = x, gx
        x = q * x + (1 - q) * gx
    return x, False


def _type(op: Op) -> str:
    return getattr(op, "kind", None) or type(op).__name__.lower()


def _port(text, ops, path):
    if not isinstance(text, str) or text.count(".") != 1:
        raise SpecError(path, f"expected 'unit.port', got {text!r}")
    unit, port = text.split(".")
    if unit not in ops:
        raise SpecError(path, f"unknown unit '{unit}'")
    return unit, port


def _thermo(d) -> Thermo:
    obj(d, "thermo", {"Cp", "T_ref"})
    Cp = {s: quantity(v, "J/mol/K", sub("thermo.Cp", s)) for s, v in obj(d.get("Cp", {}), "thermo.Cp").items()}
    return Thermo(Cp, quantity(d.get("T_ref", 298.15), "K", "thermo.T_ref"))


def _plain(x):
    if isinstance(x, dict):
        return {str(k): _plain(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [_plain(v) for v in x]
    if isinstance(x, (np.floating, np.integer)):
        return x.item()
    if isinstance(x, np.ndarray):
        return x.tolist()
    return x


def simulate(spec) -> dict:
    return Flowsheet.from_spec(spec).simulate()
