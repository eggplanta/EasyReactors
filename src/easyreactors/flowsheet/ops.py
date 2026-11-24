from dataclasses import dataclass, field, replace

import numpy as np

from ..core import ReactionSet
from ..energy.heat import Heat
from ..reactors import CSTR, PBR, PFR, Stream
from .spec import SpecError, amounts, number, obj, quantity, sub


@dataclass
class Thermo:
    Cp: dict[str, float] = field(default_factory=dict)
    T_ref: float = 298.15
    dH: dict[str, float] = field(default_factory=dict)


@dataclass
class Op:
    name: str
    path: str

    inlets: tuple = ("in",)
    outlets: tuple = ("out",)

    def check_ports(self, ins: set, outs: set) -> None:
        for port in self.inlets:
            if port not in ins:
                raise SpecError(self.path, f"inlet port '{port}' is not connected")
        for port in self.outlets:
            if port not in outs:
                raise SpecError(self.path, f"outlet port '{port}' is not connected")

    def run(self, ins: dict[str, Stream], final: bool) -> tuple[dict[str, Stream], dict]:
        raise NotImplementedError


# Feed and product

@dataclass
class Feed(Op):
    stream: Stream = None
    inlets: tuple = ()

    @classmethod
    def build(cls, name, d, path, ctx):
        obj(d, path, {"type", "phase", "T", "P", "v0", "C", "F"}, ("T",))
        phase = d.get("phase", "liquid")
        T = quantity(d["T"], "K", sub(path, "T"))
        P = quantity(d.get("P", "1 atm"), "Pa", sub(path, "P"))
        if phase == "gas":
            obj(d, path, required=("F",))
            if "C" in d or "v0" in d:
                raise SpecError(path, "gas feeds take molar flows F (and T, P), not C or v0")
            s = Stream(amounts(d["F"], "mol/s", sub(path, "F")), T, P, "gas")
        elif phase == "liquid":
            obj(d, path, required=("v0",))
            v0 = quantity(d["v0"], "m^3/s", sub(path, "v0"))
            if ("C" in d) == ("F" in d):
                raise SpecError(path, "liquid feeds take either concentrations C or molar flows F")
            F = ({s: c * v0 for s, c in amounts(d["C"], "mol/m^3", sub(path, "C")).items()} if "C" in d
                 else amounts(d["F"], "mol/s", sub(path, "F")))
            s = Stream(F, T, P, "liquid", v0)
        else:
            raise SpecError(sub(path, "phase"), "phase must be 'liquid' or 'gas'")
        return cls(name, path, stream=s)

    def run(self, ins, final):
        return {"out": self.stream}, {}


@dataclass
class Product(Op):
    outlets: tuple = ()

    @classmethod
    def build(cls, name, d, path, ctx):
        obj(d, path, {"type"})
        return cls(name, path)

    def run(self, ins, final):
        return {}, {}


# Mixer, splitter, heater

def total(s: Stream) -> float:
    return sum(s.F.values())


def mix(streams: list[Stream], thermo: Thermo, path: str) -> Stream:
    live = [s for s in streams if total(s) > 0] or streams[:1]
    first = live[0]
    if any(s.phase != first.phase for s in live):
        raise SpecError(path, "cannot mix liquid and gas streams")
    F: dict[str, float] = {}
    for s in live:
        for k, v in s.F.items():
            F[k] = F.get(k, 0.0) + v
    T = first.T
    if any(abs(s.T - first.T) > 1e-9 for s in live):
        missing = sorted({k for s in live for k, v in s.F.items() if v > 0 and k not in thermo.Cp})
        if missing:
            raise SpecError(path, f"mixing streams at different T needs thermo.Cp for {missing}")
        cap = [sum(thermo.Cp[k] * v for k, v in s.F.items() if v > 0) for s in live]
        T = sum(c * s.T for c, s in zip(cap, live)) / sum(cap)
    v0 = sum(s.v0 for s in streams) if first.phase == "liquid" else None
    return replace(first, F=F, T=T, P=min(s.P for s in live), v0=v0)


def scale(s: Stream, f: float) -> Stream:
    return replace(s, F={k: v * f for k, v in s.F.items()}, v0=None if s.v0 is None else s.v0 * f)


@dataclass
class Mixer(Op):
    thermo: Thermo = None

    def check_ports(self, ins, outs):
        if not ins:
            raise SpecError(self.path, "mixer has no inlet connected")
        self.inlets = tuple(sorted(ins))
        super().check_ports(ins, outs)

    @classmethod
    def build(cls, name, d, path, ctx):
        obj(d, path, {"type"})
        return cls(name, path, thermo=ctx.thermo)

    def run(self, ins, final):
        return {"out": mix([ins[p] for p in self.inlets], self.thermo, self.path)}, {}


@dataclass
class Splitter(Op):
    fractions: dict[str, float] = None

    @classmethod
    def build(cls, name, d, path, ctx):
        obj(d, path, {"type", "fractions"}, ("fractions",))
        fr = obj(d["fractions"], sub(path, "fractions"))
        fp = sub(path, "fractions")
        if len(fr) < 2:
            raise SpecError(fp, "give at least two outlet ports")
        open_ = [p for p, v in fr.items() if v is None]
        if len(open_) > 1:
            raise SpecError(fp, "only one fraction may be left empty (it takes the remainder)")
        out = {p: number(v, sub(fp, p), 0, 1) for p, v in fr.items() if v is not None}
        rest = 1 - sum(out.values())
        if open_:
            if rest < -1e-12:
                raise SpecError(fp, "fractions add up to more than 1")
            out[open_[0]] = max(rest, 0.0)
        elif abs(rest) > 1e-9:
            raise SpecError(fp, f"fractions must add up to 1 (got {1 - rest:.6g})")
        return cls(name, path, outlets=tuple(fr), fractions=out)

    def run(self, ins, final):
        return {p: scale(ins["in"], f) for p, f in self.fractions.items()}, {"fractions": dict(self.fractions)}


@dataclass
class Heater(Op):
    T: float = None
    thermo: Thermo = None

    @classmethod
    def build(cls, name, d, path, ctx):
        obj(d, path, {"type", "T"}, ("T",))
        return cls(name, path, T=quantity(d["T"], "K", sub(path, "T")), thermo=ctx.thermo)

    def run(self, ins, final):
        s = ins["in"]
        rep = {"T_in": s.T, "T_out": self.T}
        if all(k in self.thermo.Cp for k, v in s.F.items() if v > 0):
            rep["Q"] = sum(self.thermo.Cp[k] * v for k, v in s.F.items()) * (self.T - s.T)
        return {"out": replace(s, T=self.T)}, rep


# Reactors

@dataclass
class Reactor(Op):
    kind: str = "cstr"
    rs: ReactionSet = None
    size: float = None
    rho_b: float | None = None
    alpha: float = 0.0
    heat: Heat | None = None

    @classmethod
    def build(cls, name, d, path, ctx):
        kind = d["type"]
        size_key = "W" if kind == "pbr" else "V"
        allowed = {"type", size_key, "reactions", "energy", "rho_b"} | ({"alpha"} if kind == "pbr" else set())
        obj(d, path, allowed, (size_key,))
        names = d.get("reactions", list(ctx.reactions))
        if not isinstance(names, list) or not names:
            raise SpecError(sub(path, "reactions"), "expected a non-empty list of reaction names")
        for n in names:
            if n not in ctx.reactions:
                raise SpecError(sub(path, "reactions"), f"unknown reaction '{n}'")
        rs = ReactionSet([ctx.reaction(n) for n in names])
        size = quantity(d[size_key], "kg" if kind == "pbr" else "m^3", sub(path, size_key))
        rho_b = None if d.get("rho_b") is None else quantity(d["rho_b"], "kg/m^3", sub(path, "rho_b"))
        alpha = quantity(d.get("alpha", 0.0), "1/kg", sub(path, "alpha"))
        heat = energy(d.get("energy", "isothermal"), names, ctx, sub(path, "energy"))
        return cls(name, path, kind=kind, rs=rs, size=size, rho_b=rho_b, alpha=alpha, heat=heat)

    def model(self):
        if self.kind == "cstr":
            return CSTR(self.rs, rho_b=self.rho_b, heat=self.heat)
        if self.kind == "pfr":
            return PFR(self.rs, rho_b=self.rho_b, heat=self.heat)
        return PBR(self.rs, rho_b=self.rho_b, heat=self.heat, alpha=self.alpha)

    def run(self, ins, final):
        feed = ins["in"]
        if total(feed) <= 0:
            return {"out": feed}, {"empty": True}
        m = self.model()
        try:
            if self.kind == "cstr":
                ss = m.steady_states(feed, self.size)
                i = m.pick(ss)
                res = ss.states[i]
            else:
                ss, i, res = None, None, m.simulate(feed, self.size)
        except (ValueError, RuntimeError) as e:
            raise SpecError(self.path, str(e)) from None
        rep = self.report(feed, res, ss, i) if final else {}
        if final and ss is not None and self.heat is None and len(ss.states) > 1:
            rep["warnings"] = [m.multiplicity_note(ss)]
        return {"out": res.outlet}, rep

    def report(self, feed, res, ss, selected) -> dict:
        size_key = "W" if self.kind == "pbr" else "V"
        reactants = {s for rx in self.rs.reactions for s in rx.reactants}
        rep = {size_key: self.size,
               "X": {s: res.conversion(s) for s in res.species if s in reactants and res.n0.get(s, 0) > 0}}
        if self.kind != "pbr":
            rep["tau"] = self.size / feed.v
        if ss is not None:
            rep["steady_states"] = [{"T": st.outlet.T, "F": dict(st.outlet.F), "stable": ok,
                                     "X": {s: st.conversion(s) for s in rep["X"]}}
                                    for st, ok in zip(ss.states, ss.stable)]
            rep["selected"] = selected
        else:
            prof = {"axis": size_key, "x": res.axis.tolist(),
                    "F": {s: res.profile(s).tolist() for s in res.species}}
            for k in ("T", "p"):
                if k in res.extras:
                    prof[k] = np.asarray(res.extras[k]).tolist()
            rep["profile"] = prof
        return rep


def energy(d, names, ctx, path) -> Heat | None:
    if d == "isothermal":
        return None
    if d == "adiabatic":
        Ua, Ta = 0.0, None
    elif isinstance(d, dict):
        obj(d, path, {"Ua", "Ta"}, ("Ua", "Ta"))
        Ua, Ta = d["Ua"], quantity(d["Ta"], "K", sub(path, "Ta"))
    else:
        raise SpecError(path, "use 'isothermal', 'adiabatic' or {\"Ua\": ..., \"Ta\": ...}")
    missing = [n for n in names if n not in ctx.thermo.dH]
    if missing:
        raise SpecError(path, f"energy balance needs dH for reaction(s) {missing}")
    if not ctx.thermo.Cp:
        raise SpecError(path, "energy balance needs thermo.Cp")
    try:
        return Heat([ctx.thermo.dH[n] for n in names], dict(ctx.thermo.Cp), ctx.thermo.T_ref, Ua, Ta)
    except ValueError as e:
        raise SpecError(path, str(e)) from None


OPS = {"feed": Feed, "product": Product, "mixer": Mixer, "splitter": Splitter, "heater": Heater,
       "cstr": Reactor, "pfr": Reactor, "pbr": Reactor}
