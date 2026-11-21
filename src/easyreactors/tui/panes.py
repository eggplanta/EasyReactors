from rich.text import Text
from textual import work
from textual.app import ComposeResult
from textual.containers import Horizontal, VerticalScroll
from textual.widgets import Button, Input, Select, Static, TextArea

from dataclasses import replace

from .compute import (CORRELATIONS, GEOMETRIES, METHODS, RTD_KINDS, TWO_PARAM, ReactionSpec, Report,
                      build_reactions, build_rtd, effective_De, fit_two_param, rtd_report, run_batch, run_criteria,
                      run_flow_reactor, run_mass_transfer, run_network, run_packed_bed, run_pellet, run_rtd_models,
                      run_two_param, run_two_pellets)
from .parsing import table_file
from .widgets import FeedEditor, ReactionsEditor, ResultsView, area, box, inp, options, row, sel, value

T17 = "0 5 10 15 20 30 40 50 70 100 150 200\n112 95.8 82.2 70.6 60.9 45.6 34.5 26.3 15.7 7.67 2.55 0.9"


class Pane(Horizontal):
    def form(self) -> ComposeResult:
        yield from ()

    def compose(self) -> ComposeResult:
        with VerticalScroll(classes="form"):
            yield from self.form()
            with Horizontal(classes="actions"):
                yield Button("Run ▶", variant="primary", id="run")
                yield Button("Example", id="example")
        yield ResultsView()

    @property
    def results(self) -> ResultsView:
        return self.query_one(ResultsView)

    async def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id == "run":
            await self.run_job()
        elif event.button.id == "example":
            await self.example()

    async def run_job(self) -> None:
        try:
            job = self.job()
        except Exception as e:
            await self.results.error(str(e) or type(e).__name__)
            return
        self.results.loading = True
        self._compute(job)

    @work(thread=True, exclusive=True)
    def _compute(self, job) -> None:
        try:
            rep = job()
            self.app.call_from_thread(self.results.show, rep)
        except Exception as e:
            self.app.call_from_thread(self.results.error, str(e) or type(e).__name__)
        finally:
            self.app.call_from_thread(setattr, self.results, "loading", False)

    def job(self):
        raise NotImplementedError

    async def example(self) -> None:
        pass

    def fill(self, **values: str) -> None:
        for k, v in values.items():
            w = self.query_one(f"#{k}")
            if isinstance(w, TextArea):
                w.text = v
            else:
                w.value = v

    def specs(self) -> list[ReactionSpec]:
        return [s for s in self.query_one(ReactionsEditor).specs() if s.equation.strip()]


# Single reactor

class ReactorPane(Pane):
    def form(self):
        yield box("Reactor",
                  row("Type", sel(options(["Batch", "CSTR", "PFR"]), value="CSTR", allow_blank=False, id="kind")),
                  row("Problem", sel([("Simulate (given V or t)", "Simulate"),
                                         ("Size (V or t for a target X)", "Size"),
                                         ("Maximize an intermediate", "Maximize")],
                                        value="Simulate", allow_blank=False, id="mode")),
                  row("V", inp(placeholder="[L]  e.g. 100 · 2 m3 · 800 gallon", id="V"), "r-V"),
                  row("t", inp(placeholder="[min]  e.g. 10 h", id="t"), "r-t"),
                  row("X", inp(placeholder="target conversion, 0-1", id="X"), "r-X"),
                  row("Key species", inp(placeholder="blank = reference of reaction 1", id="key"), "r-key"),
                  row("Maximize", inp(placeholder="species, e.g. B", id="species"), "r-species"))
        yield box("Reactions", ReactionsEditor())
        yield box("Feed", FeedEditor(id="feed"), id="feed-box")

    def on_mount(self) -> None:
        self.refresh_fields()

    def on_select_changed(self, event: Select.Changed) -> None:
        if event.select.id in ("kind", "mode"):
            self.refresh_fields()

    def refresh_fields(self) -> None:
        kind, mode = value(self, "#kind"), value(self, "#mode")
        batch = kind == "Batch"
        self.query_one("#r-V").display = batch or mode == "Simulate" or (mode == "Maximize" and kind == "PFR")
        self.query_one("#r-t").display = batch and mode == "Simulate"
        self.query_one("#r-X").display = self.query_one("#r-key").display = mode == "Size"
        self.query_one("#r-species").display = mode == "Maximize" and not batch
        self.query_one("#V").placeholder = ("[L]  search range for the maximum (blank = auto)"
                                            if mode == "Maximize" else "[L]  e.g. 100 · 2 m3 · 800 gallon")
        self.query_one(FeedEditor).set_batch(batch)
        self.query_one("#feed-box").border_title = "Initial charge" if batch else "Feed"

    def job(self):
        kind, mode = value(self, "#kind"), value(self, "#mode")
        specs, f = self.specs(), {k: value(self, f"#{k}") for k in ("V", "t", "X", "key", "species")}
        feed = self.query_one(FeedEditor)
        if kind == "Batch":
            if mode == "Maximize":
                raise ValueError("Batch: use Simulate, maxima of intermediates are reported automatically")
            T, C0 = value(feed, "#T"), value(feed, "#comp")
            return lambda: run_batch(mode, build_reactions(specs), f["V"], T, C0, f["t"], f["X"], f["key"])
        stream = feed.stream()
        return lambda: run_flow_reactor(kind, mode, build_reactions(specs), stream, f["V"], f["X"], f["key"],
                                        f["species"])

    async def example(self) -> None:
        kind = value(self, "#kind")
        editor = self.query_one(ReactionsEditor)
        if kind == "PFR":
            await editor.load([ReactionSpec("C2H6 > C2H4 + H2", k="0.072 1/s", Ea="82 kcal/mol", T_ref="1000")])
            self.fill(mode="Size", X="0.8", phase="Gas", T="1100", P="6", comp="C2H6=0.425 lbmol/s")
        elif kind == "CSTR":
            await editor.load([ReactionSpec("EO + W > EG", ref="EO", orders="EO=1", k="0.311 1/min")])
            self.fill(mode="Simulate", V="800 gallon", phase="Liquid", T="300", v0="7.24 L/s",
                      comp="EO=8.05, W=27.8")
        else:
            await editor.load([ReactionSpec("A > B", k="0.5 1/h"), ReactionSpec("B > C", k="0.2 1/h")])
            self.fill(mode="Simulate", V="1000", t="10 h", T="300", comp="A=2")


# Network

PALETTE = [("CSTR", "CSTR(?)"), ("PFR", "PFR(?)"), ("Parallel", "[CSTR(?) | PFR(?)]"), ("Bypass", "bypass"),
           ("Recycle", "recycle(PFR(?), 1)"), ("Side feed", "feed")]
SYNTAX = "\n".join([
    "a > b              series",
    "[a | b]{0.3, 0.7}  parallel with split fractions ({?} = size the split, none = equal)",
    "bypass · feed      empty branch · side feed (one line per feed below)",
    "recycle(a, R)      recycle loop with ratio R",
    "CSTR(5 m3)         volume in L unless units are given; ? = unknown (same V for all)",
])


class NetworkPane(Pane):
    def form(self):
        yield box("Network",
                  inp(placeholder="CSTR(800 gallon) > [PFR(?) | bypass]{0.7, 0.3}", id="expr"),
                  Horizontal(*[Button(label, name=snippet, classes="small ins", compact=True) for label, snippet in PALETTE],
                             Button("Clear", name="", classes="small ins", variant="warning", compact=True), classes="palette"),
                  Static(Text(SYNTAX), classes="hint"))
        yield box("Problem",
                  row("X", inp(placeholder="target X (if a V or split is unknown)", id="X")),
                  row("Key species", inp(placeholder="blank = reference of reaction 1", id="key")),
                  row("Split", sel(options(["Target X", "Equal conversion"]), value="Target X",
                                      allow_blank=False, id="split")),
                  row("Side feeds", area(placeholder="one per line: v0=1 L/min, A=2  (gas: A=10 mol/min)",
                                             id="sides", classes="short")))
        yield box("Reactions", ReactionsEditor())
        yield box("Feed", FeedEditor())

    async def on_button_pressed(self, event: Button.Pressed) -> None:
        if "ins" in event.button.classes:
            expr = self.query_one("#expr", Input)
            snippet = event.button.name or ""
            if not snippet:
                expr.value = ""
            else:
                text = expr.value.rstrip()
                sep = " > " if text and text[-1] not in "[|(" else ""
                expr.value = text + sep + snippet
            expr.focus()
            return
        await super().on_button_pressed(event)

    def job(self):
        specs = self.specs()
        feed = self.query_one(FeedEditor).stream()
        sides = [l for l in self.query_one("#sides", TextArea).text.splitlines() if l.strip()]
        args = value(self, "#expr"), value(self, "#X"), value(self, "#key"), value(self, "#split")
        return lambda: run_network(args[0], build_reactions(specs), feed, sides, args[1], args[2], args[3])

    async def example(self) -> None:
        await self.query_one(ReactionsEditor).load(
            [ReactionSpec("EO + W > EG", ref="EO", orders="EO=1", k="0.311 1/min")])
        self.fill(expr="CSTR(?) > CSTR(?)", X="0.9", key="EO", phase="Liquid", T="300", v0="7.24 L/s",
                  comp="EO=8.05, W=27.8")


# RTD

class RTDPane(Pane):
    def form(self):
        yield box("Tracer data",
                  row("Data", sel(options(RTD_KINDS), value=RTD_KINDS[0], allow_blank=False, id="kind")),
                  area(placeholder="Paste t and C as two rows or two columns (spaces, commas, or ';' with "
                                       "decimal commas)", id="data"),
                  Horizontal(inp(placeholder="or a CSV/TXT file path", id="path"),
                             Button("Load", id="load", classes="small", compact=True), classes="row"),
                  row("Time unit", inp("min", id="unit")),
                  row("Step C0", inp(placeholder="same units as C (step only)", id="C0step"), "r-C0step"),
                  row("Method", sel(options(METHODS), value="trapezoid", allow_blank=False, id="method")))
        yield box("Conversion (optional: fill a reaction)",
                  ReactionsEditor(),
                  row("T", inp("300", placeholder="[K]", id="T")),
                  row("C0", inp(placeholder="A=4  [mol/L]", id="C0")),
                  row("τ = V/v0", inp(placeholder="[min] for ideal PFR/CSTR (blank = tm)", id="tau")))

    def on_mount(self) -> None:
        self.query_one("#r-C0step").display = False

    def on_select_changed(self, event: Select.Changed) -> None:
        if event.select.id == "kind":
            self.query_one("#r-C0step").display = event.value == RTD_KINDS[1]

    async def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id == "load":
            try:
                t, y = table_file(value(self, "#path"))[:2]
                self.query_one("#data", TextArea).text = " ".join(f"{v:g}" for v in t) + "\n" + " ".join(
                    f"{v:g}" for v in y)
            except Exception as e:
                await self.results.error(f"Cannot load file: {e}")
            return
        await super().on_button_pressed(event)

    def job(self):
        kind, unit, method, C0s = (value(self, f"#{k}") for k in ("kind", "unit", "method", "C0step"))
        data = self.query_one("#data", TextArea).text
        specs = self.specs()
        T, C0, tau = value(self, "#T"), value(self, "#C0"), value(self, "#tau")

        def run() -> Report:
            rtd = build_rtd(kind, data, unit or "min", method, C0s)
            rep = rtd_report(rtd, unit or "min")
            if specs:
                models = run_rtd_models(rtd, build_reactions(specs), C0, T, tau)
                rep.summary += models.summary
                rep.tables[1:1] = models.tables
            return rep
        return run

    async def example(self) -> None:
        await self.query_one(ReactionsEditor).load([ReactionSpec("2A > B", k="0.015 dm3/(mol*min)")])
        self.fill(kind=RTD_KINDS[0], data=T17, unit="min", T="300", C0="A=4", tau="40")


# Two-parameter models

class TwoParamPane(Pane):
    def form(self):
        names = list(TWO_PARAM)
        yield box("Model",
                  row("Model", sel(options(names), value=names[0], allow_blank=False, id="model")),
                  Static(TWO_PARAM[names[0]], id="desc", classes="hint"),
                  row("τ = V/v0", inp(placeholder="[min]", id="tau")),
                  row("Parameters", sel([("Fit from tracer data", "Fit"), ("Enter α and β", "Manual")],
                                           value="Fit", allow_blank=False, id="source")))
        yield box("Tracer data",
                  area(placeholder="t and C as two rows or two columns", id="data"),
                  row("Time unit", inp("min", id="unit")),
                  row("Step C0", inp(placeholder="step concentration (bypass + dead volume)", id="C0step"),
                      "r-C0step"),
                  row("t split", inp(placeholder="time between the two peaks", id="tsplit"), "r-tsplit"),
                  id="data-box")
        yield box("Parameters", row("α", inp(id="alpha")), row("β", inp(id="beta")), id="manual-box")
        yield box("Conversion (optional: fill a reaction)",
                  ReactionsEditor(),
                  row("T", inp("300", placeholder="[K]", id="T")),
                  row("C0", inp(placeholder="A=2, B=2  [mol/L]", id="C0")))

    def on_mount(self) -> None:
        self.refresh_fields()

    def on_select_changed(self, event: Select.Changed) -> None:
        if event.select.id in ("model", "source"):
            self.refresh_fields()

    def refresh_fields(self) -> None:
        model, fit = value(self, "#model"), value(self, "#source") == "Fit"
        self.query_one("#desc", Static).update(TWO_PARAM[model])
        self.query_one("#data-box").display = fit
        self.query_one("#manual-box").display = not fit
        self.query_one("#r-C0step").display = model.startswith("CSTR")
        self.query_one("#r-tsplit").display = model == "PFR + bypass"

    def job(self):
        model, tau, unit = value(self, "#model"), value(self, "#tau"), value(self, "#unit") or "min"
        fit_mode = value(self, "#source") == "Fit"
        data = self.query_one("#data", TextArea).text
        C0s, tsplit = value(self, "#C0step"), value(self, "#tsplit")
        a, b = value(self, "#alpha"), value(self, "#beta")
        specs, T, C0 = self.specs(), value(self, "#T"), value(self, "#C0")

        def run() -> Report:
            fit = fit_two_param(model, tau, data, unit, C0s, tsplit) if fit_mode else None
            alpha, beta = (fit.alpha, fit.beta) if fit else (float(a), float(b))
            return run_two_param(model, tau, alpha, beta, build_reactions(specs) if specs else None, C0, T, fit)
        return run

    async def example(self) -> None:
        await self.query_one(ReactionsEditor).load(
            [ReactionSpec("A + B > C + D", ref="A", k="0.28 m3/(kmol*min)")])
        self.fill(model=list(TWO_PARAM)[0], source="Fit", tau="10", unit="min", C0step="2000",
                  data="4 8 10 14 16 18\n1000 1333 1500 1666 1750 1800", T="300", C0="A=2, B=2")


# Diffusion

CALCS = {
    "Pellet: Thiele modulus and η": ("pellet", "kin"),
    "Criteria: Weisz-Prater and Mears": ("pellet", "crit"),
    "External mass transfer (kc)": ("mt",),
    "Two pellet sizes (find φ, η, R)": ("two",),
    "Packed bed with diffusion (Ω, W)": ("pellet", "kin", "bed", "feed"),
}
DESC = {
    "Pellet: Thiele modulus and η": "Numerical η for any rate law (exact), compared with the book formulas.",
    "Criteria: Weisz-Prater and Mears": "From an observed rate: C_WP ≫ 1 → internal limitation; MR > 0.15 → external.",
    "External mass transfer (kc)": "Re, Sc, Sh, kc and ac from a correlation; X if the bed is mass-transfer limited.",
    "Two pellet sizes (find φ, η, R)": "1st order, sphere: two observed rates at two radii give φ and η (Fogler 15-2).",
    "Packed bed with diffusion (Ω, W)": "Single reaction, external + internal resistance along the bed (Fogler 15-3).",
}
NO_RATE = "4.42e-10*530 m3/(g*s)"


class DiffusionPane(Pane):
    def form(self):
        names = list(CALCS)
        yield box("Calculation", row("Calculation", sel(options(names), value=names[0], allow_blank=False, id="calc")),
                  Static(DESC[names[0]], id="desc", classes="hint"))
        yield box("Pellet",
                  row("Geometry", sel(options(GEOMETRIES), value="sphere", allow_blank=False, id="geom")),
                  row("R", inp(placeholder="[m] radius (slab: half thickness), e.g. 3 mm", id="pR")),
                  row("De", inp(placeholder="[m²/s] effective diffusivity (blank = from DAB below)", id="pDe")),
                  row("DAB", inp(placeholder="[m²/s] to compute De = DAB·φp·σc/τ", id="pDAB"), "r-pDAB"),
                  row("φp, σc, τ", Horizontal(inp(placeholder="porosity", id="pphi"),
                                              inp(placeholder="constriction", id="psig"),
                                              inp(placeholder="tortuosity", id="ptau"), classes="multi"), "r-ptri"),
                  row("ρc", inp(placeholder="[kg/m³] pellet density, e.g. 2.8 g/cm3", id="prho")),
                  row("Rate per", sel([("Pellet volume (k)", "volume"), ("Catalyst mass (k′)", "mass")],
                                      value="mass", allow_blank=False, id="pbasis"), "r-pbasis"),
                  id="b-pellet")
        yield box("Kinetics", ReactionsEditor(),
                  row("T", inp("300", placeholder="[K]", id="kT"), "r-kT"),
                  row("Cs", inp(placeholder="surface concentrations, A=0.2  [mol/L]", id="kCs"), "r-kCs"),
                  id="b-kin")
        yield box("Criteria",
                  row("−r′A obs", inp(placeholder="[mol/kg/s] observed rate", id="crr")),
                  row("Cs", inp(placeholder="surface concentration [mol/L] (Weisz-Prater)", id="crCs")),
                  row("Cb", inp(placeholder="bulk concentration [mol/L] (Mears)", id="crCb")),
                  row("n", inp("1", placeholder="reaction order", id="crn")),
                  row("kc", inp(placeholder="[m/s]", id="crkc")),
                  row("ρb", inp(placeholder="[kg/m³] bed bulk density", id="crrhob")),
                  id="b-crit")
        yield box("External mass transfer",
                  row("Correlation", sel(options(CORRELATIONS), value=CORRELATIONS[0], allow_blank=False, id="mcorr")),
                  row("U", inp(placeholder="[m/s] superficial velocity", id="mU")),
                  row("dp", inp(placeholder="[m] particle diameter", id="mdp")),
                  row("Cylinder D, L", Horizontal(inp(placeholder="D (instead of dp)", id="mD"),
                                                  inp(placeholder="L", id="mL"), classes="multi")),
                  row("ν", inp(placeholder="[m²/s] kinematic viscosity, e.g. 0.5 cSt", id="mnu")),
                  row("DAB", inp(placeholder="[m²/s]", id="mDAB")),
                  row("Bed porosity", inp(placeholder="φ (packed bed)", id="meps")),
                  row("γ", inp(placeholder="shape factor (blank = 1 or from cylinder)", id="mgam")),
                  row("Bed length", inp(placeholder="[m] for mass-transfer-limited X", id="mLb")),
                  id="b-mt")
        yield box("Two pellet sizes",
                  row("Pellet 1", Horizontal(inp(placeholder="observed rate", id="tr1"),
                                             inp(placeholder="R1 [m]", id="tR1"), classes="multi")),
                  row("Pellet 2", Horizontal(inp(placeholder="observed rate", id="tr2"),
                                             inp(placeholder="R2 [m]", id="tR2"), classes="multi")),
                  row("Target η", inp(placeholder="e.g. 0.95 → radius needed", id="teta")),
                  id="b-two")
        yield box("Packed bed",
                  row("Problem", sel([("Size: W for a target X", "Size"), ("Simulate: X for a given W", "Simulate")],
                                     value="Size", allow_blank=False, id="bmode")),
                  row("X or W", inp(placeholder="0.998  or  450 g", id="btarget")),
                  row("ρb", inp(placeholder="[kg/m³] bed bulk density", id="brhob")),
                  row("kc", inp(placeholder="[m/s] (use External mass transfer to get it)", id="bkc")),
                  row("ac", inp(placeholder="[m²/m³] = 6(1−φ)/dp", id="bac")),
                  id="b-bed")
        yield box("Feed", FeedEditor(), id="b-feed")

    def on_mount(self) -> None:
        self.refresh_fields()

    def on_select_changed(self, event: Select.Changed) -> None:
        if event.select.id == "calc":
            self.refresh_fields()

    def refresh_fields(self) -> None:
        calc = value(self, "#calc")
        shown = CALCS[calc]
        for b in ("pellet", "kin", "crit", "mt", "two", "bed", "feed"):
            self.query_one(f"#b-{b}").display = b in shown
        self.query_one("#desc", Static).update(DESC[calc])
        bed = "bed" in shown
        self.query_one("#r-kT").display = self.query_one("#r-kCs").display = not bed
        self.query_one("#r-pbasis").display = "kin" in shown

    def _specs(self) -> list[ReactionSpec]:
        catalytic = value(self, "#pbasis") == "mass"
        return [replace(s, catalytic=catalytic) for s in self.specs()]

    def _De(self) -> float:
        return effective_De(*(value(self, f"#{k}") for k in ("pDe", "pDAB", "pphi", "psig", "ptau")))

    def job(self):
        calc = value(self, "#calc")
        v = lambda k: value(self, f"#{k}")
        if calc.startswith("Pellet"):
            specs, De = self._specs(), self._De()
            return lambda: run_pellet(build_reactions(specs), v("kCs"), v("kT"), v("pR"), De,
                                      v("prho") if value(self, "#pbasis") == "mass" else "", v("geom"))
        if calc.startswith("Criteria"):
            De = self._De() if (v("pDe") or v("pDAB")) else None
            args = (v("crr"), v("prho"), v("crrhob"), v("pR"), De, v("crCs"), v("crCb"), v("crn"), v("crkc"))
            return lambda: run_criteria(*args)
        if calc.startswith("External"):
            args = [v(k) for k in ("mcorr", "mU", "mdp", "mnu", "mDAB", "meps", "mgam", "mD", "mL", "mLb")]
            return lambda: run_mass_transfer(*args)
        if calc.startswith("Two"):
            args = [v(k) for k in ("tr1", "tR1", "tr2", "tR2", "teta")]
            return lambda: run_two_pellets(*args)
        specs, De = self._specs(), self._De()
        if not any(s.catalytic for s in specs):
            raise ValueError("Packed bed: set 'Rate per' to catalyst mass (k′)")
        feed = self.query_one(FeedEditor).stream()
        args = (v("bmode"), v("btarget"), v("pR"), De, v("prho"), v("brhob"), v("bkc"), v("bac"), v("geom"))
        return lambda: run_packed_bed(build_reactions(specs), feed, *args)

    async def example(self) -> None:
        calc = value(self, "#calc")
        no = [ReactionSpec("NO > P", k=NO_RATE)]
        pellet = dict(geom="sphere", pR="3 mm", pDe="1.82e-8", prho="2.8 g/cm3", pbasis="mass")
        if calc.startswith("Pellet"):
            await self.query_one(ReactionsEditor).load(no)
            self.fill(**pellet, kT="1173", kCs="NO=2.08e-4")
        elif calc.startswith("Criteria"):
            self.fill(**pellet, crr="2.872e-6", crCs="7.64e-5", crCb="2.078e-4", crn="1", crkc="6e-5",
                      crrhob="1.4 g/cm3")
        elif calc.startswith("External"):
            self.fill(mcorr=CORRELATIONS[0], mU="150", mdp="", mD="0.25 cm", mL="0.5 cm", mnu="4.94e-5",
                      mDAB="0.69e-4*2.06", meps="0.4", mgam="", mLb="0.05")
        elif calc.startswith("Two"):
            self.fill(tr1="3e-5", tR1="0.01", tr2="15e-5", tR2="0.001", teta="0.95")
        else:
            await self.query_one(ReactionsEditor).load(no)
            self.fill(**pellet, bmode="Size", btarget="0.998", brhob="1.4 g/cm3", bkc="6e-5", bac="500",
                      phase="Liquid", T="1173", v0="1e-6 m3/s", comp="NO=2.078e-4")
