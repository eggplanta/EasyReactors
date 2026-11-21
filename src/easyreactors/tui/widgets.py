import asyncio

import plotext as plt
from rich.markup import escape
from rich.text import Text
from textual.app import ComposeResult
from textual.containers import Horizontal, Vertical, VerticalScroll
from textual.widget import Widget
from textual.widgets import Button, DataTable, Input, Label, Markdown, Select, Static, TextArea

from ..core import parse_equation
from .compute import (DROP_MODES, ENERGY_MODES, MODELS, DropSpec, EnergySpec, Report, ReactionSpec,
                      build_feed)
from .parsing import human_K, human_k, orders


def inp(value: str = "", **kw) -> Input:
    return Input(value, compact=True, **kw)


def sel(choices, **kw) -> Select:
    return Select(choices, compact=True, **kw)


def area(text: str = "", **kw) -> TextArea:
    return TextArea(text, compact=True, **kw)


def options(values: list[str]) -> list[tuple[str, str]]:
    return [(v, v) for v in values]


def row(label: str, widget: Widget, id: str | None = None) -> Horizontal:
    return Horizontal(Label(Text(label), classes="lbl"), widget, classes="row", id=id)


def box(title: str, *children: Widget, id: str | None = None, classes: str = "") -> Vertical:
    v = Vertical(*children, classes=f"box {classes}", id=id)
    v.border_title = escape(title)
    return v


def value(root: Widget, selector: str) -> str:
    w = root.query_one(selector)
    return "" if w.value in (None, Select.BLANK) else str(w.value)


# Reactions

class ReactionRow(Vertical):
    FIELDS = {
        "Power law": ["ref", "orders", "k", "Ea", "Tref", "basis"],
        "Reversible": ["ref", "orders", "rorders", "k", "K", "Ea", "Tref", "basis"],
        "Michaelis-Menten": ["ref", "k", "K"],
        "Expression": ["ref", "expr", "params"],
    }
    LABELS = {"Michaelis-Menten": {"k": "Vmax", "K": "Km [mol/L]"}}

    def __init__(self, spec: ReactionSpec | None = None, energy: bool = False, per_mass: bool = False):
        super().__init__(classes="box reaction")
        self.spec0 = spec or ReactionSpec("")
        self.energy = energy
        self.per_mass = per_mass
        self.border_title = "Reaction"

    def compose(self) -> ComposeResult:
        s = self.spec0
        yield row("Equation", inp(s.equation, placeholder="e.g. A + B > C   ·   A <> B (reversible)", id="eq"))
        yield Static("", id="hint", classes="rhint")
        yield row("Rate law", sel(options(MODELS), value=s.model, allow_blank=False, id="model"))
        yield row("Rate of", inp(s.ref, placeholder="species whose -r is given (blank = limiting reactant)",
                                 id="ref"), "r-ref")
        yield row("Orders", inp(s.orders, placeholder="e.g. A=1, B=1 (blank = stoichiometric)", id="orders"),
                  "r-orders")
        yield row("Rev. orders", inp(s.reverse_orders, placeholder="C=1 (blank = stoichiometric)",
                                          id="rorders"), "r-rorders")
        yield row("k", inp(s.k, placeholder="value and units, see the line above", id="k"), "r-k")
        yield row("K", inp(s.K, placeholder="equilibrium constant, units above", id="K"), "r-K")
        yield row("Ea", inp(s.Ea, placeholder="optional, e.g. 82 kcal/mol · 65.7 kJ/mol", id="Ea"), "r-Ea")
        yield row("T of k [K]", inp(s.T_ref, placeholder="temperature where k was given (needed with Ea)",
                                    id="Tref"), "r-Tref")
        yield row("Basis", sel([("Concentration C", "C"), ("Partial pressure p", "P")], value=s.basis,
                                  allow_blank=False, id="basis"), "r-basis")
        yield row("-r =", inp(s.expr, placeholder="k*C_A^2/(1 + K*C_A)", id="expr"), "r-expr")
        yield row("Parameters", inp(s.params, placeholder="k=0.01, K=2  (SI)", id="params"), "r-params")
        yield row("ΔHrx", inp(s.dH, placeholder="per mol of the 'Rate of' species, e.g. -6.9 kJ/mol", id="dH"),
                  "r-dH")
        yield row("T of K", inp(s.K_T, placeholder="[K] K varies with T (van't Hoff); blank = constant K",
                                id="KT"), "r-KT")
        yield Button("✕ Remove", variant="error", classes="small", id="remove", compact=True)

    def on_mount(self) -> None:
        self.refresh_fields()

    def on_select_changed(self, event: Select.Changed) -> None:
        if event.select.id in ("model", "basis"):
            self.refresh_fields()

    def on_input_changed(self, event: Input.Changed) -> None:
        if event.input.id in ("eq", "orders", "rorders"):
            self.update_hint()

    def update_hint(self) -> None:
        hint = self.query_one("#hint", Static)
        eq = value(self, "#eq").strip()
        hint.display = bool(eq)
        if not eq:
            return
        try:
            text = self._hint(eq)
            hint.set_class(False, "bad")
        except ValueError as e:
            text = f"✗ {e}"
            hint.set_class(True, "bad")
        hint.update(Text(text))

    def _hint(self, eq: str) -> str:
        nu, rev = parse_equation(eq)
        side = lambda sign: " + ".join((f"{abs(v):g} " if abs(v) != 1 else "") + k
                                       for k, v in nu.items() if v * sign > 0) or "∅"
        model, basis = value(self, "#model"), value(self, "#basis") or "C"
        parts = [f"{side(-1)} {'⇄' if rev else '→'} {side(1)}"]
        if model in ("Power law", "Reversible"):
            text = value(self, "#orders")
            of = orders(text) if text.strip() else {k: abs(v) for k, v in nu.items() if v < 0}
            n = sum(of.values())
            parts.append(f"order {n:g}: k in {human_k(n, basis, self.per_mass)}")
            if rev or model == "Reversible":
                text = value(self, "#rorders")
                orr = orders(text) if text.strip() else {k: v for k, v in nu.items() if v > 0}
                parts.append(f"K in {human_K(of, orr, basis)}")
        elif model == "Michaelis-Menten":
            parts.append("Vmax in mol/(L*min), Km in mol/L")
        else:
            parts.append("use C_A, p_A, T and your parameters (SI units)")
        return "  ·  ".join(parts)

    def refresh_fields(self) -> None:
        model = value(self, "#model")
        shown = self.FIELDS.get(model, [])
        labels = self.LABELS.get(model, {})
        for name in ["ref", "orders", "rorders", "k", "K", "Ea", "Tref", "basis", "expr", "params"]:
            r = self.query_one(f"#r-{name}")
            r.display = name in shown
            if name in ("k", "K"):
                r.query_one(Label).update(Text(labels.get(name, name)))
        self.query_one("#r-dH").display = self.energy
        self.query_one("#r-KT").display = self.energy and model == "Reversible"
        self.update_hint()

    def set_per_mass(self, on: bool) -> None:
        self.per_mass = on
        if self.is_mounted:
            self.update_hint()


    def set_energy(self, on: bool) -> None:
        self.energy = on
        if self.is_mounted:
            self.refresh_fields()

    def spec(self) -> ReactionSpec:
        return ReactionSpec(value(self, "#eq"), value(self, "#model"), value(self, "#ref"), value(self, "#orders"),
                            value(self, "#rorders"), value(self, "#k"), value(self, "#K"), value(self, "#Ea"),
                            value(self, "#Tref"), value(self, "#basis"), value(self, "#expr"),
                            value(self, "#params"), dH=value(self, "#dH"), K_T=value(self, "#KT"))


class ReactionsEditor(Vertical):
    energy = False
    per_mass = False

    def compose(self) -> ComposeResult:
        yield Vertical(self._row(), id="rows")
        yield Button("+ Add reaction", id="add", classes="small", variant="success", compact=True)

    def _row(self, spec: ReactionSpec | None = None) -> ReactionRow:
        return ReactionRow(spec, self.energy, self.per_mass)

    def on_mount(self) -> None:
        self.renumber()

    def renumber(self) -> None:
        rows = list(self.query(ReactionRow))
        for i, r in enumerate(rows, 1):
            r.border_title = f"Reaction {i}" if len(rows) > 1 else "Reaction"

    async def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id == "add":
            event.stop()
            await self.query_one("#rows").mount(self._row())
            self.renumber()
        elif event.button.id == "remove":
            event.stop()
            await event.button.parent.remove()
            self.renumber()

    def set_per_mass(self, on: bool) -> None:
        self.per_mass = on
        for r in self.query(ReactionRow):
            r.set_per_mass(on)

    def set_energy(self, on: bool) -> None:
        self.energy = on
        for r in self.query(ReactionRow):
            r.set_energy(on)

    def set_dH(self, values: list[str]) -> None:
        for r, v in zip(self.query(ReactionRow), values):
            if v:
                r.query_one("#dH", Input).value = v

    def specs(self) -> list[ReactionSpec]:
        return [r.spec() for r in self.query(ReactionRow)]

    async def load(self, specs: list[ReactionSpec]) -> None:
        rows = self.query_one("#rows")
        await rows.remove_children()
        await rows.mount_all([self._row(s) for s in specs])
        self.renumber()


# Feed

class FeedEditor(Vertical):
    def __init__(self, batch: bool = False, **kw):
        super().__init__(**kw)
        self.batch = batch

    def compose(self) -> ComposeResult:
        yield row("Phase", sel(options(["Liquid", "Gas"]), value="Liquid", allow_blank=False, id="phase"), "f-phase")
        yield row("T [K]", inp("300", placeholder="e.g. 300 · 25 degC", id="T"))
        yield row("P [atm]", inp(placeholder="e.g. 10 · 162 kPa", id="P"), "f-P")
        yield row("v0 [L/min]", inp(placeholder="e.g. 10 · 7.24 L/s", id="v0"), "f-v0")
        yield row("C0 [mol/L]", inp(placeholder="e.g. A=2, B=0.5", id="comp"), "f-comp")

    def on_mount(self) -> None:
        self.set_batch(self.batch)

    def on_select_changed(self, event: Select.Changed) -> None:
        if event.select.id == "phase":
            self.refresh_fields()

    def set_batch(self, batch: bool) -> None:
        self.batch = batch
        self.refresh_fields()

    def refresh_fields(self) -> None:
        gas = value(self, "#phase") == "Gas" and not self.batch
        self.query_one("#f-phase").display = not self.batch
        self.query_one("#f-P").display = gas
        self.query_one("#f-v0").display = not gas and not self.batch
        self.query_one("#f-comp").query_one(Label).update(Text("F0 [mol/min]" if gas else "C0 [mol/L]"))
        self.query_one("#comp").placeholder = "e.g. A=10, I=5 · A=0.1 mol/s" if gas else "e.g. A=2, B=0.5"

    def stream(self):
        return build_feed(value(self, "#phase"), value(self, "#T"), value(self, "#P"), value(self, "#v0"),
                          value(self, "#comp"))

    def set(self, **values: str) -> None:
        for k, v in values.items():
            w = self.query_one(f"#{k}")
            w.value = v


class EnergyEditor(Vertical):
    def compose(self) -> ComposeResult:
        yield row("Mode", sel(options(ENERGY_MODES), value="Isothermal", allow_blank=False, id="e-mode"))
        yield row("Cp [J/mol/K]", inp(placeholder="every species, inerts too: A=141, B=141, I=161", id="Cp"),
                  "e-Cp")
        yield row("T of ΔH [K]", inp(placeholder="reference T of ΔHrx (blank = 298.15)", id="eTref"), "e-Tref")
        yield row("Ua [W/m3/K]", inp(placeholder="e.g. 16500 · 5000 kJ/(m3*h*K)", id="Ua"), "e-Ua")
        yield row("Ta [K]", inp(placeholder="coolant temperature (constant)", id="Ta"), "e-Ta")
        yield row("Database", Horizontal(inp(placeholder="A=n-butane, B=isobutane  → fills Cp and ΔHrx", id="names"),
                                         Button("Look up", id="lookup", classes="small", compact=True),
                                         classes="multi"), "e-names")

    def on_mount(self) -> None:
        self.refresh_fields()

    def on_select_changed(self, event: Select.Changed) -> None:
        if event.select.id == "e-mode":
            self.refresh_fields()

    @property
    def on(self) -> bool:
        return value(self, "#e-mode") != "Isothermal"

    def refresh_fields(self, total_ua: bool | None = None) -> None:
        mode = value(self, "#e-mode")
        for name in ("Cp", "Tref", "names"):
            self.query_one(f"#e-{name}").display = mode != "Isothermal"
        for name in ("Ua", "Ta"):
            self.query_one(f"#e-{name}").display = mode.startswith("Heat")
        if total_ua is not None:
            label = "UA [W/K]" if total_ua else "Ua [W/m3/K]"
            self.query_one("#e-Ua", Horizontal).query_one(Label).update(Text(label))
            self.query_one("#Ua", Input).placeholder = ("e.g. 2110 · 40000 J/(min*K)" if total_ua
                                                        else "e.g. 16500 · 5000 kJ/(m3*h*K)")

    def spec(self) -> EnergySpec:
        return EnergySpec(value(self, "#e-mode"), value(self, "#Cp"), value(self, "#Ua"), value(self, "#Ta"),
                          value(self, "#eTref"))


class DropEditor(Vertical):
    def compose(self) -> ComposeResult:
        yield row("Pressure drop", sel(options(DROP_MODES), value="None", allow_blank=False, id="d-mode"))
        yield row("α [1/kg]", inp(placeholder="e.g. 0.0367", id="alpha"), "d-alpha")
        yield row("ρb [kg/m3]", inp(placeholder="bulk density, needed with Ua or homogeneous rates", id="rho_b"),
                  "d-rho_b")
        yield row("Ac [m2]", inp(placeholder="tube cross section, e.g. 0.01414 ft^2", id="Ac"), "d-Ac")
        yield row("Dp [mm]", inp(placeholder="particle diameter, e.g. 0.25 in", id="Dp"), "d-Dp")
        yield row("Porosity φ", inp(placeholder="bed void fraction, e.g. 0.45", id="porosity"), "d-porosity")
        yield row("ρc [kg/m3]", inp(placeholder="catalyst particle density", id="rho_c"), "d-rho_c")
        yield row("μ [Pa·s]", inp(placeholder="gas viscosity, e.g. 0.0673 lb/(ft*h)", id="mu"), "d-mu")
        yield row("ρ0 [kg/m3]", inp(placeholder="inlet gas density", id="rho0"), "d-rho0")

    def on_mount(self) -> None:
        self.refresh_fields()

    def on_select_changed(self, event: Select.Changed) -> None:
        if event.select.id == "d-mode":
            self.refresh_fields()

    def refresh_fields(self) -> None:
        mode = value(self, "#d-mode")
        self.query_one("#d-alpha").display = mode == "α given"
        for name in ("Ac", "Dp", "porosity", "rho_c", "mu", "rho0"):
            self.query_one(f"#d-{name}").display = mode.startswith("Ergun")

    def spec(self) -> DropSpec:
        return DropSpec(*[value(self, f"#{k}") for k in ("d-mode", "alpha", "rho_b", "Ac", "Dp", "porosity",
                                                          "rho_c", "mu", "rho0")])


# Results

PALETTE = [(97, 175, 239), (229, 152, 75), (152, 195, 121), (198, 120, 221), (229, 192, 123), (224, 108, 117),
           (86, 182, 194), (171, 178, 191)]


def plot_text(title: str, x, series: dict, xlabel: str, ylabel: str, width: int) -> Text:
    f = plt.figure
    f.clear()
    f.theme("dark")
    f.plot_size(max(width, 40), 18)
    xs = [float(v) for v in x]
    for i, (name, y) in enumerate(series.items()):
        color = PALETTE[i % len(PALETTE)]
        f.draw(f.signal(xs, [float(v) for v in y], marker=plt.marker("braille", color)).lines().density("full"))
    f.legend(active=False)
    f.label(xlabel, "x")
    f.label(ylabel, "y")
    out = Text.from_ansi(str(f.build()).rstrip())
    if len(series) > 1:
        out.append("\n ")
        for i, name in enumerate(series):
            out.append("━━ ", style="rgb({},{},{})".format(*PALETTE[i % len(PALETTE)]))
            out.append(f"{name}   ")
    return out


def fmt(x) -> str:
    if isinstance(x, (float, int)) and not isinstance(x, bool) and abs(x) < 1e-12:
        return "0"
    if x is None:
        return "-"
    if isinstance(x, str):
        return x
    return f"{x:.4g}"


class ResultsView(VerticalScroll):
    def __init__(self, guide_text: str = "", **kw):
        super().__init__(**kw)
        self._lock = asyncio.Lock()
        self.guide_text = guide_text

    def compose(self) -> ComposeResult:
        yield Markdown(self.guide_text, classes="guide")

    async def guide(self) -> None:
        async with self._lock:
            await self.remove_children()
            await self.mount(Markdown(self.guide_text, classes="guide"))
            self.scroll_home(animate=False)

    async def show(self, rep: Report) -> None:
        async with self._lock:
            await self._show(rep)

    async def _show(self, rep: Report) -> None:
        await self.remove_children()
        widgets: list[Widget] = []
        if rep.summary:
            widgets.append(box("Summary", Static(Text("\n".join(rep.summary)), classes="summary")))
        if rep.drawing:
            widgets.append(box("Flowsheet", Static(Text(rep.drawing), classes="drawing")))
        for title, headers, rows in rep.tables:
            t = DataTable(zebra_stripes=True, cursor_type="row", show_cursor=False)
            t.add_columns(*[Text(h) for h in headers])
            t.add_rows([[Text(fmt(c)) for c in r] for r in rows])
            widgets.append(box(title, t))
        width = max(self.size.width, self.app.size.width // 2) - 8
        for title, x, series, xl, yl in rep.plots:
            widgets.append(box(title, Static(plot_text("", x, series, xl, yl, width), classes="plot")))
        await self.mount_all(widgets)
        self.scroll_home(animate=False)

    async def error(self, msg: str) -> None:
        async with self._lock:
            await self.remove_children()
            await self.mount(box("Error", Static(Text(msg), classes="error-text"), classes="error"))
