import plotext as plt
from rich.text import Text
from textual.app import ComposeResult
from textual.containers import Horizontal, Vertical, VerticalScroll
from textual.widget import Widget
from textual.widgets import Button, DataTable, Input, Label, Select, Static, TextArea

from .compute import MODELS, Report, ReactionSpec, build_feed


def inp(value: str = "", **kw) -> Input:
    return Input(value, compact=True, **kw)


def sel(choices, **kw) -> Select:
    return Select(choices, compact=True, **kw)


def area(text: str = "", **kw) -> TextArea:
    return TextArea(text, compact=True, **kw)


def options(values: list[str]) -> list[tuple[str, str]]:
    return [(v, v) for v in values]


def row(label: str, widget: Widget, id: str | None = None) -> Horizontal:
    return Horizontal(Label(label, classes="lbl"), widget, classes="row", id=id)


def box(title: str, *children: Widget, id: str | None = None, classes: str = "") -> Vertical:
    v = Vertical(*children, classes=f"box {classes}", id=id)
    v.border_title = title
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

    def __init__(self, spec: ReactionSpec | None = None):
        super().__init__(classes="box reaction")
        self.spec0 = spec or ReactionSpec("")
        self.border_title = "Reaction"

    def compose(self) -> ComposeResult:
        s = self.spec0
        yield row("Equation", inp(s.equation, placeholder="2A + B > C   (A <> B reversible)", id="eq"))
        yield row("Rate law", sel(options(MODELS), value=s.model, allow_blank=False, id="model"))
        yield row("-r of", inp(s.ref, placeholder="species the rate refers to (blank = auto)", id="ref"), "r-ref")
        yield row("Orders", inp(s.orders, placeholder="A=1, B=2 (blank = stoichiometric)", id="orders"), "r-orders")
        yield row("Reverse orders", inp(s.reverse_orders, placeholder="C=1 (blank = stoichiometric)",
                                          id="rorders"), "r-rorders")
        yield row("k", inp(s.k, placeholder="0.015 dm3/(mol*min)  ·  0.5 1/h", id="k"), "r-k")
        yield row("K", inp(s.K, placeholder="equilibrium constant with units", id="K"), "r-K")
        yield row("Ea", inp(s.Ea, placeholder="optional, e.g. 82 kcal/mol", id="Ea"), "r-Ea")
        yield row("T of k", inp(s.T_ref, placeholder="needed with Ea [K]", id="Tref"), "r-Tref")
        yield row("Basis", sel([("Concentration C", "C"), ("Partial pressure p", "P")], value=s.basis,
                                  allow_blank=False, id="basis"), "r-basis")
        yield row("-r =", inp(s.expr, placeholder="k*C_A^2/(1 + K*C_A)", id="expr"), "r-expr")
        yield row("Parameters", inp(s.params, placeholder="k=0.01, K=2  (SI)", id="params"), "r-params")
        yield Button("✕ Remove", variant="error", classes="small", id="remove", compact=True)

    def on_mount(self) -> None:
        self.refresh_fields()

    def on_select_changed(self, event: Select.Changed) -> None:
        if event.select.id == "model":
            self.refresh_fields()

    def refresh_fields(self) -> None:
        model = value(self, "#model")
        shown = self.FIELDS.get(model, [])
        labels = self.LABELS.get(model, {})
        for name in ["ref", "orders", "rorders", "k", "K", "Ea", "Tref", "basis", "expr", "params"]:
            r = self.query_one(f"#r-{name}")
            r.display = name in shown
            if name in ("k", "K"):
                r.query_one(Label).update(labels.get(name, name))

    def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id == "remove":
            event.stop()
            self.remove()

    def spec(self) -> ReactionSpec:
        return ReactionSpec(value(self, "#eq"), value(self, "#model"), value(self, "#ref"), value(self, "#orders"),
                            value(self, "#rorders"), value(self, "#k"), value(self, "#K"), value(self, "#Ea"),
                            value(self, "#Tref"), value(self, "#basis"), value(self, "#expr"),
                            value(self, "#params"))


class ReactionsEditor(Vertical):
    def compose(self) -> ComposeResult:
        yield Vertical(ReactionRow(), id="rows")
        yield Button("+ Add reaction", id="add", classes="small", variant="success", compact=True)

    def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id == "add":
            event.stop()
            self.query_one("#rows").mount(ReactionRow())

    def specs(self) -> list[ReactionSpec]:
        return [r.spec() for r in self.query(ReactionRow)]

    async def load(self, specs: list[ReactionSpec]) -> None:
        rows = self.query_one("#rows")
        await rows.remove_children()
        await rows.mount_all([ReactionRow(s) for s in specs])


# Feed

class FeedEditor(Vertical):
    def __init__(self, batch: bool = False, **kw):
        super().__init__(**kw)
        self.batch = batch

    def compose(self) -> ComposeResult:
        yield row("Phase", sel(options(["Liquid", "Gas"]), value="Liquid", allow_blank=False, id="phase"), "f-phase")
        yield row("T", inp("300", placeholder="[K] or 25 degC", id="T"))
        yield row("P", inp(placeholder="[atm] (gas)", id="P"), "f-P")
        yield row("v0", inp(placeholder="[L/min]", id="v0"), "f-v0")
        yield row("C0", inp(placeholder="A=2, B=0.5  [mol/L]", id="comp"), "f-comp")

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
        self.query_one("#f-comp").query_one(Label).update("F0" if gas else "C0")
        self.query_one("#comp").placeholder = "A=10, I=5  [mol/min]" if gas else "A=2, B=0.5  [mol/L]"

    def stream(self):
        return build_feed(value(self, "#phase"), value(self, "#T"), value(self, "#P"), value(self, "#v0"),
                          value(self, "#comp"))

    def set(self, **values: str) -> None:
        for k, v in values.items():
            w = self.query_one(f"#{k}")
            w.value = v


# Results

def plot_text(title: str, x, series: dict, xlabel: str, ylabel: str, width: int) -> Text:
    f = plt.figure
    f.clear()
    f.theme("dark")
    f.plot_size(max(width, 40), 20)
    xs = [float(v) for v in x]
    for name, y in series.items():
        sig = f.signal(xs, [float(v) for v in y], marker="braille").lines()
        f.draw(sig.label(name) if len(series) > 1 else sig)
    f.title(title)
    f.label(xlabel, "x")
    f.label(ylabel, "y")
    return Text.from_ansi(str(f.build()))


def fmt(x) -> str:
    if x is None:
        return "-"
    if isinstance(x, str):
        return x
    return f"{x:.4g}"


class ResultsView(VerticalScroll):
    def compose(self) -> ComposeResult:
        yield Static("Fill in the form and press Run (Ctrl+R).", classes="hint")

    async def show(self, rep: Report) -> None:
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
        await self.remove_children()
        await self.mount(box("Error", Static(Text(msg), classes="error-text"), classes="error"))
