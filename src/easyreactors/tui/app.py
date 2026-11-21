from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.widgets import Footer, Header, TabbedContent, TabPane

from .panes import NetworkPane, Pane, ReactorPane, RTDPane, TwoParamPane

CSS = """
Screen { background: $surface; }
.form { width: 1fr; min-width: 50; padding: 0 1; }
ResultsView { width: 1fr; padding: 0 1; border-left: tall $primary 30%; }
.box { height: auto; border: round $primary 60%; border-title-color: $accent; border-title-style: bold;
       padding: 0 1; margin: 1 0 0 0; }
.box.reaction { border: round $secondary 60%; margin: 0 0 1 0; }
.box.error { border: round $error; }
.row { height: auto; margin: 0 0 1 0; }
.lbl { width: 15; padding: 0 1 0 0; color: $text-muted; }
.row Input, .row Select, .row TextArea { width: 1fr; }
.row #path { width: 1fr; }
TextArea { height: 6; margin: 0 0 1 0; }
TextArea.short { height: 4; }
.small { min-width: 8; margin: 0 1 1 0; }
.palette { height: auto; margin: 1 0 0 0; }
.box Input, .box Select, .box TextArea { background: $boost; }
.actions { height: auto; margin: 1 0; }
.actions Button { margin-right: 2; }
.hint { color: $text-muted; padding: 1 0 0 0; }
.summary { padding: 0 0 0 1; }
.drawing, .plot { width: auto; }
.error-text { color: $error; }
DataTable { height: auto; max-height: 30; }
"""


class EasyReactors(App):
    TITLE = "EasyReactors"
    SUB_TITLE = "chemical reactor design"
    CSS = CSS
    BINDINGS = [Binding("ctrl+r", "run", "Run"), Binding("ctrl+e", "example", "Example"),
                Binding("ctrl+q", "quit", "Quit")]

    def compose(self) -> ComposeResult:
        yield Header()
        with TabbedContent():
            with TabPane("Reactor", id="tab-reactor"):
                yield ReactorPane()
            with TabPane("Network", id="tab-network"):
                yield NetworkPane()
            with TabPane("RTD", id="tab-rtd"):
                yield RTDPane()
            with TabPane("Two-parameter models", id="tab-two"):
                yield TwoParamPane()
        yield Footer()

    def pane(self) -> Pane:
        return self.query_one(TabbedContent).active_pane.query_one(Pane)

    async def action_run(self) -> None:
        await self.pane().run_job()

    async def action_example(self) -> None:
        await self.pane().example()


def main() -> None:
    EasyReactors().run()


if __name__ == "__main__":
    main()
