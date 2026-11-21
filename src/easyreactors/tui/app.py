from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.widgets import Footer, Header, TabbedContent, TabPane

from .panes import DiffusionPane, NetworkPane, Pane, ReactorPane, RTDPane, TransientPane, TwoParamPane

CSS = """
Screen { background: $surface; }
.left { width: 1fr; min-width: 50; }
.actions { height: 1; margin: 1 1 0 1; }
.actions Button { margin-right: 2; min-width: 10; }
.form { padding: 0 1; scrollbar-size-vertical: 1; }
ResultsView { width: 1fr; padding: 0 1; border-left: tall $primary 30%; scrollbar-size-vertical: 1; }
.guide { margin: 0 1; }
.box { height: auto; border: round $primary 50%; border-title-color: $accent; border-title-style: bold;
       padding: 0 1; margin: 1 0 0 0; }
.box.reaction { border: round $secondary 50%; margin: 0; }
.box.error { border: round $error; }
.row { height: auto; margin: 0; }
.lbl { width: 14; padding: 0 1 0 0; color: $text-muted; }
.row Input, .row Select, .row TextArea { width: 1fr; }
.row #path { width: 1fr; }
.multi { height: auto; width: 1fr; }
.multi Input { width: 1fr; margin-right: 1; }
TextArea { height: 6; margin: 0; }
TextArea.short { height: 4; }
.small { min-width: 8; margin: 1 1 0 0; }
#remove { margin: 1 0 0 0; }
#add { margin: 1 0 0 0; }
.palette { height: auto; margin: 0; }
.box Input, .box Select, .box TextArea { background: $boost; }
Input > .input--placeholder { text-style: italic; color: $text-disabled; }
.hint { color: $text-muted; padding: 0; }
.rhint { color: $text-muted; padding: 0 0 0 14; height: auto; }
.rhint.bad { color: $error; }
.summary { padding: 0 0 0 1; }
.drawing, .plot { width: auto; text-wrap: nowrap; text-overflow: clip; }
.sketch { margin: 1 0 0 0; color: $accent; height: auto; overflow-x: auto; }
.sketch.bad { color: $error; }
.error-text { color: $error; }
DataTable { height: auto; max-height: 30; }
ReactionsEditor, #rows, ReactionRow, FeedEditor, EnergyEditor, DropEditor { height: auto; }
ReactionsEditor { margin: 0 0 1 0; }
"""


class EasyReactors(App):
    TITLE = "EasyReactors"
    SUB_TITLE = "chemical reactor design"
    CSS = CSS
    BINDINGS = [Binding("ctrl+r", "run", "Run"), Binding("ctrl+e", "example", "Example"),
                Binding("f1", "guide", "Guide"), Binding("ctrl+q", "quit", "Quit")]

    def compose(self) -> ComposeResult:
        yield Header()
        with TabbedContent():
            with TabPane("Reactor", id="tab-reactor"):
                yield ReactorPane()
            with TabPane("Network", id="tab-network"):
                yield NetworkPane()
            with TabPane("Semibatch / Start-up", id="tab-trans"):
                yield TransientPane()
            with TabPane("RTD", id="tab-rtd"):
                yield RTDPane()
            with TabPane("Two-parameter models", id="tab-two"):
                yield TwoParamPane()
            with TabPane("Diffusion", id="tab-diff"):
                yield DiffusionPane()
        yield Footer()

    def pane(self) -> Pane:
        return self.query_one(TabbedContent).active_pane.query_one(Pane)

    async def action_run(self) -> None:
        await self.pane().run_job()

    async def action_example(self) -> None:
        await self.pane().example()

    async def action_guide(self) -> None:
        await self.pane().results.guide()


def main() -> None:
    EasyReactors().run()


if __name__ == "__main__":
    main()
