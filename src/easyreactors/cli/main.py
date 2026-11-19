from rich.markup import escape
from rich.panel import Panel

from . import network, reactors, rtd
from .inputs import Back, select
from .output import console, error

TIPS = ("Plain numbers use the unit shown in brackets; any unit can be typed (2 m3, 25 degC, 0.5 1/h).\n"
        "Arrow keys + Enter to choose  ·  Ctrl+C goes back to the main menu")

ACTIONS = {
    "Single reactor (batch, CSTR, PFR)": reactors.menu,
    "Reactor network": network.menu,
    "RTD analysis and non-ideal models": rtd.menu,
    "Two-parameter models (bypass, dead volume, interchange)": rtd.two_param_menu,
}


def app() -> None:
    console.print(Panel.fit("[bold cyan]EasyReactors[/]\n[dim]Batch · CSTR · PFR · networks · RTD · non-ideal models[/]",
                            border_style="cyan", padding=(0, 2)))
    console.print(f"[dim]{escape(TIPS)}[/]\n")
    while True:
        try:
            choice = select("Main menu", [*ACTIONS, "Quit"])
        except Back:
            return
        if choice == "Quit":
            return
        try:
            ACTIONS[choice]()
        except Back:
            console.print("[yellow]Back to main menu")
        except Exception as e:
            error(str(e) or type(e).__name__)


if __name__ == "__main__":
    app()
