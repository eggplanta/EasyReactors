from . import network, reactors, rtd
from .inputs import Back, select
from .output import console, error

BANNER = r"""
 ___                ___                 _
| __|__ _ ____  _  | _ \___ __ _ __| |_ ___ _ _ ___
| _|/ _` (_-< || | |   / -_) _` / _|  _/ _ \ '_(_-<
|___\__,_/__/\_, | |_|_\___\__,_\__|\__\___/_| /__/
             |__/
"""

ACTIONS = {
    "Single reactor (batch, CSTR, PFR)": reactors.menu,
    "Reactor network": network.menu,
    "RTD analysis and non-ideal models": rtd.menu,
    "Two-parameter models (bypass, dead volume, interchange)": rtd.two_param_menu,
}


def app() -> None:
    console.print(f"[bold cyan]{BANNER}")
    console.print("Units: plain numbers use the unit in \\[brackets]; any unit can be typed (e.g. 2 m3, 25 degC).")
    console.print("Ctrl+C returns to the main menu.\n")
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
