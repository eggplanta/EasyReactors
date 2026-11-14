import numpy as np
import plotext as plt
from rich.console import Console
from rich.markup import escape
from rich.table import Table

console = Console()


def fmt(x, digits: int = 4) -> str:
    if x is None:
        return "-"
    if isinstance(x, str):
        return x
    return f"{x:.{digits}g}"


def table(title: str, headers: list[str], rows: list[list]) -> None:
    t = Table(title=escape(title), title_justify="left", header_style="bold cyan")
    for i, h in enumerate(headers):
        t.add_column(escape(h), justify="left" if i == 0 else "right")
    for r in rows:
        t.add_row(*[fmt(x) for x in r])
    console.print(t)


def plot(x, series: dict[str, np.ndarray], xlabel: str, ylabel: str, title: str = "") -> None:
    f = plt.figure
    f.clear()
    f.theme("simple")
    f.plot_size(min(console.width, 100), 22)
    x = [float(v) for v in x]
    for name, y in series.items():
        sig = f.signal(x, [float(v) for v in y], marker="braille").lines()
        f.draw(sig.label(name) if len(series) > 1 else sig)
    f.title(title)
    f.label(xlabel, "x")
    f.label(ylabel, "y")
    f.show()


def header(text: str) -> None:
    console.rule(f"[bold]{escape(text)}")


def info(text: str) -> None:
    console.print(f"[cyan]{escape(text)}")


def error(text: str) -> None:
    console.print(f"[bold red]Error:[/] {escape(text)}")
