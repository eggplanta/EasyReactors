import re
from pathlib import Path
from typing import Callable

import numpy as np
import pint
import questionary

from ..units import Q_

_NUMBER = re.compile(r"^\s*([-+]?(?:\d+\.?\d*|\.\d+)(?:[eE][-+]?\d+)?)\s*(.*?)\s*$")


STYLE = questionary.Style([
    ("qmark", "fg:#00afaf bold"),
    ("question", "bold"),
    ("answer", "fg:#5fd75f bold"),
    ("pointer", "fg:#00afaf bold"),
    ("highlighted", "fg:#00afaf bold"),
    ("selected", "fg:#5fd75f"),
    ("instruction", "fg:#808080 italic"),
])
UI = {"qmark": "›", "style": STYLE}


class Back(Exception):
    pass


# Pure parsers

def quantity(text: str, si_unit: str, default_unit: str | None = None) -> float:
    m = _NUMBER.match(text)
    if not m:
        raise ValueError(f"Not a number with units: '{text}'")
    unit = re.sub(r"([A-Za-z])(-?\d+(?:\.\d+)?)", r"\1**\2", m[2]) or default_unit or si_unit
    try:
        return float(Q_(float(m[1]), unit).to(si_unit).magnitude)
    except pint.DimensionalityError:
        raise ValueError(f"Wrong units '{m[2] or unit}', expected something like {default_unit or si_unit}")
    except Exception:
        raise ValueError(f"Unknown unit '{m[2] or unit}'")


def numbers(text: str) -> np.ndarray:
    text = text.strip()
    if ";" in text or "\t" in text:
        tokens = re.split(r"[;\t\s]+", text.replace(",", "."))
    else:
        tokens = re.split(r"[,\s]+", text)
    try:
        return np.array([float(t) for t in tokens if t])
    except ValueError as e:
        raise ValueError(f"Invalid number list: {e}")


def mapping(text: str) -> dict[str, str]:
    out = {}
    for part in re.split(r"[,;]", text):
        if not part.strip():
            continue
        m = re.match(r"^\s*([^=:\s]+)\s*[=:]\s*(.+?)\s*$", part)
        if not m:
            raise ValueError(f"Expected 'name = value', got '{part.strip()}'")
        out[m[1]] = m[2]
    if not out:
        raise ValueError("Empty list")
    return out


def table_file(path: str | Path) -> list[np.ndarray]:
    rows = []
    for line in Path(path).expanduser().read_text().splitlines():
        try:
            row = numbers(line)
        except ValueError:
            continue
        if row.size:
            rows.append(row)
    if not rows:
        raise ValueError("No numeric data found")
    if len(rows) == 2 and rows[0].size > 2:
        return rows
    width = min(r.size for r in rows)
    if width < 2:
        raise ValueError("Need at least two columns (t and C)")
    return list(np.array([r[:width] for r in rows]).T)


def checked(parser: Callable) -> Callable[[str], bool | str]:
    def validate(text: str):
        try:
            parser(text)
            return True
        except Exception as e:
            return str(e) or type(e).__name__
    return validate


# Interactive prompts

def _ask(question):
    try:
        answer = question.unsafe_ask()
    except KeyboardInterrupt:
        raise Back
    if answer is None:
        raise Back
    return answer


def ask(message: str, parser: Callable = str, default: str = ""):
    return parser(_ask(questionary.text(message, default=default, validate=checked(parser), **UI)))


def ask_quantity(message: str, si_unit: str, default_unit: str | None = None, default: str = "") -> float:
    hint = f" [{default_unit}]" if default_unit else " [SI if no unit]"
    return ask(message + hint, lambda s: quantity(s, si_unit, default_unit), default)


def ask_float(message: str, default: str = "", lo: float | None = None, hi: float | None = None) -> float:
    def parse(s):
        x = float(s)
        if (lo is not None and x < lo) or (hi is not None and x > hi):
            raise ValueError(f"Must be between {lo} and {hi}")
        return x
    return ask(message, parse, default)


def select(message: str, choices: list[str], default: str | None = None) -> str:
    return _ask(questionary.select(message, choices=choices, default=default, **UI))


def confirm(message: str, default: bool = True) -> bool:
    return _ask(questionary.confirm(message, default=default, **UI))
