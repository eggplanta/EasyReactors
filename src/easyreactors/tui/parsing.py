import re
from pathlib import Path

import numpy as np
import pint

from ..units import Q_

_NUMBER = re.compile(r"^\s*([-+]?(?:\d+\.?\d*|\.\d+)(?:[eE][-+]?\d+)?)\s*(.*?)\s*$")


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


def amounts(text: str, si_unit: str, default_unit: str) -> dict[str, float]:
    return {s: quantity(v, si_unit, default_unit) for s, v in mapping(text).items()}


def table_text(text: str) -> list[np.ndarray]:
    rows = []
    for line in text.splitlines():
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


def table_file(path: str | Path) -> list[np.ndarray]:
    return table_text(Path(path).expanduser().read_text())


def series(text: str) -> tuple[np.ndarray, np.ndarray]:
    t, y = table_text(text)[:2]
    if len(t) != len(y):
        raise ValueError(f"t has {len(t)} values but C has {len(y)}")
    if np.any(np.diff(t) <= 0):
        raise ValueError("t must be strictly increasing")
    return t, y


# Rate constant units

def _drive(n: float, basis: str) -> str:
    drive = "mol/m^3" if basis == "C" else "Pa"
    return f"({drive})**({n:g})" if n else "dimensionless"


def k_unit(order: float, basis: str = "C") -> str:
    return f"mol/m^3/s/{_drive(order, basis)}" if order else "mol/m^3/s"


def K_unit(orders_f: dict, orders_r: dict, basis: str = "C") -> str:
    return _drive(sum(orders_r.values()) - sum(orders_f.values()), basis)


def orders(text: str) -> dict[str, float]:
    return {s: float(v) for s, v in mapping(text).items()}
