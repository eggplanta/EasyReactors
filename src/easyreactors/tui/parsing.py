import re
from fractions import Fraction
from pathlib import Path

import numpy as np
import pint

from ..units import Q_

_NUM = r"(?:\d+\.?\d*|\.\d+)(?:[eE][-+]?\d+)?"
_NUMBER = re.compile(rf"^\s*([-+]?{_NUM}(?:\s*[*/]\s*{_NUM})*)\s*(.*?)\s*$")


def _product(text: str) -> float:
    parts = re.split(r"\s*([*/])\s*", text)
    x = float(parts[0])
    for op, v in zip(parts[1::2], parts[2::2]):
        x = x * float(v) if op == "*" else x / float(v)
    return x


def quantity(text: str, si_unit: str, default_unit: str | None = None) -> float:
    m = _NUMBER.match(text)
    if not m:
        raise ValueError(f"Not a number with units: '{text}'")
    unit = re.sub(r"([A-Za-z])(-?\d+(?:\.\d+)?)", r"\1**\2", m[2]) or default_unit or si_unit
    if re.search(r"[*/]", unit):
        unit = re.sub(r"(?<!delta_)\b(deg[CF]|°[CF])\b", lambda d: "delta_deg" + d[1][-1], unit)
    try:
        return float(Q_(_product(m[1]), unit).to(si_unit).magnitude)
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


_LABEL = re.compile(r"^\s*[A-Za-z_(][^\s:=]*(\s*\[[^\]]*\])?\s*[:=]?\s*")


def table_text(text: str) -> list[np.ndarray]:
    rows = []
    for line in text.splitlines():
        line = _LABEL.sub("", line)
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


def k_unit(order: float, basis: str = "C", per_mass: bool = False) -> str:
    rate = "mol/kg/s" if per_mass else "mol/m^3/s"
    return f"{rate}/{_drive(order, basis)}" if order else rate


def K_unit(orders_f: dict, orders_r: dict, basis: str = "C") -> str:
    return _drive(sum(orders_r.values()) - sum(orders_f.values()), basis)


def orders(text: str) -> dict[str, float]:
    try:
        return {s: float(Fraction(v.strip())) for s, v in mapping(text).items()}
    except (ValueError, ZeroDivisionError):
        raise ValueError(f"Orders must be numbers or fractions, e.g. A=1, B=1/2: '{text}'")


def human_k(order: float, basis: str = "C", per_mass: bool = False) -> str:
    space = "kg" if per_mass else "L"
    n = round(order, 6)
    if n == 0:
        return f"mol/({space}*min)"
    if basis == "P":
        p = "atm" if n == 1 else f"atm^{n:g}"
        return f"mol/({space}*min*{p})"
    if per_mass:
        return "L/(kg*min)" if n == 1 else f"L^{n:g}/(mol^{n - 1:g}*kg*min)"
    if n == 1:
        return "1/min"
    if n == 2:
        return "L/(mol*min)"
    return f"L^{n - 1:g}/(mol^{n - 1:g}*min)"


def human_K(orders_f: dict, orders_r: dict, basis: str = "C") -> str:
    n = round(sum(orders_r.values()) - sum(orders_f.values()), 6)
    if n == 0:
        return "dimensionless"
    unit = "atm" if basis == "P" else "mol/L"
    return unit if n == 1 else f"({unit})^{n:g}"


def named_table(text: str) -> dict[str, np.ndarray]:
    lines = [l for l in text.splitlines() if l.strip()]
    if len(lines) < 2:
        raise ValueError("Paste a header line with column names and at least one data row")
    names = re.split(r"[,;\t\s]+", lines[0].strip())
    rows = [numbers(l) for l in lines[1:]]
    bad = [i + 2 for i, r in enumerate(rows) if r.size != len(names)]
    if bad:
        raise ValueError(f"Line {bad[0]} does not have {len(names)} values ({', '.join(names)})")
    data = np.array(rows)
    return {n: data[:, i] for i, n in enumerate(names)}
