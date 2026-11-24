import json

import pint

from ..core import LHHW, Arrhenius, Expression, MichaelisMenten, PowerLaw, Reaction, Reversible, VantHoff
from ..units import Q_, to_si


class SpecError(ValueError):
    def __init__(self, path: str, message: str):
        super().__init__(f"{path}: {message}" if path else message)
        self.path = path
        self.message = message


def load(spec) -> dict:
    if isinstance(spec, str):
        try:
            spec = json.loads(spec)
        except json.JSONDecodeError as e:
            raise SpecError("", f"invalid JSON: {e}") from None
    return obj(spec, "")


# Generic checks

def obj(d, path: str, allowed=None, required=()) -> dict:
    if not isinstance(d, dict):
        raise SpecError(path, "expected an object")
    if allowed is not None:
        extra = [k for k in d if k not in allowed and k != "ui"]
        if extra:
            raise SpecError(path, f"unknown field(s) {extra}; allowed: {sorted(allowed)}")
    missing = [k for k in required if d.get(k) is None]
    if missing:
        raise SpecError(f"{path}.{missing[0]}" if path else missing[0], "missing value")
    return d


def sub(path: str, key) -> str:
    return f"{path}.{key}" if path else str(key)


def quantity(value, si_unit: str, path: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float, str)):
        raise SpecError(path, f"expected a number or a string with units ({si_unit})")
    try:
        return to_si(value, si_unit)
    except pint.DimensionalityError:
        raise SpecError(path, f"wrong units in '{value}', expected {si_unit}") from None
    except Exception:
        raise SpecError(path, f"cannot read quantity '{value}'") from None


def base(value, path: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float, str)):
        raise SpecError(path, "expected a number or a string with units")
    if isinstance(value, str):
        try:
            return float(Q_(value).to_base_units().magnitude)
        except Exception:
            raise SpecError(path, f"cannot read quantity '{value}'") from None
    return float(value)


def number(value, path: str, lo=None, hi=None) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise SpecError(path, "expected a number")
    if (lo is not None and value < lo) or (hi is not None and value > hi):
        raise SpecError(path, f"must be within [{lo}, {hi}]")
    return float(value)


def amounts(d, si_unit: str, path: str) -> dict[str, float]:
    return {s: quantity(v, si_unit, sub(path, s)) for s, v in obj(d, path).items()}


# Kinetics

def param(value, path: str):
    if isinstance(value, dict):
        if "Ea" in value:
            obj(value, path, {"A", "Ea", "T_ref"}, ("A", "Ea"))
            T_ref = None if value.get("T_ref") is None else quantity(value["T_ref"], "K", sub(path, "T_ref"))
            return Arrhenius(base(value["A"], sub(path, "A")), quantity(value["Ea"], "J/mol", sub(path, "Ea")),
                             T_ref)
        obj(value, path, {"K_ref", "dH", "T_ref"}, ("K_ref", "dH", "T_ref"))
        return VantHoff(base(value["K_ref"], sub(path, "K_ref")), quantity(value["dH"], "J/mol", sub(path, "dH")),
                        quantity(value["T_ref"], "K", sub(path, "T_ref")))
    return base(value, path)


def orders(d, path: str) -> dict[str, float]:
    return {s: number(v, sub(path, s)) for s, v in obj(d, path).items()}


def basis(value, path: str) -> str:
    if value not in ("C", "P"):
        raise SpecError(path, "basis must be 'C' (concentration) or 'P' (partial pressure)")
    return value


def params(d, path: str) -> dict:
    return {k: param(v, sub(path, k)) for k, v in obj(d, path).items()}


RATE_FIELDS = {
    "power": ({"model", "k", "orders", "basis"}, ("k", "orders")),
    "reversible": ({"model", "kf", "orders_f", "orders_r", "K", "basis"}, ("kf", "orders_f", "orders_r", "K")),
    "lhhw": ({"model", "k", "orders", "adsorption", "n", "K_eq", "orders_r", "basis"}, ("k", "orders", "adsorption")),
    "michaelis": ({"model", "vmax", "Km", "substrate"}, ("vmax", "Km", "substrate")),
    "expression": ({"model", "expr", "params"}, ("expr",)),
}


def rate(d, path: str):
    obj(d, path, required=("model",))
    model = d["model"]
    if model not in RATE_FIELDS:
        raise SpecError(sub(path, "model"), f"unknown rate model '{model}'; use one of {sorted(RATE_FIELDS)}")
    obj(d, path, *RATE_FIELDS[model])
    p = lambda k: sub(path, k)
    b = basis(d.get("basis", "P" if model == "lhhw" else "C"), p("basis"))
    if model == "power":
        return PowerLaw(param(d["k"], p("k")), orders(d["orders"], p("orders")), b)
    if model == "reversible":
        return Reversible(param(d["kf"], p("kf")), orders(d["orders_f"], p("orders_f")),
                          orders(d["orders_r"], p("orders_r")), param(d["K"], p("K")), b)
    if model == "lhhw":
        K_eq = None if d.get("K_eq") is None else param(d["K_eq"], p("K_eq"))
        return LHHW(param(d["k"], p("k")), orders(d["orders"], p("orders")), params(d["adsorption"], p("adsorption")),
                    number(d.get("n", 1.0), p("n")), K_eq, orders(d.get("orders_r", {}), p("orders_r")), b)
    if model == "michaelis":
        return MichaelisMenten(param(d["vmax"], p("vmax")), param(d["Km"], p("Km")), str(d["substrate"]))
    try:
        return Expression(str(d["expr"]), **params(d.get("params", {}), p("params")))
    except ValueError as e:
        raise SpecError(p("expr"), str(e)) from None


def reaction(name: str, d, path: str) -> Reaction:
    obj(d, path, {"equation", "rate", "ref", "catalytic", "dH"}, ("equation", "rate"))
    try:
        return Reaction(str(d["equation"]), rate(d["rate"], sub(path, "rate")), d.get("ref"),
                        bool(d.get("catalytic", False)), name)
    except SpecError:
        raise
    except ValueError as e:
        raise SpecError(sub(path, "equation"), str(e)) from None
