import ast
import math
from dataclasses import dataclass, field
from typing import Callable, Literal

import sympy

R = 8.314462618
Param = float | Callable[[float], float]
Basis = Literal["C", "P"]


def _val(p: Param, T: float) -> float:
    return p(T) if callable(p) else float(p)


def _driving(C: dict[str, float], T: float, basis: Basis) -> dict[str, float]:
    C = {k: max(v, 0.0) for k, v in C.items()}
    return C if basis == "C" else {k: v * R * T for k, v in C.items()}


def _prod(x: dict[str, float], orders: dict[str, float]) -> float:
    return math.prod(x.get(s, 0.0) ** n for s, n in orders.items())


@dataclass
class Arrhenius:
    A: float
    Ea: float
    T_ref: float | None = None

    @classmethod
    def from_ref(cls, k_ref: float, Ea: float, T_ref: float) -> "Arrhenius":
        return cls(k_ref, Ea, T_ref)

    def __call__(self, T: float) -> float:
        if self.T_ref is None:
            return self.A * math.exp(-self.Ea / (R * T))
        return self.A * math.exp(self.Ea / R * (1 / self.T_ref - 1 / T))


def VantHoff(K_ref: float, dH: float, T_ref: float) -> Arrhenius:
    return Arrhenius.from_ref(K_ref, dH, T_ref)


@dataclass
class PowerLaw:
    k: Param
    orders: dict[str, float]
    basis: Basis = "C"

    def __call__(self, C, T):
        return _val(self.k, T) * _prod(_driving(C, T, self.basis), self.orders)


@dataclass
class Reversible:
    kf: Param
    orders_f: dict[str, float]
    orders_r: dict[str, float]
    K: Param
    basis: Basis = "C"

    def __call__(self, C, T):
        x = _driving(C, T, self.basis)
        return _val(self.kf, T) * (_prod(x, self.orders_f) - _prod(x, self.orders_r) / _val(self.K, T))


@dataclass
class LHHW:
    k: Param
    orders: dict[str, float]
    adsorption: dict[str, Param]
    n: float = 1.0
    K_eq: Param | None = None
    orders_r: dict[str, float] = field(default_factory=dict)
    basis: Basis = "P"

    def __call__(self, C, T):
        x = _driving(C, T, self.basis)
        num = _prod(x, self.orders)
        if self.K_eq is not None:
            num -= _prod(x, self.orders_r) / _val(self.K_eq, T)
        den = 1 + sum(_val(K, T) * x.get(s, 0.0) for s, K in self.adsorption.items())
        return _val(self.k, T) * num / den**self.n


@dataclass
class MichaelisMenten:
    vmax: Param
    Km: Param
    substrate: str

    def __call__(self, C, T):
        S = max(C.get(self.substrate, 0.0), 0.0)
        return _val(self.vmax, T) * S / (_val(self.Km, T) + S)


_ALLOWED_FUNCS = {"exp", "log", "sqrt", "Abs", "Max", "Min"}
_ALLOWED_NODES = (ast.Expression, ast.BinOp, ast.UnaryOp, ast.Call, ast.Name,
                  ast.Load, ast.Constant, ast.operator, ast.unaryop)


class Expression:
    def __init__(self, expr: str, **params: Param):
        tree = ast.parse(expr.replace("^", "**"), mode="eval")
        for node in ast.walk(tree):
            if not isinstance(node, _ALLOWED_NODES):
                raise ValueError(f"Invalid syntax in rate expression: {expr}")
            if isinstance(node, ast.Call) and getattr(node.func, "id", None) not in _ALLOWED_FUNCS:
                raise ValueError(f"Function not allowed: {ast.unparse(node.func)}")
        symbols = {n.id: sympy.Symbol(n.id) for n in ast.walk(tree)
                   if isinstance(n, ast.Name) and n.id not in _ALLOWED_FUNCS}
        self.expr = sympy.sympify(ast.unparse(tree), locals=symbols)
        self.params = params
        names = sorted(str(s) for s in self.expr.free_symbols)
        unknown = [n for n in names if not (n.startswith(("C_", "p_")) or n == "T" or n in params)]
        if unknown:
            raise ValueError(f"Undefined symbols: {unknown}")
        self._names = names
        self._f = sympy.lambdify([sympy.Symbol(n) for n in names], self.expr, "math")

    def __call__(self, C, T):
        args = []
        for n in self._names:
            if n == "T":
                args.append(T)
            elif n.startswith("C_"):
                args.append(max(C.get(n[2:], 0.0), 0.0))
            elif n.startswith("p_"):
                args.append(max(C.get(n[2:], 0.0), 0.0) * R * T)
            else:
                args.append(_val(self.params[n], T))
        return float(self._f(*args))
