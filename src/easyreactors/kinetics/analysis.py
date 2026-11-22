import math
from dataclasses import dataclass, field

import numpy as np
from scipy.optimize import least_squares

from ..core import Expression
from ..core.rates import R


def _r2(y: np.ndarray, fit: np.ndarray) -> float:
    ss = float(np.sum((y - np.mean(y)) ** 2))
    return 1 - float(np.sum((y - fit) ** 2)) / ss if ss > 0 else 1.0


def _line(x: np.ndarray, y: np.ndarray) -> tuple[float, float, float]:
    slope, intercept = np.polyfit(x, y, 1)
    return float(slope), float(intercept), _r2(y, slope * x + intercept)


def _arrays(*a):
    out = [np.asarray(v, float) for v in a]
    if len({len(v) for v in out}) > 1:
        raise ValueError("Data columns have different lengths")
    if len(out[0]) < 3:
        raise ValueError("Need at least 3 data points")
    return out


# Derivatives of C(t)

def finite_differences(t, C) -> np.ndarray:
    t, C = _arrays(t, C)
    n = len(t)
    d = np.empty(n)
    for i in range(n):
        j = min(max(i - 1, 0), n - 3)
        x, y = t[j:j + 3], C[j:j + 3]
        x0 = t[i]
        d[i] = sum(y[a] * math.prod(x0 - x[c] for c in range(3) if c not in (a, b)) /
                   math.prod(x[a] - x[c] for c in range(3) if c != a)
                   for a in range(3) for b in range(3) if b != a)
    return d


def polynomial_derivative(t, C, degree: int = 3) -> np.ndarray:
    t, C = _arrays(t, C)
    degree = min(degree, len(t) - 1)
    p = np.polynomial.Polynomial.fit(t, C, degree)
    return p.deriv()(t)


# Differential method

@dataclass
class OrderFit:
    order: float
    k: float
    r2: float
    method: str
    t: np.ndarray = field(repr=False)
    C: np.ndarray = field(repr=False)
    rate: np.ndarray | None = field(default=None, repr=False)

    def model(self, t: np.ndarray) -> np.ndarray:
        return integrated_conc(t, self.C[0], self.k, self.order)


def differential(t, C, method: str = "finite", degree: int = 3, order: float | None = None) -> OrderFit:
    t, C = _arrays(t, C)
    rate = -(finite_differences(t, C) if method == "finite" else polynomial_derivative(t, C, degree))
    ok = (rate > 0) & (C > 0)
    if ok.sum() < 2:
        raise ValueError("The rate -dC/dt must be positive: is C decreasing with t?")
    x, y = np.log(C[ok]), np.log(rate[ok])
    if order is None:
        order, lnk, r2 = _line(x, y)
    else:
        lnk = float(np.mean(y - order * x))
        r2 = _r2(y, lnk + order * x)
    return OrderFit(float(order), math.exp(lnk), r2, f"differential ({method})", t, C, rate)


# Integral method

def integrated_conc(t, C0: float, k: float, n: float) -> np.ndarray:
    t = np.asarray(t, float)
    if abs(n - 1) < 1e-12:
        return C0 * np.exp(-k * t)
    base = C0 ** (1 - n) + (n - 1) * k * t
    with np.errstate(invalid="ignore"):
        return np.where(base > 0, np.abs(base) ** (1 / (1 - n)), 0.0)


def linearized(t, C, n: float) -> tuple[np.ndarray, str]:
    C = np.asarray(C, float)
    if n == 0:
        return C[0] - C, "C0 - C"
    if n == 1:
        return np.log(C[0] / C), "ln(C0/C)"
    return (C ** (1 - n) - C[0] ** (1 - n)) / (n - 1), f"(C^{1 - n:g} - C0^{1 - n:g})/{n - 1:g}"


def integral(t, C, order: float) -> OrderFit:
    t, C = _arrays(t, C)
    y, _ = linearized(t, C, order)
    k = float(np.sum(t * y) / np.sum(t * t))
    return OrderFit(order, k, _r2(y, k * t), "integral", t, C)


def integral_scan(t, C, orders=(0, 1, 2)) -> list[OrderFit]:
    return [integral(t, C, n) for n in orders]


# Nonlinear regression (Fogler 7-5)

def integrated_time(C, C0: float, k: float, n: float) -> np.ndarray:
    C = np.asarray(C, float)
    if abs(n - 1) < 1e-12:
        return np.log(C0 / C) / k
    return (C0 ** (1 - n) - C ** (1 - n)) / (k * (1 - n))


def nonlinear(t, C, order: float | None = None, guess: OrderFit | None = None) -> OrderFit:
    t, C = _arrays(t, C)
    g = guess or differential(t, C)
    C0 = C[0]
    if order is None:
        f = lambda p: integrated_time(C[1:], C0, math.exp(p[0]), p[1]) - t[1:]
        sol = least_squares(f, [math.log(g.k), g.order], x_scale="jac", xtol=1e-14, ftol=1e-14)
        k, n = math.exp(sol.x[0]), float(sol.x[1])
    else:
        n = order
        f = lambda p: integrated_time(C[1:], C0, math.exp(p[0]), n) - t[1:]
        sol = least_squares(f, [math.log(g.k * C0 ** (g.order - n))], xtol=1e-14, ftol=1e-14)
        k = math.exp(sol.x[0])
    tc = integrated_time(C[1:], C0, k, n)
    return OrderFit(n, k, _r2(t[1:], tc), "nonlinear regression", t, C)


# Arrhenius

@dataclass
class ArrheniusFit:
    Ea: float
    A: float
    r2: float


def arrhenius(T, k) -> ArrheniusFit:
    T, k = _arrays(T, k)
    if (k <= 0).any() or (T <= 0).any():
        raise ValueError("k and T must be positive")
    slope, intercept, r2 = _line(1 / T, np.log(k))
    return ArrheniusFit(-slope * R, math.exp(intercept), r2)


# Rate data (differential reactor, initial rates)

@dataclass
class RateFit:
    params: dict[str, float]
    r2: float
    predicted: np.ndarray
    names: list[str]


def power_law(columns: dict[str, np.ndarray], rate) -> RateFit:
    rate = np.asarray(rate, float)
    names = list(columns)
    X = np.column_stack([np.ones(len(rate))] + [np.log(np.asarray(columns[s], float)) for s in names])
    if (rate <= 0).any():
        raise ValueError("Rates must be positive for a power-law fit")
    coef, *_ = np.linalg.lstsq(X, np.log(rate), rcond=None)
    params = {"k": math.exp(coef[0])} | {f"order_{s}": float(c) for s, c in zip(names, coef[1:])}
    pred = np.exp(X @ coef)
    return RateFit(params, _r2(np.log(rate), np.log(pred)), pred, ["k"] + [f"order_{s}" for s in names])


def fit_expression(expr: str, columns: dict[str, np.ndarray], rate, guess: dict[str, float],
                   fixed: dict[str, float] | None = None) -> RateFit:
    rate = np.asarray(rate, float)
    fixed = fixed or {}
    names = [n for n in guess if n not in fixed]
    if not names:
        raise ValueError("Give initial guesses for the parameters to fit")
    bad = [s for s in columns if not s.startswith(("C_", "p_"))]
    if bad:
        raise ValueError(f"Name data columns C_A or p_A, not: {', '.join(bad)}")
    model = Expression(expr, **{n: 1.0 for n in guess} | fixed)
    rows = [{s[2:]: float(v[i]) for s, v in columns.items()} for i in range(len(rate))]
    T = 1 / R

    def predict(p):
        model.params.update(dict(zip(names, p)))
        return np.array([model(C, T) for C in rows])

    sol = least_squares(lambda p: (predict(p) - rate) / np.abs(rate).max(), [guess[n] for n in names],
                        x_scale="jac", xtol=1e-14, ftol=1e-14)
    pred = predict(sol.x)
    return RateFit(dict(zip(names, map(float, sol.x))) | fixed, _r2(rate, pred), pred, names)
