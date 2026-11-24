import ast
import copy
import math
from dataclasses import dataclass

import numpy as np
from scipy.optimize import brentq, minimize, minimize_scalar, root
from scipy.stats import qmc

from ..units import Q_
from .graph import Flowsheet
from .spec import SpecError, load, obj, sub

DIMS = {"F": "mol/s", "C": "mol/m^3", "T": "K", "P": "Pa", "v": "m^3/s", "V": "m^3", "W": "kg", "tau": "s",
        "Q": "W", "fed": "mol/s", "products": "mol/s"}
FAILURES = (SpecError, ValueError, RuntimeError, ArithmeticError, KeyError)


@dataclass
class Var:
    path: tuple
    kind: str
    unit: str | None
    guess: float | None
    lo: float | None
    hi: float | None

    @property
    def name(self) -> str:
        return ".".join(map(str, self.path))

    @property
    def log(self) -> bool:
        return self.lo is not None and self.lo > 0 and self.hi is not None

    def spec_value(self, x: float):
        return x if self.unit is None else f"{x:.17g} {self.unit}"

    # Map [0, 1] to the bounds

    def from_unit(self, z):
        if self.log:
            return math.exp(math.log(self.lo) + z * (math.log(self.hi) - math.log(self.lo)))
        return self.lo + z * (self.hi - self.lo)

    def to_unit(self, x):
        if self.log:
            return (math.log(x) - math.log(self.lo)) / (math.log(self.hi) - math.log(self.lo))
        return (x - self.lo) / (self.hi - self.lo)


# Spec walking

def is_var(d) -> bool:
    return isinstance(d, dict) and ("unknown" in d or "vary" in d)


def find_vars(d, path=()) -> list[Var]:
    if is_var(d):
        return [make_var(d, path)]
    items = d.items() if isinstance(d, dict) else enumerate(d) if isinstance(d, list) else ()
    return [v for k, x in items for v in find_vars(x, path + (k,))]


def make_var(d, path) -> Var:
    p = ".".join(map(str, path))
    obj(d, p, {"unknown", "vary", "guess", "min", "max"})
    if ("unknown" in d) == ("vary" in d) or d.get("unknown", d.get("vary")) is not True:
        raise SpecError(p, 'use {"unknown": true} or {"vary": true}')
    raw = {k: d.get(k) for k in ("guess", "min", "max")}
    unit = next((str(Q_(v).units) for v in raw.values() if isinstance(v, str)), None)
    vals = {}
    for k, v in raw.items():
        if v is None:
            vals[k] = None
        elif isinstance(v, bool) or not isinstance(v, (int, float, str)):
            raise SpecError(sub(p, k), "expected a number or a string with units")
        else:
            try:
                vals[k] = float(Q_(v).to(unit).magnitude) if unit else float(v)
            except Exception:
                raise SpecError(sub(p, k), f"cannot read '{v}' in {unit or 'plain numbers'}") from None
    if vals["min"] is not None and vals["max"] is not None and vals["min"] >= vals["max"]:
        raise SpecError(p, "min must be smaller than max")
    return Var(path, "unknown" if "unknown" in d else "vary", unit, vals["guess"], vals["min"], vals["max"])


def put(spec, path, value):
    d = spec
    for k in path[:-1]:
        d = d[k]
    d[path[-1]] = value


# Result expressions

_NODES = (ast.Expression, ast.BinOp, ast.UnaryOp, ast.Constant, ast.Name, ast.Attribute, ast.Subscript,
          ast.Load, ast.operator, ast.unaryop, ast.Call)
_FUNCS = {"sqrt": math.sqrt, "exp": math.exp, "log": math.log, "abs": abs, "min": min, "max": max}
_ROOTS = ("streams", "units", "overall")


class Expr:
    def __init__(self, text, path):
        if not isinstance(text, str):
            raise SpecError(path, "expected an expression such as 'streams.s2.F.B'")
        self.text, self.path = text, path
        try:
            self.tree = ast.parse(text, mode="eval")
        except SyntaxError:
            raise SpecError(path, f"invalid expression '{text}'") from None
        for node in ast.walk(self.tree):
            if not isinstance(node, _NODES):
                raise SpecError(path, f"invalid syntax in '{text}'")
            if isinstance(node, ast.Call) and getattr(node.func, "id", None) not in _FUNCS:
                raise SpecError(path, f"function not allowed in '{text}'")
        body = self.tree.body
        self.dim = None
        if isinstance(body, (ast.Attribute, ast.Subscript, ast.Name)):
            keys = self._keys(body)
            field = keys[2] if keys[0] in ("streams", "units") and len(keys) > 2 else keys[1] if len(keys) > 1 else None
            self.dim = DIMS.get(field)

    def _keys(self, node) -> list:
        if isinstance(node, ast.Name):
            if node.id not in _ROOTS:
                raise SpecError(self.path, f"'{self.text}' must start with one of {list(_ROOTS)}")
            return [node.id]
        if isinstance(node, ast.Attribute):
            return self._keys(node.value) + [node.attr]
        if isinstance(node, ast.Subscript) and isinstance(node.slice, ast.Constant):
            return self._keys(node.value) + [node.slice.value]
        raise SpecError(self.path, f"invalid reference in '{self.text}'")

    def target(self, value, path) -> float:
        if isinstance(value, str):
            if self.dim is None:
                raise SpecError(path, f"'{self.text}' takes a plain number")
            try:
                return float(Q_(value).to(self.dim).magnitude)
            except Exception:
                raise SpecError(path, f"wrong units in '{value}', expected {self.dim}") from None
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise SpecError(path, "expected a number or a string with units")
        return float(value)

    def __call__(self, result) -> float:
        return float(self._eval(self.tree.body, result))

    def _eval(self, node, result):
        if isinstance(node, ast.Constant):
            return node.value
        if isinstance(node, ast.BinOp):
            a, b = self._eval(node.left, result), self._eval(node.right, result)
            ops = {ast.Add: a.__add__, ast.Sub: a.__sub__, ast.Mult: a.__mul__, ast.Div: a.__truediv__,
                   ast.Pow: a.__pow__}
            if type(node.op) not in ops:
                raise SpecError(self.path, f"operator not allowed in '{self.text}'")
            return ops[type(node.op)](b)
        if isinstance(node, ast.UnaryOp):
            v = self._eval(node.operand, result)
            return -v if isinstance(node.op, ast.USub) else v
        if isinstance(node, ast.Call):
            return _FUNCS[node.func.id](*[self._eval(a, result) for a in node.args])
        d = result
        for k in self._keys(node):
            if not isinstance(d, dict) or k not in d:
                raise SpecError(self.path, f"'{k}' not found in result for '{self.text}'")
            d = d[k]
        if not isinstance(d, (int, float)):
            raise SpecError(self.path, f"'{self.text}' is not a number")
        return d


# Solver

class Problem:
    def __init__(self, spec):
        spec = load(spec)
        obj(spec, "", {"species", "thermo", "reactions", "units", "streams", "specs", "constraints", "objective"})
        self.base = {k: v for k, v in spec.items() if k not in ("specs", "constraints", "objective")}
        self.vars = find_vars(self.base)
        self.specs = [self._spec(s, sub("specs", i), ("var", "value")) for i, s in enumerate(spec.get("specs", []))]
        self.constraints = [self._spec(c, sub("constraints", i), ("var",))
                            for i, c in enumerate(spec.get("constraints", []))]
        self.objective = self._objective(spec.get("objective"))
        self.cache, self.error = {}, None
        self._check_dof()

    def _spec(self, d, path, required):
        obj(d, path, {"var", "value", "min", "max"} if "value" not in required else {"var", "value"}, required)
        e = Expr(d["var"], sub(path, "var"))
        out = {"expr": e}
        for k in ("value", "min", "max"):
            if d.get(k) is not None:
                out[k] = e.target(d[k], sub(path, k))
        if len(out) == 1:
            raise SpecError(path, "give min and/or max")
        return out

    def _objective(self, d):
        if d is None:
            return None
        obj(d, "objective", {"maximize", "minimize"})
        if len(d) != 1:
            raise SpecError("objective", 'use {"maximize": "..."} or {"minimize": "..."}')
        (sense, text), = d.items()
        return sense, Expr(text, sub("objective", sense))

    def _check_dof(self):
        unknown = [v for v in self.vars if v.kind == "unknown"]
        vary = [v for v in self.vars if v.kind == "vary"]
        if len(unknown) != len(self.specs):
            raise SpecError("specs", f"{len(unknown)} unknown(s) {[v.name for v in unknown]} need exactly "
                                     f"{len(unknown)} spec(s), got {len(self.specs)}")
        if vary and self.objective is None:
            raise SpecError("objective", f"variables to vary {[v.name for v in vary]} need an objective")
        if self.objective and not vary:
            raise SpecError("objective", 'an objective needs at least one {"vary": true} variable')
        if self.constraints and self.objective is None:
            raise SpecError("constraints", "constraints apply only to optimization (give an objective)")
        if self.objective:
            for v in self.vars:
                if v.lo is None or v.hi is None:
                    raise SpecError(v.name, "optimization needs min and max for every variable")

    def run(self, x) -> dict | None:
        key = tuple(float(v) for v in x)
        if key not in self.cache:
            s = copy.deepcopy(self.base)
            for var, val in zip(self.vars, key):
                put(s, var.path, var.spec_value(val))
            try:
                self.cache[key] = Flowsheet.from_spec(s).simulate()
            except FAILURES as e:
                self.error = e
                self.cache[key] = None
        return self.cache[key]

    def residuals(self, x) -> np.ndarray | None:
        r = self.run(x)
        if r is None:
            return None
        return np.array([(s["expr"](r) - s["value"]) / (abs(s["value"]) or 1.0) for s in self.specs])

    def solve(self) -> dict:
        if not self.vars:
            if self.specs or self.constraints:
                raise SpecError("specs", "specs and constraints need unknown or vary variables")
            r = self.run(())
            if r is None:
                raise self.error
            return {"status": "simulated", "variables": {}, "result": r, "evaluations": 1}
        if self.objective:
            return self._optimize()
        if len(self.vars) == 1:
            return self._scan_root()
        return self._root()

    # Equations

    def _bounds_1d(self, v: Var):
        lo, hi = v.lo, v.hi
        if lo is None or hi is None:
            if v.guess is None or v.guess <= 0:
                raise SpecError(v.name, "give min and max (or a positive guess)")
            lo = v.guess / 1e3 if lo is None else lo
            hi = v.guess * 1e3 if hi is None else hi
        return lo, hi

    def _scan_root(self, n=60):
        v = self.vars[0]
        lo, hi = self._bounds_1d(v)
        xs = np.geomspace(lo, hi, n) if lo > 0 and hi / lo >= 100 else np.linspace(lo, hi, n)
        f = lambda x: self._res1(x)
        fs = [f(x) for x in xs]
        roots = [x for x, y in zip(xs, fs) if y == 0]
        for a, b, fa, fb in zip(xs[:-1], xs[1:], fs[:-1], fs[1:]):
            if np.isfinite(fa) and np.isfinite(fb) and fa * fb < 0:
                x = brentq(lambda z: _finite(f(z), fa), a, b, xtol=1e-13 * max(abs(a), abs(b), 1e-300),
                           rtol=1e-12)
                if abs(f(x)) < 1e-7:
                    roots.append(x)
        roots = sorted(set(roots))
        if not roots:
            ok = [y for y in fs if np.isfinite(y)]
            e = self.specs[0]
            if not ok:
                raise SpecError(v.name, f"simulation failed over [{lo:.4g}, {hi:.4g}]: {self.error}")
            vals = [e["value"] + y * (abs(e["value"]) or 1.0) for y in ok]
            raise SpecError("specs.0", f"no solution for {v.name} in [{lo:.4g}, {hi:.4g}] {v.unit or ''}; "
                                       f"{e['expr'].text} spans {min(vals):.4g} to {max(vals):.4g} there")
        sols = [{v.name: _report_var(v, x)} for x in roots]
        out = self._finish("solved", [roots[0]])
        if len(roots) > 1:
            out["solutions"] = sols
        return out

    def _res1(self, x) -> float:
        r = self.residuals([x])
        return np.nan if r is None else float(r[0])

    def _root(self, starts=16):
        vs = self.vars
        scale = np.array([abs(v.guess) if v.guess else (v.hi - v.lo if v.lo is not None and v.hi is not None
                                                          else 1.0) for v in vs])
        guesses = []
        if all(v.guess is not None for v in vs):
            guesses.append(np.array([v.guess for v in vs]))
        if all(v.lo is not None and v.hi is not None for v in vs):
            guesses.append(np.array([v.from_unit(0.5) for v in vs]))
            for z in qmc.Sobol(len(vs), seed=0).random(starts):
                guesses.append(np.array([v.from_unit(zi) for v, zi in zip(vs, z)]))
        if not guesses:
            raise SpecError(vs[0].name, "give a guess for every unknown, or min and max for all of them")
        big = 1e3 * np.ones(len(vs))
        for g in guesses:
            fun = lambda y: _or(self.residuals(y * scale), big)
            sol = root(fun, g / scale, method="hybr", tol=1e-12)
            x = sol.x * scale
            r = self.residuals(x)
            if r is not None and np.abs(r).max() < 1e-8 and self._inside(x):
                return self._finish("solved", x)
        raise SpecError("specs", f"no solution found for {[v.name for v in vs]}"
                                 + (f" (last error: {self.error})" if self.error else ""))

    def _inside(self, x) -> bool:
        return all((v.lo is None or xi >= v.lo - 1e-9 * abs(v.lo)) and (v.hi is None or xi <= v.hi + 1e-9 * abs(v.hi))
                   for v, xi in zip(self.vars, x))

    # Optimization

    def _optimize(self):
        sense, expr = self.objective
        sign = -1.0 if sense == "maximize" else 1.0
        vs = self.vars
        x_of = lambda z: np.array([v.from_unit(float(np.clip(zi, 0, 1))) for v, zi in zip(vs, z)])

        def f(z):
            r = self.run(x_of(z))
            return np.inf if r is None else sign * expr(r)

        if len(vs) == 1 and not self.specs and not self.constraints:
            zs = np.linspace(0, 1, 81)
            fs = np.array([f([z]) for z in zs])
            if not np.isfinite(fs).any():
                raise SpecError(vs[0].name, f"simulation failed over the whole range: {self.error}")
            k = int(np.nanargmin(np.where(np.isfinite(fs), fs, np.nan)))
            a, b = zs[max(k - 1, 0)], zs[min(k + 1, 80)]
            opt = minimize_scalar(lambda z: _finite(f([z]), fs[k] + 1e30), bounds=(a, b), method="bounded",
                                  options={"xatol": 1e-10})
            z = opt.x if opt.fun <= fs[k] else zs[k]
            return self._finish("optimized", x_of([z]))

        scale = [1.0]
        penalty = 1e6

        def fs(z):
            v = f(z)
            return penalty if not np.isfinite(v) else v / scale[0]

        def eq(z):
            r = self.residuals(x_of(z))
            return np.full(len(self.specs), -1.0) if r is None else r

        def ineq(z):
            r = self.run(x_of(z))
            if r is None:
                return -np.ones(2 * len(self.constraints))
            out = []
            for c in self.constraints:
                val, s = c["expr"](r), abs(c.get("min", 0.0)) or abs(c.get("max", 0.0)) or 1.0
                out += [(val - c["min"]) / s if "min" in c else 1.0, (c["max"] - val) / s if "max" in c else 1.0]
            return np.array(out)

        cons = ([{"type": "eq", "fun": eq}] if self.specs else []) + \
               ([{"type": "ineq", "fun": ineq}] if self.constraints else [])
        starts = [np.array([v.to_unit(v.guess) if v.guess is not None else 0.5 for v in vs])]
        starts += list(qmc.Sobol(len(vs), seed=0).random(4))
        best = None
        for z0 in starts:
            f0 = f(z0)
            scale[0] = abs(f0) if np.isfinite(f0) and f0 != 0 else 1.0
            sol = minimize(fs, z0, method="SLSQP", bounds=[(0, 1)] * len(vs), constraints=cons,
                           options={"ftol": 1e-12, "maxiter": 300})
            z = np.clip(sol.x, 0, 1)
            val = f(z)
            feasible = (not self.specs or np.abs(eq(z)).max() < 1e-6) and \
                       (not self.constraints or ineq(z).min() > -1e-6)
            if np.isfinite(val) and feasible and (best is None or val < best[0]):
                best = (val, z)
        if best is None:
            raise SpecError("objective", "no feasible point found" + (f" (last error: {self.error})"
                                                                      if self.error else ""))
        return self._finish("optimized", x_of(best[1]))

    # Output

    def _finish(self, status, x) -> dict:
        r = self.run(x)
        out = {"status": status, "variables": {v.name: _report_var(v, xi) for v, xi in zip(self.vars, x)},
               "specs": [{"var": s["expr"].text, "target": s["value"], "value": s["expr"](r)} for s in self.specs],
               "result": r, "evaluations": len(self.cache)}
        if self.constraints:
            out["constraints"] = [{"var": c["expr"].text, "min": c.get("min"), "max": c.get("max"),
                                   "value": c["expr"](r)} for c in self.constraints]
        if self.objective:
            sense, expr = self.objective
            out["objective"] = {sense: expr.text, "value": expr(r)}
        return out


def _report_var(v: Var, x) -> dict:
    return {"value": float(x), "unit": v.unit, "kind": v.kind}


def _finite(v, fallback):
    return v if np.isfinite(v) else fallback


def _or(v, fallback):
    return fallback if v is None else v


def solve(spec) -> dict:
    return Problem(spec).solve()
