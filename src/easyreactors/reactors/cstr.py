import warnings
from dataclasses import dataclass, replace

import numpy as np
from scipy.optimize import brentq, minimize_scalar, root
from scipy.stats import qmc

from ..core import ReactionSet
from ..energy.heat import Heat
from ..units import Value, to_si
from ._common import flow_setup, key_species, solve
from .result import Result
from .stream import Stream


@dataclass
class SteadyStates:
    states: list[Result]
    stable: list[bool]
    x: np.ndarray | None
    G: np.ndarray | None
    R: np.ndarray | None
    axis: str = "T"

    def best(self, key: str) -> Result:
        pool = [s for s, ok in zip(self.states, self.stable) if ok] or self.states
        return max(pool, key=lambda s: s.conversion(key))


@dataclass
class CSTR:
    reactions: ReactionSet
    V: Value | None = None
    rho_b: Value | None = None
    heat: Heat | None = None

    def _result(self, feed, species, F0, F, V, rate, T=None) -> Result:
        outlet = feed.with_F(dict(zip(species, F)))
        extras = {}
        if T is not None:
            outlet = replace(outlet, T=float(T))
            extras["T"] = np.array([T])
        return Result(species, np.array([V]), "V", F[None, :], rate(F)[None, :],
                      dict(zip(species, F0)), outlet, extras=extras)

    def _iso(self):
        return replace(self, heat=None)

    def simulate(self, feed: Stream, V: Value | None = None) -> Result:
        V = to_si(self.V if V is None else V, "m^3")
        ss = self.steady_states(feed, V)
        i = self.pick(ss)
        if self.heat is None and len(ss.states) > 1:
            warnings.warn(self.multiplicity_note(ss), stacklevel=2)
        return ss.states[i]

    def pick(self, ss: SteadyStates) -> int:
        if self.heat is not None:
            return 0
        best = ss.best(key_species(self.reactions, None))
        return next(i for i, s in enumerate(ss.states) if s is best)

    def multiplicity_note(self, ss: SteadyStates) -> str:
        key = key_species(self.reactions, None)
        Xs = ", ".join(f"{s.conversion(key):.4g}" for s in ss.states)
        return (f"CSTR has {len(ss.states)} steady states (X_{key} = {Xs}); returning the "
                f"highest-conversion stable one. Use steady_states() for all.")

    def _newton(self, feed: Stream, V: float) -> Result:
        species, F0, rate = flow_setup(self.reactions, feed, self.rho_b)
        scale = F0.sum()
        F = solve(lambda F: (F0 - F + rate(F) * V) / scale, F0, "CSTR")
        if (F < -1e-9 * scale).any():
            raise RuntimeError("CSTR converged to negative flows")
        return self._result(feed, species, F0, F, V, rate)

    # Isothermal steady states

    def _iso_states(self, feed: Stream, V: float, n: int = 400, starts: int = 32) -> SteadyStates:
        rxns = self.reactions
        species, F0, rate = flow_setup(rxns, feed, self.rho_b)
        rho = None if self.rho_b is None else to_si(self.rho_b, "kg/m^3")
        nu = np.array([[rx.nu.get(s, 0.0) for s in species] for rx in rxns.reactions])
        scale = F0.sum()

        def omega(F):
            C = feed.conc(dict(zip(species, F)))
            out = []
            for rx in rxns.reactions:
                if rx.catalytic and rho is None:
                    raise ValueError(f"Catalytic reaction needs bulk density rho_b: {rx.equation}")
                r = rx.species_rates(C, feed.T)[rx.ref] / rx.nu[rx.ref]
                out.append(r * (rho if rx.catalytic else 1.0))
            return np.array(out)

        def g(xi):
            return omega(F0 + nu.T @ xi) * V - xi

        bounds = [_extent_bounds(F0, row) for row in nu]
        if len(rxns.reactions) == 1:
            xis, x, G, R, axis = _scan_1d(g, bounds[0], n, F0, nu[0], species, rxns.reactions[0].ref, scale)
        else:
            xis, x, G, R, axis = _multistart(g, bounds, F0, nu, scale, starts), None, None, None, "X"
        if not xis:
            raise RuntimeError("No CSTR steady state found")
        states, stable = [], []
        for xi in xis:
            F = np.clip(F0 + nu.T @ xi, 0.0, None)
            states.append(self._result(feed, species, F0, F, V, rate))
            stable.append(_stable(lambda F: F0 - F + rate(F) * V, F, scale))
        return SteadyStates(states, stable, x, G, R, axis)

    def size(self, feed: Stream, X: float, key: str | None = None) -> Result:
        if self.heat is not None:
            return self._size_energy(feed, X, key)
        species, F0, rate = flow_setup(self.reactions, feed, self.rho_b)
        i = species.index(key_species(self.reactions, key))
        Fi, scale = F0[i] * (1 - X), F0.sum()

        def full(z):
            return np.concatenate([z[:i], [Fi], z[i:-1]])

        def res(z):
            return (F0 - full(z) + rate(full(z)) * np.exp(z[-1])) / scale

        guess = F0.copy()
        guess[i] = Fi
        r_i = rate(guess)[i]
        V_guess = (F0[i] - Fi) / -r_i if r_i < 0 else 1.0
        z = solve(res, np.append(np.delete(guess, i), np.log(V_guess)), "CSTR sizing")
        return self._result(feed, species, F0, full(z), float(np.exp(z[-1])), rate)

    def optimum(self, feed: Stream, species: str, V_bounds=(1e-6, 1e4)) -> tuple[float, float]:
        def neg(logV):
            return -self.simulate(feed, np.exp(logV)).profile(species)[-1]

        opt = minimize_scalar(neg, bounds=np.log(V_bounds), method="bounded")
        return float(np.exp(opt.x)), float(-opt.fun)

    # Energy balance

    def _balance(self, feed: Stream, res: Result, T: float) -> tuple[float, float]:
        h = self.heat
        C = res.outlet.conc()
        rho = None if self.rho_b is None else to_si(self.rho_b, "kg/m^3")
        G = res.end * h.generation(self.reactions, C, T, rho)
        R = sum(h.Cp[s] * f for s, f in res.n0.items()) * (T - feed.T) + h.removal(T)
        return G, R

    def T_range(self, feed: Stream) -> tuple[float, float]:
        h = self.heat
        T0 = feed.T
        lo, hi = (min(T0, h.Ta), max(T0, h.Ta)) if h.exchange else (T0, T0)
        cp0 = sum(h.Cp.get(s, 0.0) * f for s, f in feed.F.items())
        span = min(max(sum(abs(x) for x in h.dH) * sum(feed.F.values()) / cp0, 50.0), 3000.0)
        exo, endo = any(x < 0 for x in h.dH), any(x > 0 for x in h.dH)
        return max(lo - (span if endo else 1.0), 20.0), hi + (span if exo else 1.0)

    def _scan(self, feed, at, T_range, n):
        species, _, _ = flow_setup(self.reactions, feed, self.rho_b)
        self.heat.bind(self.reactions, species)
        lo, hi = T_range or self.T_range(feed)
        Ts = np.linspace(lo, hi, n)
        G, R = np.full(n, np.nan), np.full(n, np.nan)
        for j, T in enumerate(Ts):
            try:
                G[j], R[j] = self._balance(feed, at(T), T)
            except (RuntimeError, ValueError):
                pass
        roots = []
        f = lambda T: np.subtract(*self._balance(feed, at(T), T))
        for j in range(n - 1):
            a, b = G[j] - R[j], G[j + 1] - R[j + 1]
            if np.isfinite(a) and np.isfinite(b) and (a * b < 0 or a == 0):
                T = Ts[j] if a == 0 else brentq(f, Ts[j], Ts[j + 1], xtol=1e-9)
                roots.append((T, a > b))
        return Ts, G, R, roots

    def steady_states(self, feed: Stream, V: Value | None = None, T_range: tuple[float, float] | None = None,
                      n: int = 800) -> SteadyStates:
        V = to_si(self.V if V is None else V, "m^3")
        if self.heat is None:
            return self._iso_states(feed, V)
        iso = self._iso()
        at = lambda T: iso._newton(replace(feed, T=float(T)), V)
        Ts, G, R, roots = self._scan(feed, at, T_range, n)
        if not roots:
            raise RuntimeError("No steady state found in the temperature range")
        states = [self._at(feed, at(T), T) for T, _ in roots]
        return SteadyStates(states, [s for _, s in roots], Ts, G, R)

    def _at(self, feed: Stream, res: Result, T: float) -> Result:
        res.outlet = replace(res.outlet, T=float(T))
        res.extras["T"] = np.array([T])
        return res

    def _size_energy(self, feed: Stream, X: float, key: str | None, T_range=None, n: int = 400) -> Result:
        iso = self._iso()
        at = lambda T: iso.size(replace(feed, T=float(T)), X, key)
        _, _, _, roots = self._scan(feed, at, T_range, n)
        if not roots:
            raise RuntimeError(f"No temperature satisfies both balances for X = {X}")
        T = roots[0][0]
        return self._at(feed, at(T), T)


def _extent_bounds(F0, nu_row) -> tuple[float, float]:
    lo, hi = -np.inf, np.inf
    for f, v in zip(F0, nu_row):
        if v > 0:
            lo = max(lo, -f / v)
        elif v < 0:
            hi = min(hi, f / -v)
    return (0.0 if not np.isfinite(lo) else lo), (0.0 if not np.isfinite(hi) else hi)


def _scan_1d(g, bounds, n, F0, nu_row, species, ref, scale):
    lo, hi = bounds
    span = hi - lo
    if span <= 0:
        return [np.array([lo])], None, None, None, "X"
    edge = np.geomspace(1e-10, 0.5, 60)
    pts = np.unique(np.concatenate([np.linspace(lo, hi, n), lo + span * edge, hi - span * edge]))
    gv = np.array([_safe(g, np.array([p]))[0] for p in pts])
    f = lambda p: _safe(g, np.array([p]))[0]
    tol = 1e-12 * scale
    roots = [p for p, v in zip(pts, gv) if abs(v) <= tol]
    for j in range(len(pts) - 1):
        a, b = gv[j], gv[j + 1]
        if np.isfinite(a) and np.isfinite(b) and abs(a) > tol and abs(b) > tol and a * b < 0:
            roots.append(brentq(f, pts[j], pts[j + 1], xtol=1e-14 * max(span, 1.0), rtol=1e-13))
    roots = _dedupe([np.array([r]) for r in sorted(roots)], span)
    i = species.index(ref)
    nr = -nu_row[i]
    R = nr * pts
    G = R + nr * gv
    x = R / F0[i] if F0[i] > 0 else pts
    return roots, x, G, R, f"X_{ref}"


def _multistart(g, bounds, F0, nu, scale, starts):
    lo = np.array([b[0] for b in bounds])
    hi = np.array([b[1] for b in bounds])
    pts = qmc.scale(qmc.Sobol(len(bounds), seed=0).random(starts), lo, np.maximum(hi, lo + 1e-300))
    pts = np.vstack([np.zeros(len(bounds)), pts])
    pts = [p for p in pts if (F0 + nu.T @ p >= -1e-12 * scale).all()]
    found = []
    for p in pts:
        try:
            with np.errstate(all="ignore"), warnings.catch_warnings():
                warnings.simplefilter("ignore")
                sol = root(lambda xi: g(xi) / scale, p, method="hybr", tol=1e-12)
        except (ValueError, ZeroDivisionError, OverflowError):
            continue
        if (np.all(np.isfinite(sol.fun)) and np.abs(sol.fun).max() < 1e-9
                and (F0 + nu.T @ sol.x >= -1e-9 * scale).all()):
            found.append(sol.x)
    return _dedupe(found, max(np.abs(hi - lo).max(), scale * 1e-12))


def _safe(g, xi):
    try:
        with np.errstate(all="ignore"):
            return np.asarray(g(xi), float)
    except (ValueError, ZeroDivisionError, OverflowError):
        return np.full(len(xi), np.nan)


def _dedupe(xis, span, rtol=1e-6):
    out = []
    for xi in xis:
        if all(np.abs(xi - o).max() > rtol * span for o in out):
            out.append(xi)
    return out


def _stable(res, F, scale) -> bool:
    n = len(F)
    J = np.empty((n, n))
    f0 = res(F)
    for k in range(n):
        h = 1e-7 * max(F[k], 1e-6 * scale)
        Fp = F.copy()
        Fp[k] += h
        J[:, k] = (res(Fp) - f0) / h
    return bool(np.linalg.eigvals(J).real.max() < 0)
