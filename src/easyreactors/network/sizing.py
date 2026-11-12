import numpy as np
from scipy.optimize import brentq

from ..reactors import Stream
from ..reactors._common import solve
from .blocks import Block, Context, Parallel, run
from .result import NetworkResult


def _simulate(net: Block, feed: Stream, ctx: Context) -> NetworkResult:
    ctx.add_fed(feed)
    outlet, node = run(net, feed, ctx)
    return NetworkResult(feed, outlet, node, ctx)


def _key(net: Block, feed: Stream, key: str | None) -> str:
    if key:
        return key
    rx = net.reactors()[0].reactions
    rx.resolve_refs(feed.F)
    return rx.reactions[0].ref


def size_network(net: Block, feed: Stream, X: float, key: str | None = None,
                 V_max: float = 1e6) -> NetworkResult:
    unknown = [r for r in net.reactors() if r.V is None]
    if not unknown:
        raise ValueError("No reactor with V=None to size")
    key = _key(net, feed, key)

    def f(logV):
        V = np.exp(logV)
        return _simulate(net, feed, Context(V={id(r): V for r in unknown})).conversion(key) - X

    lo, hi = np.log(1e-9), np.log(1e-3)
    while f(hi) < 0:
        if hi >= np.log(V_max):
            raise ValueError(f"X = {X} not reachable for {key} with V <= {V_max} m3")
        hi += np.log(10)
    V = np.exp(brentq(f, lo, hi, xtol=1e-12, rtol=1e-10))
    return _simulate(net, feed, Context(V={id(r): V for r in unknown}))


def _branch_conversions(node, key: str) -> np.ndarray:
    return np.array([(i.F[key] - o.F.get(key, 0.0)) / i.F[key]
                     for i, o in zip(node.inlets, node.outlets)])


def size_split(net: Block, par: Parallel, feed: Stream, X: float | None = None,
               key: str | None = None, equal_conversion: bool = False) -> NetworkResult:
    key = _key(net, feed, key)
    n = len(par.blocks)
    eps = 1e-9

    def sim(split):
        return _simulate(net, feed, Context(split={id(par): list(split)}))

    if equal_conversion:
        def res(x):
            split = np.append(x, 1 - x.sum())
            Xb = _branch_conversions(sim(split).ctx.nodes[id(par)], key)
            return Xb[1:] - Xb[0]

        x = solve(res, np.full(n - 1, 1 / n), "Split sizing")
        return sim(np.append(x, 1 - x.sum()))

    if X is None or n != 2:
        raise ValueError("Target X sizing needs X and exactly 2 branches")

    def f(a):
        return sim([a, 1 - a]).conversion(key) - X

    if f(eps) * f(1 - eps) > 0:
        raise ValueError(f"X = {X} not reachable by changing the split")
    a = brentq(f, eps, 1 - eps, xtol=1e-12)
    return sim([a, 1 - a])
