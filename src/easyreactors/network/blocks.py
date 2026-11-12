from dataclasses import dataclass, field, replace

import numpy as np

from ..reactors import CSTR, PFR, Result, Stream
from ..reactors._common import solve
from ..units import to_si


def mix(streams: list[Stream]) -> Stream:
    first = streams[0]
    if any(s.phase != first.phase for s in streams):
        raise ValueError("Cannot mix streams of different phases")
    if any(abs(s.T - first.T) > 1e-9 for s in streams):
        raise ValueError("Mixing streams at different T requires the energy balance (stage 5)")
    F: dict[str, float] = {}
    for s in streams:
        for k, v in s.F.items():
            F[k] = F.get(k, 0.0) + v
    v0 = sum(s.v0 for s in streams) if first.phase == "liquid" else None
    return replace(first, F=F, P=min(s.P for s in streams), v0=v0)


def scale(stream: Stream, f: float) -> Stream:
    v0 = None if stream.v0 is None else stream.v0 * f
    return replace(stream, F={k: v * f for k, v in stream.F.items()}, v0=v0)


@dataclass
class UnitNode:
    name: str
    kind: str
    V: float
    inlet: Stream
    outlet: Stream
    result: Result


@dataclass
class SeriesNode:
    children: list


@dataclass
class ParallelNode:
    children: list
    split: list[float]
    inlets: list[Stream]
    outlets: list[Stream]


@dataclass
class SideFeedNode:
    stream: Stream


@dataclass
class BypassNode:
    pass


@dataclass
class RecycleNode:
    child: object
    R: float


@dataclass
class Context:
    V: dict[int, float] = field(default_factory=dict)
    split: dict[int, list[float]] = field(default_factory=dict)
    fed: dict[str, float] = field(default_factory=dict)
    units: list[UnitNode] = field(default_factory=list)
    nodes: dict[int, object] = field(default_factory=dict)
    counters: dict[str, int] = field(default_factory=dict)

    def add_fed(self, stream: Stream) -> None:
        for k, v in stream.F.items():
            self.fed[k] = self.fed.get(k, 0.0) + v

    def name(self, kind: str) -> str:
        self.counters[kind] = self.counters.get(kind, 0) + 1
        return f"{kind}-{self.counters[kind]}"


def run(block, feed: Stream, ctx: Context):
    if isinstance(block, (PFR, CSTR)):
        V = ctx.V.get(id(block), block.V)
        if V is None:
            raise ValueError("Reactor volume not set (use size_network)")
        res = block.simulate(feed, V)
        kind = type(block).__name__
        node = UnitNode(ctx.name(kind), kind, float(to_si(V, "m^3")), feed, res.outlet, res)
        ctx.units.append(node)
        return res.outlet, node
    return block.run(feed, ctx)


def _reactors(block):
    if isinstance(block, (PFR, CSTR)):
        return [block]
    return [r for child in block.children() for r in _reactors(child)]


class Block:
    def children(self) -> list:
        return []

    def simulate(self, feed: Stream) -> "NetworkResult":
        from .result import NetworkResult
        ctx = Context()
        ctx.add_fed(feed)
        outlet, node = run(self, feed, ctx)
        return NetworkResult(feed, outlet, node, ctx)

    def reactors(self) -> list:
        return _reactors(self)


@dataclass
class Series(Block):
    blocks: list

    def children(self):
        return self.blocks

    def run(self, feed, ctx):
        nodes = []
        for b in self.blocks:
            feed, node = run(b, feed, ctx)
            nodes.append(node)
        node = SeriesNode(nodes)
        ctx.nodes[id(self)] = node
        return feed, node


@dataclass
class Parallel(Block):
    blocks: list
    split: list[float] | None = None

    def children(self):
        return self.blocks

    def run(self, feed, ctx):
        n = len(self.blocks)
        split = ctx.split.get(id(self)) or self.split or [1 / n] * n
        if len(split) != n or abs(sum(split) - 1) > 1e-9 or min(split) < 0:
            raise ValueError(f"Invalid split fractions: {split}")
        inlets, outlets, nodes = [], [], []
        for b, f in zip(self.blocks, split):
            inlet = scale(feed, f)
            out, node = run(b, inlet, ctx)
            inlets.append(inlet)
            outlets.append(out)
            nodes.append(node)
        node = ParallelNode(nodes, list(split), inlets, outlets)
        ctx.nodes[id(self)] = node
        return mix(outlets), node


@dataclass
class Bypass(Block):
    def run(self, feed, ctx):
        return feed, BypassNode()


@dataclass
class SideFeed(Block):
    stream: Stream

    def run(self, feed, ctx):
        ctx.add_fed(self.stream)
        return mix([feed, self.stream]), SideFeedNode(self.stream)


@dataclass
class Recycle(Block):
    block: object
    R: float

    def children(self):
        return [self.block]

    def _recycle_stream(self, feed, F):
        v0 = None if feed.phase == "gas" else feed.v0 * self.R
        return replace(feed, F=F, v0=v0)

    def run(self, feed, ctx):
        frac = self.R / (1 + self.R)
        probe, _ = run(self.block, feed, Context(V=ctx.V, split=ctx.split))
        species = list(dict.fromkeys(list(feed.F) + list(probe.F)))
        scale_F = sum(feed.F.values()) or 1.0

        def residual(x):
            rec = self._recycle_stream(feed, dict(zip(species, x)))
            out, _ = run(self.block, mix([feed, rec]), Context(V=ctx.V, split=ctx.split))
            return (x - frac * np.array([out.F.get(s, 0.0) for s in species])) / scale_F

        x0 = frac * np.array([probe.F.get(s, 0.0) for s in species])
        rec = self._recycle_stream(feed, dict(zip(species, solve(residual, x0, "Recycle"))))
        out, child = run(self.block, mix([feed, rec]), ctx)
        node = RecycleNode(child, self.R)
        ctx.nodes[id(self)] = node
        return scale(out, 1 / (1 + self.R)), node
