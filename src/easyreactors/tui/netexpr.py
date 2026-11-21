import re
from dataclasses import dataclass, field

from ..core import ReactionSet
from ..network import Bypass, Parallel, Recycle, Series, SideFeed
from ..reactors import CSTR, PFR, Stream
from .parsing import numbers, quantity

# Network expression

@dataclass
class UnitSpec:
    kind: str
    V: str | None = None


@dataclass
class ParSpec:
    branches: list[list]
    split: list[float] | str | None = None


@dataclass
class RecycleSpec:
    body: list
    R: float


@dataclass
class BypassSpec:
    pass


@dataclass
class FeedSpec:
    stream: Stream | None = field(default=None, repr=False)


class _Parser:
    def __init__(self, text: str):
        self.s, self.i = text, 0

    def fail(self, msg: str):
        raise ValueError(f"{msg} at position {self.i + 1}: ...{self.s[self.i:self.i + 15]!r}")

    def skip(self):
        while self.i < len(self.s) and self.s[self.i].isspace():
            self.i += 1

    def peek(self, token: str) -> bool:
        self.skip()
        return self.s.startswith(token, self.i)

    def eat(self, token: str):
        if not self.peek(token):
            self.fail(f"Expected '{token}'")
        self.i += len(token)

    def until(self, stop: str) -> str:
        j = self.s.find(stop, self.i)
        if j < 0:
            self.fail(f"Missing '{stop}'")
        text, self.i = self.s[self.i:j].strip(), j + 1
        return text

    def word(self) -> str:
        self.skip()
        m = re.compile(r"[A-Za-z]+").match(self.s, self.i)
        if not m:
            self.fail("Expected a unit")
        self.i = m.end()
        return m[0].lower()

    def series(self) -> list:
        items = [self.item()]
        while self.peek(">"):
            self.eat(">")
            items.append(self.item())
        return items

    def item(self):
        if self.peek("["):
            self.eat("[")
            branches = [self.series()]
            while self.peek("|"):
                self.eat("|")
                branches.append(self.series())
            self.eat("]")
            if len(branches) < 2:
                self.fail("Parallel needs at least 2 branches")
            split = None
            if self.peek("{"):
                self.eat("{")
                text = self.until("}")
                split = "?" if text == "?" else list(numbers(text))
                if split != "?" and len(split) != len(branches):
                    self.fail("Split must have one fraction per branch")
            return ParSpec(branches, split)
        w = self.word()
        if w in ("cstr", "pfr"):
            V = None
            if self.peek("("):
                self.eat("(")
                V = self.until(")") or None
                if V == "?":
                    V = None
                if V is not None:
                    quantity(V, "m^3", "L")
            return UnitSpec(w.upper(), V)
        if w == "bypass":
            return BypassSpec()
        if w in ("feed", "side"):
            return FeedSpec()
        if w == "recycle":
            self.eat("(")
            body = self.series()
            self.eat(",")
            R = float(self.until(")"))
            return RecycleSpec(body, R)
        self.fail(f"Unknown element '{w}'")


def parse(text: str) -> list:
    p = _Parser(text)
    spec = p.series()
    p.skip()
    if p.i != len(p.s):
        p.fail("Unexpected text")
    return spec


def to_text(spec: list) -> str:
    def item(x):
        if isinstance(x, UnitSpec):
            return f"{x.kind}({x.V or '?'})"
        if isinstance(x, ParSpec):
            body = "[" + " | ".join(to_text(b) for b in x.branches) + "]"
            if x.split == "?":
                return body + "{?}"
            return body + ("{" + ", ".join(f"{f:g}" for f in x.split) + "}" if x.split else "")
        if isinstance(x, RecycleSpec):
            return f"recycle({to_text(x.body)}, {x.R:g})"
        return "bypass" if isinstance(x, BypassSpec) else "feed"
    return " > ".join(item(x) for x in spec)


def walk(spec: list):
    for x in spec:
        yield x
        if isinstance(x, ParSpec):
            for b in x.branches:
                yield from walk(b)
        elif isinstance(x, RecycleSpec):
            yield from walk(x.body)


@dataclass
class Built:
    net: Series
    sized_split: Parallel | None = None


def build(spec: list, rs: ReactionSet) -> Built:
    out = Built(None)

    def block(x):
        if isinstance(x, UnitSpec):
            V = None if x.V is None else quantity(x.V, "m^3", "L")
            return (CSTR if x.kind == "CSTR" else PFR)(rs, V=V)
        if isinstance(x, ParSpec):
            par = Parallel([series(b) for b in x.branches], None if x.split == "?" else x.split)
            if x.split == "?":
                if out.sized_split is not None:
                    raise ValueError("Only one parallel split can be sized")
                out.sized_split = par
            return par
        if isinstance(x, RecycleSpec):
            return Recycle(series(x.body), x.R)
        if isinstance(x, BypassSpec):
            return Bypass()
        if x.stream is None:
            raise ValueError("Side feed stream not set")
        return SideFeed(x.stream)

    def series(items):
        blocks = [block(x) for x in items]
        return blocks[0] if len(blocks) == 1 else Series(blocks)

    out.net = Series([block(x) for x in spec])
    if out.sized_split is not None and any(r.V is None for r in out.net.reactors()):
        raise ValueError("Size either an unknown volume or a split, not both")
    return out
