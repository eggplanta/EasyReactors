from dataclasses import dataclass

from .blocks import BypassNode, ParallelNode, RecycleNode, SeriesNode, SideFeedNode, UnitNode


@dataclass
class Box:
    lines: list[str]
    row: int

    @property
    def width(self) -> int:
        return max(len(l) for l in self.lines)

    def padded(self, above: int, below: int, width: int | None = None) -> list[str]:
        w = width or self.width
        lines = [""] * (above - self.row) + self.lines
        lines += [""] * (below - (len(self.lines) - self.row - 1))
        r = above
        return [l.ljust(w, "-") if i == r else l.ljust(w) for i, l in enumerate(lines)]


def fmt_volume(V: float) -> str:
    return f"V={V * 1000:.4g} L" if V < 1 else f"V={V:.4g} m3"


def unit_box(name: str, kind: str, text: str) -> Box:
    edge = "=" if kind == "PFR" else "-"
    w = max(len(name) + 8, len(text) + 4)
    top = "+" + f"[{name}]".center(w - 2, edge) + "+"
    return Box([top, "|" + text.center(w - 2) + "|", "+" + edge * (w - 2) + "+"], 1)


def _unit(node: UnitNode) -> Box:
    return unit_box(node.name, node.kind, fmt_volume(node.V))


def _hcat(boxes: list[Box], sep: str = "--->") -> Box:
    above = max(b.row for b in boxes)
    below = max(len(b.lines) - b.row - 1 for b in boxes)
    cols = [b.padded(above, below) for b in boxes]
    lines = []
    for i in range(above + below + 1):
        gap = sep if i == above else " " * len(sep)
        lines.append(gap.join(c[i] for c in cols))
    return Box(lines, above)


def parallel_box(boxes: list[Box], labels: list[str]) -> Box:
    w = max(b.width for b in boxes)
    lines, rows = [], []
    for k, b in enumerate(boxes):
        if k:
            lines.append("")
        rows.append(len(lines) + b.row)
        lines += b.padded(b.row, len(b.lines) - b.row - 1, w)
    first, last = rows[0], rows[-1]
    out = []
    for i, l in enumerate(lines):
        if i in rows:
            out.append(f"+-{labels[rows.index(i)]:-^4}-->" + l + "-+")
        elif first < i < last:
            out.append("|" + " " * 8 + l.ljust(w) + " |")
        else:
            out.append(" " * 9 + l.ljust(w) + "  ")
    return Box(out, first)


def _parallel(node: ParallelNode) -> Box:
    return parallel_box([_box(c) for c in node.children], [f"{f:.2f}" for f in node.split])


def feed_box(label: str) -> Box:
    w = max(len(label), 7)
    c = w // 2
    return Box([label.ljust(w), (" " * c + "|").ljust(w), (" " * c + "v").ljust(w),
                "-" * c + "+" + "-" * (w - c - 1)], 3)


def _side_feed(node: SideFeedNode) -> Box:
    F = sum(node.stream.F.values())
    names = ",".join(k for k, v in node.stream.F.items() if v > 0)
    return feed_box(f"+{names} F={F * 60:.4g} mol/min")


def recycle_box(b: Box, R: str) -> Box:
    w, r, h = b.width, b.row, len(b.lines)
    body = b.padded(r, h - r - 1, w)
    lines = []
    for i, l in enumerate(body):
        if i == r:
            lines.append("-+->" + l + "-+--")
        elif i > r:
            lines.append((" ^  " if i == r + 1 else " |  ") + l + " |  ")
        else:
            lines.append("    " + l + "    ")
    lines.append((" ^  " if r == h - 1 else " |  ") + f"R={R}".center(w) + " |  ")
    lines.append(" +" + "-" * (w + 3) + "+  ")
    return Box(lines, r)


def _recycle(node: RecycleNode) -> Box:
    return recycle_box(_box(node.child), f"{node.R:g}")


BYPASS = Box(["-----"], 0)


def _box(node) -> Box:
    if isinstance(node, UnitNode):
        return _unit(node)
    if isinstance(node, SeriesNode):
        return _hcat([_box(c) for c in node.children])
    if isinstance(node, ParallelNode):
        return _parallel(node)
    if isinstance(node, SideFeedNode):
        return _side_feed(node)
    if isinstance(node, RecycleNode):
        return _recycle(node)
    if isinstance(node, BypassNode):
        return BYPASS
    raise TypeError(f"Cannot draw {node!r}")


def frame(box: Box, left: list[str], right: list[str]) -> str:
    above, below = max(box.row, 1), max(len(box.lines) - box.row - 1, 1)
    body = box.padded(above, below)
    lw = max(len(s) for s in left)
    out = []
    for i, l in enumerate(body):
        j = i - (above - 1)
        lab = left[j] if 0 <= j < len(left) else ""
        rab = right[j] if 0 <= j < len(right) else ""
        arrow_in, arrow_out = ("--->", "---> ") if i == above else ("    ", "     ")
        out.append((lab.ljust(lw) + " " + arrow_in + l + arrow_out + rab).rstrip())
    return "\n".join(out)


def draw(res, key: str | None = None) -> str:
    key = key or next(iter(res.feed.F))
    left = [f"F0={sum(res.feed.F.values()) * 60:.4g} mol/min", "X=0", f"T={res.feed.T:.4g} K"]
    right = [f"F={sum(res.outlet.F.values()) * 60:.4g} mol/min", f"X={res.conversion(key):.4g}",
             f"T={res.outlet.T:.4g} K"]
    return frame(_box(res.tree), left, right)
