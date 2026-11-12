from dataclasses import dataclass

from ..reactors import Stream
from .blocks import Context, UnitNode


@dataclass
class NetworkResult:
    feed: Stream
    outlet: Stream
    tree: object
    ctx: Context

    @property
    def units(self) -> list[UnitNode]:
        return self.ctx.units

    @property
    def fed(self) -> dict[str, float]:
        return self.ctx.fed

    def conversion(self, s: str) -> float:
        return (self.fed[s] - self.outlet.F.get(s, 0.0)) / self.fed[s]

    def draw(self, key: str | None = None) -> str:
        from .ascii import draw
        return draw(self, key)
