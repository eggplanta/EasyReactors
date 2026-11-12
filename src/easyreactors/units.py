import pint

ureg = pint.UnitRegistry()
ureg.define("lbmol = 453.59237 * mol = lb_mol = pound_mole")
Q_ = ureg.Quantity

Value = float | str | pint.Quantity


def to_si(value: Value, si_unit: str) -> float:
    if isinstance(value, str):
        value = Q_(value)
    if isinstance(value, pint.Quantity):
        return float(value.to(si_unit).magnitude)
    return float(value)


def from_si(value: float, si_unit: str, target: str) -> pint.Quantity:
    return Q_(value, si_unit).to(target)
