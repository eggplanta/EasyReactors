COMMON = """
### Typing values
- **Grey text is only a hint**: an empty field is empty. Press **Example** to see a filled form.
- Numbers without units use the unit in brackets, e.g. `[L]`. Any unit works: `2 m3`, `800 gallon`, `25 degC`.
- Reactions: `A + B > C` (one way), `A <> B` (reversible), `2A > D`.
- Lists: `A=1, B=1.2`. Orders accept fractions: `A=1/3`.

### Keys
`Ctrl+R` run · `Ctrl+E` example · `F1` this guide · `Tab` next field · `Ctrl+Q` quit
"""

REACTOR = """
## Single reactor
Batch, CSTR, PFR or packed bed (PBR, rates per kg of catalyst).

1. Pick the **Type** and the **Problem**:
   - *Simulate*: give V (or W, or t) and get the outlet.
   - *Size*: give a target X and get V (or W, or t).
   - *Maximize*: find the V that maximizes an intermediate.
2. Write the **reactions**. The line under each one shows the k units it expects.
3. Fill in the **feed** (or initial charge for batch).
4. Optional: **Energy balance** (adiabatic or constant Ta) and, for a PBR, **pressure drop** (Ergun).
5. Press **Run**.

With an energy balance, a CSTR reports every steady state and the G(T) / R(T) curves.
""" + COMMON

NETWORK = """
## Reactor networks
Type the flowsheet as an expression. The sketch below the box updates as you type.

| Write | Means |
|---|---|
| `CSTR(100) > PFR(50)` | series, volumes in L |
| `[CSTR(?) \\| PFR(?)]` | parallel, equal split |
| `[a \\| b]{0.3, 0.7}` | given split fractions |
| `[a \\| b]{?}` | size the split |
| `recycle(PFR(10), 2)` | recycle with ratio R |
| `bypass`, `feed` | empty branch, side feed |

`?` marks an unknown volume. All unknown volumes get the **same V**, sized to reach the target X.
""" + COMMON

TRANSIENT = """
## Semibatch and CSTR start-up
Isothermal, liquid phase.

**Semibatch**: the vessel starts with *V0* and the *initial C*. The feed stream (v0, C0) flows in for a time *t*.

**CSTR start-up**: the reactor starts full (initial C, blank = only solvent). The feed starts at t = 0.
The result gives *t_s*, the time to cover a *fraction* (0.99) of the way to the steady state.
""" + COMMON

RTD = """
## Residence time distribution
1. Paste tracer data (t and C columns, or two rows) or load a file.
2. Run to get E(t), F(t), the mean residence time and the variance.
3. Add a reaction to compare ideal reactors, segregation and maximum mixedness,
   tanks in series and dispersion.
""" + COMMON

TWO_PARAM = """
## Two-parameter models
Fit α and β of a combination of ideal reactors to tracer data, then predict the conversion.
The description under the model tells which tracer test it needs.
""" + COMMON

DIFFUSION = """
## Diffusion and reaction
- **Pellet**: Thiele modulus and effectiveness factor η for any rate law (numerical), compared with the formulas.
- **Criteria**: Weisz-Prater and Mears.
- **Mass transfer**: kc from Thoenes-Kramers, Frössling or Colburn, and the mass-transfer limited X.
- **Two pellets**: φ and η from rates measured with two pellet sizes.
- **Packed bed**: catalyst weight with internal and external resistances.
""" + COMMON
