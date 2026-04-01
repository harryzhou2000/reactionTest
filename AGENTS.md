# AGENTS.md

## Project Overview

Research codebase exploring implicit ODE methods for advection-reaction PDEs.
The goal is to evaluate alternatives to Strang splitting -- including fully
implicit, embedded splitting, and exponential DITR -- for eventual use in
reaction-enabled Navier-Stokes solvers. The current test problem is 1D linear
advection + bistable reaction on a uniform grid, serving as a model problem.

**Language:** Python 3.12  
**Package manager:** Conda (configured via `.vscode/settings.json`)  
**Dependencies:** numpy, scipy, matplotlib, scienceplots  
**No formal build system** -- no `pyproject.toml`, `setup.py`, or `requirements.txt`.

## Project Structure

```
reactionTest/               # Main Python package (working directory for scripts)
  __init__.py
  PlotEnv.py                # Matplotlib plotting utilities (scienceplots styles)
  Solver/                   # Core solver subpackage
    __init__.py
    AdvReactUni.py          # Advection-reaction evaluator + solver (~390 lines)
    AdvReactUniFunctors.py  # Functor classes (Frhs, Fsolve, FrhsDITRExp, FsolveDITR)
    ESDIRK_Data.py          # Butcher tableau coefficients for ESDIRK methods
    FVUni2nd.py             # 2nd-order finite volume discretization on uniform 1D grid
    ODE.py                  # ODE integrator framework (ESDIRK, DITRExp) + ABCs
  test_ODE.py               # Script test for ODE integrators (dense-matrix exponential)
  test_AdvReactUni1D.py     # Script test for advection-reaction solver
  test_AdvReactUni1D.ipynb  # PRIMARY test: bistable reaction, all 7 method variants
  test_AdvReactUni1D_DITR.ipynb  # DITRExp long-time pure advection test
  test_BrusselatorUni1D.py  # Brusselator 2-species test with probe time series
```

## Running Tests

There is no test framework (no pytest/unittest). Tests are plain scripts that
instantiate solvers, run them, and print output. They must be run from the
`reactionTest/` subdirectory because they use package-relative imports.

```bash
# Run from the reactionTest/ subdirectory
cd reactionTest

# Run a single test script
python test_ODE.py
python test_AdvReactUni1D.py

# Run a module directly (some files have __main__ blocks)
python Solver/FVUni2nd.py
python Solver/AdvReactUni.py
```

The primary comparison is in `test_AdvReactUni1D.ipynb` (run in Jupyter). It
compares 7 configurations on the bistable problem: reference (ESDIRK4, fine dt),
Strang splitting, fully implicit ESDIRK3, DITR, Exponential DITR, Embed ESDIRK3,
and Embed DITR.

There is no lint, format, type-check, or build command configured.

## Solver Architecture

All ODE integrators are **implicit**. The two integrator families are:

- **ESDIRK** -- singly diagonally implicit Runge-Kutta. Each stage is solved
  independently via pseudo-time Jacobi iteration until convergence.
- **DITR** (`DITRExp`) -- a fully implicit 2-stage method (matches Lobatto IIIA
  in some configurations). Its two coupled stages are solved one-by-one in a
  sweep, iterating the whole sweep until convergence (not a 2x-large Newton).

Three splitting modes exist in `stepInterval()`:
- `"full"` -- monolithic implicit (flow + source coupled).
- `"strang"` -- classical Strang operator splitting (half-flow, source, half-flow).
- `"embed"` -- embedded splitting: source sub-steps at each RK stage node,
  flow solved implicitly with forcing correction.

### Probe Recording

`AdvReactUni1DSolver` supports recording time series at specified spatial
locations via the probe API:

- `set_probes(x_locations)` -- set probe locations (mapped to nearest cell center).
- `clear_probes()` -- clear recorded data for all probes.
- `get_probe_data(x=None)` -- retrieve recorded data; returns dict with `"t"` and `"u"`.
- `stepInterval(..., record_probes=True)` -- enable recording during time integration.

### Exponential Jacobian Abstraction

The exponential variant (`use_exp=True`) extracts a stable linear part `A` from
the source Jacobian and handles it via `exp(A*dt)` and phi-functions, leaving
only the nonlinear remainder for the implicit solver. The abstraction uses
overridable methods on `ODE_F_RHS`:

- `JacobianExpo(u)` -- returns `A` (stored in `self.currentA`), also stores
  `self.currentU` as the linearization point.
- `JacobianExpoMult(A, v)` -- `A * v` (element-wise) or `A @ v` (matrix).
- `JacobianExpoEye(u)` -- identity: `np.ones_like(u)` or `np.eye(n)`.
- `JacobianExpoExp(u, dt)` -- `exp(A*dt)`: `np.exp` or `scipy.linalg.expm`.
- `JacobianExpoPhikSeq(u, dt, k_max)` -- phi-function sequence with fallback
  to Taylor limits (`1/k!`) when `|A*dt| < 1e-3`.

The defaults make the exponential degenerate to regular DITR (A ~ 0).
When `A = 0`, exponential DITR falls back to standard DITR.

**Current Jacobian dimension support** (`invert_jacobian_diag` / `jacobian_diag_mult`):

| `ndim` | Shape              | Meaning                  | Inversion     | Multiply        |
|--------|--------------------|--------------------------|---------------|-----------------|
| 2      | `(nVars, nx)`      | Per-element scalar diag  | `1.0 / JD`   | `JD * u`        |
| 3      | `(nVars, nVars, nx)` | Per-point dense matrix | `pinv` batch  | `einsum ij..,j..->i..` |

**Known limitation:** The exponential Jacobian methods in `FrhsDITRExp` (inside
`AdvReactUni1DSolver`) currently only handle `ndim==2` (per-element scalars using
`np.exp` and `*`). The `test_ODE.py` version uses full matrices (`expm`, `@`).
Future work needs the `ndim==3` path (per-point diagonal or dense matrix) in the
exponential methods -- `JacobianExpoExp`, `JacobianExpoPhikSeq`, and the
`a22Invb2 = 1.0 / a22` line in `DITRExp.step()` (marked `# TODO`).

**Stability clamping:** `FrhsDITRExp.JacobianExpo` clamps `currentA` to strictly
negative values via `np.minimum(JDSource, abs(JDSource).max() * -1e-4)`.
Positive eigenvalues would cause `exp(A*dt)` to blow up.

## Code Style

### Imports

- Standard library first, then third-party (`numpy`, `scipy`), then local.
- Conventional aliases: `import numpy as np`, `import scipy.linalg as spl`.
- Relative imports within the `Solver/` subpackage:
  `from .FVUni2nd import FVUni2nd1D`, `from . import ODE`.
- Files with `__main__` blocks use a conditional pattern for dual import support:
  ```python
  if __name__ == "__main__":
      from FVUni2nd import FVUni2nd1D
  else:
      from .FVUni2nd import FVUni2nd1D
  ```
- Test scripts use absolute-from-package imports: `import Solver.ODE as ODE`.

### Naming Conventions

- **Classes:** PascalCase, preserving uppercase acronyms:
  `FVUni2nd1D`, `AdvReactUni1DEval`, `ESDIRK`, `DITRExp`.
- **ABC interfaces:** Uppercase with underscores:
  `ODE_F_RHS`, `ODE_F_SOLVE_SingleStage`, `ImplicitOdeIntegrator`.
- **Methods:** snake_case: `rhs_flow()`, `rhs_source()`, `get_shape_u()`,
  `stepInterval()` -- some camelCase methods exist (e.g., `recGrad`, `cellOthers`).
- **Local variables:** camelCase following math/physics notation:
  `uGrad`, `uRec`, `nVars`, `dtC`, `resN`, `JDFlow`, `JDFullInv`.
- **Loop counters:** Short names: `nx`, `dx`, `iStage`, `iJ`, `iter`.
- **Constants/config:** camelCase: `butcherA`, `butcherB`, `butcherC`.

When adding code, match the surrounding file's existing conventions. The codebase
prioritizes mathematical readability over PEP 8 strict naming.

### Type Annotations

- Sparse but used on key interfaces: `def __init__(self, nx: int)`,
  `def dt(self, u: np.ndarray, cStage, iStage) -> float`.
- `np.ndarray` for array parameters. Return types are rarely annotated.
- No mypy or type checker is configured. Match the annotation density of
  surrounding code -- add hints on public method signatures, skip on internals.

### Formatting

- No formatter is configured (no black, ruff, autopep8).
- 4-space indentation throughout.
- Lines are not strictly length-limited; long expressions (especially math) are
  kept on one line for readability.
- Blank lines separate logical blocks within methods.

### Error Handling

- `ValueError` for invalid inputs:
  `raise ValueError("mode not valid")`,
  `raise ValueError("Jacobian Diag shape not valid: " + f"{JD.shape}")`.
- `NotImplementedError` for unimplemented abstract branches.
- `assert` for preconditions on numerical data:
  `assert butcherC[0] == 0`, `assert (butcherA[0, :] == 0).all()`.
- Convergence failures print warnings rather than raising:
  `print("did not converge")`.
- No `try/except` blocks in the codebase. Keep error handling minimal and direct.

### Numerical Conventions

- Heavy use of numpy broadcasting, `np.einsum`, `np.linalg.pinv`.
- Small epsilon values used to avoid division by zero: `1e-300`, `1e-100`.
- Variable naming follows physics/math conventions: `u` (solution), `rhs`
  (right-hand side), `dt` (time step), `dx` (spatial step), `nx` (grid count).
- f-strings for iteration output:
  `f"iter [{iStage},{iter}], resN [{resN:.4e} / {resN0:.4e}]"`.
- Default mutable arguments are used (e.g., `params={}`, `solve_opts={}`).
  This is intentional in this codebase -- do not "fix" it without being asked.

## Common Pitfalls

- Scripts must run from `reactionTest/` directory, not the repo root.
- `AdvReactUni.py` has deeply nested inner classes -- read carefully
  before modifying.
- The `DITRExp` solver passes lists of arrays and nested alpha coefficient lists
  to `fSolve` -- the interface differs from `ESDIRK`'s scalar `alphaRHS`.
- `recGrad` in `FVUni2nd.py` applies Barth-Jespersen limiting -- do not remove
  the limiter logic without understanding its role in preventing oscillations.
- `FsolveDITR` uses `isinstance(fRHS, FrhsDITRExp)` to correct the Jacobian
  preconditioner when the exponential RHS subtracts the `A*(u - u_ref)` term.
