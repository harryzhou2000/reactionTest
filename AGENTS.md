# AGENTS.md

## Project Overview

Research codebase exploring implicit ODE methods for advection-diffusion-reaction
PDEs. The goal is to evaluate alternatives to Strang splitting -- including fully
implicit, embedded splitting, and exponential DITR -- for eventual use in
reaction-enabled Navier-Stokes solvers. Test problems include 1D bistable
reaction, Brusselator oscillator, and premixed combustion flame.

**Language:** Python 3.12
**Package manager:** Conda (configured via `.vscode/settings.json`)
**Dependencies:** numpy, scipy, matplotlib, scienceplots
**No formal build system** -- no `pyproject.toml`, `setup.py`, or `requirements.txt`.

For physical models, spatial discretization, boundary conditions, and test
problem descriptions, see [`reactionTest/PHYSICS.md`](reactionTest/PHYSICS.md).

## Project Structure

```
reactionTest/                       # Main Python package (working directory)
  __init__.py
  PlotEnv.py                        # Matplotlib helpers (scienceplots styles)
  PHYSICS.md                        # Models, numerics, and test descriptions
  Solver/
    __init__.py
    FVUni2nd.py                     # 2nd-order FV grid, BCs, reconstruction
    AdvReactUni.py                  # Evaluator (rhs_flow, models) + solver wrapper
    AdvReactUniFunctors.py          # Functor classes for ODE integrators
    ODE.py                          # ESDIRK, DITRExp integrator framework + ABCs
    ESDIRK_Data.py                  # Butcher tableau coefficients
  test_ODE.py                       # Unit test for ODE integrators
  test_AdvReactUni1D.py             # Bistable advection-reaction
  test_AdvDiffReactUni1D.py         # Bistable advection-diffusion-reaction
  test_BrusselatorUni1D.py          # Brusselator 2-species, periodic BC, probes
  test_PremixedUni1D.py             # Premixed combustion, Dirichlet BC, probes
  sweep_brusselator.py              # Brusselator parameter sweep
  test_AdvReactUni1D.ipynb          # Primary 7-method comparison (Jupyter)
  test_AdvReactUni1D_DITR.ipynb     # DITRExp pure advection (Jupyter)
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
python test_PremixedUni1D.py

# Run a module directly (some files have __main__ blocks)
python Solver/FVUni2nd.py
```

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

### Source Function Dispatch

`AdvReactUni1DEval` binds model-specific `rhs_source` and `rhs_source_jacobian`
methods at construction time based on the `model` string. This avoids if-elif
branching on every call during the implicit iteration. The bound methods are
named `_rhs_source_<model>` and `_rhs_source_jacobian_<model>`.

### Boundary Conditions

`FVUni2nd1D` defaults to periodic BC. Call `set_bc_dirichlet(uL, uR)` to switch
to Dirichlet. The `cellOthers` generator accepts `homogeneous=True` to zero out
boundary ghosts -- used internally by Jacobi iterations on the correction `du`.

### Probe Recording

`AdvReactUni1DSolver` records time series at specified spatial locations:

- `set_probes(x_locations)` -- set probe locations (mapped to nearest cell center).
- `clear_probes()` -- clear recorded data for all probes.
- `get_probe_data(x=None)` -- retrieve recorded data (returns a deep copy).
- `stepInterval(..., record_probes=True)` -- enable recording during integration.

`get_probe_data()` returns a deep copy, so callers are not affected by
subsequent `clear_probes()` calls on the same solver instance.

### Per-Component Diffusion

`AdvReactUni1DEval` stores diffusion coefficients as `self.eps` with shape
`(nVars, 1)`. Pass `nVars=` to the constructor when using per-component values
(e.g., `params={"eps": [0.02, 0.01]}, nVars=2`). A scalar `eps` broadcasts to
all components.

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

When `A = 0`, exponential DITR falls back to standard DITR.

**Jacobian dimension support** (`invert_jacobian_diag` / `jacobian_diag_mult`):

| `ndim` | Shape              | Meaning                  | Inversion     | Multiply        |
|--------|--------------------|--------------------------|---------------|-----------------|
| 2      | `(nVars, nx)`      | Per-element scalar diag  | `1.0 / JD`   | `JD * u`        |
| 3      | `(nVars, nVars, nx)` | Per-point dense matrix | `pinv` batch  | `einsum ij..,j..->i..` |

**Stability clamping:** `FrhsDITRExp.JacobianExpo` clamps `currentA` to strictly
negative values. Positive eigenvalues would cause `exp(A*dt)` to blow up.

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
- The `DITRExp` solver passes lists of arrays and nested alpha coefficient lists
  to `fSolve` -- the interface differs from `ESDIRK`'s scalar `alphaRHS`.
- `recGrad` in `FVUni2nd.py` applies Barth-Jespersen limiting -- do not remove
  the limiter logic without understanding its role in preventing oscillations.
- `FsolveDITR` uses `isinstance(fRHS, FrhsDITRExp)` to correct the Jacobian
  preconditioner when the exponential RHS subtracts the `A*(u - u_ref)` term.
- When multiple methods share one solver instance, `get_probe_data()` returns
  a deep copy to prevent `clear_probes()` from clobbering earlier results.
- Jacobi iterations on correction `du` must use `cellOthers(..., homogeneous=True)`
  for Dirichlet BCs. The boundary ghost for `du` is zero (homogeneous), not
  the physical BC value.
