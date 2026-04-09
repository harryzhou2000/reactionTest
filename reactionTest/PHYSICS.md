# PHYSICS.md

Physical models and numerical methods in this codebase.

## Governing Equations

The general 1D system solved is:

    u_t + a * u_x = eps * u_xx + S(u)

where `u` is the solution vector (scalar or multi-component), `a` is advection
speed, `eps` is per-component diffusion, and `S(u)` is the reaction source.

The spatial domain is `[0, 1]` on a uniform grid of `nx` cells. Time
integration uses implicit methods (ESDIRK or DITRExp). The flow term
(advection + diffusion) is evaluated together in `rhs_flow`; the source
`S(u)` is evaluated in `rhs_source`.

## Reaction Models

Each model is selected by passing `model="..."` to `AdvReactUni1DEval`.
The source function and its Jacobian are bound once at construction time
(no per-call branching).

### Bistable (`model="bistable"`)

Single scalar variable `u`, 1 component.

    S(u) = k * u * (1 - u) * (u - a)

Parameters: `a` (threshold, typically 0.5), `k` (rate).

Jacobian (scalar per point):

    dS/du = k * (2u - a + 2au - 3u^2)

Test scripts: `test_AdvReactUni1D.py`, `test_AdvDiffReactUni1D.py`.

### Brusselator (`model="brusselator"`)

Two-component system `[u, v]`.

    S_u = k * (A - (B+1)*u + u^2*v)
    S_v = k * (B*u - u^2*v)

Parameters: `A`, `B`, `k`. Limit cycle oscillations occur when `B > 1 + A^2`.

Jacobian (2x2 per point):

    J = k * [[ -(B+1) + 2uv,   u^2      ],
             [  B - 2uv,       -u^2      ]]

Test scripts: `test_BrusselatorUni1D.py`, `sweep_brusselator.py`.

### Premixed combustion (`model="premixed"`)

Two-component system `[T, Y]` where `T` is temperature and `Y` is fuel
mass fraction. The Arrhenius reaction rate is:

    omega = B * Y * exp(-Ze * (Tb / T - 1))

where `Ze = E / (R * Tb)` is the Zeldovich number. The sources are:

    S_T = Q * omega       (heat release)
    S_Y = -omega          (fuel consumption)

Parameters (key in `params` dict):
- `B` -- pre-exponential factor (reaction time scale).
- `Q_div_rho_cp` -- heat release per unit fuel, expressed as temperature rise.
- `Tb` -- burnt (adiabatic flame) temperature.
- `E_div_RTb` -- Zeldovich number.

Derived quantity: `T0 = Tb - Q` (unburnt temperature).

Jacobian (2x2 per point):

    domega/dT = B * Y * Ze * Tb / T^2 * exp(...)
    domega/dY = B * exp(...)
    J = [[ Q * domega/dT,   Q * domega/dY ],
         [  -domega/dT,      -domega/dY   ]]

Test script: `test_PremixedUni1D.py`.

### No reaction (`model=""`)

Returns zero source and near-zero Jacobian (`1e-100`).

## Spatial Discretization

`FVUni2nd1D` implements a 2nd-order finite volume method on a uniform 1D
grid of `nx` cells over `[0, 1]`.

### Advection

Upwind flux with 2nd-order MUSCL reconstruction. The gradient is limited
by Barth-Jespersen to prevent oscillations near discontinuities.

### Diffusion

Central-difference stencil: `eps * (u[i-1] - 2*u[i] + u[i+1]) / hx^2`.
The diffusion coefficient `eps` is per-component, stored as shape
`(nVars, 1)` in `AdvReactUni1DEval.eps`. It is set via the `"eps"` key
in `params`:
- Scalar: broadcast to all components.
- Array `[eps_0, eps_1, ...]`: one value per variable. Pass `nVars=`
  to the constructor so the broadcast is correct.

### Boundary Conditions

Controlled by `FVUni2nd1D`. Default is **periodic** (cyclic `np.roll`).

**Dirichlet** BCs are activated by calling `fv.set_bc_dirichlet(uL, uR)`
before the solve. The implementation:

- **Reconstruction:** ghost cell value = BC value, ghost gradient = 0.
  The reconstructed face value at the boundary equals the BC value.
- **Flux:** the boundary face sees `uR = bcValue` from the ghost side.
- **Diffusion:** the rolled neighbor at the boundary is replaced by the
  BC value, giving a standard one-sided FD stencil.

The `cellOthers` generator accepts a `homogeneous` flag. When `True`,
boundary ghosts are set to zero instead of BC values. This is used
internally by the Jacobi iteration for the correction `du`, which must
satisfy homogeneous Dirichlet (the BC is already incorporated in the
residual).

## Time Integration

### ESDIRK

Singly diagonally implicit Runge-Kutta. Available tableaux (from
`ESDIRK_Data.py`): BackwardEuler, Trapezoid, ESDIRK3, ESDIRK4.

Each implicit stage is solved by pseudo-time Jacobi iteration. The
preconditioner includes both flow and source Jacobian diagonals.

### DITRExp

Fully implicit 2-stage method. Its two coupled stages are solved in
alternating sweeps until convergence. Supports an exponential variant
(`use_exp=True`) that extracts the stable linear part of the source
Jacobian into `exp(A*dt)` and phi-functions.

### Splitting Modes

`stepInterval()` supports three modes:
- `"full"` -- monolithic: flow and source coupled in one implicit system.
- `"strang"` -- Strang splitting: half-flow, source, half-flow.
- `"embed"` -- embedded splitting: source sub-steps at each RK stage
  node, flow solved implicitly with forcing correction.

### Pseudo-time CFL

The implicit stage solve uses pseudo-time relaxation. The pseudo-time
step is `dTau = CFL * dt_char`, where `dt_char` is the minimum of:
- `hx / |ax|` (advection CFL limit, skipped when `ax = 0`)
- `hx^2 / max(eps)` (diffusion CFL limit, skipped when `eps = 0`)

`CFL` is set via `solve_opts={"CFL": value}`. Typical values:
~1000 for well-conditioned advection-dominated problems, ~1--100 for
diffusion-reaction problems.

## Test Scripts

All scripts run from the `reactionTest/` subdirectory.

| Script | Model | Features |
|--------|-------|----------|
| `test_ODE.py` | Dense-matrix exponential ODE | Unit test for ESDIRK and DITRExp |
| `test_AdvReactUni1D.py` | Bistable | Advection + reaction, DITR |
| `test_AdvDiffReactUni1D.py` | Bistable | Advection + diffusion + reaction |
| `test_BrusselatorUni1D.py` | Brusselator | 2-species, periodic BC, probes |
| `test_PremixedUni1D.py` | Premixed | Diffusion-reaction, Dirichlet BC, probes |
| `sweep_brusselator.py` | Brusselator | Parameter sweep (k, CFL) |
| `test_AdvReactUni1D.ipynb` | Bistable | Primary 7-method comparison (Jupyter) |
| `test_AdvReactUni1D_DITR.ipynb` | None | DITRExp pure advection (Jupyter) |

## Project Structure

```
reactionTest/
  __init__.py
  PlotEnv.py                    # Matplotlib helpers (scienceplots styles)
  Solver/
    __init__.py
    FVUni2nd.py                 # 2nd-order FV grid, BCs, reconstruction
    AdvReactUni.py              # Evaluator (rhs_flow, models) + solver wrapper
    AdvReactUniFunctors.py      # Functor classes for ODE integrators
    ODE.py                      # ESDIRK, DITRExp integrator framework
    ESDIRK_Data.py              # Butcher tableau coefficients
  test_ODE.py
  test_AdvReactUni1D.py
  test_AdvDiffReactUni1D.py
  test_BrusselatorUni1D.py
  test_PremixedUni1D.py
  sweep_brusselator.py
  test_AdvReactUni1D.ipynb
  test_AdvReactUni1D_DITR.ipynb
```
