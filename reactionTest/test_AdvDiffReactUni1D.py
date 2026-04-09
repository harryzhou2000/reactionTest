"""
Test: 1D advection-diffusion-reaction with bistable source.

    u_t + a u_x = eps * u_xx + k * u(1-u)(u-a)

The diffusion smears the reaction front, creating an internal layer of
width ~ sqrt(eps/k).  Splitting errors produce observable front speed
errors -- a clean scalar metric to compare methods.
"""

import numpy as np
import pathlib
import matplotlib.pyplot as plt
from Solver.AdvReactUni import AdvReactUni1DSolver, AdvReactUni1DEval
from Solver.FVUni2nd import FVUni2nd1D
from Solver.FVUniWENO5Z import FVUniWENO5Z1D
from Solver.ODE import ESDIRK, DITRExp
import PlotEnv

# ╔══════════════════════════════════════════════════════════════════╗
# ║  Problem configuration -- change only this block                 ║
# ╚══════════════════════════════════════════════════════════════════╝

# Grid
Nx = 128
rec_scheme = "weno5z"  # "muscl2" or "weno5z"
fmt_fig = "pdf"  # output figure format: "pdf", "png", etc.

# Time stepping
CFLt = 2  # CFL multiplier for coarse dt
dt = 1 / Nx / 2 * 2 * CFLt  # coarse time step
dtRef = 1 / Nx / 2 / 4  # fine reference time step
tEnd = 1

# Bistable reaction + diffusion parameters
a_react = 0.5  # bistable threshold
k_react = 1000  # reaction stiffness
eps_diff = 1e-1  # diffusion coefficient

# Solver tuning
CFL_ref = 1000  # pseudo-time CFL for reference
CFL_coarse = 10  # pseudo-time CFL for coarse runs
rel_tol = 1e-4
max_iter_exp = 50  # max iterations for exponential DITR

# Methods to run (comment out lines to skip)
enabled_methods = [
    "ref",
    "Strang",
    "Strang DITR",
    "fully implicit",
    "DITR",
    "Exp DITR",
    # "Embed implicit",
    # "Embed DITR",
]

# ═══════════════════════════════════════════════════════════════════

# ── Output directory ────────────────────────────────────────────────
script_dir = pathlib.Path(__file__).resolve().parent
pic_dir = script_dir / "pics" / "advdiffreact"
pic_dir.mkdir(parents=True, exist_ok=True)

# ── Setup ───────────────────────────────────────────────────────────
fv = {"muscl2": FVUni2nd1D, "weno5z": FVUniWENO5Z1D}[rec_scheme](nx=Nx)
ev = AdvReactUni1DEval(
    fv=fv,
    model="bistable",
    params={"a": a_react, "k": k_react, "eps": eps_diff},
)

solver4 = AdvReactUni1DSolver(eval=ev, ode=ESDIRK("ESDIRK4"))
solver = AdvReactUni1DSolver(eval=ev, ode=ESDIRK("ESDIRK3"))
solverDITR = AdvReactUni1DSolver(eval=ev, ode=DITRExp())

# Initial condition
u = np.array([np.sin(fv.xcs * np.pi * 2) * 0.5 + 0.5])

# ── Method registry ────────────────────────────────────────────────
method_runners = {
    "ref": lambda: solver4.stepInterval(
        dtRef,
        u,
        0.0,
        tEnd,
        mode="full",
        solve_opts={"CFL": CFL_ref},
    ),
    "Strang": lambda: solver.stepInterval(
        dt,
        u,
        0.0,
        tEnd,
        mode="strang",
        solve_opts={"CFL": CFL_coarse},
    ),
    "Strang DITR": lambda: solverDITR.stepInterval(
        dt,
        u,
        0.0,
        tEnd,
        mode="strang",
        solve_opts={"rel_tol": rel_tol, "CFL": CFL_coarse},
    ),
    "fully implicit": lambda: solver.stepInterval(
        dt,
        u,
        0.0,
        tEnd,
        solve_opts={"rel_tol": rel_tol, "CFL": CFL_coarse},
    ),
    "DITR": lambda: solverDITR.stepInterval(
        dt,
        u,
        0.0,
        tEnd,
        solve_opts={"rel_tol": rel_tol, "CFL": CFL_coarse},
        use_exp=False,
    ),
    "Exp DITR": lambda: solverDITR.stepInterval(
        dt,
        u,
        0.0,
        tEnd,
        solve_opts={"rel_tol": rel_tol, "CFL": CFL_coarse, "max_iter": max_iter_exp},
        use_exp=True,
    ),
    "Embed implicit": lambda: solver.stepInterval(
        dt,
        u,
        0.0,
        tEnd,
        mode="embed",
        solve_opts={"CFL": CFL_coarse},
    ),
    "Embed DITR": lambda: solverDITR.stepInterval(
        dt,
        u,
        0.0,
        tEnd,
        mode="embed",
        solve_opts={"rel_tol": rel_tol, "CFL": CFL_coarse},
        use_exp=False,
    ),
}

# ── Run selected methods ────────────────────────────────────────────
results = {}

for name in enabled_methods:
    runner = method_runners.get(name)
    if runner is None:
        print(f"WARNING: unknown method '{name}', skipping")
        continue
    print("=" * 60)
    print(name)
    print("=" * 60)
    try:
        sol = runner()
        results[name] = sol
        print(f"  >> {name} completed, uNorm = {np.linalg.norm(sol):.6e}")
    except Exception as e:
        print(f"  >> {name} FAILED: {e}")
        results[name] = None

# ── Plot ────────────────────────────────────────────────────────────
plotEnv = PlotEnv.PlotEnv(dpi=180, markEvery=10)
tag = f"k{k_react}_eps{eps_diff}_CFL{CFLt}_T{tEnd}_{rec_scheme}"

fig = plotEnv.figure(101, figsize=(6, 4))
for i, name in enumerate(enabled_methods):
    sol = results.get(name)
    if sol is not None:
        plotEnv.plot(fv.xcs, sol[0], plotIndex=i, label=name)
plt.legend()
plt.title(
    f"Adv-Diff-React  (k={k_react}, eps={eps_diff}, CFL={CFLt})"
    + (" WENO5" if rec_scheme == "weno5z" else "")
)
plt.xlabel("x")
plt.ylabel("u")
plt.savefig(pic_dir / f"advdiffreact_{tag}.{fmt_fig}", dpi=180, bbox_inches="tight")
plt.show()

# ── Error norms ─────────────────────────────────────────────────────
u1_ref = results.get("ref")
if u1_ref is not None:
    print("\n" + "=" * 60)
    print(
        f"L2 errors vs reference  (k={k_react}, eps={eps_diff}, CFL={CFLt}, T={tEnd}):"
    )
    for name in enabled_methods:
        if name == "ref":
            continue
        sol = results.get(name)
        if sol is not None:
            err = np.linalg.norm(sol - u1_ref) / np.linalg.norm(u1_ref)
            print(f"  {name:25s}: {err:.6e}")
        else:
            print(f"  {name:25s}: FAILED")
