"""
Test: 1D advection + Brusselator reaction (2-species system).

    u_t + a u_x = k * (A - (B+1)*u + u^2 * v)
    v_t + a v_x = k * (B*u - u^2 * v)

The Brusselator has a limit cycle for B > 1 + A^2, producing persistent
oscillatory structure where splitting errors accumulate.  The source
Jacobian is a dense 2x2 matrix per point, exercising the ndim==3 code
paths in the exponential integrator.
"""

import numpy as np
import pathlib
import matplotlib.pyplot as plt
from Solver.AdvReactUni import AdvReactUni1DSolver, AdvReactUni1DEval
from Solver.FVUni2nd import FVUni2nd1D
from Solver.ODE import ESDIRK, DITRExp
import PlotEnv

# ╔══════════════════════════════════════════════════════════════════╗
# ║  Problem configuration -- change only this block               ║
# ╚══════════════════════════════════════════════════════════════════╝

# Grid
Nx = 128

# Time stepping
CFLt = 1                        # CFL multiplier for coarse dt
dt = 1 / Nx / 2 * 2 * CFLt       # coarse time step
dtRef = 1 / Nx / 2 / 8           # fine reference time step
tEnd = 4.0

# Brusselator parameters  (limit cycle when B > 1 + A^2)
A_br = 1.0
B_br = 3.0
k_br = 50

# Solver tuning
CFL_ref = 1000                    # pseudo-time CFL for reference
CFL_coarse = 10                  # pseudo-time CFL for coarse runs
rel_tol = 1e-4
max_iter_exp = 50                # max iterations for exponential DITR

# Methods to run (comment out lines to skip)
enabled_methods = [
    "ref",
    "Strang",
    "Strang DITR",
    # "fully implicit",
    "DITR",
    "Exp DITR",
    # "Embed implicit",
    # "Embed DITR",
]

# ═══════════════════════════════════════════════════════════════════

# ── Output directory ────────────────────────────────────────────────
script_dir = pathlib.Path(__file__).resolve().parent
pic_dir = script_dir / "pics" / "brusselator"
pic_dir.mkdir(parents=True, exist_ok=True)

# ── Setup ───────────────────────────────────────────────────────────
fv = FVUni2nd1D(nx=Nx)
ev = AdvReactUni1DEval(
    fv=fv,
    model="brusselator",
    params={"A": A_br, "B": B_br, "k": k_br},
)

solver4 = AdvReactUni1DSolver(eval=ev, ode=ESDIRK("ESDIRK4"))
solver = AdvReactUni1DSolver(eval=ev, ode=ESDIRK("ESDIRK3"))
solverDITR = AdvReactUni1DSolver(eval=ev, ode=DITRExp())

# Initial condition: perturbation of steady state (u_ss=A, v_ss=B/A)
u0_u = A_br + 0.5 * np.sin(fv.xcs * np.pi * 2)
u0_v = B_br / A_br + 0.5 * np.cos(fv.xcs * np.pi * 2)
u0 = np.array([u0_u, u0_v])

# ── Method registry ────────────────────────────────────────────────
method_runners = {
    "ref": lambda: solver4.stepInterval(
        dtRef, u0, 0.0, tEnd, mode="full",
        solve_opts={"CFL": CFL_ref},
    ),
    "Strang": lambda: solver.stepInterval(
        dt, u0, 0.0, tEnd, mode="strang",
        solve_opts={"CFL": CFL_coarse},
    ),
    "Strang DITR": lambda: solverDITR.stepInterval(
        dt, u0, 0.0, tEnd, mode="strang",
        solve_opts={"rel_tol": rel_tol, "CFL": CFL_coarse},
    ),
    "fully implicit": lambda: solver.stepInterval(
        dt, u0, 0.0, tEnd,
        solve_opts={"rel_tol": rel_tol, "CFL": CFL_coarse},
    ),
    "DITR": lambda: solverDITR.stepInterval(
        dt, u0, 0.0, tEnd,
        solve_opts={"rel_tol": rel_tol, "CFL": CFL_coarse}, use_exp=False,
    ),
    "Exp DITR": lambda: solverDITR.stepInterval(
        dt, u0, 0.0, tEnd,
        solve_opts={"rel_tol": rel_tol, "CFL": CFL_coarse,
                    "max_iter": max_iter_exp},
        use_exp=True,
    ),
    "Embed implicit": lambda: solver.stepInterval(
        dt, u0, 0.0, tEnd, mode="embed",
        solve_opts={"CFL": CFL_coarse},
    ),
    "Embed DITR": lambda: solverDITR.stepInterval(
        dt, u0, 0.0, tEnd, mode="embed",
        solve_opts={"rel_tol": rel_tol, "CFL": CFL_coarse}, use_exp=False,
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
tag = f"k{k_br}_CFL{CFLt}_T{tEnd}"

# Species u
fig = plotEnv.figure(201, figsize=(6, 4))
for i, name in enumerate(enabled_methods):
    sol = results.get(name)
    if sol is not None:
        plotEnv.plot(fv.xcs, sol[0], plotIndex=i, label=name)
plt.legend()
plt.title(f"Brusselator u  (k={k_br}, CFL={CFLt}, T={tEnd})")
plt.xlabel("x")
plt.ylabel("u")
plt.savefig(pic_dir / f"brusselator_u_{tag}.png", dpi=180, bbox_inches="tight")
plt.show()

# Species v
fig = plotEnv.figure(202, figsize=(6, 4))
for i, name in enumerate(enabled_methods):
    sol = results.get(name)
    if sol is not None:
        plotEnv.plot(fv.xcs, sol[1], plotIndex=i, label=name)
plt.legend()
plt.title(f"Brusselator v  (k={k_br}, CFL={CFLt}, T={tEnd})")
plt.xlabel("x")
plt.ylabel("v")
plt.savefig(pic_dir / f"brusselator_v_{tag}.png", dpi=180, bbox_inches="tight")
plt.show()

# ── Error norms ─────────────────────────────────────────────────────
u1_ref = results.get("ref")
if u1_ref is not None:
    print("\n" + "=" * 60)
    print(f"L2 errors vs reference  (k={k_br}, CFL={CFLt}, T={tEnd}):")
    for name in enabled_methods:
        if name == "ref":
            continue
        sol = results.get(name)
        if sol is not None:
            err = np.linalg.norm(sol - u1_ref) / np.linalg.norm(u1_ref)
            print(f"  {name:25s}: {err:.6e}")
        else:
            print(f"  {name:25s}: FAILED")
