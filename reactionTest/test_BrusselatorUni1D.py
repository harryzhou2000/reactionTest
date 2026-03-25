"""
Test: 1D advection + Brusselator reaction (2-species system).

    u_t + a u_x = k * (A - (B+1)*u + u^2 * v)
    v_t + a v_x = k * (B*u - u^2 * v)

The Brusselator has a limit cycle for B > 1 + A^2, producing persistent
oscillatory structure where splitting errors accumulate.  The source
Jacobian is a dense 2x2 matrix per point, exercising the ndim==3 code
paths in the exponential integrator.

This script tests all 7 solver variants.
"""

import numpy as np
import matplotlib.pyplot as plt
from Solver.AdvReactUni import AdvReactUni1DSolver, AdvReactUni1DEval
from Solver.FVUni2nd import FVUni2nd1D
from Solver.ODE import ESDIRK, DITRExp
import PlotEnv

# ── Grid and time stepping ──────────────────────────────────────────
Nx = 128
CFLt = 1
dt = 1 / Nx / 2 * 2 * CFLt
dtRef = 1 / Nx / 2 / 4
tEnd = 0.25

# ── Physics ─────────────────────────────────────────────────────────
# Brusselator: limit cycle when B > 1 + A^2 = 2
# Steady state: u_ss = A, v_ss = B/A
A_br = 1.0
B_br = 3.0
k_br = 20  # stiffness (moderate: still stiff enough for splitting errors)

fv = FVUni2nd1D(nx=Nx)
ev = AdvReactUni1DEval(
    fv=fv,
    model="brusselator",
    params={"A": A_br, "B": B_br, "k": k_br},
)

# ── Solvers ─────────────────────────────────────────────────────────
solver4 = AdvReactUni1DSolver(eval=ev, ode=ESDIRK("ESDIRK4"))
solver = AdvReactUni1DSolver(eval=ev, ode=ESDIRK("ESDIRK3"))
solverDITR = AdvReactUni1DSolver(eval=ev, ode=DITRExp())

# ── Initial condition: perturbation of steady state ────────────────
# u_ss = A, v_ss = B/A
u_ss = A_br
v_ss = B_br / A_br
u0_u = u_ss + 0.5 * np.sin(fv.xcs * np.pi * 2)
u0_v = v_ss + 0.5 * np.cos(fv.xcs * np.pi * 2)
u0 = np.array([u0_u, u0_v])

# ── Helper to run each method safely ───────────────────────────────
results = {}


def run_method(name, func):
    print("=" * 60)
    print(name)
    print("=" * 60)
    try:
        sol = func()
        results[name] = sol
        print(f"  >> {name} completed, uNorm = {np.linalg.norm(sol):.6e}")
    except Exception as e:
        print(f"  >> {name} FAILED: {e}")
        results[name] = None


# ── Reference (ESDIRK4, fine dt, fully implicit) ───────────────────
run_method("ref", lambda: solver4.stepInterval(
    dtRef, u0, 0.0, tEnd, mode="full", solve_opts={"CFL": 1000},
))

# ── Strang splitting ────────────────────────────────────────────────
run_method("Strang", lambda: solver.stepInterval(
    dt, u0, 0.0, tEnd, mode="strang", solve_opts={"CFL": 100},
))

# ── Fully implicit ESDIRK3 ─────────────────────────────────────────
run_method("fully implicit", lambda: solver.stepInterval(
    dt, u0, 0.0, tEnd, solve_opts={"rel_tol": 1e-4, "CFL": 100},
))

# ── DITR ────────────────────────────────────────────────────────────
run_method("DITR", lambda: solverDITR.stepInterval(
    dt, u0, 0.0, tEnd, solve_opts={"rel_tol": 1e-4, "CFL": 100}, use_exp=False,
))

# ── Exponential DITR ────────────────────────────────────────────────
run_method("Exp DITR", lambda: solverDITR.stepInterval(
    dt, u0, 0.0, tEnd,
    solve_opts={"rel_tol": 1e-4, "CFL": 100, "max_iter": 100}, use_exp=True,
))

# ── Embedded ESDIRK3 ───────────────────────────────────────────────
run_method("Embed implicit", lambda: solver.stepInterval(
    dt, u0, 0.0, tEnd, mode="embed", solve_opts={"CFL": 100},
))

# ── Embedded DITR ───────────────────────────────────────────────────
run_method("Embed DITR", lambda: solverDITR.stepInterval(
    dt, u0, 0.0, tEnd, mode="embed",
    solve_opts={"rel_tol": 1e-4, "CFL": 100}, use_exp=False,
))

# ── Plot ────────────────────────────────────────────────────────────
plotEnv = PlotEnv.PlotEnv(dpi=180, markEvery=10)
plotLabels = ["ref", "Strang", "fully implicit", "DITR",
              "Exp DITR", "Embed implicit", "Embed DITR"]

# Species u
fig = plotEnv.figure(201, figsize=(6, 4))
for i, name in enumerate(plotLabels):
    sol = results.get(name)
    if sol is not None:
        plotEnv.plot(fv.xcs, sol[0], plotIndex=i, label=name)
plt.legend()
plt.title(f"Brusselator: species u (k={k_br})")
plt.xlabel("x")
plt.ylabel("u")
plt.savefig("test_BrusselatorUni1D_u.png", dpi=180, bbox_inches="tight")
plt.show()

# Species v
fig = plotEnv.figure(202, figsize=(6, 4))
for i, name in enumerate(plotLabels):
    sol = results.get(name)
    if sol is not None:
        plotEnv.plot(fv.xcs, sol[1], plotIndex=i, label=name)
plt.legend()
plt.title(f"Brusselator: species v (k={k_br})")
plt.xlabel("x")
plt.ylabel("v")
plt.savefig("test_BrusselatorUni1D_v.png", dpi=180, bbox_inches="tight")
plt.show()

# ── Error norms ─────────────────────────────────────────────────────
u1_ref = results.get("ref")
if u1_ref is not None:
    print("\n" + "=" * 60)
    print("L2 errors vs reference (both species):")
    for name in plotLabels[1:]:
        sol = results.get(name)
        if sol is not None:
            err = np.linalg.norm(sol - u1_ref) / np.linalg.norm(u1_ref)
            print(f"  {name:25s}: {err:.6e}")
        else:
            print(f"  {name:25s}: FAILED")
