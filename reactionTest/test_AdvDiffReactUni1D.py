"""
Test: 1D advection-diffusion-reaction with bistable source.

    u_t + a u_x = eps * u_xx + k * u(1-u)(u-a)

The diffusion smears the reaction front, creating an internal layer of
width ~ sqrt(eps/k).  Splitting errors produce observable front speed
errors -- a clean scalar metric to compare methods.

This script mirrors test_AdvReactUni1D.ipynb but adds viscous diffusion.
"""

import numpy as np
import matplotlib.pyplot as plt
from Solver.AdvReactUni import AdvReactUni1DSolver, AdvReactUni1DEval
from Solver.FVUni2nd import FVUni2nd1D
from Solver.ODE import ESDIRK, DITRExp
import PlotEnv

# ── Grid and time stepping ──────────────────────────────────────────
Nx = 128
CFLt = 2
dt = 1 / Nx / 2 * 2 * CFLt
dtRef = 1 / Nx / 2 / 4
tEnd = 0.5  # shorter than pure-advection test (front doesn't wrap)

# ── Physics ─────────────────────────────────────────────────────────
fv = FVUni2nd1D(nx=Nx)
ev = AdvReactUni1DEval(
    fv=fv,
    model="bistable",
    params={"a": 0.5, "k": 1000, "eps": 1e-3},
)

# ── Solvers ─────────────────────────────────────────────────────────
solver4 = AdvReactUni1DSolver(eval=ev, ode=ESDIRK("ESDIRK4"))
solver = AdvReactUni1DSolver(eval=ev, ode=ESDIRK("ESDIRK3"))
solverDITR = AdvReactUni1DSolver(eval=ev, ode=DITRExp())

# ── Initial condition ───────────────────────────────────────────────
u = np.array([np.sin(fv.xcs * np.pi * 2) * 0.5 + 0.5])

# ── Reference (ESDIRK4, fine dt, fully implicit) ───────────────────
print("=" * 60)
print("Reference: ESDIRK4, fully implicit, fine dt")
print("=" * 60)
u1_ref = solver4.stepInterval(
    dtRef, u, 0.0, tEnd,
    mode="full",
    solve_opts={"CFL": 1000},
)

# ── Strang splitting ────────────────────────────────────────────────
print("=" * 60)
print("Strang splitting")
print("=" * 60)
u1_strang = solver.stepInterval(
    dt, u, 0.0, tEnd,
    mode="strang",
    solve_opts={"CFL": 10},
)

# ── Fully implicit ESDIRK3 ─────────────────────────────────────────
print("=" * 60)
print("Fully implicit ESDIRK3")
print("=" * 60)
u1 = solver.stepInterval(
    dt, u, 0.0, tEnd,
    solve_opts={"rel_tol": 1e-4, "CFL": 10},
)

# ── DITR ────────────────────────────────────────────────────────────
print("=" * 60)
print("DITR")
print("=" * 60)
u1Ditr = solverDITR.stepInterval(
    dt, u, 0.0, tEnd,
    solve_opts={"rel_tol": 1e-4, "CFL": 10},
    use_exp=False,
)

# ── Exponential DITR ────────────────────────────────────────────────
print("=" * 60)
print("Exponential DITR")
print("=" * 60)
u1DitrExp = solverDITR.stepInterval(
    dt, u, 0.0, tEnd,
    solve_opts={"rel_tol": 1e-4, "CFL": 10, "max_iter": 50},
    use_exp=True,
)

# ── Embedded ESDIRK3 ───────────────────────────────────────────────
print("=" * 60)
print("Embedded ESDIRK3")
print("=" * 60)
u1_embed = solver.stepInterval(
    dt, u, 0.0, tEnd,
    mode="embed",
    solve_opts={"CFL": 10},
)

# ── Embedded DITR ───────────────────────────────────────────────────
print("=" * 60)
print("Embedded DITR")
print("=" * 60)
u1DitrEmbed = solverDITR.stepInterval(
    dt, u, 0.0, tEnd,
    mode="embed",
    solve_opts={"rel_tol": 1e-4, "CFL": 10},
    use_exp=False,
)

# ── Plot ────────────────────────────────────────────────────────────
plotEnv = PlotEnv.PlotEnv(dpi=180, markEvery=10)

fig = plotEnv.figure(101, figsize=(6, 4))
plotEnv.plot(fv.xcs, u1_ref[0], plotIndex=0, label="ref")
plotEnv.plot(fv.xcs, u1_strang[0], plotIndex=1, label="Strang splitting")
plotEnv.plot(fv.xcs, u1[0], plotIndex=2, label="fully implicit")
plotEnv.plot(fv.xcs, u1Ditr[0], plotIndex=3, label="DITR")
plotEnv.plot(fv.xcs, u1DitrExp[0], plotIndex=4, label="Exponential DITR")
plotEnv.plot(fv.xcs, u1_embed[0], plotIndex=5, label="Embed implicit")
plotEnv.plot(fv.xcs, u1DitrEmbed[0], plotIndex=6, label="Embed DITR")
plt.legend()
plt.title("Advection-Diffusion-Reaction (bistable, eps=1e-3)")
plt.xlabel("x")
plt.ylabel("u")
plt.savefig("test_AdvDiffReactUni1D.png", dpi=180, bbox_inches="tight")
plt.show()

# ── Error norms ─────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("L2 errors vs reference:")
for label, sol in [
    ("Strang splitting", u1_strang),
    ("Fully implicit", u1),
    ("DITR", u1Ditr),
    ("Exponential DITR", u1DitrExp),
    ("Embed implicit", u1_embed),
    ("Embed DITR", u1DitrEmbed),
]:
    err = np.linalg.norm(sol - u1_ref) / np.linalg.norm(u1_ref)
    print(f"  {label:25s}: {err:.6e}")
