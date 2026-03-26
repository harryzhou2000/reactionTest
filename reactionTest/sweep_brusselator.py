"""
Quick sweep to find where Strang splitting is significantly bad
for the Brusselator test case.
"""

import numpy as np
import sys, io, contextlib
from Solver.AdvReactUni import AdvReactUni1DSolver, AdvReactUni1DEval
from Solver.FVUni2nd import FVUni2nd1D
from Solver.ODE import ESDIRK

Nx = 64  # coarser for speed
fv = FVUni2nd1D(nx=Nx)

A_br = 1.0
B_br = 3.0
u0_u = A_br + 0.5 * np.sin(fv.xcs * np.pi * 2)
u0_v = B_br / A_br + 0.5 * np.cos(fv.xcs * np.pi * 2)
u0 = np.array([u0_u, u0_v])

dtRef_base = 1 / Nx / 2 / 8

def run_silent(func):
    with contextlib.redirect_stdout(io.StringIO()):
        return func()

print(f"{'k':>6s} {'CFLt':>6s} {'tEnd':>6s} | {'E_strang':>12s} {'E_implicit':>12s} {'ratio':>8s}")
print("-" * 72)

for k_br in [5, 10, 20, 50, 100]:
    for CFLt in [0.5, 1, 2, 4]:
        tEnd = 0.2
        dt = 1 / Nx / 2 * 2 * CFLt

        ev = AdvReactUni1DEval(
            fv=fv, model="brusselator",
            params={"A": A_br, "B": B_br, "k": k_br},
        )
        solver4 = AdvReactUni1DSolver(eval=ev, ode=ESDIRK("ESDIRK4"))
        solver = AdvReactUni1DSolver(eval=ev, ode=ESDIRK("ESDIRK3"))

        try:
            u1_ref = run_silent(lambda: solver4.stepInterval(
                dtRef_base, u0, 0.0, tEnd, mode="full",
                solve_opts={"CFL": 1000},
            ))
            ref_norm = np.linalg.norm(u1_ref)

            u1_strang = run_silent(lambda: solver.stepInterval(
                dt, u0, 0.0, tEnd, mode="strang",
                solve_opts={"CFL": 100},
            ))
            e_strang = np.linalg.norm(u1_strang - u1_ref) / ref_norm

            u1_impl = run_silent(lambda: solver.stepInterval(
                dt, u0, 0.0, tEnd,
                solve_opts={"rel_tol": 1e-4, "CFL": 100},
            ))
            e_impl = np.linalg.norm(u1_impl - u1_ref) / ref_norm

            ratio = e_strang / max(e_impl, 1e-30)
            print(f"{k_br:6d} {CFLt:6.1f} {tEnd:6.2f} | {e_strang:12.4e} {e_impl:12.4e} {ratio:8.2f}")
        except Exception as e:
            print(f"{k_br:6d} {CFLt:6.1f} {tEnd:6.2f} | FAILED: {type(e).__name__}")
        sys.stdout.flush()
