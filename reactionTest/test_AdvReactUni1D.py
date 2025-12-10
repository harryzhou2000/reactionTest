import numpy as np
from Solver.AdvReactUni import AdvReactUni1DSolver, AdvReactUni1DEval
from Solver.FVUni2nd import FVUni2nd1D
from Solver.ODE import ESDIRK, DITRExp

Nx = 128

fv = FVUni2nd1D(nx=Nx)
ev = AdvReactUni1DEval(
    fv=fv,
    # model="bistable",
    # params={"a": 0.5, "k": 100},
)
# solver = AdvReactUni1DSolver(eval=ev, ode=ESDIRK("BackwardEuler"))
solver = AdvReactUni1DSolver(eval=ev, ode=DITRExp())

dt = 1 / Nx * 0.5 * 1
tEnd = 0.5

u = np.array([np.sin(fv.xcs * np.pi * 2)])

# u1 = solver.stepInterval(dt, u, 0.0, tEnd)

solverDITR = AdvReactUni1DSolver(eval=ev, ode=DITRExp())

u1Ditr = solverDITR.stepInterval(
    dt,
    u,
    0.0,
    tEnd,
    solve_opts={
        "rel_tol": 1e-4,
        "CFL": 10,
    },
    use_exp=True,
)
