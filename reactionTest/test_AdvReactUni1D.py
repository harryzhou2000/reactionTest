"""
Test: 1D advection-reaction with bistable source.

    u_t + a u_x = k * u(1-u)(u-a)

Quick single-run test with DITRExp.
"""

import numpy as np
from Solver.AdvReactUni import AdvReactUni1DEval
import TestCommon as TC

# ╔══════════════════════════════════════════════════════════════════╗
# ║  Configuration                                                  ║
# ╚══════════════════════════════════════════════════════════════════╝

Nx = 128
rec_scheme = "muscl2"  # "muscl2" or "weno5z"

# ═══════════════════════════════════════════════════════════════════

fv = TC.make_fv(rec_scheme, Nx)

ev_params = dict(model="bistable", params={"a": 0.5, "k": 1000})
ev = TC.make_ev(fv, **ev_params)
ev_ps = TC.make_ev(fv, **ev_params, source_quadrature=3)

solver_sets = {
    "": TC.SolverSet(ev),
    " p-source": TC.SolverSet(ev_ps),
}

dt = 1 / Nx * 0.5
tEnd = 0.5
u0 = np.array([np.sin(fv.xcs * np.pi * 2)]) * 0.5 + 1

enabled_methods = [
    "DITR U2R2",
    "DITR U2R1",
    "DITR U2R2 p-source",
    "DITR U2R1 p-source",
]

runners = TC.build_method_runners(
    solver_sets, dt, dt, u0, tEnd,
    CFL_ref=10, CFL_coarse=10, rel_tol=1e-4,
    record_probes=False,
)
results, _ = TC.run_methods(runners, enabled_methods)
