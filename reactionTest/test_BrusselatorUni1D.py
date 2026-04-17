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
from Solver.AdvReactUni import AdvReactUni1DEval
import TestCommon as TC

# ╔══════════════════════════════════════════════════════════════════╗
# ║  Problem configuration -- change only this block               ║
# ╚══════════════════════════════════════════════════════════════════╝

# Grid
Nx = 128
rec_scheme = "weno5z"  # "muscl2" or "weno5z"
fmt_fig = "png"  # output figure format: "pdf", "png", etc.
show_title = False  # show titles on plots

# Time stepping
CFLt = 1  # CFL multiplier for coarse dt
dt = 1 / Nx / 2 * 2 * CFLt  # coarse time step
dtRef = 1 / Nx / 2 / 4  # fine reference time step
tEnd = 1.0

# Brusselator parameters  (limit cycle when B > 1 + A^2)
A_br = 1.0
B_br = 3.0
k_br = 50

# Solver tuning
CFL_ref = 1000  # pseudo-time CFL for reference
CFL_coarse = 5  # pseudo-time CFL for coarse runs
rel_tol = 1e-4
max_iter_exp = 50  # max iterations for exponential DITR
ref_suffix = " p-source"  # "" = base evaluator, " p-source" = quadrature source for ref
# ref_suffix = ""
chi_split_width = None
chi_split_threshold = None

# Methods to run
enabled_methods = [
    "ref",
    "DITR U2R2",
    # "DITR U2R1",
    # "ESDIRK3",
    # "ESDIRK4",
    # "Strang ESDIRK3",
    # "Strang DITR U2R2",
    "Masked Strang ESDIRK3",
    "Masked Strang DITR U2R2",
    # "Strang DITR U2R1",
    # "DITR U2R2 p-source",
    # "Strang DITR U2R2 p-source",
    # "fully implicit p-source",
    # "DITR U2R2 p-source",
    # "Exp DITR U2R2",
    # "Embed DITR U2R2",
]

# Probe locations
probe_locations = [0.0, 0.5]

# Plot limits (None for auto)
xlim = None
ylim = [0, 5]

# ═══════════════════════════════════════════════════════════════════

# ── Output directory ────────────────────────────────────────────────
script_dir = pathlib.Path(__file__).resolve().parent
pic_dir = script_dir / "pics" / "brusselator"
pic_dir.mkdir(parents=True, exist_ok=True)

# ── Setup ───────────────────────────────────────────────────────────
fv = TC.make_fv(rec_scheme, Nx)

ev_params = dict(model="brusselator",
                 params={"A": A_br, "B": B_br, "k": k_br})

ev = TC.make_ev(fv, **ev_params)
ev_ps = TC.make_ev(fv, **ev_params, source_quadrature=3)

solver_sets = {
    "": TC.SolverSet(
        ev,
        probe_locations,
        chi_split_width=chi_split_width,
        chi_split_threshold=chi_split_threshold,
    ),
    " p-source": TC.SolverSet(
        ev_ps,
        probe_locations,
        chi_split_width=chi_split_width,
        chi_split_threshold=chi_split_threshold,
    ),
    # chi=0 forced: use very large threshold so sigmoid always outputs ~0
    " chi0": TC.SolverSet(
        ev,
        probe_locations,
        chi_split_width=chi_split_width,
        chi_split_threshold=1e10,
    ),
}

# Initial condition: perturbation of steady state (u_ss=A, v_ss=B/A)
u0_u = A_br + 0.5 * np.sin(fv.xcs * np.pi * 2)
u0_v = B_br / A_br + 0.5 * np.cos(fv.xcs * np.pi * 2)
u0 = np.array([u0_u, u0_v])

# ── Build & run ────────────────────────────────────────────────────
runners = TC.build_method_runners(
    solver_sets, dt, dtRef, u0, tEnd,
    CFL_ref, CFL_coarse, rel_tol, max_iter_exp,
    ref_suffix=ref_suffix,
)
results, probe_results = TC.run_methods(runners, enabled_methods)

# ── Plot & errors ──────────────────────────────────────────────────
tag = f"brusselator_k{k_br}_CFL{CFLt}_T{tEnd}_{rec_scheme}"

TC.plot_profiles(fv, results, enabled_methods, ["u", "v"],
                 f"Brusselator  (k={k_br}, CFL={CFLt}, T={tEnd})",
                 tag, pic_dir, fmt_fig, rec_scheme,
                 show_title=show_title, xlim=xlim, ylim=ylim)

TC.print_errors(results, enabled_methods,
                header=f"k={k_br}, CFL={CFLt}, T={tEnd}")
TC.write_latex_errors(results, enabled_methods, tag)

TC.plot_probes(probe_results, probe_locations, enabled_methods,
               ["u", "v"], tag, pic_dir, fmt_fig, rec_scheme,
               show_title=show_title, xlim=xlim, ylim=ylim)

TC.plot_chi_split(
    fv,
    ev,
    results,
    enabled_methods,
    dt,
    tag,
    pic_dir,
    fmt_fig,
    chi_split_threshold=chi_split_threshold,
    chi_split_width=chi_split_width,
    show_title=show_title,
    xlim=xlim,
)
