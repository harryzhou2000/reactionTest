"""
Test: 1D advection-diffusion-reaction with bistable source.

    u_t + a u_x = eps * u_xx + k * u(1-u)(u-a)

The diffusion smears the reaction front, creating an internal layer of
width ~ sqrt(eps/k).  Splitting errors produce observable front speed
errors -- a clean scalar metric to compare methods.
"""

import numpy as np
import pathlib
import TestCommon as TC

# ╔══════════════════════════════════════════════════════════════════╗
# ║  Problem configuration -- change only this block                 ║
# ╚══════════════════════════════════════════════════════════════════╝

# Grid
Nx = 128
rec_scheme = "weno5z"  # "muscl2" or "weno5z"
fmt_fig = "png"  # output figure format: "pdf", "png", etc.
show_title = False  # show titles on plots

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
    # "Masked Strang DITR U2R2 chi0",  # chi=0 forced via large threshold
    # "Strang DITR U2R1",
    # "DITR U2R2 p-source",
    # "Strang DITR U2R2 p-source",
    # "fully implicit p-source",
    # "DITR U2R2 p-source",
    # "Exp DITR U2R2",
    # "Embed DITR U2R2",
]

# Probe locations (empty list to disable)
probe_locations = []

# Plot limits (None for auto)
xlim = [0.0, 1.0]
ylim = None

# ═══════════════════════════════════════════════════════════════════

# ── Output directory ────────────────────────────────────────────────
script_dir = pathlib.Path(__file__).resolve().parent
pic_dir = script_dir / "pics" / "advdiffreact"
pic_dir.mkdir(parents=True, exist_ok=True)

# ── Setup ───────────────────────────────────────────────────────────
fv = TC.make_fv(rec_scheme, Nx)

ev_params = dict(model="bistable", params={"a": a_react, "k": k_react, "eps": eps_diff})

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

# Initial condition
u0 = np.array([np.sin(fv.xcs * np.pi * 2) * 0.5 + 0.5])

# ── Build & run ────────────────────────────────────────────────────
runners = TC.build_method_runners(
    solver_sets,
    dt,
    dtRef,
    u0,
    tEnd,
    CFL_ref,
    CFL_coarse,
    rel_tol,
    max_iter_exp,
    record_probes=bool(probe_locations),
    ref_suffix=ref_suffix,
)
results, probe_results = TC.run_methods(runners, enabled_methods)

# ── Plot & errors ──────────────────────────────────────────────────
tag = f"advdiffreact_k{k_react}_eps{eps_diff}_CFL{CFLt}_T{tEnd}_{rec_scheme}"

TC.plot_profiles(
    fv,
    results,
    enabled_methods,
    ["u"],
    f"Adv-Diff-React  (k={k_react}, eps={eps_diff}, CFL={CFLt})",
    tag,
    pic_dir,
    fmt_fig,
    rec_scheme,
    show_title=show_title,
    xlim=xlim,
    ylim=ylim,
)

TC.print_errors(
    results,
    enabled_methods,
    header=f"k={k_react}, eps={eps_diff}, CFL={CFLt}, T={tEnd}",
)
TC.write_latex_errors(results, enabled_methods, tag)


if probe_locations:
    TC.plot_probes(
        probe_results,
        probe_locations,
        enabled_methods,
        ["u"],
        tag,
        pic_dir,
        fmt_fig,
        rec_scheme,
        show_title=show_title,
        xlim=xlim,
        ylim=ylim,
    )

TC.plot_chi_split(
    fv, ev, results, enabled_methods, dt, tag, pic_dir, fmt_fig,
    chi_split_threshold=chi_split_threshold, chi_split_width=chi_split_width,
    show_title=show_title, xlim=xlim,
)
