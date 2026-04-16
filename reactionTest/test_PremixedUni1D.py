"""
Test: 1D diffusion-reaction with premixed combustion model (no advection).

    T_t = eps_T * T_xx + Q * omega
    Y_t = eps_Y * Y_xx - omega

    omega = B * Y * exp(-Ze * (Tb / T - 1))

where Ze = E/(R*Tb) is the Zeldovich number.

Initial condition: tanh jump from unburnt (left) to burnt (right).
    Left:  T = T0,      Y = 1
    Right: T = T0 + Q,  Y = 0
with Q = Q_div_rho_cp, T0 = Tb - Q.

Dirichlet BCs matching the left/right initial states.
"""

import numpy as np
import pathlib
from Solver.AdvReactUni import AdvReactUni1DEval
import TestCommon as TC

# ╔══════════════════════════════════════════════════════════════════╗
# ║  Problem configuration                                         ║
# ╚══════════════════════════════════════════════════════════════════╝

# Grid
Nx = 256
rec_scheme = "weno5z"  # "muscl2" or "weno5z"
fmt_fig = "pdf"  # output figure format: "pdf", "png", etc.
show_title = False  # show titles on plots

Da = 1e2

# Initial jump parameters
x_jump = 0.5  # jump center location
delta_jump = 0.05  # characteristic width of tanh jump

# Diffusion coefficients (per component: [T, Y])
eps_T = 1e-1
eps_Y = 1e-1

# Premixed combustion parameters
Q = 0.9  # heat release (temperature rise)
Tb = 1.0  # burnt temperature
Ze = 14.0  # Zeldovich number E/(R*Tb)
B_react = Da * eps_Y / delta_jump**2  # pre-exponential factor
T0 = Tb - Q  # unburnt temperature

print(f"tau diff { delta_jump ** 2 / eps_Y:.4e}")
print(f"tau reac {1 / B_react:.4e}")

# Time stepping
dt = 5e-3
dtRef = 1e-4
tEnd = 50e-3

# Solver tuning
CFL_ref = 1000
CFL_coarse = 100
rel_tol = 1e-4
max_iter_exp = 50

ref_suffix = " p-source"  # "" = base evaluator, " p-source" = quadrature source for ref
# ref_suffix = ""

# Methods to run
enabled_methods = [
    "ref",
    "DITR U2R2",
    "DITR U2R1",
    "ESDIRK3",
    "ESDIRK4",
    "Strang ESDIRK3",
    "Strang DITR U2R2",
    # "Strang DITR U2R1",
    # "DITR U2R2 p-source",
    # "Strang DITR U2R2 p-source",
    # "fully implicit p-source",
    # "DITR U2R2 p-source",
    # "Exp DITR U2R2",
    # "Embed DITR U2R2",
]

# Probe locations
probe_locations = [0.5]

# Plot limits (None for auto)
xlim = None
ylim = None

# ═══════════════════════════════════════════════════════════════════

# ── Output directory ────────────────────────────────────────────────
script_dir = pathlib.Path(__file__).resolve().parent
pic_dir = script_dir / "pics" / "premixed"
pic_dir.mkdir(parents=True, exist_ok=True)

# ── Setup ───────────────────────────────────────────────────────────
fv = TC.make_fv(rec_scheme, Nx)

# Dirichlet BCs: left = unburnt, right = burnt
bcL = np.array([T0, 1.0])  # [T_unburnt, Y_unburnt]
bcR = np.array([T0 + Q, 0.0])  # [T_burnt,   Y_burnt]
fv.set_bc_dirichlet(uL=bcL, uR=bcR)

ev_params = dict(
    model="premixed",
    params={
        "B": B_react,
        "Q_div_rho_cp": Q,
        "Tb": Tb,
        "E_div_RTb": Ze,
        "eps": [eps_T, eps_Y],
    },
    nVars=2,
    ax=0.0,
)

ev = TC.make_ev(fv, **ev_params)
ev_ps = TC.make_ev(fv, **ev_params, source_quadrature=3)

solver_sets = {
    "": TC.SolverSet(ev, probe_locations),
    " p-source": TC.SolverSet(ev_ps, probe_locations),
}

# ── Initial condition: tanh jump ────────────────────────────────────
xi = (fv.xcs - x_jump) / delta_jump
phi = 0.5 * (1.0 + np.tanh(xi))  # 0 on left, 1 on right

T_init = T0 + Q * phi
Y_init = 1.0 - phi
u0 = np.array([T_init, Y_init])

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
    ref_suffix=ref_suffix,
)
results, probe_results = TC.run_methods(runners, enabled_methods)

# ── Plot & errors ──────────────────────────────────────────────────
tag = f"premixed_Ze{Ze:.2g}_B{B_react:.2g}_eps{eps_T:.2g}_T{tEnd:.2g}_{rec_scheme}"

TC.plot_profiles(
    fv,
    results,
    enabled_methods,
    ["T", "Y"],
    f"Premixed  (Ze={Ze:.2g}, B={B_react:.2g}, T={tEnd:.2g})",
    tag,
    pic_dir,
    fmt_fig,
    rec_scheme,
    show_title=show_title,
    xlim=xlim,
    ylim=ylim,
)

TC.print_errors(results, enabled_methods, header=f"Ze={Ze}, B={B_react}, T={tEnd}")
TC.write_latex_errors(results, enabled_methods, tag)

TC.plot_probes(
    probe_results,
    probe_locations,
    enabled_methods,
    ["T", "Y"],
    tag,
    pic_dir,
    fmt_fig,
    rec_scheme,
    show_title=show_title,
    xlim=xlim,
    ylim=ylim,
)
