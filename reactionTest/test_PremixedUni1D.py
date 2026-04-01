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
import matplotlib.pyplot as plt
from Solver.AdvReactUni import AdvReactUni1DSolver, AdvReactUni1DEval
from Solver.FVUni2nd import FVUni2nd1D
from Solver.ODE import ESDIRK, DITRExp
import PlotEnv

# ╔══════════════════════════════════════════════════════════════════╗
# ║  Problem configuration                                         ║
# ╚══════════════════════════════════════════════════════════════════╝

# Grid
Nx = 256

Da = 1e2

# Initial jump parameters
x_jump = 0.5            # jump center location
delta_jump = 0.05       # characteristic width of tanh jump

# Diffusion coefficients (per component: [T, Y])
eps_T = 1e-1
eps_Y = 1e-1

# Premixed combustion parameters
Q = 0.9                 # heat release (temperature rise)
Tb = 1.0                # burnt temperature
Ze = 14.0                # Zeldovich number E/(R*Tb)
B_react = Da * eps_Y / delta_jump ** 2  # pre-exponential factor (reaction time scale)
T0 = Tb - Q             # unburnt temperature

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

# Methods to run
enabled_methods = [
    "ref",
    "fully implicit",
    "Strang",
    "DITR",
    "Strang DITR",
    # "Exp DITR",
    # "Embed DITR",
]

# ═══════════════════════════════════════════════════════════════════

# ── Output directory ────────────────────────────────────────────────
script_dir = pathlib.Path(__file__).resolve().parent
pic_dir = script_dir / "pics" / "premixed"
pic_dir.mkdir(parents=True, exist_ok=True)

# ── Setup ───────────────────────────────────────────────────────────
fv = FVUni2nd1D(nx=Nx)

# Dirichlet BCs: left = unburnt, right = burnt
bcL = np.array([T0, 1.0])      # [T_unburnt, Y_unburnt]
bcR = np.array([T0 + Q, 0.0])  # [T_burnt,   Y_burnt]
fv.set_bc_dirichlet(uL=bcL, uR=bcR)

ev = AdvReactUni1DEval(
    fv=fv,
    model="premixed",
    params={
        "B": B_react,
        "Q_div_rho_cp": Q,
        "Tb": Tb,
        "E_div_RTb": Ze,
        "eps": [eps_T, eps_Y],
    },
    nVars=2,
)
ev.ax = 0.0  # no advection

solver4 = AdvReactUni1DSolver(eval=ev, ode=ESDIRK("ESDIRK4"))
solver = AdvReactUni1DSolver(eval=ev, ode=ESDIRK("ESDIRK3"))
solverDITR = AdvReactUni1DSolver(eval=ev, ode=DITRExp())

# Set up probes
probe_locations = [0.5]
solver4.set_probes(probe_locations)
solver.set_probes(probe_locations)
solverDITR.set_probes(probe_locations)

# ── Initial condition: tanh jump ────────────────────────────────────
xi = (fv.xcs - x_jump) / delta_jump
phi = 0.5 * (1.0 + np.tanh(xi))  # 0 on left, 1 on right

T_init = T0 + Q * phi
Y_init = 1.0 - phi
u0 = np.array([T_init, Y_init])

# ── Method registry ────────────────────────────────────────────────
method_runners = {
    "ref": (
        lambda: solver4.stepInterval(
            dtRef,
            u0,
            0.0,
            tEnd,
            mode="full",
            solve_opts={"CFL": CFL_ref},
            record_probes=True,
        ),
        solver4,
    ),
    "fully implicit": (
        lambda: solver.stepInterval(
            dt,
            u0,
            0.0,
            tEnd,
            mode="full",
            solve_opts={"rel_tol": rel_tol, "CFL": CFL_coarse},
            record_probes=True,
        ),
        solver,
    ),
    "Strang": (
        lambda: solver.stepInterval(
            dt,
            u0,
            0.0,
            tEnd,
            mode="strang",
            solve_opts={"CFL": CFL_coarse},
            record_probes=True,
        ),
        solver,
    ),
    "DITR": (
        lambda: solverDITR.stepInterval(
            dt,
            u0,
            0.0,
            tEnd,
            mode="full",
            solve_opts={"rel_tol": rel_tol, "CFL": CFL_coarse},
            use_exp=False,
            record_probes=True,
        ),
        solverDITR,
    ),
    "Strang DITR": (
        lambda: solverDITR.stepInterval(
            dt,
            u0,
            0.0,
            tEnd,
            mode="strang",
            solve_opts={"rel_tol": rel_tol, "CFL": CFL_coarse},
            record_probes=True,
        ),
        solverDITR,
    ),
    "Exp DITR": (
        lambda: solverDITR.stepInterval(
            dt,
            u0,
            0.0,
            tEnd,
            mode="full",
            solve_opts={"rel_tol": rel_tol, "CFL": CFL_coarse, "max_iter": 50},
            use_exp=True,
            record_probes=True,
        ),
        solverDITR,
    ),
    "Embed DITR": (
        lambda: solverDITR.stepInterval(
            dt,
            u0,
            0.0,
            tEnd,
            mode="embed",
            solve_opts={"rel_tol": rel_tol, "CFL": CFL_coarse},
            use_exp=False,
            record_probes=True,
        ),
        solverDITR,
    ),
}

# ── Run selected methods ────────────────────────────────────────────
results = {}
probe_results = {}

for name in enabled_methods:
    entry = method_runners.get(name)
    if entry is None:
        print(f"WARNING: unknown method '{name}', skipping")
        continue
    runner, solver_inst = entry
    solver_inst.clear_probes()
    print("=" * 60)
    print(name)
    print("=" * 60)
    try:
        sol = runner()
        results[name] = sol
        probe_results[name] = solver_inst.get_probe_data()
        print(f"  >> {name} completed, uNorm = {np.linalg.norm(sol):.6e}")
    except Exception as e:
        print(f"  >> {name} FAILED: {e}")
        import traceback
        traceback.print_exc()
        results[name] = None
        probe_results[name] = None

# ── Plot spatial profiles ───────────────────────────────────────────
plotEnv = PlotEnv.PlotEnv(dpi=180, markEvery=max(1, Nx // 20))
tag = f"Ze{Ze:.2g}_B{B_react:.2g}_eps{eps_T:.2g}_T{tEnd:.2g}"

# Temperature
fig = plotEnv.figure(201, figsize=(6, 4))
for i, name in enumerate(enabled_methods):
    sol = results.get(name)
    if sol is not None:
        plotEnv.plot(fv.xcs, sol[0], plotIndex=i, label=name)
plt.legend()
plt.title(f"Premixed T  (Ze={Ze:.2g}, B={B_react:.2g}, T={tEnd:.2g})")
plt.xlabel("x")
plt.ylabel("T")
plt.savefig(pic_dir / f"premixed_T_{tag}.png", dpi=180, bbox_inches="tight")
plt.show()

# Fuel fraction
fig = plotEnv.figure(202, figsize=(6, 4))
for i, name in enumerate(enabled_methods):
    sol = results.get(name)
    if sol is not None:
        plotEnv.plot(fv.xcs, sol[1], plotIndex=i, label=name)
plt.legend()
plt.title(f"Premixed Y  (Ze={Ze:.2g}, B={B_react:.2g}, T={tEnd:.2g})")
plt.xlabel("x")
plt.ylabel("Y")
plt.savefig(pic_dir / f"premixed_Y_{tag}.png", dpi=180, bbox_inches="tight")
plt.show()

# ── Error norms ─────────────────────────────────────────────────────
u1_ref = results.get("ref")
if u1_ref is not None:
    print("\n" + "=" * 60)
    print(f"L2 errors vs reference  (Ze={Ze}, B={B_react}, T={tEnd}):")
    for name in enabled_methods:
        if name == "ref":
            continue
        sol = results.get(name)
        if sol is not None:
            err = np.linalg.norm(sol - u1_ref) / np.linalg.norm(u1_ref)
            print(f"  {name:25s}: {err:.6e}")
        else:
            print(f"  {name:25s}: FAILED")

# ── Probe time series plots ─────────────────────────────────────────
for x_probe in probe_locations:
    # Temperature at probe
    fig = plotEnv.figure(300 + int(x_probe * 100), figsize=(6, 4))
    for i, name in enumerate(enabled_methods):
        pdata = probe_results.get(name)
        if pdata is not None and x_probe in pdata:
            t_arr = np.array(pdata[x_probe]["t"])
            u_arr = np.array(pdata[x_probe]["u"])
            plotEnv.plot(t_arr, u_arr[:, 0], plotIndex=i, label=name)
    plt.legend()
    plt.title(f"T at x={x_probe:.2f}")
    plt.xlabel("t")
    plt.ylabel("T")
    plt.savefig(
        pic_dir / f"premixed_T_x{x_probe}_{tag}.png",
        dpi=180, bbox_inches="tight",
    )
    plt.show()

    # Fuel fraction at probe
    fig = plotEnv.figure(400 + int(x_probe * 100), figsize=(6, 4))
    for i, name in enumerate(enabled_methods):
        pdata = probe_results.get(name)
        if pdata is not None and x_probe in pdata:
            t_arr = np.array(pdata[x_probe]["t"])
            u_arr = np.array(pdata[x_probe]["u"])
            plotEnv.plot(t_arr, u_arr[:, 1], plotIndex=i, label=name)
    plt.legend()
    plt.title(f"Y at x={x_probe:.2f}")
    plt.xlabel("t")
    plt.ylabel("Y")
    plt.savefig(
        pic_dir / f"premixed_Y_x{x_probe}_{tag}.png",
        dpi=180, bbox_inches="tight",
    )
    plt.show()
