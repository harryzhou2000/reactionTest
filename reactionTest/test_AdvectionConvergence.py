"""
Test: pure advection convergence -- MUSCL2 vs WENO5-Z on refining grids.

    u_t + a u_x = 0,   x in [0, 1], periodic BC

Initial condition: u(x, 0) = sin(2 pi x).
Exact solution at t = 1 (with a = 1): u(x, 1) = sin(2 pi x).

For each grid resolution, both reconstruction schemes are run with
ESDIRK4 at a dt small enough that temporal error is negligible
compared to spatial error.  L1, L2, and Linf errors are printed
and plotted with ideal convergence slopes.
"""

import numpy as np
import pathlib
import io, contextlib
import matplotlib.pyplot as plt
from Solver.AdvReactUni import AdvReactUni1DSolver, AdvReactUni1DEval
from Solver.FVUni2nd import FVUni2nd1D
from Solver.FVUniWENO5Z import FVUniWENO5Z1D
from Solver.ODE import ESDIRK
import PlotEnv

# ╔══════════════════════════════════════════════════════════════════╗
# ║  Configuration                                                  ║
# ╚══════════════════════════════════════════════════════════════════╝

fmt_fig = "pdf"              # output figure format: "pdf", "png", etc.
tEnd = 1.0                   # advection speed a=1, domain [0,1] => 1 period
CFL_solve = 1000             # pseudo-time CFL (implicit solver tuning)
rel_tol = 1e-6               # tight tolerance so solver error is small
ode_name = "ESDIRK4"         # high-order time integrator

# Grid resolutions for the convergence study
Nx_list = [16, 32, 64, 128, 256]

# dt schedule: dt = C * hx keeps CFL ~ 1 which is fine for implicit,
# and the 4th-order time error (dt^4 ~ hx^4) stays below the
# 2nd-order MUSCL spatial error.
dt_CFL = 0.5                 # dt = dt_CFL * hx

# ═══════════════════════════════════════════════════════════════════

script_dir = pathlib.Path(__file__).resolve().parent
pic_dir = script_dir / "pics" / "advection_convergence"
pic_dir.mkdir(parents=True, exist_ok=True)

schemes = {
    "muscl2": FVUni2nd1D,
    "weno5z": FVUniWENO5Z1D,
}


def run_silent(func):
    with contextlib.redirect_stdout(io.StringIO()):
        return func()


def exact_solution(xcs):
    """Exact solution at t = tEnd (one full period)."""
    return np.sin(2 * np.pi * xcs)


# ── Run convergence study ──────────────────────────────────────────
# results[scheme] = {"Nx": [...], "L1": [...], "L2": [...], "Linf": [...]}
# finest_u1[scheme] = (fv, u1)  -- cache for profile plot
results = {}
finest_u1 = {}

for scheme_name, FVClass in schemes.items():
    res = {"Nx": [], "L1": [], "L2": [], "Linf": []}
    for Nx in Nx_list:
        hx = 1.0 / Nx
        dt = dt_CFL * hx

        fv = FVClass(nx=Nx)
        ev = AdvReactUni1DEval(fv=fv, model="", params={})
        solver = AdvReactUni1DSolver(eval=ev, ode=ESDIRK(ode_name))

        u0 = np.array([np.sin(2 * np.pi * fv.xcs)])

        print(f"{scheme_name:8s}  Nx={Nx:4d}  dt={dt:.4e}  ", end="", flush=True)
        u1 = run_silent(lambda: solver.stepInterval(
            dt, u0, 0.0, tEnd,
            mode="full",
            solve_opts={"CFL": CFL_solve, "rel_tol": rel_tol},
        ))

        uExact = exact_solution(fv.xcs)
        err = u1[0] - uExact
        eL1 = np.mean(np.abs(err))
        eL2 = np.sqrt(np.mean(err**2))
        eLinf = np.max(np.abs(err))

        res["Nx"].append(Nx)
        res["L1"].append(eL1)
        res["L2"].append(eL2)
        res["Linf"].append(eLinf)

        print(f"L1={eL1:.4e}  L2={eL2:.4e}  Linf={eLinf:.4e}")

        if Nx == Nx_list[-1]:
            finest_u1[scheme_name] = (fv, u1)

    results[scheme_name] = res

# ── Print convergence table ────────────────────────────────────────
print("\n" + "=" * 72)
print(f"{'scheme':8s}  {'Nx':>5s}  {'L1':>11s}  {'L2':>11s}  {'Linf':>11s}")
print("-" * 72)

for scheme_name in schemes:
    res = results[scheme_name]
    for i, Nx in enumerate(res["Nx"]):
        print(f"{scheme_name:8s}  {Nx:5d}  {res['L1'][i]:11.4e}  "
              f"{res['L2'][i]:11.4e}  {res['Linf'][i]:11.4e}")
        if i < len(res["Nx"]) - 1:
            rL1 = np.log2(res["L1"][i] / res["L1"][i + 1])
            rL2 = np.log2(res["L2"][i] / res["L2"][i + 1])
            rLinf = np.log2(res["Linf"][i] / res["Linf"][i + 1])
            print(f"{'':8s}  {'order':>5s}  {rL1:11.2f}  {rL2:11.2f}  {rLinf:11.2f}")
    print("-" * 72)

# ── Plot convergence ───────────────────────────────────────────────
plotEnv = PlotEnv.PlotEnv(dpi=180, markEvery=0, msc=6, lwc=1.0)

fig = plotEnv.figure(1, figsize=(6, 5))
pIdx = 0
for scheme_name in schemes:
    res = results[scheme_name]
    Nxs = np.array(res["Nx"])
    hxs = 1.0 / Nxs
    for norm_name in ["L1", "L2", "Linf"]:
        errs = np.array(res[norm_name])
        plotEnv.plot(hxs, errs, plotIndex=pIdx,
                     label=f"{scheme_name} {norm_name}")
        pIdx += 1

# Reference slopes anchored to the midpoint of the muscl2 L2 data
hxs_all = 1.0 / np.array(Nx_list)
hx_ref = np.array([hxs_all[0], hxs_all[-1]])
mid = len(Nx_list) // 2
for order, ls, lbl, anchor_scheme in [
    (2, "--", r"$O(h^2)$", "muscl2"),
    (5, ":", r"$O(h^5)$", "weno5z"),
]:
    anchor_err = results[anchor_scheme]["L2"][mid]
    anchor_hx = hxs_all[mid]
    c = anchor_err / anchor_hx**order
    plt.plot(hx_ref, c * hx_ref**order, ls, color="gray", lw=0.8, label=lbl)

plt.xscale("log")
plt.yscale("log")
plt.xlabel("h")
plt.ylabel("error")
plt.title("Pure advection convergence")
plt.legend(fontsize=7, ncol=2)
plt.grid(True, which="both", ls=":", lw=0.3)
plt.savefig(pic_dir / f"advection_convergence.{fmt_fig}", dpi=180, bbox_inches="tight")
plt.show()

# ── Plot final solutions at finest grid ────────────────────────────
Nx_fine = Nx_list[-1]
fig = plotEnv.figure(2, figsize=(6, 4))

fv0 = list(finest_u1.values())[0][0]
plotEnv.plot(fv0.xcs, exact_solution(fv0.xcs), plotIndex=0, label="exact")

pIdx = 1
for scheme_name in schemes:
    fv, u1 = finest_u1[scheme_name]
    plotEnv.plot(fv.xcs, u1[0], plotIndex=pIdx, label=scheme_name)
    pIdx += 1

plt.legend()
plt.xlabel("x")
plt.ylabel("u")
plt.title(f"Pure advection at t={tEnd}, Nx={Nx_fine}")
plt.savefig(pic_dir / f"advection_profiles_Nx{Nx_fine}.{fmt_fig}",
            dpi=180, bbox_inches="tight")
plt.show()
