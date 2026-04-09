"""
Verification: plot the reconstructed polynomial inside 3 consecutive
cells for both MUSCL2 and WENO5-Z.

Panel (a): smooth sin(2 pi x) -- cells on the steep slope, away from
           extrema where the limiter would flatten the gradient.
Panel (b): step function -- cells straddling the discontinuity to test
           shock capture behavior.
"""

import numpy as np
import pathlib
import matplotlib.pyplot as plt
from Solver.FVUni2nd import FVUni2nd1D
from Solver.FVUniWENO5Z import FVUniWENO5Z1D
from Solver.FVUni1D import FVUni1D
import PlotEnv

# ╔══════════════════════════════════════════════════════════════════╗
# ║  Configuration                                                  ║
# ╚══════════════════════════════════════════════════════════════════╝

fmt_fig = "pdf"
Nx = 16                 # coarse grid so reconstruction shape is visible
iCells_smooth = [0, 1, 2]   # steep slope of sin(2 pi x), away from peak
nGauss = 4              # number of Gauss quadrature points to show
nPlot = 201             # dense points per cell for the reconstruction curve

# ═══════════════════════════════════════════════════════════════════

script_dir = pathlib.Path(__file__).resolve().parent
pic_dir = script_dir / "pics" / "rec_verify"
pic_dir.mkdir(parents=True, exist_ok=True)

hx = 1.0 / Nx
xiPlot = np.linspace(-0.5, 0.5, nPlot)
xiGauss, wGauss = FVUni1D.gaussPoints(nGauss)


def plot_cells(ax, plotEnv, fv_dict, u_dict, iCells, exact_fn, title):
    """Plot reconstruction for consecutive cells on a given axes."""
    fv0 = list(fv_dict.values())[0]
    xL_range = fv0.xcs[iCells[0]] - 0.5 * hx
    xR_range = fv0.xcs[iCells[-1]] + 0.5 * hx

    # Exact function (wider range for context)
    xFine = np.linspace(xL_range - 0.3 * hx, xR_range + 0.3 * hx, 1001)
    ax.plot(xFine, exact_fn(xFine), "k-", lw=0.6, label="exact", zorder=1)

    # Cell boundaries and averages
    for iCell in iCells:
        xc = fv0.xcs[iCell]
        xL_cell = xc - 0.5 * hx
        xR_cell = xc + 0.5 * hx
        ax.axvline(xL_cell, color="gray", ls="--", lw=0.4)
        ax.axvline(xR_cell, color="gray", ls="--", lw=0.4)
        u_avg_val = u_dict[list(u_dict.keys())[0]][0, iCell]
        lbl = "cell avg" if iCell == iCells[0] else None
        ax.plot([xL_cell, xR_cell], [u_avg_val, u_avg_val],
                color="gray", ls="-", lw=1.5, alpha=0.5, label=lbl, zorder=2)

    # Reconstructions per cell
    pIdx = 1
    for name in fv_dict:
        fv = fv_dict[name]
        u = u_dict[name]
        uPtsAll = fv.recPointValues(u, xiPlot)    # (1, Nx, nPlot)
        uGaussAll = fv.recPointValues(u, xiGauss)  # (1, Nx, nGauss)
        c = plotEnv.color_seq[pIdx % len(plotEnv.color_seq)]
        marker = "s" if pIdx == 1 else "D"

        for j, iCell in enumerate(iCells):
            xc = fv0.xcs[iCell]
            xPlotC = xc + xiPlot * hx
            xGaussC = xc + xiGauss * hx
            lbl_line = name if j == 0 else None
            lbl_gauss = f"{name} Gauss" if j == 0 else None
            ax.plot(xPlotC, uPtsAll[0, iCell, :], color=c, lw=1.0,
                    label=lbl_line, zorder=3)
            ax.plot(xGaussC, uGaussAll[0, iCell, :], marker, color=c,
                    ms=4.5, mfc="none", mew=0.8, label=lbl_gauss, zorder=5)
        pIdx += 1

    ax.set_xlim(xL_range - 0.15 * hx, xR_range + 0.15 * hx)
    ax.set_xlabel("x")
    ax.set_ylabel("u")
    ax.set_title(title)
    ax.legend(fontsize=6, loc="best")
    ax.grid(True, ls=":", lw=0.3)

    # Print errors
    for name in fv_dict:
        fv = fv_dict[name]
        u = u_dict[name]
        uGaussAll = fv.recPointValues(u, xiGauss)
        for iCell in iCells:
            xGaussC = fv0.xcs[iCell] + xiGauss * hx
            uExactG = exact_fn(xGaussC)
            uG = uGaussAll[0, iCell, :]
            err = np.max(np.abs(uG - uExactG))
            print(f"  {name:8s} cell {iCell:2d}: max Gauss-pt error = {err:.6e}")


# ── Build data ─────────────────────────────────────────────────────
plotEnv = PlotEnv.PlotEnv(dpi=180, markEvery=0, msc=5, lwc=1.0)

# (a) Smooth: exact cell averages of sin(2 pi x)
def sin_exact(x):
    return np.sin(2 * np.pi * x)

def make_sin_avg(fv):
    xL = fv.xs[:-1]
    xR = fv.xs[1:]
    return (-np.cos(2 * np.pi * xR) + np.cos(2 * np.pi * xL)) / (2 * np.pi * hx)

fv_smooth = {"MUSCL2": FVUni2nd1D(Nx), "WENO5-Z": FVUniWENO5Z1D(Nx)}
u_smooth = {name: make_sin_avg(fv).reshape(1, -1) for name, fv in fv_smooth.items()}

# (b) Step: discontinuity at x = x_jump (off-grid to get a fractional cell)
x_jump = 0.53

def step_exact(x):
    return np.where(x < x_jump, 0.0, 1.0)

def make_step_avg(fv):
    """Exact cell averages of Heaviside(x - x_jump)."""
    xL = fv.xs[:-1]
    xR = fv.xs[1:]
    avg = np.clip((xR - x_jump), 0, hx) / hx
    avg = np.where(xL >= x_jump, 1.0, avg)
    avg = np.where(xR <= x_jump, 0.0, avg)
    return avg

fv_step = {"MUSCL2": FVUni2nd1D(Nx), "WENO5-Z": FVUniWENO5Z1D(Nx)}
u_step = {name: make_step_avg(fv).reshape(1, -1) for name, fv in fv_step.items()}

# 3 cells straddling the jump: the cell containing x_jump and its neighbors
iMid = int(x_jump * Nx)
iCells_step = [iMid - 1, iMid, iMid + 1]

# ── Plot ───────────────────────────────────────────────────────────
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.5), dpi=180)
plt.style.use(["science"])

print("(a) Smooth:")
plot_cells(ax1, plotEnv, fv_smooth, u_smooth, iCells_smooth, sin_exact,
           f"(a) smooth, cells {iCells_smooth}, Nx={Nx}")

print("(b) Step:")
plot_cells(ax2, plotEnv, fv_step, u_step, iCells_step, step_exact,
           f"(b) step, cells {iCells_step}, Nx={Nx}")

plt.tight_layout()
plt.savefig(pic_dir / f"rec_verify_Nx{Nx}.{fmt_fig}", dpi=180, bbox_inches="tight")
plt.show()
