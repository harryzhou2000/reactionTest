"""
Exact orthodox configuration reproduction for chi_split verification.
Each case uses the IDENTICAL parameters from test_*Uni1D.py.
"""

import numpy as np
import sys
from Solver.AdvReactUni import AdvReactUni1DSolver, AdvReactUni1DEval
from Solver.FVUni2nd import FVUni2nd1D
from Solver.ODE import ESDIRK

class DevNull:
    def write(self, msg): pass
    def flush(self): pass

orig_stdout = sys.stdout

def run_silent(func):
    sys.stdout = DevNull()
    try:
        return func()
    finally:
        sys.stdout = orig_stdout


def run_case(fv, ev_params, u0, dt, dtRef, tEnd, CFL_ref, CFL_coarse, thr=None, wid=None):
    ev = AdvReactUni1DEval(fv, **ev_params)
    solver4 = AdvReactUni1DSolver(eval=ev, ode=ESDIRK("ESDIRK4"))
    solver = AdvReactUni1DSolver(eval=ev, ode=ESDIRK("ESDIRK3"))
    if thr is not None:
        solver.chi_split_threshold = thr
        solver.chi_split_width = wid
    try:
        u_ref = run_silent(lambda: solver4.stepInterval(
            dtRef, u0, 0.0, tEnd, mode="full", solve_opts={"CFL": CFL_ref}
        ))
        ref_norm = np.linalg.norm(u_ref)
    except Exception as e:
        return {"error": str(e), "ref_norm": None}
    results = {}
    for mode, label in [("full", "impl"), ("strang", "strang"), ("masked_strang", "masked")]:
        try:
            u_res = run_silent(lambda: solver.stepInterval(
                dt, u0, 0.0, tEnd, mode=mode, solve_opts={"rel_tol": 1e-4, "CFL": CFL_coarse}
            ))
            results[label] = np.linalg.norm(u_res - u_ref) / ref_norm
        except Exception as e:
            results[label] = float('inf')
    return results


# ==== A. AdvDiffReact (orthodox from test_AdvDiffReactUni1D.py) ====
def test_advdiffreact(thr=None, wid=None):
    Nx = 128
    fv = FVUni2nd1D(nx=Nx)
    CFLt = 2
    dt = 1 / Nx / 2 * 2 * CFLt
    dtRef = 1 / Nx / 2 / 4
    tEnd = 1.0
    a_react = 0.5
    k_react = 1000
    eps_diff = 1e-1
    evp = dict(model="bistable", params={"a": a_react, "k": k_react, "eps": eps_diff}, nVars=1)
    u0 = np.array([np.sin(fv.xcs * np.pi * 2) * 0.5 + 0.5])
    return run_case(fv, evp, u0, dt, dtRef, tEnd, 1000, 10, thr, wid)


# ==== B. Brusselator (orthodox from test_BrusselatorUni1D.py) ====
def test_brusselator(thr=None, wid=None):
    Nx = 128
    fv = FVUni2nd1D(nx=Nx)
    CFLt = 1
    dt = 1 / Nx / 2 * 2 * CFLt
    dtRef = 1 / Nx / 2 / 4
    tEnd = 1.0
    A_br = 1.0
    B_br = 3.0
    k_br = 50
    evp = dict(model="brusselator", params={"A": A_br, "B": B_br, "k": k_br}, nVars=2)
    u0_u = A_br + 0.5 * np.sin(fv.xcs * np.pi * 2)
    u0_v = B_br / A_br + 0.5 * np.cos(fv.xcs * np.pi * 2)
    u0 = np.array([u0_u, u0_v])
    return run_case(fv, evp, u0, dt, dtRef, tEnd, 1000, 5, thr, wid)


# ==== C. Premixed (orthodox from test_PremixedUni1D.py) ====
def test_premixed(thr=None, wid=None):
    Nx = 256
    fv = FVUni2nd1D(nx=Nx)
    Da = 1e2
    x_jump = 0.5
    delta_jump = 0.05
    eps_T = 1e-1
    eps_Y = 1e-1
    Q = 0.9
    Tb = 1.0
    Ze = 14.0
    B_react = Da * eps_Y / delta_jump**2
    T0 = Tb - Q
    dt = 5e-3
    dtRef = 1e-4
    tEnd = 50e-3
    xi = (fv.xcs - x_jump) / delta_jump
    phi = 0.5 * (1.0 + np.tanh(xi))
    T_init = T0 + Q * phi
    Y_init = 1.0 - phi
    u0 = np.array([T_init, Y_init])
    fv.set_bc_dirichlet(uL=np.array([T0, 1.0]), uR=np.array([Tb, 0.0]))
    evp = dict(
        model="premixed",
        params={"B": B_react, "Q_div_rho_cp": Q, "Tb": Tb, "E_div_RTb": Ze, "eps": [eps_T, eps_Y]},
        nVars=2,
    )
    ev = AdvReactUni1DEval(fv, **evp)
    ev.ax = 0.0
    
    solver4 = AdvReactUni1DSolver(eval=ev, ode=ESDIRK("ESDIRK4"))
    solver = AdvReactUni1DSolver(eval=ev, ode=ESDIRK("ESDIRK3"))
    if thr is not None:
        solver.chi_split_threshold = thr
        solver.chi_split_width = wid
    try:
        u_ref = run_silent(lambda: solver4.stepInterval(
            dtRef, u0, 0.0, tEnd, mode="full", solve_opts={"CFL": 1000}
        ))
        ref_norm = np.linalg.norm(u_ref)
    except Exception as e:
        return {"error": str(e), "ref_norm": None}
    results = {}
    for mode, label in [("full", "impl"), ("strang", "strang"), ("masked_strang", "masked")]:
        try:
            u_res = run_silent(lambda: solver.stepInterval(
                dt, u0, 0.0, tEnd, mode=mode, solve_opts={"rel_tol": 1e-4, "CFL": 100}
            ))
            results[label] = np.linalg.norm(u_res - u_ref) / ref_norm
        except Exception as e:
            results[label] = float('inf')
    return results


def fmt(r):
    if 'error' in r:
        return f"FAIL: {r['error']}"
    return f"i={r['impl']:.3e}  s={r['strang']:.3e}  m={r['masked']:.3e}"


print("Running EXACT orthodox configurations:")
print(f"{'Case':<20s} | {'Implicit':<12s} | {'Strang':<12s} | {'Masked':<12s}")
print("-" * 70)

ra = test_advdiffreact()
print(f"{'A. AdvDiffReact':<20s} | {ra['impl']:.3e}    | {ra['strang']:.3e}    | {ra['masked']:.3e}")

rb = test_brusselator()
print(f"{'B. Brusselator':<20s} | {rb['impl']:.3e}    | {rb['strang']:.3e}    | {rb['masked']:.3e}")

rc = test_premixed()
print(f"{'C. Premixed':<20s} | {rc['impl']:.3e}    | {rc['strang']:.3e}    | {rc['masked']:.3e}")

print("\n" + "="*70)
print("Summary (masked vs best baseline):")
print(f"  A: masked={ra['masked']:.3e}  impl={ra['impl']:.3e}  ratio={ra['masked']/ra['impl']:.2f}x")
print(f"  B: masked={rb['masked']:.3e}  strang={rb['strang']:.3e}  ratio={rb['masked']/rb['strang']:.2f}x")
print(f"  C: masked={rc['masked']:.3e}  impl={rc['impl']:.3e}  ratio={rc['masked']/rc['impl']:.2f}x")
