"""
Compute reference solutions for all three orthodox cases and save them.
Then evaluate indicator formulations directly on the final states
(without solving ODEs) to inspect chi distributions.
"""

import numpy as np
import pickle
import sys
from pathlib import Path
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


def compute_refs():
    refs = {}
    
    # ==== A. AdvDiffReact ====
    Nx = 128
    fv = FVUni2nd1D(nx=Nx)
    CFLt = 2
    dt = 1 / Nx / 2 * 2 * CFLt
    dtRef = 1 / Nx / 2 / 4
    tEnd = 1.0
    evp = dict(model="bistable", params={"a": 0.5, "k": 1000, "eps": 1e-1}, nVars=1)
    u0 = np.array([np.sin(fv.xcs * np.pi * 2) * 0.5 + 0.5])
    ev = AdvReactUni1DEval(fv, **evp)
    solver4 = AdvReactUni1DSolver(eval=ev, ode=ESDIRK("ESDIRK4"))
    u_ref = run_silent(lambda: solver4.stepInterval(dtRef, u0, 0.0, tEnd, mode="full", solve_opts={"CFL": 1000}))
    refs['A'] = {'u': u_ref, 'fv': fv, 'dt': dt, 'ev': ev}
    
    # ==== B. Brusselator ====
    Nx = 128
    fv = FVUni2nd1D(nx=Nx)
    CFLt = 1
    dt = 1 / Nx / 2 * 2 * CFLt
    dtRef = 1 / Nx / 2 / 4
    tEnd = 1.0
    A_br, B_br, k_br = 1.0, 3.0, 50
    evp = dict(model="brusselator", params={"A": A_br, "B": B_br, "k": k_br}, nVars=2)
    u0_u = A_br + 0.5 * np.sin(fv.xcs * np.pi * 2)
    u0_v = B_br / A_br + 0.5 * np.cos(fv.xcs * np.pi * 2)
    u0 = np.array([u0_u, u0_v])
    ev = AdvReactUni1DEval(fv, **evp)
    solver4 = AdvReactUni1DSolver(eval=ev, ode=ESDIRK("ESDIRK4"))
    u_ref = run_silent(lambda: solver4.stepInterval(dtRef, u0, 0.0, tEnd, mode="full", solve_opts={"CFL": 1000}))
    refs['B'] = {'u': u_ref, 'fv': fv, 'dt': dt, 'ev': ev}
    
    # ==== C. Premixed ====
    Nx = 256
    fv = FVUni2nd1D(nx=Nx)
    Da = 1e2
    x_jump, delta_jump = 0.5, 0.05
    eps_T, eps_Y = 1e-1, 1e-1
    Q, Tb, Ze = 0.9, 1.0, 14.0
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
    ev = AdvReactUni1DEval(
        fv, model="premixed",
        params={"B": B_react, "Q_div_rho_cp": Q, "Tb": Tb, "E_div_RTb": Ze, "eps": [eps_T, eps_Y]},
        nVars=2,
    )
    ev.ax = 0.0
    solver4 = AdvReactUni1DSolver(eval=ev, ode=ESDIRK("ESDIRK4"))
    u_ref = run_silent(lambda: solver4.stepInterval(dtRef, u0, 0.0, tEnd, mode="full", solve_opts={"CFL": 1000}))
    refs['C'] = {'u': u_ref, 'fv': fv, 'dt': dt, 'ev': ev}
    
    # Save
    Path("experiment").mkdir(exist_ok=True)
    with open("experiment/ref_solutions.pkl", "wb") as f:
        pickle.dump({k: {'u': v['u'], 'dt': v['dt'], 'nx': v['fv'].nx} for k, v in refs.items()}, f)
    
    return refs


def eval_indicator(refs, ind_name, ind_func):
    print(f"\n=== {ind_name} ===")
    for case in ['A', 'B', 'C']:
        u = refs[case]['u']
        dt = refs[case]['dt']
        ev = refs[case]['ev']
        chi = ind_func(ev, u, dt)
        print(f"  {case}: chi range [{chi.min():.3f}, {chi.max():.3f}], mean={chi.mean():.3f}, median={np.median(chi):.3f}")
        # Show how many cells have chi > 0.1 and > 0.5
        print(f"      chi>0.1: {np.sum(chi>0.1)}/{len(chi)}, chi>0.5: {np.sum(chi>0.5)}/{len(chi)}, chi>0.9: {np.sum(chi>0.9)}/{len(chi)}")


# ---- Indicator formulations ----

def ind_bandpass_plain(ev, u, dt):
    JD = ev.rhs_source_jacobian(u)
    nVars, nx = ev.fv.get_shape_u(u)
    if JD.ndim == 2:
        lambda_max = np.max(np.abs(JD), axis=0)
    elif JD.ndim == 3:
        lambda_max = np.array([np.max(np.abs(np.linalg.eigvals(JD[:,:,ix]))) for ix in range(nx)])
    else:
        return np.zeros(nx)
    x = lambda_max * dt
    val = x / (1.0 + x**2)
    arg = (val - 0.2) / 0.03
    chi = np.clip(arg, 0.0, 1e300)
    return chi / (1.0 + chi)


def ind_bandpass_contrast(ev, u, dt):
    JD = ev.rhs_source_jacobian(u)
    nVars, nx = ev.fv.get_shape_u(u)
    if JD.ndim == 2:
        lambda_max = np.max(np.abs(JD), axis=0)
    elif JD.ndim == 3:
        lambda_max = np.array([np.max(np.abs(np.linalg.eigvals(JD[:,:,ix]))) for ix in range(nx)])
    else:
        return np.zeros(nx)
    x = lambda_max * dt
    val = x / (1.0 + x**2)
    # 5-cell contrast
    win = 2
    lam_max_local = lambda_max.copy()
    lam_min_local = lambda_max.copy()
    for w in range(1, win+1):
        lam_max_local = np.maximum(lam_max_local, np.roll(lambda_max, w))
        lam_max_local = np.maximum(lam_max_local, np.roll(lambda_max, -w))
        lam_min_local = np.minimum(lam_min_local, np.roll(lambda_max, w))
        lam_min_local = np.minimum(lam_min_local, np.roll(lambda_max, -w))
    contrast = lam_max_local / (lam_min_local + 1e-300)
    arg_c = (contrast - 20.0) / 5.0
    penalty = np.clip(arg_c, 0.0, 1e300)
    penalty = penalty / (1.0 + penalty)
    val = val * (1.0 - penalty)
    if ev.fv.bcL is not None:
        for i in range(win):
            val[i] = 0.0
    if ev.fv.bcR is not None:
        for i in range(win):
            val[-(i+1)] = 0.0
    arg = (val - 0.2) / 0.03
    chi = np.clip(arg, 0.0, 1e300)
    return chi / (1.0 + chi)


def ind_bandpass_log_contrast(ev, u, dt):
    JD = ev.rhs_source_jacobian(u)
    nVars, nx = ev.fv.get_shape_u(u)
    if JD.ndim == 2:
        lambda_max = np.max(np.abs(JD), axis=0)
    elif JD.ndim == 3:
        lambda_max = np.array([np.max(np.abs(np.linalg.eigvals(JD[:,:,ix]))) for ix in range(nx)])
    else:
        return np.zeros(nx)
    x = lambda_max * dt
    val = x / (1.0 + x**2)
    # log contrast over 5-cell window
    win = 2
    lam_max_local = lambda_max.copy()
    lam_min_local = lambda_max.copy()
    for w in range(1, win+1):
        lam_max_local = np.maximum(lam_max_local, np.roll(lambda_max, w))
        lam_max_local = np.maximum(lam_max_local, np.roll(lambda_max, -w))
        lam_min_local = np.minimum(lam_min_local, np.roll(lambda_max, w))
        lam_min_local = np.minimum(lam_min_local, np.roll(lambda_max, -w))
    contrast = np.log10(lam_max_local / (lam_min_local + 1e-300))
    arg_c = (contrast - 1.0) / 0.3
    penalty = np.clip(arg_c, 0.0, 1e300)
    penalty = penalty / (1.0 + penalty)
    val = val * (1.0 - penalty)
    if ev.fv.bcL is not None:
        for i in range(win):
            val[i] = 0.0
    if ev.fv.bcR is not None:
        for i in range(win):
            val[-(i+1)] = 0.0
    arg = (val - 0.2) / 0.03
    chi = np.clip(arg, 0.0, 1e300)
    return chi / (1.0 + chi)


def ind_lambda_hard_cutoff(ev, u, dt):
    JD = ev.rhs_source_jacobian(u)
    nVars, nx = ev.fv.get_shape_u(u)
    if JD.ndim == 2:
        lambda_max = np.max(np.abs(JD), axis=0)
    elif JD.ndim == 3:
        lambda_max = np.array([np.max(np.abs(np.linalg.eigvals(JD[:,:,ix]))) for ix in range(nx)])
    else:
        return np.zeros(nx)
    # For bistable/premixed: lambda*dt is huge at front -> want 0
    # For brusselator: lambda*dt is ~0.4-2 -> want 1
    # Hard bandpass around lambda*dt ~ 1-3
    x = lambda_max * dt
    val = np.zeros(nx)
    # Only keep cells where x is in [0.5, 3.0]
    mask = (x >= 0.5) & (x <= 3.0)
    val[mask] = 1.0 - np.abs(x[mask] - 1.75) / 1.25
    arg = (val - 0.2) / 0.1
    chi = np.clip(arg, 0.0, 1e300)
    return chi / (1.0 + chi)


def ind_combined_curvature_bandpass(ev, u, dt):
    JD = ev.rhs_source_jacobian(u)
    nVars, nx = ev.fv.get_shape_u(u)
    if JD.ndim == 2:
        lambda_max = np.max(np.abs(JD), axis=0)
    elif JD.ndim == 3:
        lambda_max = np.array([np.max(np.abs(np.linalg.eigvals(JD[:,:,ix]))) for ix in range(nx)])
    else:
        return np.zeros(nx)
    rhs = ev._rhs_source_raw(u)
    rhs_norm = np.linalg.norm(rhs, axis=0)
    
    # Bandpass
    x = lambda_max * dt
    bp = x / (1.0 + x**2)
    
    # Curvature (log10)
    mask = rhs_norm > 1e-300
    curv = np.zeros(nx)
    if np.any(mask):
        if JD.ndim == 2:
            J_dot_S = JD * rhs
        else:
            J_dot_S = np.einsum('ijv,jv->iv', JD, rhs)
        J_dot_S_norm = np.linalg.norm(J_dot_S, axis=0)
        curv[mask] = np.log10(J_dot_S_norm[mask] / rhs_norm[mask] * dt)
    
    # Brusselator: curv ~ -0.3 to 0.2 (modest)
    # Bistable front: curv ~ 2-3 (very high)
    # Use curv to suppress bistable/premixed but keep brusselator
    curv_penalty = np.zeros(nx)
    curv_penalty[mask] = np.clip((curv[mask] - 0.0) / 1.0, 0.0, 1e300)
    curv_penalty = curv_penalty / (1.0 + curv_penalty)
    
    val = bp * (1.0 - curv_penalty)
    arg = (val - 0.2) / 0.03
    chi = np.clip(arg, 0.0, 1e300)
    return chi / (1.0 + chi)


def ind_combined_activity_bandpass(ev, u, dt):
    JD = ev.rhs_source_jacobian(u)
    nVars, nx = ev.fv.get_shape_u(u)
    if JD.ndim == 2:
        lambda_max = np.max(np.abs(JD), axis=0)
    elif JD.ndim == 3:
        lambda_max = np.array([np.max(np.abs(np.linalg.eigvals(JD[:,:,ix]))) for ix in range(nx)])
    else:
        return np.zeros(nx)
    rhs = ev._rhs_source_raw(u)
    rhs_norm = np.linalg.norm(rhs, axis=0)
    u_ref = ev._get_u_ref(u)
    
    x = lambda_max * dt
    bp = x / (1.0 + x**2)
    
    # Activity suppression: very high for bistable/premixed, mild for brusselator
    activity = rhs_norm * dt / (u_ref + 1e-300)
    # Use quadratic suppression with large ref
    val = bp / (1.0 + (activity / 20.0) ** 2)
    
    arg = (val - 0.2) / 0.03
    chi = np.clip(arg, 0.0, 1e300)
    return chi / (1.0 + chi)


def ind_bandpass_windowed_contrast(ev, u, dt):
    JD = ev.rhs_source_jacobian(u)
    nVars, nx = ev.fv.get_shape_u(u)
    if JD.ndim == 2:
        lambda_max = np.max(np.abs(JD), axis=0)
    elif JD.ndim == 3:
        lambda_max = np.array([np.max(np.abs(np.linalg.eigvals(JD[:,:,ix]))) for ix in range(nx)])
    else:
        return np.zeros(nx)
    x = lambda_max * dt
    bp = x / (1.0 + x**2)
    
    # Larger window contrast (9 cells)
    win = 4
    lam_max_local = lambda_max.copy()
    lam_min_local = lambda_max.copy()
    for w in range(1, win+1):
        lam_max_local = np.maximum(lam_max_local, np.roll(lambda_max, w))
        lam_max_local = np.maximum(lam_max_local, np.roll(lambda_max, -w))
        lam_min_local = np.minimum(lam_min_local, np.roll(lambda_max, w))
        lam_min_local = np.minimum(lam_min_local, np.roll(lambda_max, -w))
    contrast = lam_max_local / (lam_min_local + 1e-300)
    arg_c = (contrast - 20.0) / 5.0
    penalty = np.clip(arg_c, 0.0, 1e300)
    penalty = penalty / (1.0 + penalty)
    val = bp * (1.0 - penalty)
    if ev.fv.bcL is not None:
        for i in range(win):
            val[i] = 0.0
    if ev.fv.bcR is not None:
        for i in range(win):
            val[-(i+1)] = 0.0
    arg = (val - 0.2) / 0.03
    chi = np.clip(arg, 0.0, 1e300)
    return chi / (1.0 + chi)


if __name__ == "__main__":
    print("Computing reference solutions...")
    refs = compute_refs()
    print("Done. Evaluating indicators on final states:\n")
    
    eval_indicator(refs, "bandpass_plain", ind_bandpass_plain)
    eval_indicator(refs, "bandpass_contrast", ind_bandpass_contrast)
    eval_indicator(refs, "bandpass_log_contrast", ind_bandpass_log_contrast)
    eval_indicator(refs, "lambda_hard_cutoff", ind_lambda_hard_cutoff)
    eval_indicator(refs, "combined_curvature_bandpass", ind_combined_curvature_bandpass)
    eval_indicator(refs, "combined_activity_bandpass", ind_combined_activity_bandpass)
    eval_indicator(refs, "bandpass_windowed_contrast", ind_bandpass_windowed_contrast)
