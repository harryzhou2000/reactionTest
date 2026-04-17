"""
Verify the updated core implementation works correctly on all three cases.
"""

import numpy as np
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from Solver.AdvReactUni import AdvReactUni1DSolver, AdvReactUni1DEval
from Solver.FVUni2nd import FVUni2nd1D
from Solver.ODE import ESDIRK, DITRExp

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


def run_one_case(case_name):
    if case_name == 'A':
        Nx = 128
        fv = FVUni2nd1D(nx=Nx)
        dt = 1 / Nx / 2 * 2 * 2
        dtRef = 1 / Nx / 2 / 4
        ev = AdvReactUni1DEval(fv, model="bistable", params={"a":0.5,"k":1000,"eps":1e-1}, nVars=1)
        u0 = np.array([np.sin(fv.xcs * np.pi * 2) * 0.5 + 0.5])
        s4 = AdvReactUni1DSolver(eval=ev, ode=ESDIRK("ESDIRK4"))
        s = AdvReactUni1DSolver(eval=ev, ode=DITRExp())
        u_ref = run_silent(lambda: s4.stepInterval(dtRef, u0, 0.0, 1.0, mode="full", solve_opts={"CFL": 1000}))
        ref_norm = np.linalg.norm(u_ref)
        r = {}
        for mode, label in [("full", "impl"), ("strang", "strang"), ("masked_strang", "masked")]:
            u_res = run_silent(lambda: s.stepInterval(dt, u0, 0.0, 1.0, mode=mode, solve_opts={"rel_tol": 1e-4, "CFL": 10}))
            r[label] = np.linalg.norm(u_res - u_ref) / ref_norm
    
    elif case_name == 'B':
        Nx = 128
        fv = FVUni2nd1D(nx=Nx)
        dt = 1 / Nx / 2 * 2 * 1
        dtRef = 1 / Nx / 2 / 4
        ev = AdvReactUni1DEval(fv, model="brusselator", params={"A":1.0,"B":3.0,"k":50}, nVars=2)
        u0 = np.array([1.0 + 0.5*np.sin(fv.xcs*np.pi*2), 3.0 + 0.5*np.cos(fv.xcs*np.pi*2)])
        s4 = AdvReactUni1DSolver(eval=ev, ode=ESDIRK("ESDIRK4"))
        s = AdvReactUni1DSolver(eval=ev, ode=DITRExp())
        u_ref = run_silent(lambda: s4.stepInterval(dtRef, u0, 0.0, 1.0, mode="full", solve_opts={"CFL": 1000}))
        ref_norm = np.linalg.norm(u_ref)
        r = {}
        for mode, label in [("full", "impl"), ("strang", "strang"), ("masked_strang", "masked")]:
            u_res = run_silent(lambda: s.stepInterval(dt, u0, 0.0, 1.0, mode=mode, solve_opts={"rel_tol": 1e-4, "CFL": 5}))
            r[label] = np.linalg.norm(u_res - u_ref) / ref_norm
    
    else:
        Nx = 256
        fv = FVUni2nd1D(nx=Nx)
        xi = (fv.xcs - 0.5) / 0.05
        phi = 0.5 * (1.0 + np.tanh(xi))
        u0 = np.array([0.1 + 0.9*phi, 1.0 - phi])
        fv.set_bc_dirichlet(uL=np.array([0.1, 1.0]), uR=np.array([1.0, 0.0]))
        ev = AdvReactUni1DEval(fv, model="premixed", params={"B":1e2*1e-1/0.05**2,"Q_div_rho_cp":0.9,"Tb":1.0,"E_div_RTb":14.0,"eps":[1e-1,1e-1]}, nVars=2)
        ev.ax = 0.0
        s4 = AdvReactUni1DSolver(eval=ev, ode=ESDIRK("ESDIRK4"))
        s = AdvReactUni1DSolver(eval=ev, ode=DITRExp())
        u_ref = run_silent(lambda: s4.stepInterval(1e-4, u0, 0.0, 50e-3, mode="full", solve_opts={"CFL": 1000}))
        ref_norm = np.linalg.norm(u_ref)
        r = {}
        for mode, label in [("full", "impl"), ("strang", "strang"), ("masked_strang", "masked")]:
            u_res = run_silent(lambda: s.stepInterval(5e-3, u0, 0.0, 50e-3, mode=mode, solve_opts={"rel_tol": 1e-4, "CFL": 100}))
            r[label] = np.linalg.norm(u_res - u_ref) / ref_norm
    
    return case_name, r


def fmt(r):
    return f"i={r['impl']:.2e} s={r['strang']:.2e} m={r['masked']:.2e}"


with ProcessPoolExecutor(max_workers=3) as exe:
    futures = {exe.submit(run_one_case, c): c for c in ['A', 'B', 'C']}
    print(f"{'case':<10s} | {'A (bistable)':<35s} | {'B (bruss)':<35s} | {'C (premix)':<35s}")
    print("-" * 125)
    for future in as_completed(futures):
        case, r = future.result()
        print(f"{case:<10s} | {fmt(r):<35s}")
        sys.stdout.flush()
