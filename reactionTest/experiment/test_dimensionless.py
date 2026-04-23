"""
Test explicitly dimensionless penalty formulations.
H has units 1/([u]*[time]), so we need to multiply by [u]*[time] to get dimensionless.
"""

import numpy as np
import sys
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


def make_compute_chi_split(penalty_type, thr):
    def compute_chi_split(self, u, dt, threshold=None, width=None, transition="inv"):
        threshold = 0.27
        width = 0.03
        
        JD = self.rhs_source_jacobian(u)
        nVars, nx = self.fv.get_shape_u(u)
        
        if JD.ndim == 2:
            lambda_max = np.max(np.abs(JD), axis=0)
        elif JD.ndim == 3:
            lambda_max = np.zeros(nx)
            for ix in range(nx):
                lambda_max[ix] = np.max(np.abs(np.linalg.eigvals(JD[:,:,ix])))
        else:
            return np.zeros(nx)
        
        H = self._fd_hessian_source(u)
        if H.size > 0:
            if JD.ndim == 2:
                H_norm = np.sqrt(np.sum(H**2, axis=(0,1)))
            else:
                H_norm = np.sqrt(np.sum(H**2, axis=(0,1,2)))
        else:
            H_norm = np.zeros(nx)
        
        # Spatial gradient penalty
        dx = self.fv.hx
        grad = np.abs(np.roll(lambda_max, -1) - np.roll(lambda_max, 1)) / (2*dx)
        rel_grad = grad / (lambda_max + 1e-300)
        arg_g = (rel_grad - 30.0) / 40.0
        grad_pen = np.clip(arg_g, 0.0, 1e300)
        grad_pen = grad_pen / (1.0 + grad_pen)
        penalty_grad = grad_pen
        
        # Dimensionless Hessian penalty
        if penalty_type == 'H_u_dt':
            # H * |u| * dt (dimensionless)
            u_mag = np.max(np.abs(u), axis=0)
            measure = H_norm * u_mag * dt
            penalty_highH = np.where(measure > thr, 1.0, 0.0)
        elif penalty_type == 'H_S_dt2':
            # H * |S| * dt^2 (dimensionless)
            rhs = self._rhs_source_raw(u)
            rhs_mag = np.linalg.norm(rhs, axis=0)
            measure = H_norm * rhs_mag * dt**2
            penalty_highH = np.where(measure > thr, 1.0, 0.0)
        elif penalty_type == 'H_Jinv_dt':
            # H / J_norm * dt = H * J^{-1} * dt... wait, J^{-1} has units [u]*[time]
            # H * [u]*[time] * [time] = [time]/[u], not dimensionless
            # Actually H / lambda_max has units 1/[u], not dimensionless
            pass
        else:
            penalty_highH = np.zeros(nx)
        
        penalty = np.maximum(penalty_grad, penalty_highH)
        
        x = lambda_max * dt
        bp = x / (1.0 + x**2)
        
        val = bp * (1.0 - penalty)
        
        def _max_filter(v, w):
            out = v.copy()
            for i in range(-w, w+1):
                if i == 0:
                    continue
                out = np.maximum(out, np.roll(v, i))
            return out
        
        val = _max_filter(val, 2)
        
        arg = (val - threshold) / width
        if transition == "inv":
            chi = np.clip(arg, 0.0, 1e300)
            chi = chi / (1.0 + chi)
        else:
            chi = np.clip(arg, 0.0, 1.0)
        
        for _ in range(2):
            chi = _max_filter(chi, 1)
        
        return chi
    return compute_chi_split


def run_case(case_name, penalty_type, thr):
    AdvReactUni1DEval.compute_chi_split = make_compute_chi_split(penalty_type, thr)
    
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
        u_res = run_silent(lambda: s.stepInterval(dt, u0, 0.0, 1.0, mode="masked_strang", solve_opts={"rel_tol": 1e-4, "CFL": 10}))
        return np.linalg.norm(u_res - u_ref) / ref_norm
    
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
        u_res = run_silent(lambda: s.stepInterval(dt, u0, 0.0, 1.0, mode="masked_strang", solve_opts={"rel_tol": 1e-4, "CFL": 5}))
        return np.linalg.norm(u_res - u_ref) / ref_norm
    
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
        u_res = run_silent(lambda: s.stepInterval(5e-3, u0, 0.0, 50e-3, mode="masked_strang", solve_opts={"rel_tol": 1e-4, "CFL": 100}))
        return np.linalg.norm(u_res - u_ref) / ref_norm


# First, compute the actual values for each measure
print("Computing dimensionless measures on reference solutions...")
print()

import pickle

with open("experiment/ref_solutions.pkl", "rb") as f:
    data = pickle.load(f)

for case in ['A', 'B', 'C']:
    nx = data[case]['nx']
    fv = FVUni2nd1D(nx=nx)
    if case == 'A':
        ev = AdvReactUni1DEval(fv, model="bistable", params={"a":0.5,"k":1000,"eps":1e-1}, nVars=1)
    elif case == 'B':
        ev = AdvReactUni1DEval(fv, model="brusselator", params={"A":1.0,"B":3.0,"k":50}, nVars=2)
    else:
        xi = (fv.xcs - 0.5) / 0.05
        phi = 0.5 * (1.0 + np.tanh(xi))
        fv.set_bc_dirichlet(uL=np.array([0.1, 1.0]), uR=np.array([1.0, 0.0]))
        ev = AdvReactUni1DEval(fv, model="premixed", params={"B":1e2*1e-1/0.05**2,"Q_div_rho_cp":0.9,"Tb":1.0,"E_div_RTb":14.0,"eps":[1e-1,1e-1]}, nVars=2)
        ev.ax = 0.0
    
    u = data[case]['u']
    dt = data[case]['dt']
    nx = ev.fv.nx
    
    JD = ev.rhs_source_jacobian(u)
    if JD.ndim == 2:
        lambda_max = np.max(np.abs(JD), axis=0)
    else:
        lambda_max = np.array([np.max(np.abs(np.linalg.eigvals(JD[:,:,ix]))) for ix in range(nx)])
    
    H = ev._fd_hessian_source(u)
    if H.size > 0:
        if JD.ndim == 2:
            H_norm = np.sqrt(np.sum(H**2, axis=(0,1)))
        else:
            H_norm = np.sqrt(np.sum(H**2, axis=(0,1,2)))
    else:
        H_norm = np.zeros(nx)
    
    # Dimensionless measures
    u_mag = np.max(np.abs(u), axis=0)
    measure1 = H_norm * u_mag * dt
    
    rhs = ev._rhs_source_raw(u)
    rhs_mag = np.linalg.norm(rhs, axis=0)
    measure2 = H_norm * rhs_mag * dt**2
    
    print(f"=== Case {case} ===")
    print(f"{'measure':<20} {'min':<12} {'p10':<12} {'p50':<12} {'p90':<12} {'max':<12}")
    for name, arr in [
        ('H*|u|*dt', measure1),
        ('H*|S|*dt^2', measure2),
    ]:
        # Filter out zeros for percentile
        arr_nonzero = arr[arr > 1e-300]
        if len(arr_nonzero) > 0:
            print(f"{name:<20} {arr.min():<12.2e} {np.percentile(arr_nonzero,10):<12.2e} "
                  f"{np.percentile(arr_nonzero,50):<12.2e} {np.percentile(arr_nonzero,90):<12.2e} {arr.max():<12.2e}")
        else:
            print(f"{name:<20} {arr.min():<12.2e} {'all zero':<12} {'all zero':<12} {'all zero':<12} {arr.max():<12.2e}")
    print()

# Now test ODE solves
print("\nTesting H*|u|*dt penalty:")
print(f"{'thr':<8s} | {'A':<12s} {'B':<12s} {'C':<12s}")
print("-" * 45)
for thr in [5.0, 10.0, 15.0, 20.0, 30.0, 50.0]:
    errs = {case: run_case(case, 'H_u_dt', thr) for case in ['A', 'B', 'C']}
    print(f"{thr:<8.1f} | {errs['A']:.2e}   {errs['B']:.2e}   {errs['C']:.2e}")
    sys.stdout.flush()

print("\nTesting H*|S|*dt^2 penalty:")
print(f"{'thr':<8s} | {'A':<12s} {'B':<12s} {'C':<12s}")
print("-" * 45)
for thr in [0.01, 0.05, 0.1, 0.5, 1.0, 5.0]:
    errs = {case: run_case(case, 'H_S_dt2', thr) for case in ['A', 'B', 'C']}
    print(f"{thr:<8.2f} | {errs['A']:.2e}   {errs['B']:.2e}   {errs['C']:.2e}")
    sys.stdout.flush()
