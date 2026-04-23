"""
Test necessity of each remaining penalty with the updated flow-aware indicator.
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


def make_compute_chi_split(disable):
    def compute_chi_split(self, u, dt, threshold=None, width=None, transition="inv"):
        threshold = 0.27
        width = 0.03
        
        JD = self.rhs_source_jacobian(u)
        nVars, nx = self.fv.get_shape_u(u)
        
        if JD.ndim == 2:
            lambda_source = np.max(np.abs(JD), axis=0)
        elif JD.ndim == 3:
            lambda_source = np.zeros(nx)
            for ix in range(nx):
                lambda_source[ix] = np.max(np.abs(np.linalg.eigvals(JD[:,:,ix])))
        else:
            return np.zeros(nx)
        
        JD_flow = self.rhs_flow_jacobian_diag(u)
        if JD_flow.ndim == 2:
            lambda_flow = np.max(np.abs(JD_flow), axis=0)
        else:
            lambda_flow = np.max(np.abs(JD_flow), axis=(0,1))
        lambda_max = np.maximum(lambda_source, lambda_flow)
        
        H = self._fd_hessian_source(u)
        if H.size > 0:
            if JD.ndim == 2:
                H_norm = np.sqrt(np.sum(H**2, axis=(0,1)))
            else:
                H_norm = np.sqrt(np.sum(H**2, axis=(0,1,2)))
        else:
            H_norm = np.zeros(nx)
        
        # Spatial gradient penalty
        if not disable.get('grad', False):
            dx = self.fv.hx
            grad = np.abs(np.roll(lambda_max, -1) - np.roll(lambda_max, 1)) / (2*dx)
            rel_grad = grad / (lambda_max + 1e-300)
            arg_g = (rel_grad - 30.0) / 40.0
            grad_pen = np.clip(arg_g, 0.0, 1e300)
            grad_pen = grad_pen / (1.0 + grad_pen)
            penalty_grad = grad_pen
        else:
            penalty_grad = np.zeros(nx)
        
        # Absolute Hessian penalty
        if not disable.get('highH', False):
            penalty_highH = np.where(H_norm > 800.0, 1.0, 0.0)
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
        
        # Max-filter on val
        if not disable.get('val_mf', False):
            val = _max_filter(val, 2)
        
        arg = (val - threshold) / width
        if transition == "inv":
            chi = np.clip(arg, 0.0, 1e300)
            chi = chi / (1.0 + chi)
        else:
            chi = np.clip(arg, 0.0, 1.0)
        
        # Chi hole-filling
        if not disable.get('chi_mf', False):
            for _ in range(2):
                chi = _max_filter(chi, 1)
        
        return chi
    return compute_chi_split


def run_case(case_name, disable):
    AdvReactUni1DEval.compute_chi_split = make_compute_chi_split(disable)
    
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


# Baseline
print("Baseline (flow-aware, all penalties):")
for case in ['A', 'B', 'C']:
    err = run_case(case, {})
    print(f"  {case}: {err:.2e}")
print()

# Test each removal
configs = [
    ("No gradient penalty", {'grad': True}),
    ("No highH penalty", {'highH': True}),
    ("No val max-filter", {'val_mf': True}),
    ("No chi max-filter", {'chi_mf': True}),
]

for name, disable in configs:
    print(f"{name}:")
    for case in ['A', 'B', 'C']:
        err = run_case(case, disable)
        print(f"  {case}: {err:.2e}")
    print()

# Combinations
print("Combinations:")
for name, disable in [
    ("No grad+highH", {'grad': True, 'highH': True}),
    ("No val+chi mf", {'val_mf': True, 'chi_mf': True}),
    ("Only flow+grad", {'highH': True, 'val_mf': True, 'chi_mf': True}),
    ("Only flow+highH", {'grad': True, 'val_mf': True, 'chi_mf': True}),
]:
    print(f"{name}:")
    for case in ['A', 'B', 'C']:
        err = run_case(case, disable)
        print(f"  {case}: {err:.2e}")
    print()
