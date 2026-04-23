"""
Test using actual combined eigenvalue (flow + source, with sign) vs max(abs).
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


def make_compute_chi_split(mode):
    def compute_chi_split(self, u, dt, threshold=None, width=None, transition="inv"):
        threshold = 0.27
        width = 0.03
        
        JD_source = self.rhs_source_jacobian(u)
        nVars, nx = self.fv.get_shape_u(u)
        
        if JD_source.ndim == 2:
            lambda_source = np.max(np.abs(JD_source), axis=0)
            # Actual combined: flow_diag + source_diag
            JD_flow = self.rhs_flow_jacobian_diag(u)
            lambda_combined_signed = JD_flow[0] + JD_source[0] if JD_flow.ndim == 2 else JD_flow + JD_source[0]
        elif JD_source.ndim == 3:
            lambda_source = np.zeros(nx)
            lambda_combined_signed = np.zeros(nx)
            for ix in range(nx):
                lambda_source[ix] = np.max(np.abs(np.linalg.eigvals(JD_source[:,:,ix])))
                # Combined: source + flow_diagonal
                JD_flow = self.rhs_flow_jacobian_diag(u)
                if JD_flow.ndim == 3:
                    J_comb = JD_source[:,:,ix] + np.diag(JD_flow[:,ix])
                else:
                    J_comb = JD_source[:,:,ix] + np.diag(JD_flow[:,ix])
                lambda_combined_signed[ix] = np.max(np.abs(np.linalg.eigvals(J_comb)))
        else:
            return np.zeros(nx)
        
        if mode == 'source_only':
            lambda_max = lambda_source
        elif mode == 'max_abs':
            JD_flow = self.rhs_flow_jacobian_diag(u)
            if JD_flow.ndim == 2:
                lambda_flow = np.max(np.abs(JD_flow), axis=0)
            else:
                lambda_flow = np.max(np.abs(JD_flow), axis=(0,1))
            lambda_max = np.maximum(lambda_source, lambda_flow)
        elif mode == 'combined_signed':
            lambda_max = np.abs(lambda_combined_signed)
        else:
            lambda_max = lambda_source
        
        H = self._fd_hessian_source(u)
        if H.size > 0:
            if JD_source.ndim == 2:
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
        
        # Hessian penalty
        penalty_highH = np.where(H_norm > 800.0, 1.0, 0.0)
        
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


def run_case(case_name, mode):
    AdvReactUni1DEval.compute_chi_split = make_compute_chi_split(mode)
    
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


print("Testing different Jacobian combinations for bandpass:")
print()

for mode in ['source_only', 'max_abs', 'combined_signed']:
    errs = {case: run_case(case, mode) for case in ['A', 'B', 'C']}
    print(f"{mode:<20s}: A={errs['A']:.2e}, B={errs['B']:.2e}, C={errs['C']:.2e}")
