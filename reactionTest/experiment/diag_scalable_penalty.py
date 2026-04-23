"""
Test scalable alternatives to absolute H_norm > 800 penalty.
Compare H*dt, H*dt^2, H/lambda_max, etc.
"""

import numpy as np
import pickle
from Solver.AdvReactUni import AdvReactUni1DEval
from Solver.FVUni2nd import FVUni2nd1D


def load_refs():
    with open("experiment/ref_solutions.pkl", "rb") as f:
        data = pickle.load(f)
    refs = {}
    for case in ['A','B','C']:
        nx = data[case]['nx']
        fv = FVUni2nd1D(nx=nx)
        if case == 'A':
            evp = dict(model="bistable", params={"a":0.5,"k":1000,"eps":1e-1}, nVars=1)
        elif case == 'B':
            evp = dict(model="brusselator", params={"A":1.0,"B":3.0,"k":50}, nVars=2)
        else:
            xi = (fv.xcs - 0.5) / 0.05
            phi = 0.5 * (1.0 + np.tanh(xi))
            fv.set_bc_dirichlet(uL=np.array([0.1, 1.0]), uR=np.array([1.0, 0.0]))
            evp = dict(model="premixed", params={"B":1e2*1e-1/0.05**2,"Q_div_rho_cp":0.9,"Tb":1.0,"E_div_RTb":14.0,"eps":[1e-1,1e-1]}, nVars=2)
        ev = AdvReactUni1DEval(fv, **evp)
        if case == 'C':
            ev.ax = 0.0
        refs[case] = {'u': data[case]['u'], 'fv': fv, 'dt': data[case]['dt'], 'ev': ev}
    return refs


def analyze_scalable_penalties(refs):
    for case in ['A', 'B', 'C']:
        u = refs[case]['u']
        dt = refs[case]['dt']
        ev = refs[case]['ev']
        nx = ev.fv.nx
        
        JD = ev.rhs_source_jacobian(u)
        if JD.ndim == 2:
            lambda_max = np.max(np.abs(JD), axis=0)
            J_norm = np.linalg.norm(JD, axis=0)
        else:
            lambda_max = np.array([np.max(np.abs(np.linalg.eigvals(JD[:,:,ix]))) for ix in range(nx)])
            J_norm = np.linalg.norm(JD, axis=(0,1))
        
        H = ev._fd_hessian_source(u)
        if H.size > 0:
            if JD.ndim == 2:
                H_norm = np.sqrt(np.sum(H**2, axis=(0,1)))
            else:
                H_norm = np.sqrt(np.sum(H**2, axis=(0,1,2)))
        else:
            H_norm = np.zeros(nx)
        
        # Scalable measures
        H_dt = H_norm * dt
        H_dt2 = H_norm * dt**2
        H_over_lam = H_norm / (lambda_max + 1e-300)
        H_over_J = H_norm / (J_norm + 1e-300)
        
        print(f"\n=== Case {case} (dt={dt:.4e}) ===")
        print(f"{'measure':<15} {'min':<10} {'p10':<10} {'p50':<10} {'p90':<10} {'max':<10}")
        
        for name, arr in [
            ('H_norm', H_norm),
            ('H*dt', H_dt),
            ('H*dt^2', H_dt2),
            ('H/lam', H_over_lam),
            ('H/J', H_over_J),
        ]:
            print(f"{name:<15} {arr.min():<10.2e} {np.percentile(arr,10):<10.2e} "
                  f"{np.percentile(arr,50):<10.2e} {np.percentile(arr,90):<10.2e} {arr.max():<10.2e}")
        
        # Show cells with highest H*dt
        print(f"\n  Top 10 H*dt cells:")
        order = np.argsort(H_dt)[::-1][:10]
        for ix in order:
            print(f"    idx={ix:3d} H*dt={H_dt[ix]:.2f}  H={H_norm[ix]:.1f}  lam={lambda_max[ix]:.1f}  bp={lambda_max[ix]*dt/(1+(lambda_max[ix]*dt)**2):.3f}")


refs = load_refs()
analyze_scalable_penalties(refs)
