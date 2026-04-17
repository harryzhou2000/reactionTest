"""
Compare H_norm distributions across A IC, A final, B final, C final.
Look for absolute thresholds that separate A from B.
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


def get_H_norm(ev, u):
    JD = ev.rhs_source_jacobian(u)
    H = ev._fd_hessian_source(u)
    if H.size == 0:
        return np.array([])
    if JD.ndim == 2:
        return np.sqrt(np.sum(H**2, axis=(0,1)))
    else:
        return np.sqrt(np.sum(H**2, axis=(0,1,2)))


refs = load_refs()

# A IC
Nx = 128
fvA = FVUni2nd1D(nx=Nx)
evA_ic = AdvReactUni1DEval(fvA, model="bistable", params={"a":0.5,"k":1000,"eps":1e-1}, nVars=1)
u0A = np.array([np.sin(fvA.xcs * np.pi * 2) * 0.5 + 0.5])
H_A_ic = get_H_norm(evA_ic, u0A)

H_A_fin = get_H_norm(refs['A']['ev'], refs['A']['u'])
H_B_fin = get_H_norm(refs['B']['ev'], refs['B']['u'])
H_C_fin = get_H_norm(refs['C']['ev'], refs['C']['u'])

cases = [
    ('A IC', H_A_ic),
    ('A fin', H_A_fin),
    ('B fin', H_B_fin),
    ('C fin', H_C_fin),
]

print(f"{'case':<10} {'min':<10} {'p10':<10} {'p25':<10} {'p50':<10} {'p75':<10} {'p90':<10} {'max':<10}")
for name, H in cases:
    print(f"{name:<10} {H.min():<10.1f} {np.percentile(H,10):<10.1f} {np.percentile(H,25):<10.1f} "
          f"{np.percentile(H,50):<10.1f} {np.percentile(H,75):<10.1f} {np.percentile(H,90):<10.1f} {H.max():<10.1f}")

# Show where H is high in each case
print("\n=== Fraction with H_norm > threshold ===")
for thr in [200, 400, 600, 800, 1000, 1200]:
    print(f"\nH_thr={thr}:")
    for name, H in cases:
        frac = np.mean(H > thr)
        print(f"  {name:<8}: {frac:.2%}")

# Show top 10 H values for B
print(f"\n=== B top 10 H_norm cells ===")
order = np.argsort(H_B_fin)[::-1][:10]
for ix in order:
    print(f"  idx={ix:3d} H={H_B_fin[ix]:.1f}")

# Show top 10 H values for A IC
print(f"\n=== A IC top 10 H_norm cells ===")
order = np.argsort(H_A_ic)[::-1][:10]
for ix in order:
    print(f"  idx={ix:3d} H={H_A_ic[ix]:.1f}")

# Show top 10 H values for A final
print(f"\n=== A final top 10 H_norm cells ===")
order = np.argsort(H_A_fin)[::-1][:10]
for ix in order:
    print(f"  idx={ix:3d} H={H_A_fin[ix]:.1f}")
