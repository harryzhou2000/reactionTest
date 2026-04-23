# Universal Stiffness Indicator for Masked Strang Splitting

## Executive Summary

A universal stiffness indicator for `compute_chi_split` was developed that works across all three orthodox test cases with a single configuration. The key insight is that the **flow Jacobian acts as a natural stiffness floor** — when flow dominates source, the bandpass is automatically suppressed, making additional penalties unnecessary.

## Test Cases

### A. Bistable Advection-Diffusion-Reaction
- **Model**: `bistable` with `k=1000`, `eps=0.1`
- **Grid**: 128 cells, periodic BC
- **Time step**: `dt = 2/Nx`
- **Behavior**: Sharp fronts form and propagate; needs implicit treatment everywhere
- **Target**: Masked Strang should work nearly as good as fully implicit

### B. Brusselator Oscillator
- **Model**: `brusselator` with `A=1.0, B=3.0, k=50`
- **Grid**: 128 cells, periodic BC
- **Time step**: `dt = 1/Nx`
- **Behavior**: Oscillatory reaction-diffusion with smooth spatial variation
- **Target**: Masked Strang should work nearly as good as classical Strang splitting

### C. Premixed Combustion Flame
- **Model**: `premixed` with `B~4000`, `Ze=14`
- **Grid**: 256 cells, Dirichlet BC
- **Time step**: `dt = 5e-3`
- **Behavior**: Steep flame front with very stiff chemistry
- **Target**: Masked Strang should work nearly as good as fully implicit

## Final Results (DITR U2R2)

| Case | Implicit | Strang | Masked Strang |
|------|----------|--------|---------------|
| A (bistable) | 3.26e-03 | 1.42e-01 | **3.26e-03** |
| B (brusselator) | 1.18e-01 | 2.97e-03 | **4.88e-03** |
| C (premixed) | 6.66e-04 | 1.53e-01 | **6.66e-04** |

**Assessment**:
- **A**: Masked Strang **exactly matches** implicit
- **B**: Masked Strang is **1.6x Strang** but **24x better** than implicit
- **C**: Masked Strang **exactly matches** implicit

## Indicator Formulation

The indicator uses only two ingredients:

1. **Bandpass on max(source_lambda, flow_lambda) * dt**
2. **Max-filters** (val w=2, chi w=2)

```python
# 1. Compute both source and flow spectral radii
lambda_source = max(abs(eigvals(J_source)))
lambda_flow = max(abs(J_flow_diag))
lambda_max = max(lambda_source, lambda_flow)

# 2. Bandpass
bp = (lambda_max * dt) / (1 + (lambda_max * dt)^2)

# 3. Threshold
chi = inv_transition((bp - 0.27) / 0.03)

# 4. Max-filters (fill holes)
chi = max_filter(chi, w=2)
```

### Why This Works

| Case | lambda_flow | lambda_source | lambda_max | bp range | Effect |
|------|-------------|---------------|------------|----------|--------|
| A | 3405 (diffusion) | 47-500 | **3405** | ~0.02 | chi=0 everywhere |
| B | 128 (advection) | 21-481 | **128-481** | 0.25-0.50 | chi varies with source |
| C | 13107 (diffusion) | 0-4000 | **13107** | ~0.02 | chi=0 everywhere |

- **A & C**: Flow dominates → `lambda_max` is uniform and very high → bandpass ≈ 0.02 → `chi = 0` → fully implicit
- **B**: Flow is weak → `lambda_max` follows source → bandpass peaks at ~0.5 → `chi ≈ 1` in stiff cells → effective splitting

The flow Jacobian provides a **natural stiffness floor** that automatically suppresses splitting in diffusion-dominated problems.

## Removed Components

All of the following were tested and found unnecessary:

### H/J Penalty
```
penalty_hj = inv_transition((log10(H_norm/J_norm) - 0.9) / 0.2)
```
**Status**: Removed. Unnecessary; slightly hurt B.

### Low-Hessian Penalty
```
penalty_lowH = 1.0 if H_norm < 200 and scalar else 0.0
```
**Status**: Removed. Unnecessary.

### Oscillation Boost
```
osc = -discriminant / (tr^2 + 4|det|)  for complex eigenvalues
```
**Status**: Removed. Unnecessary; slightly hurt B.

### Spatial Gradient Penalty
```
penalty_grad = inv_transition((rel_grad(lambda_max) - 30) / 40)
```
**Status**: Removed. Redundant with flow-aware bandpass.

### Absolute Hessian Penalty
```
penalty_highH = 1.0 if H_norm > 800 else 0.0
```
**Status**: Removed. Redundant with flow-aware bandpass.

## Files

- `Solver/AdvReactUni.py`: Core implementation
- `docs/indicator.md`: This document
- `docs/old_indicator_reference.py`: Commented old indicator code for reference
- `experiment/`: Diagnostic and verification scripts

## References

See `docs/old_indicator_reference.py` for the evolution of the indicator code.
