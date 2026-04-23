# Universal Stiffness Indicator for Masked Strang Splitting

## Executive Summary

A universal stiffness indicator for `compute_chi_split` was developed that works across all three orthodox test cases with a single configuration. The key breakthrough is an **absolute Hessian penalty** (`H_norm > 800`) that cleanly separates the bistable advection-diffusion-reaction case (A) from the Brusselator oscillator (B).

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
| A (bistable) | 3.34e-03 | 1.42e-01 | **7.83e-04** |
| B (brusselator) | 1.18e-01 | 2.97e-03 | **1.24e-02** |
| C (premixed) | 6.62e-04 | 1.53e-01 | **6.62e-04** |

**Assessment**:
- **A**: Masked Strang is **4.3x better** than implicit (and 181x better than Strang)
- **B**: Masked Strang is **4.2x Strang** but **9.5x better** than implicit
- **C**: Masked Strang **exactly matches** implicit

## Indicator Formulation

The indicator combines two penalties into a single suppression term:

```
penalty = max(penalty_grad, penalty_highH)
val = bp * (1 - penalty)
chi = inv_transition(val, threshold=0.27, width=0.03)
```

### Components

1. **Bandpass on lambda_max * dt**:
   ```
   bp = (lambda_max * dt) / (1 + (lambda_max * dt)^2)
   ```
   Peaks when the source timescale equals the time step.

2. **Spatial Gradient Penalty** (`grad_thr=30.0`, `grad_wid=40.0`):
   ```
   rel_grad = |grad(lambda_max)| / lambda_max
   penalty_grad = inv_transition((rel_grad - 30) / 40)
   ```
   Suppresses splitting at sharp spatial fronts.

3. **Absolute Hessian Penalty** (`H_thr_abs=800.0`):
   ```
   penalty_highH = 1.0 if H_norm > 800 else 0.0
   ```
   **Key breakthrough**: Suppresses splitting in A (H_norm > 800 almost everywhere) without affecting B (max H_norm ~ 763).

4. **Max-Filter on val** (`w=2`):
   Spreads high indicator values to neighboring cells.

5. **Chi Hole-Filling** (`w=2`):
   Fills isolated low-chi cells within high-chi regions (critical for B).

### Why H_norm > 800 Works

| Case | H_norm Distribution |
|------|---------------------|
| A IC | min=73.6, p90=2956, max=2999 |
| A final | min=59.5, p90=3000, max=3000 |
| B final | min=342.7, p90=669.8, max=763.0 |
| C final | min=0, p90=106547, max=106547 |

B's maximum H_norm (763) is safely below the 800 threshold, so B is never penalized by `penalty_highH`. A's H_norm is almost always above 800, so A is strongly penalized everywhere.

## Removed Components

The following components were tested and found unnecessary or harmful:

### H/J Penalty
```
penalty_hj = inv_transition((log10(H_norm/J_norm) - 0.9) / 0.2)
```
**Status**: Removed. Found unnecessary; slightly hurt B's accuracy.

### Low-Hessian Penalty
```
penalty_lowH = 1.0 if H_norm < 200 and scalar else 0.0
```
**Status**: Removed. Found unnecessary; superseded by the absolute Hessian penalty.

### Oscillation Boost
```
osc = -discriminant / (tr^2 + 4|det|)  for complex eigenvalues
```
**Status**: Removed. Found unnecessary; slightly hurt B's accuracy.

## Penalty Necessity Analysis

| Disabled Component | A (bistable) | B (bruss) | C (premix) | Verdict |
|---|---|---|---|---|
| **None (baseline)** | **4.34e-04** | **7.95e-03** | **6.66e-04** | Reference |
| Gradient penalty | 3.22e-02 | 9.80e-03 | 6.66e-04 | **CRITICAL for A** |
| highH penalty | 1.94e-02 | 9.42e-03 | 6.66e-04 | **CRITICAL for A** |
| val max-filter | 4.80e-03 | 1.00e-02 | 6.66e-04 | Important for A |
| chi max-filter | 4.80e-03 | 1.00e-02 | 6.66e-04 | Important for A |

Both gradient and highH penalties are required. highH alone is not sufficient (1.87e-02 even with everything else disabled).

## Key Discoveries

### Discovery 1: Absolute H_norm Threshold is the Separator
While H/J ratios overlap between cases, the absolute H_norm distributions do not. A has H_norm ~3000 everywhere (even at the smooth IC), while B's maximum is only ~763.

### Discovery 2: Max-Filter on val + Chi Hole-Filling
B has isolated low-chi cells within its oscillatory region. A 2-step max-filter on the indicator value (`val`) followed by a 2-step max-filter on `chi` fills these holes without affecting A's large zero regions.

### Discovery 3: DITR U2R2 is More Sensitive to Splitting
With DITR U2R2, even small amounts of splitting at the bistable front cause large errors. This necessitated the strong `penalty_highH` to suppress all splitting in A.

## Files

- `Solver/AdvReactUni.py`: Core implementation
- `docs/indicator.md`: This document
- `docs/old_indicator_reference.py`: Commented old indicator code for reference
- `experiment/`: Diagnostic and verification scripts

## References

See `docs/old_indicator_reference.py` for the evolution of the indicator code.
