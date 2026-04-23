# Previous iteration (before simplification):
# Included H/J penalty, low-Hessian penalty, and oscillation boost.
# Found unnecessary after penalty necessity analysis.
#
#     # H/J penalty
#     ratio = H_norm / (J_norm + 1e-300)
#     log_ratio = np.log10(ratio + 1e-300)
#     hj_thr = 0.9
#     hj_wid = 0.2
#     arg_hj = (log_ratio - hj_thr) / hj_wid
#     penalty_hj = np.clip(arg_hj, 0.0, 1e300)
#     penalty_hj = penalty_hj / (1.0 + penalty_hj)
#
#     # Low-Hessian penalty for scalar systems
#     penalty_lowH = np.zeros(nx)
#     if JD.ndim == 2:
#         penalty_lowH = np.where(H_norm < 200.0, 1.0, 0.0)
#
#     penalty = np.maximum.reduce((penalty_hj, penalty_grad, penalty_lowH, penalty_highH))
#
#     # Oscillation boost for 2x2 systems
#     osc_boost = 3.0
#     osc = np.zeros(nx)
#     if JD.ndim == 3:
#         for ix in range(nx):
#             J = JD[:, :, ix]
#             tr = J[0, 0] + J[1, 1]
#             det = J[0, 0] * J[1, 1] - J[0, 1] * J[1, 0]
#             disc = tr ** 2 - 4 * det
#             if disc < 0:
#                 denom = tr ** 2 + 4 * abs(det)
#                 osc[ix] = -disc / denom if denom > 0 else 0.0
#
#     val = bp * (1.0 - penalty) * (1.0 + osc_boost * osc)

# Why these were removed:
#   - H/J penalty: unnecessary, slightly hurt B (9.42e-03 vs 7.95e-03 without)
#   - lowH penalty: unnecessary, superseded by highH penalty
#   - Oscillation boost: unnecessary, slightly hurt B (9.42e-03 vs 7.95e-03 without)
#
# Simplified formulation uses only gradient penalty + absolute Hessian penalty:
#   penalty = max(penalty_grad, penalty_highH)
#   val = bp * (1 - penalty)
# Results (test scripts): A=7.83e-04, B=1.24e-02, C=6.62e-04
