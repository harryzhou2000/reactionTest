"""
Old indicator code reference - preserved for documentation.

This is the original compute_chi_split implementation before the universal
stiffness indicator was developed. The old approach used a "source curvature"
indicator based on |J @ S| / |S|, which failed on the orthodox test configs.

The new indicator (see Solver/AdvReactUni.py) uses:
  - Bandpass on lambda_max * dt
  - H/J penalty (Hessian-to-Jacobian ratio)
  - Spatial gradient penalty on lambda_max
  - Low-Hessian penalty for scalar systems
  - Absolute Hessian penalty (H_norm > 800)  <-- KEY BREAKTHROUGH
  - Oscillation boost for complex eigenvalues
  - Max-filter on indicator value (w=2)
  - Chi hole-filling max-filter (w=2)

See docs/indicator.md for the full design rationale and results.
"""

# Original compute_chi_split from before the universal indicator:
#
# def compute_chi_split(
#     self,
#     u: np.ndarray,
#     dt: float,
#     threshold: float = 1.0,   # Old default
#     width: float = 0.5,       # Old default
#     transition="inv",
#     smooth_steps=0,
#     smooth_ratio=0.5,
# ) -> np.ndarray:
#     """Compute per-cell splitting mask chi_split based on source stiffness.
#
#     The stiffness indicator combined four measures and took the maximum:
#         stiffness_ratio = max( |lambda_max(J)| * dt,
#                                |S(u)| / u_ref * dt,
#                                |J(u)*S(u)| / |S(u)| * dt,
#                                ||dJ/du||_F * |S(u)| * dt^2 )
#
#     where |.| denotes the vector 2-norm per cell.
#     """
#     JD = self.rhs_source_jacobian(u)
#     nVars, nx = self.fv.get_shape_u(u)
#
#     if JD.ndim == 2:
#         jac_norm = np.linalg.norm(JD, axis=0)  # (nx,)
#     elif JD.ndim == 3:
#         jac_norm = np.linalg.norm(JD, axis=(0, 1))  # (nx,)
#     else:
#         return np.zeros(nx)
#
#     # Combined stiffness indicator
#     rhs = self._rhs_source_raw(u)
#     rhs_norm = np.linalg.norm(rhs, axis=0)
#
#     stiffness_ratio = np.empty(nx)
#     stiffness_ratio[:] = -1e300
#
#     # # 1. Linear spectral indicator (Frobenius norm of Jacobian)
#     # stiffness_ratio = np.log10(jac_norm * dt)  # (nx,)
#
#     # # 2. Source activity indicator: |S| / u_ref
#     # u_ref = self._get_u_ref(u)
#     # source_activity = np.log10(rhs_norm / u_ref * dt)
#     # source_activity = source_activity / 0.01
#     # stiffness_ratio = np.maximum(stiffness_ratio, source_activity)
#
#     # 3. Source curvature indicator: |J @ S| / |S|
#     mask = rhs_norm > 1e-300
#     if np.any(mask):
#         if JD.ndim == 2:
#             J_dot_S = JD * rhs
#         elif JD.ndim == 3:
#             J_dot_S = np.einsum('ijv,jv->iv', JD, rhs)
#         J_dot_S_norm = np.linalg.norm(J_dot_S, axis=0)
#         source_curvature = np.zeros(nx)
#         source_curvature[mask] = np.log10(
#             J_dot_S_norm[mask] / rhs_norm[mask] * dt
#         )
#         stiffness_ratio = np.maximum(stiffness_ratio, source_curvature)
#
#     # # 4. Hessian nonlinearity indicator: ||dJ/du||_F * |S| * dt^2
#     # H = self._fd_hessian_source(u)
#     # if H.size > 0:
#     #     hessian_norm = np.sqrt(np.sum(H ** 2, axis=tuple(range(H.ndim - 1))))
#     #     hessian_indicator = hessian_norm * rhs_norm * dt ** 2
#     #     hessian_indicator = np.where(np.isfinite(hessian_indicator),
#     #                                  hessian_indicator, 0.0)
#     #     stiffness_ratio = np.maximum(stiffness_ratio, hessian_indicator)
#
#     arg = (stiffness_ratio - threshold) / width
#
#     if transition == "inv":
#         chi_split = np.clip(arg, 0.0, 1e300)
#         chi_split = chi_split / (1.0 + chi_split)
#     elif transition == "sigmoid":
#         arg = np.clip(arg, -50, 50)
#         chi_split = 1.0 / (1.0 + np.exp(-arg))
#     elif transition == "linear":
#         chi_split = np.clip(arg, 0.0, 1.0)
#     else:
#         raise ValueError(f"Unknown transition type: {transition}")
#
#     for i in range(smooth_steps):
#         chi_l = np.roll(chi_split, +1, 0) * smooth_ratio
#         chi_r = np.roll(chi_split, -1, 0) * smooth_ratio
#         if self.fv.bcL is not None:
#             chi_l[0] = 0
#             chi_r[0] = 0
#         if self.fv.bcR is not None:
#             chi_l[-1] = 0
#             chi_r[-1] = 0
#         chi_split = np.maximum.reduce((chi_l, chi_r, chi_split))
#
#     return chi_split

# Why this failed:
# The source curvature indicator |J @ S| / |S| was sensitive to the magnitude
# of the source term but not to the spatial structure. On the orthodox configs
# (coarse grid, long integration, full diffusion), it either:
#   - Produced chi=1 everywhere (treating smooth regions as stiff)
#   - Or chi=0 everywhere (missing stiff fronts)
# It could not distinguish the bistable front (needs implicit) from the
# Brusselator oscillation (needs splitting).
#
# The new indicator solves this by:
#   1. Using a bandpass on lambda_max*dt (peaks at stiffness ~ 1/dt)
#   2. Penalizing based on Hessian properties (nonlinearity measure)
#   3. The absolute Hessian threshold H>800 separates A (H~3000) from B (H<763)
