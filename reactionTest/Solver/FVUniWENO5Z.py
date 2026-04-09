"""WENO5-Z finite volume reconstruction on a uniform 1D grid.

NOTE: This entire module assumes a uniform grid (constant hx).  The
reconstruction coefficients, smoothness indicators, and ideal weights
are all derived for equal-sized cells and are NOT valid on non-uniform
meshes.  A non-uniform extension would require recomputing these
quantities per cell.
"""

import numpy as np

try:
    from .FVUni1D import FVUni1D
except ImportError:
    from FVUni1D import FVUni1D


class FVUniWENO5Z1D(FVUni1D):
    """5th-order WENO-Z reconstruction (Borges et al.) on a uniform 1D grid.

    Assumes a uniform grid with constant cell width hx = 1/nx.  All
    reconstruction coefficients, Jiang-Shu smoothness indicators, and
    ideal weights (d0=1/10, d1=6/10, d2=3/10) are hardcoded for the
    uniform-grid case.  On a non-uniform mesh every one of these would
    need to be recomputed per cell from the local cell widths.

    Uses 3 ghost cells per side (obtained via padGhosts).  For periodic BCs
    the ghosts wrap around; for Dirichlet BCs they use constant extrapolation
    from the boundary value.
    """

    _nGhost = 3  # ghost cells needed per side

    # Precomputed inverse cell-average matrices for sub-stencil polynomial
    # reconstruction (uniform grid only).  For each sub-stencil k, Ainv_k
    # maps the 3 cell averages to polynomial coefficients [c0, c1, c2] in
    # p(xi) = c0 + c1*xi + c2*xi^2 where xi = (x - xc_i) / hx is the
    # local coordinate centered on cell i.  The integration limits used to
    # build the matrices assume unit-width cells in xi-space (hx cancels),
    # which is exact only when all cells have the same width.
    #
    # S0: cells {i-2, i-1, i},  S1: cells {i-1, i, i+1},  S2: cells {i, i+1, i+2}
    _Ainv3 = None  # lazily computed, shape (3, 3, 3): [stencil, coeff, cell_avg]

    # Full 5-cell degree-4 polynomial (uniform grid only): Ainv5 maps 5
    # cell averages {i-2..i+2} to polynomial coefficients [c0..c4].
    _Ainv5 = None  # lazily computed, shape (5, 5)

    @classmethod
    def _get_Ainv3(cls):
        if cls._Ainv3 is not None:
            return cls._Ainv3

        def avg_row(jshift):
            a = jshift - 0.5
            b = jshift + 0.5
            return [b - a, (b**2 - a**2) / 2, (b**3 - a**3) / 3]

        stencil_shifts = [[-2, -1, 0], [-1, 0, 1], [0, 1, 2]]
        Ainv = np.empty((3, 3, 3))
        for k, shifts in enumerate(stencil_shifts):
            A = np.array([avg_row(s) for s in shifts])
            Ainv[k] = np.linalg.inv(A)
        cls._Ainv3 = Ainv
        return Ainv

    @classmethod
    def _get_Ainv5(cls):
        if cls._Ainv5 is not None:
            return cls._Ainv5

        def avg_row(jshift):
            a = jshift - 0.5
            b = jshift + 0.5
            return [b - a, (b**2 - a**2) / 2, (b**3 - a**3) / 3,
                    (b**4 - a**4) / 4, (b**5 - a**5) / 5]

        A = np.array([avg_row(s) for s in [-2, -1, 0, 1, 2]])
        cls._Ainv5 = np.linalg.inv(A)
        return cls._Ainv5

    def __init__(self, nx: int):
        super().__init__(nx)

    # ── WENO5-Z core (operates on padded 1D arrays) ─────────────────

    @staticmethod
    def _weno5z_left(up):
        """Left-biased WENO5-Z reconstruction at face i+1/2.

        Uniform grid only.  The candidate reconstruction coefficients
        (2, -7, 11)/6 etc., the Jiang-Shu smoothness indicator
        coefficients (13/12, 1/4), and the ideal weights (1/10, 6/10,
        3/10) are all specific to equal-sized cells.

        Args:
            up: Padded array, shape (..., nGhost + nx + nGhost).
                Ghost width must be >= 3.

        Returns:
            Reconstructed values at faces, shape (..., nx+1).
            Face j corresponds to the interface between padded cells
            (nGhost-1 + j) and (nGhost + j), i.e. the j-th interior face
            counting from the left boundary.
        """
        ng = 3
        # Stencil: for face j (0-indexed among nx+1 faces), the five cells are
        #   up[..., ng-3+j], up[..., ng-2+j], up[..., ng-1+j],
        #   up[..., ng+j],   up[..., ng+1+j]
        # which are (i-2, i-1, i, i+1, i+2) where i = ng-1+j is the left cell.
        # Number of faces = nx + 1 for Dirichlet; for periodic we trim later.
        nFaces = up.shape[-1] - 2 * ng + 1

        # Extract shifted views (all shape (..., nFaces))
        im2 = up[..., ng - 3: ng - 3 + nFaces]
        im1 = up[..., ng - 2: ng - 2 + nFaces]
        ic  = up[..., ng - 1: ng - 1 + nFaces]
        ip1 = up[..., ng:     ng + nFaces]
        ip2 = up[..., ng + 1: ng + 1 + nFaces]

        # Candidate reconstructions at x_{i+1/2} (left-biased)
        q0 = (2.0 * im2 - 7.0 * im1 + 11.0 * ic) / 6.0
        q1 = (-im1 + 5.0 * ic + 2.0 * ip1) / 6.0
        q2 = (2.0 * ic + 5.0 * ip1 - ip2) / 6.0

        # Smoothness indicators (Jiang-Shu, uniform grid)
        beta0 = (13.0 / 12.0) * (im2 - 2.0 * im1 + ic) ** 2 \
              + (1.0 / 4.0) * (im2 - 4.0 * im1 + 3.0 * ic) ** 2
        beta1 = (13.0 / 12.0) * (im1 - 2.0 * ic + ip1) ** 2 \
              + (1.0 / 4.0) * (im1 - ip1) ** 2
        beta2 = (13.0 / 12.0) * (ic - 2.0 * ip1 + ip2) ** 2 \
              + (1.0 / 4.0) * (3.0 * ic - 4.0 * ip1 + ip2) ** 2

        # WENO-Z weights (Borges et al., JCP 2008)
        eps = 1e-40
        tau5 = np.abs(beta0 - beta2)
        alpha0 = (1.0 / 10.0) * (1.0 + (tau5 / (beta0 + eps)) ** 2)
        alpha1 = (6.0 / 10.0) * (1.0 + (tau5 / (beta1 + eps)) ** 2)
        alpha2 = (3.0 / 10.0) * (1.0 + (tau5 / (beta2 + eps)) ** 2)
        alphaSum = alpha0 + alpha1 + alpha2
        w0 = alpha0 / alphaSum
        w1 = alpha1 / alphaSum
        w2 = alpha2 / alphaSum

        return w0 * q0 + w1 * q1 + w2 * q2

    @staticmethod
    def _weno5z_right(up):
        """Right-biased WENO5-Z reconstruction at face i+1/2.

        Mirror of _weno5z_left: reconstruct from the right side of each
        face.  Same uniform-grid assumption as _weno5z_left.
        """
        ng = 3
        nFaces = up.shape[-1] - 2 * ng + 1

        # For right-biased: the "right cell" is i+1, and the stencil is
        # {i+3, i+2, i+1, i, i-1} mirrored.
        im1 = up[..., ng - 2: ng - 2 + nFaces]
        ic  = up[..., ng - 1: ng - 1 + nFaces]
        ip1 = up[..., ng:     ng + nFaces]
        ip2 = up[..., ng + 1: ng + 1 + nFaces]
        ip3 = up[..., ng + 2: ng + 2 + nFaces]

        # Candidate reconstructions at x_{i+1/2} (right-biased, from cell i+1)
        q0 = (2.0 * ip3 - 7.0 * ip2 + 11.0 * ip1) / 6.0
        q1 = (-ip2 + 5.0 * ip1 + 2.0 * ic) / 6.0
        q2 = (2.0 * ip1 + 5.0 * ic - im1) / 6.0

        # Smoothness indicators (mirrored, uniform grid)
        beta0 = (13.0 / 12.0) * (ip3 - 2.0 * ip2 + ip1) ** 2 \
              + (1.0 / 4.0) * (ip3 - 4.0 * ip2 + 3.0 * ip1) ** 2
        beta1 = (13.0 / 12.0) * (ip2 - 2.0 * ip1 + ic) ** 2 \
              + (1.0 / 4.0) * (ip2 - ic) ** 2
        beta2 = (13.0 / 12.0) * (ip1 - 2.0 * ic + im1) ** 2 \
              + (1.0 / 4.0) * (3.0 * ip1 - 4.0 * ic + im1) ** 2

        # WENO-Z weights (uniform grid ideal weights: 1/10, 6/10, 3/10)
        eps = 1e-40
        tau5 = np.abs(beta0 - beta2)
        alpha0 = (1.0 / 10.0) * (1.0 + (tau5 / (beta0 + eps)) ** 2)
        alpha1 = (6.0 / 10.0) * (1.0 + (tau5 / (beta1 + eps)) ** 2)
        alpha2 = (3.0 / 10.0) * (1.0 + (tau5 / (beta2 + eps)) ** 2)
        alphaSum = alpha0 + alpha1 + alpha2
        w0 = alpha0 / alphaSum
        w1 = alpha1 / alphaSum
        w2 = alpha2 / alphaSum

        return w0 * q0 + w1 * q1 + w2 * q2

    # ── Public interface ─────────────────────────────────────────────

    def recFaceValues(self, u: np.ndarray):
        """Reconstruct left/right face values via WENO5-Z.

        Returns:
            (uL, uR): shape (nVars, nFaces).
                Periodic: nFaces = nx, face i sits between cell i-1 and cell i.
                Dirichlet: nFaces = nx+1, face 0 = left boundary, face nx = right.
        """
        ng = self._nGhost
        up = self.padGhosts(u, ng)

        if self.bcL is None:
            # Periodic: padded has ng + nx + ng cells.
            # _weno5z produces nx+1 faces; for periodic, faces 0 and nx
            # are the same physical face -- take faces 0..nx-1.
            uL = self._weno5z_left(up)[..., :self.nx]
            uR = self._weno5z_right(up)[..., :self.nx]
            return uL, uR
        else:
            # Dirichlet: nx+1 faces
            uL = self._weno5z_left(up)
            uR = self._weno5z_right(up)

            # Override boundary face values with exact BCs
            uL[:, 0] = self.bcL
            uR[:, -1] = self.bcR
            return uL, uR

    def recPointValues(self, u: np.ndarray, xiPts: np.ndarray):
        """Evaluate WENO5-Z reconstruction at internal points.

        In smooth regions the full degree-4 polynomial (5-cell stencil) is
        used, giving 5th-order pointwise accuracy.  Near discontinuities
        the WENO-Z weights blend toward the sub-stencil quadratics to
        suppress oscillations.

        The blending parameter sigma measures how close the WENO-Z weights
        are to the ideal weights (sigma ~ 0 in smooth regions, ~ 1 near
        shocks).  The output is (1 - sigma) * p4(xi) + sigma * p_weno3(xi).

        Args:
            u: Cell-averaged solution, shape (nVars, nx).
            xiPts: Local coordinates in [-1/2, 1/2], shape (nPts,).

        Returns:
            uPts: shape (nVars, nx, nPts).
        """
        nVars, nx = self.get_shape_u(u)
        ng = self._nGhost
        up = self.padGhosts(u, ng)  # (nVars, ng + nx + ng)
        nPts = len(xiPts)
        Ainv3 = self._get_Ainv3()  # (3, 3, 3)
        Ainv5 = self._get_Ainv5()  # (5, 5)

        # ── Full degree-4 polynomial from 5 cell averages ───────────
        # uFull[j]: cell average at shift j in {-2,-1,0,1,2}
        uFull = np.empty((5, nVars, nx))
        for j, s in enumerate([-2, -1, 0, 1, 2]):
            uFull[j] = up[:, ng + s: ng + s + nx]

        # Polynomial coefficients: c[k] for xi^k, k=0..4
        # coeffs5: (5, nVars, nx)
        coeffs5 = np.einsum("kj,jvn->kvn", Ainv5, uFull)

        # Evaluate: p4(xi) = sum_k coeffs5[k] * xi^k
        basis5 = np.array([xiPts**k for k in range(5)])  # (5, nPts)
        p4 = np.einsum("kvn,kp->vnp", coeffs5, basis5)   # (nVars, nx, nPts)

        # ── Sub-stencil quadratics with WENO-Z weights ──────────────
        uStencil = np.empty((3, 3, nVars, nx))
        stencil_shifts = [[-2, -1, 0], [-1, 0, 1], [0, 1, 2]]
        for k, shifts in enumerate(stencil_shifts):
            for j, s in enumerate(shifts):
                uStencil[k, j] = up[:, ng + s: ng + s + nx]

        coeffs3 = np.einsum("kcj,kjvn->kcvn", Ainv3, uStencil)  # (3, 3, nVars, nx)
        basis3 = np.array([np.ones(nPts), xiPts, xiPts**2])      # (3, nPts)
        pk = np.einsum("kcvn,cp->kvnp", coeffs3, basis3)         # (3, nVars, nx, nPts)

        # Smoothness indicators (Jiang-Shu, uniform grid).
        # The coefficients 13/12 and 1/4 are specific to equal-sized cells.
        im2 = up[:, ng - 2: ng - 2 + nx]
        im1 = up[:, ng - 1: ng - 1 + nx]
        ic  = up[:, ng:     ng + nx]
        ip1 = up[:, ng + 1: ng + 1 + nx]
        ip2 = up[:, ng + 2: ng + 2 + nx]

        beta0 = (13.0 / 12.0) * (im2 - 2.0 * im1 + ic) ** 2 \
              + (1.0 / 4.0) * (im2 - 4.0 * im1 + 3.0 * ic) ** 2
        beta1 = (13.0 / 12.0) * (im1 - 2.0 * ic + ip1) ** 2 \
              + (1.0 / 4.0) * (im1 - ip1) ** 2
        beta2 = (13.0 / 12.0) * (ic - 2.0 * ip1 + ip2) ** 2 \
              + (1.0 / 4.0) * (3.0 * ic - 4.0 * ip1 + ip2) ** 2

        eps = 1e-40
        tau5 = np.abs(beta0 - beta2)
        alpha0 = (1.0 / 10.0) * (1.0 + (tau5 / (beta0 + eps)) ** 2)
        alpha1 = (6.0 / 10.0) * (1.0 + (tau5 / (beta1 + eps)) ** 2)
        alpha2 = (3.0 / 10.0) * (1.0 + (tau5 / (beta2 + eps)) ** 2)
        alphaSum = alpha0 + alpha1 + alpha2
        w0 = alpha0 / alphaSum  # (nVars, nx)
        w1 = alpha1 / alphaSum
        w2 = alpha2 / alphaSum

        # WENO-weighted sub-stencil polynomial
        pW = (w0[:, :, np.newaxis] * pk[0]
              + w1[:, :, np.newaxis] * pk[1]
              + w2[:, :, np.newaxis] * pk[2])  # (nVars, nx, nPts)

        # Blending parameter: distance from ideal weights
        # sigma ~ 0 in smooth regions, ~ 1 near discontinuities
        sigma = (np.abs(w0 - 0.1) + np.abs(w1 - 0.6) + np.abs(w2 - 0.3)) / 2.0
        sigma = np.clip(sigma, 0.0, 1.0)

        uPts = (1.0 - sigma[:, :, np.newaxis]) * p4 + sigma[:, :, np.newaxis] * pW
        return uPts


if __name__ == "__main__":
    # Quick sanity check: smooth function, periodic BC
    nx = 20
    fv = FVUniWENO5Z1D(nx)
    u = np.array([np.sin(fv.xcs * 2 * np.pi)])
    uL, uR = fv.recFaceValues(u)
    # At face centers, exact sin should be close
    xf = fv.xs[:nx]  # periodic: nx faces
    uExact = np.sin(xf * 2 * np.pi)
    print("WENO5-Z periodic test:")
    print(f"  max |uL - exact| = {np.max(np.abs(uL[0] - uExact)):.6e}")
    print(f"  max |uR - exact| = {np.max(np.abs(uR[0] - uExact)):.6e}")
