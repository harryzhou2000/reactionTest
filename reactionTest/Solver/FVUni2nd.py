import numpy as np

try:
    from .FVUni1D import FVUni1D
except ImportError:
    from FVUni1D import FVUni1D


class FVUni2nd1D(FVUni1D):
    """2nd-order FV with MUSCL reconstruction and Barth-Jespersen limiter."""

    def __init__(self, nx: int):
        super().__init__(nx)

    def recGrad(self, u: np.ndarray):
        nVars, nx = self.get_shape_u(u)
        uGrad = np.zeros((1, nVars, nx))

        for nx, dx, (uN,) in self.cellOthers([u]):
            du = uN - u
            uGrad[0] += 0.5 * du / dx

        uMin = np.copy(u)
        uMax = np.copy(u)
        uRecMin = np.zeros_like(u) + 1e300
        uRecMax = np.zeros_like(u) - 1e300

        for nx, dx, (uN,) in self.cellOthers([u]):
            uRecInc = dx * 0.5 * uGrad[0]
            uRecMax = np.maximum(uRecMax, uRecInc)
            uRecMin = np.minimum(uRecMin, uRecInc)
            uMax = np.maximum(uMax, uN)
            uMin = np.minimum(uMin, uN)

        alphaBJMax = np.minimum(np.abs(uMax - u) / (np.abs(uRecMax) + 1e-300), 1.0)
        alphaBJMin = np.minimum(np.abs(uMin - u) / (np.abs(uRecMin) + 1e-300), 1.0)
        uGrad[0] *= np.minimum(alphaBJMax, alphaBJMin)

        return uGrad

    def recFaceValues(self, u: np.ndarray):
        """Reconstruct left/right face values via MUSCL + Barth-Jespersen.

        Returns:
            (uL, uR): shape (nVars, nFaces).
                Periodic: nFaces = nx, face i sits between cell i-1 and cell i.
                Dirichlet: nFaces = nx+1, face 0 = left boundary, face nx = right.
        """
        nVars, nx = self.get_shape_u(u)
        uGrad = self.recGrad(u)

        if self.bcL is None:
            # Periodic: nFaces = nx
            # Face i sits between cell (i-1) % nx and cell i.
            # uL at face i = u[i-1] + hx/2 * grad[i-1]   (reconstructed from left cell)
            # uR at face i = u[i]   - hx/2 * grad[i]      (reconstructed from right cell)
            uL = np.roll(u + 0.5 * self.hx * uGrad[0], 1, axis=-1)
            uR = u - 0.5 * self.hx * uGrad[0]
            return uL, uR
        else:
            # Dirichlet: nFaces = nx + 1
            nFaces = nx + 1
            uL = np.zeros((nVars, nFaces))
            uR = np.zeros((nVars, nFaces))

            # Interior faces 1..nx-1: face i between cell i-1 and cell i
            uL[:, 1:nx] = u[:, :nx-1] + 0.5 * self.hx * uGrad[0, :, :nx-1]
            uR[:, 1:nx] = u[:, 1:nx] - 0.5 * self.hx * uGrad[0, :, 1:nx]

            # Face 0 (left boundary): uL = bcL, uR = u[0] - hx/2 * grad[0]
            uL[:, 0] = self.bcL
            uR[:, 0] = u[:, 0] - 0.5 * self.hx * uGrad[0, :, 0]

            # Face nx (right boundary): uL = u[-1] + hx/2 * grad[-1], uR = bcR
            uL[:, nx] = u[:, -1] + 0.5 * self.hx * uGrad[0, :, -1]
            uR[:, nx] = self.bcR

            return uL, uR


    def recPointValues(self, u: np.ndarray, xiPts: np.ndarray):
        """Evaluate MUSCL linear reconstruction at internal points.

        Args:
            u: Cell-averaged solution, shape (nVars, nx).
            xiPts: Local coordinates in [-1/2, 1/2], shape (nPts,).

        Returns:
            uPts: shape (nVars, nx, nPts).
        """
        nVars, nx = self.get_shape_u(u)
        uGrad = self.recGrad(u)  # (1, nVars, nx)
        nPts = len(xiPts)
        # p(xi) = u_i + uGrad_i * hx * xi
        # u: (nVars, nx) -> (nVars, nx, 1)
        # uGrad[0]: (nVars, nx) -> (nVars, nx, 1)
        # xiPts: (nPts,) -> (1, 1, nPts)
        uPts = (u[:, :, np.newaxis]
                + uGrad[0, :, :, np.newaxis] * self.hx
                * xiPts[np.newaxis, np.newaxis, :])
        return uPts


if __name__ == "__main__":
    fv = FVUni2nd1D(20)
    u = np.array([np.sin(fv.xcs * 2 * np.pi)])
    print(fv.recGrad(u))
