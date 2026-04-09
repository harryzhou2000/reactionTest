"""Abstract base class for 1D uniform finite volume discretizations."""

from abc import ABC, abstractmethod
import numpy as np


class FVUni1D(ABC):
    """ABC for uniform 1D finite volume grids with reconstruction.

    Subclasses must implement:
        recFaceValues(u) -- reconstruct left/right states at cell faces.
        recPointValues(u, xiPts) -- evaluate reconstruction at internal points.

    Common infrastructure (grid, BCs, cellOthers, diffusion stencil)
    is provided here.
    """

    def __init__(self, nx: int):
        self.xs = np.linspace(0, 1, nx + 1)
        self.xcs = (self.xs[0:-1] + self.xs[1:]) * 0.5
        self.hx: float = 1.0 / nx
        self.nx = nx
        self.vol = self.hx

        # Boundary conditions: None = periodic (default)
        # Set via set_bc_dirichlet(uL, uR) where uL, uR are (nVars,) arrays.
        self.bcL = None  # (nVars,) or None
        self.bcR = None  # (nVars,) or None

    def set_bc_dirichlet(self, uL: np.ndarray, uR: np.ndarray):
        """Set Dirichlet boundary values at left and right faces.

        Args:
            uL: Left boundary value, shape (nVars,).
            uR: Right boundary value, shape (nVars,).
        """
        self.bcL = np.asarray(uL, dtype=float)
        self.bcR = np.asarray(uR, dtype=float)

    def cellOthers(self, us: list[np.ndarray], homogeneous: bool = False):
        """Yield (direction, dx, neighbor_values) for left and right neighbors.

        For periodic BCs, neighbors wrap around.  For Dirichlet BCs, the
        boundary ghost cell is filled with the BC value (or zero when
        homogeneous=True, used by Jacobi iterations on the correction du).
        """
        if self.bcL is None:
            # Periodic BC (original behavior)
            uN = [np.roll(u, 1, axis=-1) for u in us]
            yield -1, -self.hx, uN

            uN = [np.roll(u, -1, axis=-1) for u in us]
            yield 1, self.hx, uN
        else:
            # Dirichlet BC: replace rolled boundary values with ghost values
            # homogeneous=True: zero boundary ghost (for du corrections)
            # homogeneous=False: use bcL/bcR (for u evaluation)

            # Left neighbor (nx=-1): roll +1 wraps cell[-1] into position [0]
            uN = [np.roll(u, 1, axis=-1) for u in us]
            for uNi in uN:
                if homogeneous:
                    uNi[..., 0] = 0.0
                elif uNi.ndim == 2:
                    uNi[:, 0] = self.bcL
                else:
                    uNi[..., 0] = 0.0  # zero ghost gradient
            yield -1, -self.hx, uN

            # Right neighbor (nx=+1): roll -1 wraps cell[0] into position [-1]
            uN = [np.roll(u, -1, axis=-1) for u in us]
            for uNi in uN:
                if homogeneous:
                    uNi[..., -1] = 0.0
                elif uNi.ndim == 2:
                    uNi[:, -1] = self.bcR
                else:
                    uNi[..., -1] = 0.0  # zero ghost gradient
            yield 1, self.hx, uN

    def get_shape_u(self, u):
        nVars = u.shape[0]
        nx = u.shape[1]
        if len(u.shape) != 2 or nx != self.nx:
            raise ValueError("u size not compatible: " + f"{u.shape}")
        return nVars, nx

    def padGhosts(self, u: np.ndarray, nGhost: int, homogeneous: bool = False):
        """Pad u with ghost cells on each side.

        Args:
            u: Solution array, shape (nVars, nx).
            nGhost: Number of ghost cells per side.
            homogeneous: If True, ghost values are zero (for du corrections).

        Returns:
            Padded array of shape (nVars, nGhost + nx + nGhost).
        """
        nVars, nx = self.get_shape_u(u)
        if self.bcL is None:
            # Periodic: wrap around
            return np.concatenate([u[..., -nGhost:], u, u[..., :nGhost]], axis=-1)
        else:
            # Dirichlet: constant extrapolation from boundary value
            if homogeneous:
                padL = np.zeros((nVars, nGhost))
                padR = np.zeros((nVars, nGhost))
            else:
                padL = np.tile(self.bcL.reshape(-1, 1), (1, nGhost))
                padR = np.tile(self.bcR.reshape(-1, 1), (1, nGhost))
            return np.concatenate([padL, u, padR], axis=-1)

    @staticmethod
    def gaussPoints(nPts: int):
        """Gauss-Legendre quadrature points and weights on [-1/2, 1/2].

        Args:
            nPts: Number of quadrature points.

        Returns:
            (xi, w): Points and weights, each of length nPts.
                Points are in [-1/2, 1/2]; weights sum to 1.
        """
        xi_ref, w_ref = np.polynomial.legendre.leggauss(nPts)
        # Map from [-1, 1] to [-1/2, 1/2]
        xi = xi_ref * 0.5
        w = w_ref * 0.5
        return xi, w

    @abstractmethod
    def recFaceValues(self, u: np.ndarray):
        """Reconstruct left and right states at all cell faces.

        Args:
            u: Cell-averaged solution, shape (nVars, nx).

        Returns:
            (uL, uR): Left and right reconstructed values at each face.
                For periodic BCs: shape (nVars, nx), face i is between
                    cell i-1 (left) and cell i (right), with wrapping.
                For Dirichlet BCs: shape (nVars, nx+1), face 0 is the
                    left boundary and face nx is the right boundary.
        """
        ...

    @abstractmethod
    def recPointValues(self, u: np.ndarray, xiPts: np.ndarray):
        """Evaluate the reconstructed polynomial at internal points.

        Args:
            u: Cell-averaged solution, shape (nVars, nx).
            xiPts: Local coordinates within each cell, shape (nPts,).
                Values in [-1/2, 1/2] where 0 = cell center.

        Returns:
            uPts: Reconstructed values, shape (nVars, nx, nPts).
        """
        ...
