"""Functor classes for AdvReactUni1DSolver.

These classes implement the RHS evaluation and implicit solve interfaces
required by the ODE integrators (ESDIRK, DITRExp).
"""
from __future__ import annotations
from typing import TYPE_CHECKING
import numpy as np

from . import ODE

if TYPE_CHECKING:
    from .AdvReactUni import AdvReactUni1DEval


class Frhs(ODE.ODE_F_RHS):
    """RHS functor for advection-reaction system."""

    def __init__(self, eval: AdvReactUni1DEval, mode: str = "full",
                 chi_split: np.ndarray = None):
        super().__init__()
        self.eval = eval
        self.mode = mode
        self.chi_split = chi_split  # (nx,) array or None
        if mode not in {"full", "flow", "source", "masked_implicit", "masked_split"}:
            raise ValueError("mode not valid")

    def __call__(self, u, cStage, iStage):
        if self.mode == "full":
            return self.eval.rhs_flow(u) + self.eval.rhs_source(u)
        elif self.mode == "flow":
            return self.eval.rhs_flow(u)
        elif self.mode == "source":
            return self.eval.rhs_source(u)
        elif self.mode == "masked_implicit":
            # F(u) + (1 - chi_split) * S(u)
            flow = self.eval.rhs_flow(u)
            source = self.eval.rhs_source(u)
            if self.chi_split is None:
                return flow + source
            # chi_split is (nx,), broadcast to (nVars, nx)
            chi = self.chi_split[np.newaxis, :]
            return flow + (1.0 - chi) * source
        elif self.mode == "masked_split":
            # chi_split * S(u)
            source = self.eval.rhs_source(u)
            if self.chi_split is None:
                return 0.0 * source
            chi = self.chi_split[np.newaxis, :]
            return chi * source
        else:
            raise ValueError()

    def dt(self, u, cStage, iStage):
        hx = self.eval.fv.hx
        ax = self.eval.ax
        eps_max = float(np.max(self.eval.eps))
        dt_adv = hx / ax if ax != 0 else 1e300
        dt_diff = hx**2 / eps_max if eps_max != 0 else 1e300
        return min(dt_adv, dt_diff)

    def Jacobian(self, u, cStage, iStage):
        raise NotImplementedError()


class Fsolve(ODE.ODE_F_SOLVE_SingleStage):
    """Implicit solver functor using pseudo-time Jacobi iteration."""

    def __init__(
        self,
        eval: AdvReactUni1DEval,
        CFL: float = 10,
        rel_tol: float = 1e-4,
        rel_tol_rhs: float = None,
        max_iter: int = 1000,
        n_print: int = 10,
        mode: str = "full",
        chi_split: np.ndarray = None,
    ):
        super().__init__()
        self.eval = eval
        self.CFL = CFL
        self.rel_tol = rel_tol
        self.rel_tol_rhs = rel_tol_rhs if rel_tol_rhs is not None else 0.1 * rel_tol
        self.max_iter = max_iter
        self.n_print = n_print
        self.mode = mode
        self.chi_split = chi_split  # (nx,) array or None

        if mode not in {"full", "flow", "source", "full_split", "masked_implicit",
                        "masked_split"}:
            raise ValueError("mode not valid")

    def __call__(
        self, u0, dt, alphaRHS, fRHS: ODE.ODE_F_RHS, fRes, cStage, iStage
    ):
        u = np.copy(u0)
        nVars, nx = self.eval.fv.get_shape_u(u)

        resN0 = None
        rhs = None

        for iter in range(1, self.max_iter + 1):
            dTau = fRHS.dt(u, cStage, iStage) * self.CFL
            rhs = fRHS(u, cStage, iStage)
            res = -(u) / dt + alphaRHS * rhs + fRes

            du = np.zeros_like(u)
            if self.mode == "full":
                JDFlow = -alphaRHS * self.eval.rhs_flow_jacobian_diag(u)
                JDFlow += 1 / (dTau) + 1 / dt
                # JDFlow_inv = self.eval.invert_jacobian_diag(JDFlow)
                JDSource = -alphaRHS * self.eval.rhs_source_jacobian(u)
                JDFlow = self.eval.add_jacobian_diags(JDFlow, JDSource)
                JDFullInv = self.eval.invert_jacobian_diag(JDFlow)

                for iJ in range(3):
                    du = self.eval.rhs_flow_jacobian_jacobiIter(
                        u, res, du, alphaRHS, JDFullInv
                    )
            elif self.mode == "flow":
                JDFlow = -alphaRHS * self.eval.rhs_flow_jacobian_diag(u)
                JDFlow += 1 / (dTau) + 1 / dt
                JDFlow_inv = self.eval.invert_jacobian_diag(JDFlow)

                for iJ in range(3):
                    du = self.eval.rhs_flow_jacobian_jacobiIter(
                        u, res, du, alphaRHS, JDFlow_inv
                    )
            elif self.mode == "source":
                JDSource = -alphaRHS * self.eval.rhs_source_jacobian(u)
                if JDSource.ndim == 3:
                    nV = JDSource.shape[0]
                    eyeNx = np.eye(nV).reshape(nV, nV, 1) * np.ones(
                        (1, 1, JDSource.shape[2])
                    )
                    JDSource += (1 / (dTau) + 1 / dt) * eyeNx
                else:
                    JDSource += 1 / (dTau) + 1 / dt
                JDSourceInv = self.eval.invert_jacobian_diag(JDSource)
                du = self.eval.jacobian_diag_mult(JDSourceInv, res)
            elif self.mode == "masked_implicit":
                # Jacobian for F(u) + (1 - chi_split) * S(u)
                JDFlow = -alphaRHS * self.eval.rhs_flow_jacobian_diag(u)
                JDFlow += 1 / (dTau) + 1 / dt
                JDSource = -alphaRHS * self.eval.rhs_source_jacobian(u)
                # Scale source Jacobian by (1 - chi_split)
                if self.chi_split is not None:
                    chi = self.chi_split  # (nx,)
                    one_minus_chi = 1.0 - chi
                    if JDSource.ndim == 2:
                        # (nVars, nx) * (nx,) broadcast
                        JDSource = JDSource * one_minus_chi[np.newaxis, :]
                    elif JDSource.ndim == 3:
                        # (nVars, nVars, nx) * (nx,) broadcast
                        JDSource = JDSource * one_minus_chi[np.newaxis, np.newaxis, :]
                JDFlow = self.eval.add_jacobian_diags(JDFlow, JDSource)
                JDFullInv = self.eval.invert_jacobian_diag(JDFlow)

                for iJ in range(3):
                    du = self.eval.rhs_flow_jacobian_jacobiIter(
                        u, res, du, alphaRHS, JDFullInv
                    )
            elif self.mode == "masked_split":
                # Jacobian for chi_split * S(u)
                JDSource = -alphaRHS * self.eval.rhs_source_jacobian(u)
                # Scale source Jacobian by chi_split
                if self.chi_split is not None:
                    chi = self.chi_split  # (nx,)
                    if JDSource.ndim == 2:
                        JDSource = JDSource * chi[np.newaxis, :]
                    elif JDSource.ndim == 3:
                        JDSource = JDSource * chi[np.newaxis, np.newaxis, :]
                if JDSource.ndim == 3:
                    nV = JDSource.shape[0]
                    eyeNx = np.eye(nV).reshape(nV, nV, 1) * np.ones(
                        (1, 1, JDSource.shape[2])
                    )
                    JDSource += (1 / (dTau) + 1 / dt) * eyeNx
                else:
                    JDSource += 1 / (dTau) + 1 / dt
                JDSourceInv = self.eval.invert_jacobian_diag(JDSource)
                du = self.eval.jacobian_diag_mult(JDSourceInv, res)
            else:
                raise NotImplementedError()
            u += du

            resN = np.linalg.norm(res)
            rhsN = np.linalg.norm(rhs)
            duN = np.linalg.norm(du)
            if iter == 1:
                resN0 = resN
            stop = False
            # Converged if:
            # 1. Residual is small relative to initial residual, OR
            # 2. Residual is small relative to RHS (handles zero initial residual), OR
            # 3. Increment is at machine precision (absolute tolerance on du)
            abs_tol = 1e-12
            if (resN < self.rel_tol * resN0
                or resN < self.rel_tol_rhs * (rhsN + 1e-300)
                or duN < abs_tol):
                stop = True
            if iter % self.n_print == 0 or stop:
                print(
                    f"iter [{iStage},{iter}], resN [{resN:.4e} / {resN0:.4e}]"
                )
            if stop:
                break
        else:
            # raise UserWarning("did not converge")
            print("did not converge")
        return u, rhs


class FrhsDITRExp(Frhs):
    """RHS functor with exponential Jacobian extraction for DITRExp."""

    def __init__(self, eval: AdvReactUni1DEval, mode: str = "full",
                 chi_split: np.ndarray = None):
        super().__init__(eval=eval, mode=mode, chi_split=chi_split)
        self.currentA = None
        self.currentU = None
        self._eigV = None
        self._eigVinv = None
        self._eigvals = None

    def dt(self, u, cStage, iStage):
        hx = self.eval.fv.hx
        ax = self.eval.ax
        eps_max = float(np.max(self.eval.eps))
        dt_adv = hx / ax if ax != 0 else 1e300
        dt_diff = hx**2 / eps_max if eps_max != 0 else 1e300
        return min(dt_adv, dt_diff)

    def Jacobian(self, u, cStage, iStage):
        raise NotImplementedError()

    def JacobianExpo(self, u, cStage, iStage):
        self.currentU = u.copy()
        JDSource = self.eval.rhs_source_jacobian(u)
        if JDSource.ndim == 2:
            # Per-element scalar diagonal (original path)
            if self.eval.model == "":
                JDSource = self.eval.rhs_flow_jacobian_diag(u)
            self.currentA = (
                np.minimum(JDSource, np.abs(JDSource).max() * -1e-4) * 1
            )
            self._eigV = None  # not used for ndim==2
            self._eigVinv = None
            self._eigvals = None
            return self.currentA
        elif JDSource.ndim == 3:
            # Per-point dense matrix: shape (nVars, nVars, nx)
            # Eigendecompose and extract decaying portion.
            # Store eigenbasis for reuse in exp/phi computations.
            nVars, _, nx = JDSource.shape
            Jt = np.moveaxis(JDSource, (0, 1), (-2, -1))  # (nx, nV, nV)

            # --- Eigenvalue treatment strategy ---
            # "complex": keep complex eigenvalues, project out Re>=0
            # "real":    drop imaginary parts first, then project out >=0
            # expo_eig_mode = "real"
            expo_eig_mode = "complex"

            # Batch eigendecomposition
            Vt = np.empty((nx, nVars, nVars), dtype=complex)
            Vinvt = np.empty_like(Vt)
            kept = np.empty((nx, nVars), dtype=complex)
            for ix in range(nx):
                eigvals_ix, V_ix = np.linalg.eig(Jt[ix])

                if expo_eig_mode == "real":
                    # Drop imaginary parts: use only the decay rate.
                    # For a conjugate pair a +/- bi, both collapse to a,
                    # so A degenerates to a*I in that subspace -- the
                    # oscillatory direction is removed, leaving only the
                    # physical decay rate for the exponential integrator.
                    proj = eigvals_ix.real.copy()
                    proj[proj >= 0] = 0.0
                    kept[ix] = proj
                else:  # "complex"
                    # Keep complex eigenvalues, project out Re>=0
                    proj = eigvals_ix.copy()
                    proj[eigvals_ix.real >= 0] = 0.0
                    kept[ix] = proj

                Vt[ix] = V_ix
                Vinvt[ix] = np.linalg.inv(V_ix)

            # Alternative: clamp instead of project (uncomment to use)
            # maxAbsEig = max(np.abs(kept).max(), 1e-30)
            # eps_shift = maxAbsEig * 1e-4
            # for ix in range(nx):
            #     kept[ix].real = np.minimum(kept[ix].real, -eps_shift)

            # Reconstruct A = V @ diag(kept) @ V^{-1}  (real)
            At = np.empty_like(Jt)
            for ix in range(nx):
                At[ix] = (Vt[ix] @ np.diag(kept[ix]) @ Vinvt[ix]).real

            # Store eigenbasis (nVars, nVars, nx) and (nVars, nx) layout
            self._eigV = np.moveaxis(Vt, (-2, -1), (0, 1))       # (nV, nV, nx) complex
            self._eigVinv = np.moveaxis(Vinvt, (-2, -1), (0, 1)) # (nV, nV, nx) complex
            self._eigvals = np.moveaxis(kept, -1, 0)             # (nV, nx) complex
            self.currentA = np.moveaxis(At, (-2, -1), (0, 1))
            return self.currentA

    def JacobianExpoEye(self, u):
        nVars = u.shape[0]
        if nVars == 1:
            return np.ones_like(u)
        nx = u.shape[1]
        return np.eye(nVars).reshape(nVars, nVars, 1) * np.ones((1, 1, nx))

    def JacobianExpoMult(self, JExpo, u):
        if JExpo.ndim == 2 and u.ndim == 2:
            return JExpo * u
        elif JExpo.ndim == 3 and u.ndim == 2:
            return np.einsum("ij...,j...->i...", JExpo, u)
        elif JExpo.ndim == 3 and u.ndim == 3:
            return np.einsum("ij...,jk...->ik...", JExpo, u)
        return JExpo * u

    def _reconstruct_from_eigvals(self, scalar_per_eigval):
        """Reconstruct (nVars, nVars, nx) real matrix from per-eigenvalue
        scalar function values.

        Args:
            scalar_per_eigval: (nVars, nx) complex array of f(lambda_i)
                for each eigenvalue at each spatial point.

        Returns:
            (nVars, nVars, nx) real array:  V @ diag(f(lambda)) @ V^{-1}
        """
        # V: (nV, nV, nx), scalar_per_eigval: (nV, nx)
        # diag broadcast: V[:,j,ix] * scalar[j,ix] then @ Vinv
        # = einsum("ij...,j...,jk...->ik...", V, s, Vinv)
        Vs = self._eigV * scalar_per_eigval[np.newaxis, :, :]  # (nV, nV, nx)
        result = np.einsum("ij...,jk...->ik...", Vs, self._eigVinv)
        return result.real

    def JacobianExpoExp(self, u, dt, cStage, iStage):
        if self.currentA.ndim == 2:
            Ah = self.currentA * dt
            return np.exp(Ah)
        elif self.currentA.ndim == 3:
            # exp(A*dt) via eigenbasis: V @ diag(exp(lambda_i * dt)) @ V^{-1}
            zh = self._eigvals * dt  # (nVars, nx) complex
            return self._reconstruct_from_eigvals(np.exp(zh))

    def JacobianExpoPhikSeq(self, u, dt, k_max, cStage, iStage):
        if self.currentA.ndim == 2:
            Ah = self.currentA * dt
            ifFix = np.abs(Ah) < 1e-3
            AhInv = self.eval.invert_jacobian_diag(Ah)
            ret = [self.JacobianExpoExp(u, dt, cStage, iStage)]
            ret[0][ifFix] = ODE.expo_quad_phik0(0)
            for k in range(k_max):
                ret.append(AhInv * (ret[k] - ODE.expo_quad_phik0(k)))
                ret[-1][ifFix] = ODE.expo_quad_phik0(k + 1)
            return ret
        elif self.currentA.ndim == 3:
            # Compute scalar phi_k(z_i) for each eigenvalue z_i = lambda_i * dt,
            # then reconstruct matrix phi_k via eigenbasis.
            # This avoids matrix inverse of A*dt entirely.
            nVars, nx = self._eigvals.shape
            zh = self._eigvals * dt  # (nVars, nx) complex

            # Scalar phi sequence per eigenvalue
            tol = 1e-6
            small = np.abs(zh) < tol
            # Safe inverse: use 1.0 for small entries (overwritten by Taylor)
            zh_safe = np.where(small, 1.0, zh)
            zh_inv = 1.0 / zh_safe
            exp_zh = np.exp(zh)

            # phi_0 = exp(z)
            phi_scalars = [exp_zh.copy()]
            phi_scalars[0][small] = ODE.expo_quad_phik0(0)
            for k in range(k_max):
                next_phi = zh_inv * (phi_scalars[k] - ODE.expo_quad_phik0(k))
                next_phi[small] = ODE.expo_quad_phik0(k + 1)
                phi_scalars.append(next_phi)

            # Reconstruct matrix phi_k = V @ diag(phi_k(z_i)) @ V^{-1}
            ret = []
            for phi_s in phi_scalars:
                ret.append(self._reconstruct_from_eigvals(phi_s))
            return ret

    def __call__(self, u, cStage, iStage):
        return super().__call__(u, cStage, iStage) - self.JacobianExpoMult(
            self.currentA, u - self.currentU
        )


class FsolveDITR(Fsolve):
    """Implicit solver functor for DITRExp (2-stage coupled solve)."""

    def __init__(
        self,
        eval: AdvReactUni1DEval,
        CFL: float = 10,
        rel_tol: float = 1e-4,
        rel_tol_rhs: float = None,
        max_iter: int = 1000,
        n_print: int = 10,
        mode: str = "full",
        chi_split: np.ndarray = None,
        **kwargs,
    ):
        super().__init__(
            eval=eval,
            CFL=CFL,
            rel_tol=rel_tol,
            rel_tol_rhs=rel_tol_rhs,
            max_iter=max_iter,
            n_print=n_print,
            chi_split=chi_split,
            **kwargs,
        )

        if mode not in {
            "full",
            "flow",
            "source",
            "masked_implicit",
            "masked_split",
        }:
            raise ValueError("mode not valid")

    def __call__(
        self,
        u0,
        dt,
        alphaRHS,
        fRHS: ODE.ODE_F_RHS,
        fRes,
        cStage,
        iStage,
    ):
        us = [np.copy(u0c) for u0c in u0]
        nVars, nx = self.eval.fv.get_shape_u(us[0])

        resN0 = None
        rhs0 = fRHS(u0[0], cStage, iStage)
        rhs = [rhs0, rhs0]

        for iter in range(1, self.max_iter + 1):
            # make a ref
            for iStageI in range(2):
                u = us[iStageI]
                cStageC = cStage[iStageI]
                iStageC = iStage[iStageI]
                dTau = fRHS.dt(u, cStageC, iStageC) * self.CFL

                res = (
                    -(u) / dt
                    + fRHS.JacobianExpoMult(alphaRHS[iStageI][0], rhs[0])
                    + fRHS.JacobianExpoMult(alphaRHS[iStageI][1], rhs[1])
                    + fRHS.JacobianExpoMult(alphaRHS[iStageI + 2][0], us[0])
                    / dt
                    + fRHS.JacobianExpoMult(alphaRHS[iStageI + 2][1], us[1])
                    / dt
                    + fRes[iStageI]
                )
                alphaRHSDiag = alphaRHS[iStageI][iStageI]

                du = np.zeros_like(u)
                if self.mode == "full":
                    JDFlow = -fRHS.JacobianExpoMult(
                        alphaRHSDiag, self.eval.rhs_flow_jacobian_diag(u)
                    )

                    scalarDiag = 1 / (dTau) + 1 / dt
                    if JDFlow.ndim == 3:
                        nV = JDFlow.shape[0]
                        eyeNx = np.eye(nV).reshape(nV, nV, 1) * np.ones(
                            (1, 1, JDFlow.shape[2])
                        )
                        JDFlow += scalarDiag * eyeNx
                    else:
                        JDFlow += scalarDiag
                    # JDFlow_inv = self.eval.invert_jacobian_diag(JDFlow)
                    JDSource = -fRHS.JacobianExpoMult(
                        alphaRHSDiag, self.eval.rhs_source_jacobian(u)
                    )
                    # if self.eval.model != "" and isinstance(fRHS, FrhsDITRExp):
                    #     JDSource -= fRHS.JacobianExpo(u0[0], 0.0, 0)
                    if isinstance(fRHS, FrhsDITRExp):
                        JDSource += fRHS.JacobianExpoMult(
                            alphaRHSDiag, fRHS.JacobianExpo(u0[0], 0.0, 0)
                        )
                    JDFlow = self.eval.add_jacobian_diags(JDFlow, JDSource)
                    JDFullInv = self.eval.invert_jacobian_diag(JDFlow)

                elif self.mode == "flow":
                    JDFlow = -fRHS.JacobianExpoMult(
                        alphaRHSDiag, self.eval.rhs_flow_jacobian_diag(u)
                    )

                    scalarDiag = 1 / (dTau) + 1 / dt
                    if JDFlow.ndim == 3:
                        nV = JDFlow.shape[0]
                        eyeNx = np.eye(nV).reshape(nV, nV, 1) * np.ones(
                            (1, 1, JDFlow.shape[2])
                        )
                        JDFlow += scalarDiag * eyeNx
                    else:
                        JDFlow += scalarDiag
                    JDFullInv = self.eval.invert_jacobian_diag(JDFlow)

                elif self.mode == "source":
                    JDFlow = (
                        -fRHS.JacobianExpoMult(
                            alphaRHSDiag, self.eval.rhs_flow_jacobian_diag(u)
                        )
                        * 0.0
                    )

                    scalarDiag = 1 / (dTau) + 1 / dt
                    if JDFlow.ndim == 3:
                        nV = JDFlow.shape[0]
                        eyeNx = np.eye(nV).reshape(nV, nV, 1) * np.ones(
                            (1, 1, JDFlow.shape[2])
                        )
                        JDFlow += scalarDiag * eyeNx
                    else:
                        JDFlow += scalarDiag
                    # JDFlow_inv = self.eval.invert_jacobian_diag(JDFlow)
                    JDSource = -fRHS.JacobianExpoMult(
                        alphaRHSDiag, self.eval.rhs_source_jacobian(u)
                    )
                    # if self.eval.model != "" and isinstance(fRHS, FrhsDITRExp):
                    #     JDSource -= fRHS.JacobianExpo(u0[0], 0.0, 0)
                    if isinstance(fRHS, FrhsDITRExp):
                        JDSource += fRHS.JacobianExpoMult(
                            alphaRHSDiag, fRHS.JacobianExpo(u0[0], 0.0, 0)
                        )
                    JDFlow = self.eval.add_jacobian_diags(JDFlow, JDSource)
                    JDFullInv = self.eval.invert_jacobian_diag(JDFlow)
                else:
                    raise NotImplementedError()

                for iJ in range(3):
                    du = self.eval.rhs_flow_jacobian_jacobiIterExpo(
                        u, res, du, alphaRHSDiag, JDFullInv
                    )
                u += du
                rhs[iStageI] = fRHS(u, cStageC, iStageC)

            resN = np.linalg.norm(res)
            rhsN = np.linalg.norm(rhs[0]) + np.linalg.norm(rhs[1])
            duN = np.linalg.norm(du)
            if iter == 1:
                resN0 = resN

            stop = False
            # Converged if:
            # 1. Residual is small relative to initial residual, OR
            # 2. Residual is small relative to RHS (handles zero initial residual), OR
            # 3. Increment is at machine precision (absolute tolerance on du)
            abs_tol = 1e-12
            if (resN < self.rel_tol * resN0
                or resN < self.rel_tol_rhs * (rhsN + 1e-300)
                or duN < abs_tol):
                stop = True
            if iter % self.n_print == 0 or stop:
                print(
                    f"iter [{iStage},{iter}], resN [{resN:.4e} / {resN0:.4e}]"
                )
            if stop:
                break
        else:
            # raise UserWarning("did not converge")
            print("did not converge")
        return us, rhs
