if __name__ == "__main__":
    from FVUni2nd import FVUni2nd1D
    import ODE
else:
    from .FVUni2nd import FVUni2nd1D
    from . import ODE
import numpy as np


class AdvReactUni1DEval:
    def __init__(self, fv: FVUni2nd1D, model: str = "", params={}):
        self.fv: FVUni2nd1D = fv

        self.ax: float = 1.0

        self.model = model.lower()
        self.params = params

    def rhs_flow(self, u: np.ndarray):
        uGrad = self.fv.recGrad(u)
        rhs = np.zeros_like(u)
        for nx, dx, (uN, uGradN) in self.fv.cellOthers((u, uGrad)):
            uRec = u + dx * 0.5 * uGrad[0]
            uRecN = uN - dx * 0.5 * uGradN[0]
            an = self.ax * nx
            f = 0.5 * an * (uRec + uRecN) - 0.5 * abs(an) * (uRecN - uRec)
            rhs += -f * 1.0
        rhs /= self.fv.vol

        # Diffusion: eps * u_xx via central difference
        eps = self.params.get("eps", 0.0)
        if eps != 0.0:
            diff = np.zeros_like(u)
            for nx, dx, (uN,) in self.fv.cellOthers((u,)):
                diff += uN - u
            rhs += eps / (self.fv.hx**2) * diff

        return rhs

    def rhs_flow_jacobian_diag(self, u: np.ndarray):
        nVars, nx = self.fv.get_shape_u(u)

        rhsJD = np.zeros((nVars, nx))
        for nx, dx, (uN,) in self.fv.cellOthers((u,)):
            uRec = u
            uRecN = uN
            an = self.ax * nx
            f = -0.5 * abs(an) * (-1)
            rhsJD += -f * 1.0
        rhsJD /= self.fv.vol

        eps = self.params.get("eps", 0.0)
        if eps != 0.0:
            nV, nX = self.fv.get_shape_u(u)
            rhsJD += -2.0 * eps / (self.fv.hx**2)

        return rhsJD

    def rhs_flow_jacobian_matvec(self, u: np.ndarray, du: np.ndarray):

        drhs = np.zeros_like(u)
        for nx, dx, (duN,) in self.fv.cellOthers((du,)):
            duRec = du
            duRecN = duN
            an = self.ax * nx
            df = 0.5 * an * (duRec + duRecN) - 0.5 * abs(an) * (duRecN - duRec)
            drhs += -df * 1.0
        drhs /= self.fv.vol

        eps = self.params.get("eps", 0.0)
        if eps != 0.0:
            ddiff = np.zeros_like(du)
            for nx, dx, (duN,) in self.fv.cellOthers((du,)):
                ddiff += duN - du
            drhs += eps / (self.fv.hx**2) * ddiff

        return drhs

    def rhs_flow_jacobian_jacobiIter(
        self,
        u: np.ndarray,
        res: np.ndarray,
        du: np.ndarray,
        alphaDiag: float,
        rhsJDInv: np.ndarray,
    ):
        duNew = np.copy(res)
        for nx, dx, (duN,) in self.fv.cellOthers((du,)):
            duRec = du
            duRecN = duN
            an = self.ax * nx
            df = 0.5 * an * (duRecN) - 0.5 * abs(an) * (duRecN)
            duNew += -alphaDiag * df * 1.0 / self.fv.vol

        # Diffusion off-diagonal Jacobi contribution
        eps = self.params.get("eps", 0.0)
        if eps != 0.0:
            for nx, dx, (duN,) in self.fv.cellOthers((du,)):
                duNew += alphaDiag * eps / (self.fv.hx**2) * duN

        duNew = self.jacobian_diag_mult(rhsJDInv, duNew)
        return duNew

    def rhs_flow_jacobian_jacobiIterExpo(
        self,
        u: np.ndarray,
        res: np.ndarray,
        du: np.ndarray,
        alphaDiag: np.ndarray,
        rhsJDInv: np.ndarray,
    ):
        duNew = np.zeros_like(res)
        for nx, dx, (duN,) in self.fv.cellOthers((du,)):
            duRec = du
            duRecN = duN
            an = self.ax * nx
            df = 0.5 * an * (duRecN) - 0.5 * abs(an) * (duRecN)
            duNew += -df * 1.0 / self.fv.vol

        # Diffusion off-diagonal Jacobi contribution
        eps = self.params.get("eps", 0.0)
        if eps != 0.0:
            for nx, dx, (duN,) in self.fv.cellOthers((du,)):
                duNew += eps / (self.fv.hx**2) * duN

        duNew = self.jacobian_diag_mult(alphaDiag, duNew)
        duNew += res
        duNew = self.jacobian_diag_mult(rhsJDInv, duNew)
        return duNew

    def rhs_source(self, u: np.ndarray):
        if self.model == "bistable":
            a = self.params["a"]
            k = self.params["k"]
            return u * (1 - u) * (u - a) * k
        elif self.model == "brusselator":
            A = self.params["A"]
            B = self.params["B"]
            k = self.params["k"]
            uu = u[0:1]
            vv = u[1:2]
            Su = k * (A - (B + 1) * uu + uu**2 * vv)
            Sv = k * (B * uu - uu**2 * vv)
            return np.concatenate([Su, Sv], axis=0)
        return 0 * u

    def rhs_source_jacobian(self, u: np.ndarray):
        if self.model == "bistable":
            a = self.params["a"]
            k = self.params["k"]
            return (2 * u - a + 2 * a * u - 3 * u**2) * k
        elif self.model == "brusselator":
            A = self.params["A"]
            B = self.params["B"]
            k = self.params["k"]
            uu = u[0:1]
            vv = u[1:2]
            nVars, nx = self.fv.get_shape_u(u)
            # Jacobian is 2x2 per point: [[dSu/du, dSu/dv], [dSv/du, dSv/dv]]
            J = np.zeros((nVars, nVars, nx))
            J[0, 0] = k * (-(B + 1) + 2 * uu[0] * vv[0])
            J[0, 1] = k * (uu[0] ** 2)
            J[1, 0] = k * (B - 2 * uu[0] * vv[0])
            J[1, 1] = k * (-(uu[0] ** 2))
            return J
        return 1e-100 + u * 0

    def invert_jacobian_diag(self, JD: np.ndarray):
        badShape = ValueError("Jacobian Diag shape not valid: " + f"{JD.shape}")
        if JD.ndim == 2:
            return 1.0 / JD
        if JD.ndim == 3:
            nVars = JD.shape[0]
            if JD.shape[1] != nVars:
                raise badShape
            JDI = np.linalg.pinv(np.moveaxis(JD, (0, 1), (-2, -1)))
            return np.moveaxis(JDI, (-2, -1), (0, 1))

        raise badShape

    def promote_to_matrix_diag(self, JD: np.ndarray):
        """Promote (nVars, nx) -> (nVars, nVars, nx) diagonal matrix."""
        if JD.ndim == 3:
            return JD
        nVars, nx = JD.shape
        out = np.zeros((nVars, nVars, nx))
        for i in range(nVars):
            out[i, i] = JD[i]
        return out

    def add_jacobian_diags(self, A: np.ndarray, B: np.ndarray):
        """Add two Jacobian diagonals, promoting if dimensions differ."""
        if A.ndim == B.ndim:
            return A + B
        if A.ndim == 2 and B.ndim == 3:
            return self.promote_to_matrix_diag(A) + B
        if A.ndim == 3 and B.ndim == 2:
            return A + self.promote_to_matrix_diag(B)
        raise ValueError(f"Cannot add Jacobian diags of ndim {A.ndim} and {B.ndim}")

    def jacobian_diag_mult(self, JD: np.ndarray, u: np.ndarray):
        badShape = ValueError("Jacobian Diag shape not valid: " + f"{JD.shape}")
        nVars, N = self.fv.get_shape_u(u)
        if JD.ndim == 2:
            return JD * u
        if JD.ndim == 3:
            if JD.shape != (nVars, nVars, N):
                raise badShape
            return np.einsum("ij...,j...->i...", JD, u)

        raise badShape


class AdvReactUni1DSolver:
    def __init__(self, eval: AdvReactUni1DEval, ode: ODE.ImplicitOdeIntegrator):
        self.eval = eval
        self.ode = ode

        class Frhs(ODE.ODE_F_RHS):
            def __init__(self, eval: AdvReactUni1DEval, mode="full"):
                super().__init__()
                self.eval = eval
                self.mode = mode
                if mode not in {"full", "flow", "source"}:
                    raise ValueError("mode not valid")

            def __call__(self, u, cStage, iStage):
                if self.mode == "full":
                    return self.eval.rhs_flow(u) + self.eval.rhs_source(u)
                elif self.mode == "flow":
                    return self.eval.rhs_flow(u)
                elif self.mode == "source":
                    return self.eval.rhs_source(u)
                else:
                    raise ValueError()

            def dt(self, u, cStage, iStage):
                return self.eval.fv.hx / self.eval.ax

            def Jacobian(self, u, cStage, iStage):
                raise NotImplementedError()

        class Fsolve(ODE.ODE_F_SOLVE_SingleStage):
            def __init__(
                self,
                eval: AdvReactUni1DEval,
                CFL=10,
                rel_tol=1e-4,
                max_iter=1000,
                n_print=10,
                mode="full",
            ):
                super().__init__()
                self.eval = eval
                self.CFL = CFL
                self.rel_tol = rel_tol
                self.max_iter = max_iter
                self.n_print = n_print
                self.mode = mode

                if mode not in {"full", "flow", "source", "full_split"}:
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
                    else:
                        raise NotImplementedError()
                    u += du

                    resN = np.linalg.norm(res)
                    if iter == 1:
                        resN0 = resN
                    stop = False
                    if resN < self.rel_tol * resN0:
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

        self.Frhs = Frhs
        self.Fsolve = Fsolve

        class FrhsDITRExp(Frhs):
            def __init__(self, eval: AdvReactUni1DEval, mode="full"):
                super().__init__(eval=eval, mode=mode)
                self.currentA = None

            def dt(self, u, cStage, iStage):
                return self.eval.fv.hx / self.eval.ax

            def Jacobian(self, u, cStage, iStage):
                raise NotImplementedError()

            def JacobianExpo(self, u, cStage, iStage):
                self.currentU = u.copy()
                JDSource = eval.rhs_source_jacobian(u)
                if JDSource.ndim == 2:
                    # Per-element scalar diagonal (original path)
                    if self.eval.model == "":
                        JDSource = eval.rhs_flow_jacobian_diag(u)
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
            def __init__(
                self,
                eval: AdvReactUni1DEval,
                CFL=10,
                rel_tol=1e-4,
                max_iter=1000,
                n_print=10,
                mode="full",
                **kwargs,
            ):
                super().__init__(
                    eval=eval,
                    CFL=CFL,
                    rel_tol=rel_tol,
                    max_iter=max_iter,
                    n_print=n_print,
                    **kwargs,
                )

                if mode not in {
                    "full",
                    "flow",
                    "source",
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
                    if iter == 1:
                        resN0 = resN

                    stop = False
                    if resN < self.rel_tol * resN0:
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

        self.FrhsDITRExp = FrhsDITRExp
        self.FsolveDITR = FsolveDITR

    def step(
        self,
        dt: float,
        u: np.ndarray,
        mode="full",
        use_exp=False,
        solve_opts={},
        uForce=lambda c2: 0.0,
    ):
        mode_frhs = mode
        mode_fsolve = mode
        if isinstance(self.ode, ODE.DITRExp):
            return self.ode.step(
                dt,
                u,
                (
                    self.FrhsDITRExp(self.eval, mode=mode_frhs)
                    if use_exp
                    else self.Frhs(self.eval, mode=mode_frhs)
                ),
                self.FsolveDITR(self.eval, mode=mode_fsolve, **solve_opts),
                fForce=uForce,
            )
        return self.ode.step(
            dt,
            u,
            self.Frhs(self.eval, mode=mode_frhs),
            self.Fsolve(self.eval, mode=mode_fsolve, **solve_opts),
            fForce=uForce,
        )

    def stepInterval(
        self,
        dt: float,
        u: np.ndarray,
        t0: float,
        t1: float,
        init_step=1,
        max_step=10000000,
        use_exp=False,
        solve_opts={},
        mode="full",
    ):
        mode = mode.lower()
        t = t0
        for iStep in range(init_step, max_step + 1):
            dtC = dt
            tNew = t + dtC
            stop = False
            if tNew + 1e-10 * dt >= t1:
                dtC = t1 - t
                tNew = t1
                stop = True
            if mode == "full":
                u = self.step(
                    dtC, u, mode="full", use_exp=use_exp, solve_opts=solve_opts
                )
            elif mode == "strang":
                N_react = 2
                u = self.step(dtC * 0.5, u, mode="flow", solve_opts=solve_opts)
                for i_react in range(N_react):
                    u = self.step(
                        dtC / N_react, u, mode="source", solve_opts=solve_opts
                    )
                u = self.step(dtC * 0.5, u, mode="flow", solve_opts=solve_opts)
            elif mode == "embed":
                cs = self.ode.get_cs()
                N_react = 2
                u_cur = u.copy()
                assert cs[0] == 0, "cs must start with 0"
                c_cur = cs[0]
                uForces = {c_cur: u_cur}
                for c in cs[1:]:
                    dtI = dtC * (c - c_cur)
                    for i_react in range(N_react):
                        u_cur = self.step(
                            dtI / N_react, u_cur, mode="source", solve_opts=solve_opts
                        )
                    c_cur = c
                    uForces[c_cur] = u_cur.copy()
                u = self.step(
                    dtC,
                    u,
                    mode="flow",
                    solve_opts=solve_opts,
                    uForce=lambda c2: (uForces[c2] - uForces[cs[0]]) / dtC,
                )

            t = tNew
            print(f"Step [{iStep}], t = [{t:.4e}] uNorm [{np.linalg.vector_norm(u)}]")
            if stop:
                break
        return u


if __name__ == "__main__":
    fv = FVUni2nd1D(20)
    u = np.array([np.sin(fv.xcs * 2 * np.pi)])

    advReactEval = AdvReactUni1DEval(fv)
    print(advReactEval.rhs_flow(u))

    JD = np.eye(3).reshape(3, 3, 1) @ np.ones((1, 20))
    JD[0, 1, :] = 1
    JDI = advReactEval.invert_jacobian_diag(JD)
    print(np.moveaxis(JDI, (0, 1), (-2, -1)))
    print(advReactEval.jacobian_diag_mult(JDI, np.ones((3, 20))))
