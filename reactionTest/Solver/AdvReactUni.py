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

        return rhs

    def rhs_flow_jacobian_diag(self, u: np.ndarray):
        nVars, nx = self.fv.get_shape_u(u)

        rhsJD = np.zeros((nVars, nx))
        for nx, dx, (uN,) in self.fv.cellOthers((u)):
            uRec = u
            uRecN = uN
            an = self.ax * nx
            f = -0.5 * abs(an) * (-1)
            rhsJD += -f * 1.0
        rhsJD /= self.fv.vol

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
        duNew = self.jacobian_diag_mult(alphaDiag, duNew)
        duNew += res
        duNew = self.jacobian_diag_mult(rhsJDInv, duNew)
        return duNew

    def rhs_source(self, u: np.ndarray):
        if self.model == "bistable":
            a = self.params["a"]
            k = self.params["k"]
            return u * (1 - u) * (u - a) * k
        return 0 * u

    def rhs_source_jacobian(self, u: np.ndarray):
        if self.model == "bistable":
            a = self.params["a"]
            k = self.params["k"]
            return (2 * u - a + 2 * a * u - 3 * u**2) * k
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
                        JDFlow += JDSource  # TODO: make compatible with matrix
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
                self.currentAh = None

            def dt(self, u, cStage, iStage):
                return self.eval.fv.hx / self.eval.ax

            def Jacobian(self, u, cStage, iStage):
                raise NotImplementedError()

            def JacobianExpo(self, u, dt, cStage, iStage):
                JDSource = eval.rhs_source_jacobian(u)
                # TODO: handle if JDSource is a matrix
                if JDSource.ndim == 2:
                    if self.eval.model == "":
                        JDSource = eval.rhs_flow_jacobian_diag(u)
                    # self.currentA = np.minimum(JDSource, np.abs(JDSource).max() * -1e-8) * 10/128
                    self.currentA = np.ones_like(u) * (-20.0)

                    self.currentAh = self.currentA * dt
                    return self.currentAh

            def JacobianExpoEye(self, u):
                return np.ones_like(u)

            def JacobianExpoMult(self, JExpo, u):
                return JExpo * u

            def JacobianExpoExp(self, u, dt, cStage, iStage):
                Ah = self.currentAh
                return np.exp(Ah)

            def JacobianExpoPhikSeq(self, u, dt, k_max, cStage, iStage):
                Ah = self.currentAh
                AhInv = self.eval.invert_jacobian_diag(Ah)
                ret = [self.JacobianExpoExp(u, dt, cStage, iStage)]
                for k in range(k_max):
                    ret.append(AhInv * (ret[k] - ODE.expo_quad_phik0(k)))
                return ret

            def __call__(self, u, cStage, iStage):
                return super().__call__(u, cStage, iStage) - self.JacobianExpoMult(
                    self.currentA, u
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

                if mode not in {"full"}:
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
                            + fRes[iStageI]
                        )
                        alphaRHSDiag = alphaRHS[iStageI][iStageI]

                        du = np.zeros_like(u)
                        if self.mode == "full":
                            JDFlow = -fRHS.JacobianExpoMult(
                                alphaRHSDiag, self.eval.rhs_flow_jacobian_diag(u)
                            )

                            JDFlow += 1 / (dTau) + 1 / dt
                            # JDFlow_inv = self.eval.invert_jacobian_diag(JDFlow)
                            JDSource = -fRHS.JacobianExpoMult(
                                alphaRHSDiag, self.eval.rhs_source_jacobian(u)
                            )
                            JDFlow += JDSource  # TODO: make compatible with matrix
                            JDFullInv = self.eval.invert_jacobian_diag(JDFlow)

                            for iJ in range(3):
                                du = self.eval.rhs_flow_jacobian_jacobiIterExpo(
                                    u, res, du, alphaRHSDiag, JDFullInv
                                )
                        else:
                            raise NotImplementedError()
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

    def step(self, dt: float, u: np.ndarray, mode="full", use_exp=False, solve_opts={}):
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
            )
        return self.ode.step(
            dt,
            u,
            self.Frhs(self.eval, mode=mode_frhs),
            self.Fsolve(self.eval, mode=mode_fsolve, **solve_opts),
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
