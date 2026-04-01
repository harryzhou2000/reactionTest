import Solver.ODE as ODE
import numpy as np
import scipy as sp

import scipy.linalg as spl


class Frhs(ODE.ODE_F_RHS):
    def __init__(self):
        pass

    def dt(self, u, cStage, iStage):
        return 0.1

    def Jacobian(self, u, cStage, iStage):
        return np.eye(u.shape[0]) * -1

    def __call__(self, u, cStage, iStage):
        return self.Jacobian(u, cStage, iStage) @ u


class FrhsDITRExp(Frhs):
    def __init__(self):
        pass

    def Jacobian(self, u, cStage, iStage): 
        return super().Jacobian(u, cStage, iStage) 

    def JacobianExpo(self, u, cStage, iStage):
        self.currentA = super().Jacobian(u, cStage, iStage) * 0.001 + np.eye(u.shape[0]) * -0
        self.currentU = u.copy()
        return self.currentA

    def JacobianExpoEye(self, u):
        return np.eye(u.shape[0])

    def JacobianExpoMult(self, JExpo, u):
        return JExpo @ u

    def JacobianExpoExp(self, u, dt, cStage, iStage):
        Ah = self.currentA * dt
        return spl.expm(Ah)

    def JacobianExpoPhikSeq(self, u, dt, k_max, cStage, iStage):
        Ah = self.currentA * dt
        AhInv = np.linalg.pinv(Ah)
        ret = [self.JacobianExpoExp(u, dt, cStage, iStage)]
        for k in range(k_max):
            ret.append(AhInv @ (ret[k] - ODE.expo_quad_phik0(k)))
        return ret

    def __call__(self, u, cStage, iStage):
        return super().__call__(u, cStage, iStage) - self.JacobianExpoMult(
            self.currentA, u - self.currentU
        )


class FsolveDITR(ODE.ODE_F_SOLVE_SingleStage):
    def __init__(
        self,
    ):
        self.CFL = 100
        self.rel_tol = 1e-9
        self.max_iter = 1000
        self.n_print = 1
        pass

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
                    + fRHS.JacobianExpoMult(alphaRHS[iStageI + 2][0], us[0]) / dt
                    + fRHS.JacobianExpoMult(alphaRHS[iStageI + 2][1], us[1]) / dt
                    + fRes[iStageI]
                )
                alphaRHSDiag = alphaRHS[iStageI][iStageI]

                du = np.zeros_like(u)
                JDFlow = -fRHS.JacobianExpoMult(
                    alphaRHSDiag, fRHS.Jacobian(u, cStage, iStage)
                ) + 1 / dt * np.eye(u.shape[0])

                JDFullInv = np.linalg.pinv(JDFlow)
                du = JDFullInv @ res

                u += du
                rhs[iStageI] = fRHS(u, cStageC, iStageC)

            resN = np.linalg.norm(res)
            if iter == 1:
                resN0 = resN

            stop = False
            if resN < self.rel_tol * resN0:
                stop = True
            if iter % self.n_print == 0 or stop:
                print(f"iter [{iStage},{iter}], resN [{resN:.4e} / {resN0:.4e}]")
            if stop:
                break
        else:
            # raise UserWarning("did not converge")
            print("did not converge")
        return us, rhs


u0 = np.ones((1,))

ode = ODE.DITRExp()


# u1 = ode.step(1.0, u0, fRHS=Frhs(), fSolve=FsolveDITR())

u1 = ode.step(1.0, u0, fRHS=FrhsDITRExp(), fSolve=FsolveDITR(), fForce=lambda ct: 0)


print(f"{np.exp(-1.0)} --- {np.exp(-1e-3)}")
print(u1)
