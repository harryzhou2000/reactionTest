import numpy as np
from abc import ABC, abstractmethod
import math


def expo_quad_phik0(k):
    return 1.0 / math.factorial(k)


def expo_quad_phik(k: int, z, zinv, expz):
    if k == 0:
        return expz
    return zinv * (expo_quad_phik(k - 1, z, zinv, expz) - expo_quad_phik0(k - 1))


def expo_quad_phi_kseq(k_max: int, z, zinv, expz):
    ret = [expz]
    for k in range(k_max):
        ret.append(zinv * (ret[k] - expo_quad_phik0(k)))
    return ret


class ODE_F_RHS(ABC):
    @abstractmethod
    def __call__(self, u, cStage, iStage):  # -> f(u), f == du/dt(u)
        pass

    @abstractmethod
    def Jacobian(self, u: np.ndarray, cStage, iStage):
        pass

    @abstractmethod
    def dt(self, u: np.ndarray, cStage, iStage) -> float:
        pass

    def JacobianExpo(self, u: np.ndarray, dt, cStage, iStage):
        self.currentA = np.zeros_like(u) - 1e-300
        return self.currentA * dt

    def JacobianExpoEye(self, u: np.ndarray):
        return np.ones_like(u)

    def JacobianExpoExp(self, u: np.ndarray, dt, cStage, iStage):
        return np.ones_like(u)

    def JacobianExpoPhikSeq(self, u: np.ndarray, dt, k_max, cStage, iStage):
        ret = [expo_quad_phik0(0) * np.ones_like(u)]
        for k in range(k_max):
            ret.append(expo_quad_phik0(k + 1) * np.ones_like(u))
        return ret

    def JacobianExpoMult(self, JExpo, u):
        return JExpo * u


class ODE_F_SOLVE_SingleStage(ABC):
    @abstractmethod
    def __call__(self, u0, dt, alphaRHS, fRHS, fRes, cStage, iStage):  # -> (u,f(u))
        """solves: -(u) / dt + alphaRHS * fRHS(u) + fRes == 0

        Warning: should treat u0 as immutable

        Args:
            u0 (_type_): _description_
            alphaRHS (_type_): _description_
            fRHS (_type_): _description_
            fRes (_type_): _description_
        """
        pass


class ImplicitOdeIntegrator(ABC):
    @abstractmethod
    def step(self, dt: float, u, fRHS: ODE_F_RHS, fSolve: ODE_F_SOLVE_SingleStage):
        pass


class ESDIRK(ImplicitOdeIntegrator):

    def __init__(self, method: str):
        super().__init__()
        from . import ESDIRK_Data

        butherAMap = {
            "ESDIRK4": ESDIRK_Data._ESDIRK_ButherA_ESDIRK4(),
            "ESDIRK3": ESDIRK_Data._ESDIRK_ButherA_ESDIRK3(),
            "Trapezoid": ESDIRK_Data._ESDIRK_ButherA_Trapezoid(),
            "BackwardEuler": ESDIRK_Data._ESDIRK_ButherA_BackwardEuler(),
        }
        if method in butherAMap:
            butcherA = butherAMap[method]
        else:
            raise ValueError(f"Method {method} not found!")

        butcherA = np.array(butcherA, dtype=np.float64)

        butcherB = butcherA[-1, :]
        butcherC = butcherA.sum(axis=1)
        self.butcherA = butcherA
        self.butcherB = butcherB
        self.butcherC = butcherC

        assert butcherA.shape[0] == butcherA.shape[1]
        self.nStage = butcherA.shape[0]
        assert butcherC[0] == 0
        assert (butcherA[0, :] == 0).all()
        self.rhsSeq = [None for _ in range(self.nStage)]
        self.uSeq = [None for _ in range(self.nStage)]

    def step(self, dt: float, u, fRHS: ODE_F_RHS, fSolve: ODE_F_SOLVE_SingleStage):
        uLast = u
        for iStage in range(1, self.nStage + 1):
            if iStage == 1:
                self.rhsSeq[iStage - 1] = fRHS(
                    u=u, cStage=self.butcherC[iStage - 1], iStage=iStage
                )
                continue

            fRes = uLast * (1 / dt)
            for jStage in range(1, iStage):
                fRes += self.butcherA[iStage - 1, jStage - 1] * self.rhsSeq[jStage - 1]
            self.uSeq[iStage - 1], self.rhsSeq[iStage - 1] = fSolve(
                u0=u,
                dt=dt,
                alphaRHS=self.butcherA[iStage - 1, iStage - 1],
                fRHS=fRHS,
                fRes=fRes,
                cStage=self.butcherC[iStage - 1],
                iStage=iStage,
            )
            u = self.uSeq[iStage - 1]

        return u


class DITRExp(ImplicitOdeIntegrator):

    def __init__(self, c2=0.5):
        super().__init__()

        self.c2 = c2

        self.d0 = 1 - (3 * c2**2 - 2 * c2**3)
        self.d1 = 1 - self.d0
        self.d2 = c2 - 2 * c2**2 + c2**3
        self.d3 = -(c2**2) + c2**3

    def step(self, dt: float, u, fRHS: ODE_F_RHS, fSolve: ODE_F_SOLVE_SingleStage):
        uLast = u
        umid = uLast.copy()
        u = uLast.copy()

        Ah = fRHS.JacobianExpo(u, dt, cStage=0.0, iStage=1)
        expc2hA = fRHS.JacobianExpoExp(u, dt * self.c2, cStage=0.0, iStage=1)
        phi0, phi1, phi2, phi3 = fRHS.JacobianExpoPhikSeq(
            u, dt, 3, cStage=0.0, iStage=1
        )
        rhsLast = fRHS(
            u=u, cStage=0, iStage=0
        )  # ! warning: we need a fRHS excluding the Au part

        eye = fRHS.JacobianExpoEye(u)

        uLastBc2 = fRHS.JacobianExpoMult(expc2hA, uLast)
        uLastB1 = fRHS.JacobianExpoMult(phi0, uLast)

        b1 = 1 / self.c2 * (self.c2 * phi1 - (1 + self.c2) * phi2 + 2 * phi3)
        b2 = (phi2 - 2 * phi3) / ((1 - self.c2) * self.c2)
        b3 = 1 / (1 - self.c2) * (-self.c2 * phi2 + 2 * phi3)
        d1pd3mhA = self.d1 * eye + Ah * self.d3
        a21 = eye * self.d2 + d1pd3mhA * b1
        a22 = d1pd3mhA * b2
        a23 = eye * self.d3 + d1pd3mhA * b3

        alphas = [[a22, a23], [b2, b3]]

        fResMid = uLastBc2 / dt + fRHS.JacobianExpoMult(a21, rhsLast)
        fRes1 = uLastB1 / dt + fRHS.JacobianExpoMult(b1, rhsLast)

        (umid, u1), (rhsMid, rhs1) = fSolve(
            u0=[umid, u],
            dt=dt,
            alphaRHS=alphas,
            fRHS=fRHS,
            fRes=[fResMid, fRes1],
            cStage=[self.c2, 1.0],
            iStage=[0.5, 1],
        )

        return u1
