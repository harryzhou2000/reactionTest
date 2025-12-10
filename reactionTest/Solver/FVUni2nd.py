import numpy as np


class FVUni2nd1D:
    def __init__(self, nx: int):
        self.xs = np.linspace(0, 1, nx + 1)
        self.xcs = (self.xs[0:-1] + self.xs[1:]) * 0.5
        self.hx: float = 1.0 / nx
        self.nx = nx
        self.vol = self.hx

    def cellOthers(self, us: list[np.ndarray]):
        uN = [np.roll(u, 1, axis=-1) for u in us]
        yield -1, -self.hx, uN

        uN = [np.roll(u, -1, axis=-1) for u in us]
        yield 1, self.hx, uN

    def get_shape_u(self, u):
        nVars = u.shape[0]
        nx = u.shape[1]
        if len(u.shape) != 2 or nx != self.nx:
            raise ValueError("u size not compatible: " + f"{u.shape}")
        return nVars, nx

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


if __name__ == "__main__":
    fv = FVUni2nd1D(20)
    u = np.array([np.sin(fv.xcs * 2 * np.pi)])
    print(fv.recGrad(u))

    
