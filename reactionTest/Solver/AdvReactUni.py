if __name__ == "__main__":
    from FVUni2nd import FVUni2nd1D
    from AdvReactUniFunctors import Frhs, Fsolve, FrhsDITRExp, FsolveDITR
    import ODE
else:
    from .FVUni2nd import FVUni2nd1D
    from .AdvReactUniFunctors import Frhs, Fsolve, FrhsDITRExp, FsolveDITR
    from . import ODE
import numpy as np


class AdvReactUni1DEval:
    def __init__(self, fv: FVUni2nd1D, model: str = "", params={}, nVars: int = 1):
        self.fv: FVUni2nd1D = fv

        self.ax: float = 1.0

        self.model = model.lower()
        self.params = params

        # Per-component diffusion coefficients: always (nVars, 1) for broadcasting
        eps_raw = params.get("eps", 0.0)
        eps_arr = np.atleast_1d(np.asarray(eps_raw, dtype=float))
        if eps_arr.size == 1:
            eps_arr = np.broadcast_to(eps_arr, (nVars,)).copy()
        self.eps = eps_arr.reshape(-1, 1)  # (nVars, 1)

        # Bind model-specific source functions to avoid if-elif per call
        if self.model == "bistable":
            self.rhs_source = self._rhs_source_bistable
            self.rhs_source_jacobian = self._rhs_source_jacobian_bistable
        elif self.model == "brusselator":
            self.rhs_source = self._rhs_source_brusselator
            self.rhs_source_jacobian = self._rhs_source_jacobian_brusselator
        elif self.model == "premixed":
            self.rhs_source = self._rhs_source_premixed
            self.rhs_source_jacobian = self._rhs_source_jacobian_premixed
        else:
            self.rhs_source = self._rhs_source_none
            self.rhs_source_jacobian = self._rhs_source_jacobian_none

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

        # Diffusion: eps * u_xx via central difference (per-component)
        if np.any(self.eps != 0.0):
            diff = np.zeros_like(u)
            for nx, dx, (uN,) in self.fv.cellOthers((u,)):
                diff += uN - u
            rhs += self.eps / (self.fv.hx**2) * diff

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

        if np.any(self.eps != 0.0):
            rhsJD += -2.0 * self.eps / (self.fv.hx**2)

        return rhsJD

    def rhs_flow_jacobian_matvec(self, u: np.ndarray, du: np.ndarray):

        drhs = np.zeros_like(u)
        for nx, dx, (duN,) in self.fv.cellOthers((du,), homogeneous=True):
            duRec = du
            duRecN = duN
            an = self.ax * nx
            df = 0.5 * an * (duRec + duRecN) - 0.5 * abs(an) * (duRecN - duRec)
            drhs += -df * 1.0
        drhs /= self.fv.vol

        if np.any(self.eps != 0.0):
            ddiff = np.zeros_like(du)
            for nx, dx, (duN,) in self.fv.cellOthers((du,), homogeneous=True):
                ddiff += duN - du
            drhs += self.eps / (self.fv.hx**2) * ddiff

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
        for nx, dx, (duN,) in self.fv.cellOthers((du,), homogeneous=True):
            duRec = du
            duRecN = duN
            an = self.ax * nx
            df = 0.5 * an * (duRecN) - 0.5 * abs(an) * (duRecN)
            duNew += -alphaDiag * df * 1.0 / self.fv.vol

        # Diffusion off-diagonal Jacobi contribution
        if np.any(self.eps != 0.0):
            for nx, dx, (duN,) in self.fv.cellOthers((du,), homogeneous=True):
                duNew += alphaDiag * self.eps / (self.fv.hx**2) * duN

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
        for nx, dx, (duN,) in self.fv.cellOthers((du,), homogeneous=True):
            duRec = du
            duRecN = duN
            an = self.ax * nx
            df = 0.5 * an * (duRecN) - 0.5 * abs(an) * (duRecN)
            duNew += -df * 1.0 / self.fv.vol

        # Diffusion off-diagonal Jacobi contribution
        if np.any(self.eps != 0.0):
            for nx, dx, (duN,) in self.fv.cellOthers((du,), homogeneous=True):
                duNew += self.eps / (self.fv.hx**2) * duN

        duNew = self.jacobian_diag_mult(alphaDiag, duNew)
        duNew += res
        duNew = self.jacobian_diag_mult(rhsJDInv, duNew)
        return duNew

    def _rhs_source_none(self, u: np.ndarray):
        return 0 * u

    def _rhs_source_bistable(self, u: np.ndarray):
        a = self.params["a"]
        k = self.params["k"]
        return u * (1 - u) * (u - a) * k

    def _rhs_source_brusselator(self, u: np.ndarray):
        A = self.params["A"]
        B = self.params["B"]
        k = self.params["k"]
        uu = u[0:1]
        vv = u[1:2]
        Su = k * (A - (B + 1) * uu + uu**2 * vv)
        Sv = k * (B * uu - uu**2 * vv)
        return np.concatenate([Su, Sv], axis=0)

    def _rhs_source_premixed(self, u: np.ndarray):
        # Premixed combustion: [T, Y]
        # omega = B * Y * exp(-E_div_RTb * (Tb / T - 1))
        # S_T = Q_div_rho_cp * omega,  S_Y = -omega
        Bp = self.params["B"]
        Q = self.params["Q_div_rho_cp"]
        Tb = self.params["Tb"]
        Ze = self.params["E_div_RTb"]
        T = u[0:1]
        Y = u[1:2]
        omega = Bp * Y * np.exp(-Ze * (Tb / T - 1))
        ST = Q * omega
        SY = -omega
        return np.concatenate([ST, SY], axis=0)

    def _rhs_source_jacobian_none(self, u: np.ndarray):
        return 1e-100 + u * 0

    def _rhs_source_jacobian_bistable(self, u: np.ndarray):
        a = self.params["a"]
        k = self.params["k"]
        return (2 * u - a + 2 * a * u - 3 * u**2) * k

    def _rhs_source_jacobian_brusselator(self, u: np.ndarray):
        A = self.params["A"]
        B = self.params["B"]
        k = self.params["k"]
        uu = u[0:1]
        vv = u[1:2]
        nVars, nx = self.fv.get_shape_u(u)
        J = np.zeros((nVars, nVars, nx))
        J[0, 0] = k * (-(B + 1) + 2 * uu[0] * vv[0])
        J[0, 1] = k * (uu[0] ** 2)
        J[1, 0] = k * (B - 2 * uu[0] * vv[0])
        J[1, 1] = k * (-(uu[0] ** 2))
        return J

    def _rhs_source_jacobian_premixed(self, u: np.ndarray):
        Bp = self.params["B"]
        Q = self.params["Q_div_rho_cp"]
        Tb = self.params["Tb"]
        Ze = self.params["E_div_RTb"]
        T = u[0:1]
        Y = u[1:2]
        expTerm = np.exp(-Ze * (Tb / T - 1))
        domega_dT = Bp * Y * Ze * Tb / (T**2) * expTerm
        domega_dY = Bp * expTerm
        nVars, nx = self.fv.get_shape_u(u)
        J = np.zeros((nVars, nVars, nx))
        J[0, 0] = Q * domega_dT[0]
        J[0, 1] = Q * domega_dY[0]
        J[1, 0] = -domega_dT[0]
        J[1, 1] = -domega_dY[0]
        return J

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

        # Probe storage: list of x locations to record
        self._probe_xs = []
        self._probe_indices = []
        self._probe_data = {}  # {x: {"t": [], "u": []}}

        # Functor classes (imported from AdvReactUniFunctors)
        self.Frhs = Frhs
        self.Fsolve = Fsolve
        self.FrhsDITRExp = FrhsDITRExp
        self.FsolveDITR = FsolveDITR

    def set_probes(self, x_locations: list):
        """Set probe locations for time series recording.

        Args:
            x_locations: List of x coordinates to probe. Each location is
                mapped to the nearest cell center.
        """
        self._probe_xs = list(x_locations)
        self._probe_indices = []
        self._probe_data = {}
        xcs = self.eval.fv.xcs
        for x in self._probe_xs:
            idx = np.argmin(np.abs(xcs - x))
            self._probe_indices.append(idx)
            self._probe_data[x] = {"t": [], "u": []}

    def clear_probes(self):
        """Clear all recorded probe data."""
        for x in self._probe_xs:
            self._probe_data[x] = {"t": [], "u": []}

    def _record_probes(self, t: float, u: np.ndarray):
        """Record probe data at current time."""
        for x, idx in zip(self._probe_xs, self._probe_indices):
            self._probe_data[x]["t"].append(t)
            # Store all variables at the probe location
            self._probe_data[x]["u"].append(u[:, idx].copy())

    def get_probe_data(self, x: float = None):
        """Get recorded probe data.

        Args:
            x: Specific probe location to retrieve. If None, returns all probes.

        Returns:
            If x is specified: dict with "t" (times) and "u" (values array).
            If x is None: dict mapping each probe x to its data.
            Always returns a deep copy so callers are not affected by
            subsequent clear_probes() calls.
        """
        import copy
        if x is not None:
            data = self._probe_data.get(x, None)
            return copy.deepcopy(data)
        return copy.deepcopy(self._probe_data)

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
        record_probes=False,
    ):
        mode = mode.lower()
        t = t0

        # Record initial state if probes are enabled
        if record_probes and self._probe_xs:
            self._record_probes(t, u)

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

            # Record probe data after each step
            if record_probes and self._probe_xs:
                self._record_probes(t, u)

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
