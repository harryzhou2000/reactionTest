if __name__ == "__main__":
    from FVUni1D import FVUni1D
    from FVUni2nd import FVUni2nd1D
    from AdvReactUniFunctors import Frhs, Fsolve, FrhsDITRExp, FsolveDITR
    import ODE
else:
    from .FVUni1D import FVUni1D
    from .FVUni2nd import FVUni2nd1D
    from .AdvReactUniFunctors import Frhs, Fsolve, FrhsDITRExp, FsolveDITR
    from . import ODE
import numpy as np


class AdvReactUni1DEval:
    def __init__(
        self,
        fv: FVUni1D,
        model: str = "",
        params={},
        nVars: int = 1,
        source_quadrature: int = 0,
    ):
        self.fv: FVUni1D = fv

        self.ax: float = 1.0

        self.model = model.lower()
        self.params = params
        self.source_quadrature = source_quadrature

        # Per-component diffusion coefficients: always (nVars, 1) for broadcasting
        eps_raw = params.get("eps", 0.0)
        eps_arr = np.atleast_1d(np.asarray(eps_raw, dtype=float))
        if eps_arr.size == 1:
            eps_arr = np.broadcast_to(eps_arr, (nVars,)).copy()
        self.eps = eps_arr.reshape(-1, 1)  # (nVars, 1)

        # Bind model-specific source functions to avoid if-elif per call.
        # _rhs_source_raw is always the pointwise source (used by quadrature).
        # rhs_source is what the solver calls; it may be the quadrature wrapper.
        # rhs_source_jacobian is always evaluated at cell averages (unchanged).
        if self.model == "bistable":
            self._rhs_source_raw = self._rhs_source_bistable
            self.rhs_source_jacobian = self._rhs_source_jacobian_bistable
        elif self.model == "brusselator":
            self._rhs_source_raw = self._rhs_source_brusselator
            self.rhs_source_jacobian = self._rhs_source_jacobian_brusselator
        elif self.model == "premixed":
            self._rhs_source_raw = self._rhs_source_premixed
            self.rhs_source_jacobian = self._rhs_source_jacobian_premixed
        else:
            self._rhs_source_raw = self._rhs_source_none
            self.rhs_source_jacobian = self._rhs_source_jacobian_none

        if source_quadrature > 0:
            self._source_xi, self._source_w = FVUni1D.gaussPoints(source_quadrature)
            self.rhs_source = self._rhs_source_quadrature
        else:
            self.rhs_source = self._rhs_source_raw

    def _rhs_source_quadrature(self, u: np.ndarray):
        """Evaluate source via Gauss quadrature over the reconstructed polynomial.

        Reconstructs u(x) at Gauss points within each cell using
        fv.recPointValues, evaluates the pointwise source S(u(x)) at
        each quadrature point, and integrates to obtain the cell-average
        source.  More accurate than S(u_avg) for nonlinear sources.

        The source Jacobian is NOT affected by this option; it remains
        evaluated at the cell-averaged u.
        """
        # uPts: (nVars, nx, nPts)
        uPts = self.fv.recPointValues(u, self._source_xi)
        nVars, nx = self.fv.get_shape_u(u)
        nPts = len(self._source_xi)

        # Evaluate S at each quadrature point
        # Reshape to (nVars, nx*nPts), evaluate, reshape back
        uFlat = uPts.reshape(nVars, nx * nPts)
        sFlat = self._rhs_source_raw(uFlat)
        sPts = sFlat.reshape(nVars, nx, nPts)

        # Integrate: cell-average source = sum_p w_p * S(u(xi_p))
        return np.einsum("vnp,p->vn", sPts, self._source_w)

    def rhs_flow(self, u: np.ndarray):
        nVars, nx = self.fv.get_shape_u(u)

        # Reconstruct left/right face values via the FV scheme
        uL, uR = self.fv.recFaceValues(u)

        # Rusanov (local Lax-Friedrichs) flux at each face
        a = self.ax
        fFlux = 0.5 * a * (uL + uR) - 0.5 * abs(a) * (uR - uL)

        # Accumulate net flux: rhs[i] = -(F_{i+1} - F_i) / vol
        if self.fv.bcL is None:
            # Periodic: nFaces = nx, face i between cell (i-1)%nx and cell i
            rhs = -(np.roll(fFlux, -1, axis=-1) - fFlux) / self.fv.vol
        else:
            # Dirichlet: nFaces = nx+1, face 0..nx
            rhs = -(fFlux[:, 1:] - fFlux[:, :-1]) / self.fv.vol

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

    def _get_u_ref(self, u: np.ndarray) -> np.ndarray:
        """Return a model-dependent reference scale for the solution.

        This reference is used to non-dimensionalise the source activity
        indicator in compute_chi_split.  Currently returns 1.0 for all
        models, but can be overridden per-model in the future.

        Args:
            u: Solution array of shape (nVars, nx).

        Returns:
            u_ref: Array of shape (nx,) with the reference scale per cell.
        """
        nVars, nx = self.fv.get_shape_u(u)
        return np.ones(nx)

    def _fd_hessian_source(self, u: np.ndarray) -> np.ndarray:
        """Finite-difference approximation of the source Hessian dJ/du.

        Returns a tensor H where H[..., i, :] ~= dJ/du_i.  The leading
        dimensions match the Jacobian layout (scalar diagonal or dense
        matrix).  Invalid entries (NaN/Inf) are zeroed out.

        Args:
            u: Solution array of shape (nVars, nx).

        Returns:
            H: Array of shape (nVars, nVars, nx) for scalar-diagonal J,
               or (nVars, nVars, nVars, nx) for dense J.
        """
        JD = self.rhs_source_jacobian(u)
        nVars, nx = self.fv.get_shape_u(u)
        u_ref = self._get_u_ref(u)
        eps = np.sqrt(np.finfo(float).eps)

        if JD.ndim == 2:
            H = np.zeros((nVars, nVars, nx))
            for i in range(nVars):
                scale = np.maximum(np.abs(u[i]), u_ref) * eps
                du = np.zeros_like(u)
                du[i] = scale
                JD_pert = self.rhs_source_jacobian(u + du)
                dJ = (JD_pert - JD) / (scale + 1e-300)
                H[:, i, :] = dJ
        elif JD.ndim == 3:
            H = np.zeros((nVars, nVars, nVars, nx))
            for i in range(nVars):
                scale = np.maximum(np.abs(u[i]), u_ref) * eps
                du = np.zeros_like(u)
                du[i] = scale
                JD_pert = self.rhs_source_jacobian(u + du)
                dJ = (JD_pert - JD) / (scale + 1e-300)
                H[:, :, i, :] = dJ
        else:
            return np.array([])

        # Filter invalid ranges
        H = np.where(np.isfinite(H), H, 0.0)
        return H

    def compute_chi_split(
        self,
        u: np.ndarray,
        dt: float,
        threshold: float = 1.0,
        width: float = 0.5,
        transition="inv",
        smooth_steps=0,
        smooth_ratio=0.5,
    ) -> np.ndarray:
        """Compute per-cell splitting mask chi_split based on source stiffness.

        chi_split = 1 where source is stiff (tau_source << dt), meaning the
        source should be split out and solved separately.
        chi_split = 0 where source is slow (tau_source >= dt), meaning the
        source can be handled implicitly with the flow.

        The stiffness indicator combines four measures and takes the maximum:
            stiffness_ratio = max( |lambda_max(J)| * dt,
                                   |S(u)| / u_ref * dt,
                                   |J(u)*S(u)| / |S(u)| * dt,
                                   ||dJ/du||_F * |S(u)| * dt^2 )

        where |.| denotes the vector 2-norm per cell, |lambda_max(J)| is the
        spectral radius of the source Jacobian, and u_ref is a model-dependent
        reference scale returned by _get_u_ref(u).

        The mask uses a smooth sigmoid transition:
            chi_split = sigmoid((stiffness_ratio - threshold) / width)

        Args:
            u: Solution array of shape (nVars, nx).
            dt: Time step size.
            threshold: Stiffness ratio at which chi_split = 0.5.
                       Default 1.0 means transition when stiffness_ratio ~ 1.
            width: Width of the sigmoid transition. Smaller = sharper.
                   Default 0.5 for smooth blending.

        Returns:
            chi_split: Array of shape (nx,) with values in [0, 1].
        """
        JD = self.rhs_source_jacobian(u)
        nVars, nx = self.fv.get_shape_u(u)

        if JD.ndim == 2:
            jac_norm = np.linalg.norm(JD, axis=0)  # (nx,)
        elif JD.ndim == 3:
            jac_norm = np.linalg.norm(JD, axis=(0, 1))  # (nx,)
        else:
            return np.zeros(nx)

        # Combined stiffness indicator
        rhs = self._rhs_source_raw(u)
        rhs_norm = np.linalg.norm(rhs, axis=0)

        stiffness_ratio = np.empty(nx)
        stiffness_ratio[:] = -1e300

        # # 1. Linear spectral indicator (Frobenius norm of Jacobian)
        # stiffness_ratio = np.log10(jac_norm * dt)  # (nx,)

        # # 2. Source activity indicator: |S| / u_ref
        # u_ref = self._get_u_ref(u)
        # source_activity = np.log10(rhs_norm / u_ref * dt)
        # source_activity = source_activity / 0.01
        # stiffness_ratio = np.maximum(stiffness_ratio, source_activity)
        # # print(source_activity)
        # # exit()

        # 3. Source curvature indicator: |J @ S| / |S|
        mask = rhs_norm > 1e-300
        if np.any(mask):
            if JD.ndim == 2:
                J_dot_S = JD * rhs
            elif JD.ndim == 3:
                J_dot_S = np.einsum('ijv,jv->iv', JD, rhs)
            J_dot_S_norm = np.linalg.norm(J_dot_S, axis=0)
            source_curvature = np.zeros(nx)
            source_curvature[mask] = np.log10(J_dot_S_norm[mask] / rhs_norm[mask] * dt)
            stiffness_ratio = np.maximum(stiffness_ratio, source_curvature)

        # # 4. Hessian nonlinearity indicator: ||dJ/du||_F * |S| * dt^2
        # H = self._fd_hessian_source(u)
        # if H.size > 0:
        #     hessian_norm = np.sqrt(np.sum(H ** 2, axis=tuple(range(H.ndim - 1))))
        #     hessian_indicator = hessian_norm * rhs_norm * dt ** 2
        #     # Filter invalid ranges
        #     hessian_indicator = np.where(np.isfinite(hessian_indicator),
        #                                  hessian_indicator, 0.0)
        #     stiffness_ratio = np.maximum(stiffness_ratio, hessian_indicator)

        arg = (stiffness_ratio - threshold) / width

        if transition == "inv":
            chi_split = np.clip(arg, 0.0, 1e300)
            chi_split = chi_split / (1.0 + chi_split)
        elif transition == "sigmoid":
            arg = np.clip(arg, -50, 50)
            chi_split = 1.0 / (1.0 + np.exp(-arg))
        elif transition == "linear":
            chi_split = np.clip(arg, 0.0, 1.0)
        else:
            raise ValueError(f"Unknown transition type: {transition}")

        for i in range(smooth_steps):
            chi_l = np.roll(chi_split, +1, 0) * smooth_ratio
            chi_r = np.roll(chi_split, -1, 0) * smooth_ratio
            if self.fv.bcL is not None:
                chi_l[0] = 0
                chi_r[0] = 0
            if self.fv.bcR is not None:
                chi_l[-1] = 0
                chi_r[-1] = 0
            chi_split = np.maximum.reduce((chi_l, chi_r, chi_split))

        return chi_split

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
    def __init__(
        self,
        eval: AdvReactUni1DEval,
        ode: ODE.ImplicitOdeIntegrator,
        N_react: int = 4,
        chi_split_threshold: float = None,
        chi_split_width: float = None,
    ):
        self.eval = eval
        self.ode = ode
        self.N_react = N_react

        if chi_split_threshold is None:
            chi_split_threshold = 10
        if chi_split_width is None:
            chi_split_width = 0.5

        # Masked Strang parameters for chi_split computation
        self.chi_split_threshold = chi_split_threshold
        self.chi_split_width = chi_split_width

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
        chi_split: np.ndarray = None,
    ):
        mode_frhs = mode
        mode_fsolve = mode
        if isinstance(self.ode, ODE.DITRExp):
            return self.ode.step(
                dt,
                u,
                (
                    self.FrhsDITRExp(self.eval, mode=mode_frhs, chi_split=chi_split)
                    if use_exp
                    else self.Frhs(self.eval, mode=mode_frhs, chi_split=chi_split)
                ),
                self.FsolveDITR(
                    self.eval, mode=mode_fsolve, chi_split=chi_split, **solve_opts
                ),
                fForce=uForce,
            )
        return self.ode.step(
            dt,
            u,
            self.Frhs(self.eval, mode=mode_frhs, chi_split=chi_split),
            self.Fsolve(self.eval, mode=mode_fsolve, chi_split=chi_split, **solve_opts),
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
                N_react = self.N_react
                for i_react in range(N_react):
                    u = self.step(
                        dtC * 0.5 / N_react, u, mode="source", solve_opts=solve_opts
                    )
                u = self.step(dtC, u, mode="flow", solve_opts=solve_opts)
                for i_react in range(N_react):
                    u = self.step(
                        dtC * 0.5 / N_react, u, mode="source", solve_opts=solve_opts
                    )
            elif mode == "embed":
                cs = self.ode.get_cs()
                N_react = self.N_react
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
            elif mode == "masked_strang":
                # Masked Strang: source-flow-source with per-cell adaptive mask
                # chi_split = 1 where stiff (split out), chi_split = 0 where slow (implicit)
                N_react = self.N_react
                chi_split = self.eval.compute_chi_split(
                    u,
                    dtC,
                    threshold=self.chi_split_threshold,
                    width=self.chi_split_width,
                )

                # Half-step: solve chi_split * S(u) via sub-stepping
                for i_react in range(N_react):
                    u = self.step(
                        dtC * 0.5 / N_react,
                        u,
                        mode="masked_split",
                        solve_opts=solve_opts,
                        chi_split=chi_split,
                    )

                # Full-step: implicit solve of F(u) + (1 - chi_split) * S(u)
                u = self.step(
                    dtC,
                    u,
                    mode="masked_implicit",
                    solve_opts=solve_opts,
                    chi_split=chi_split,
                )

                # Half-step: solve chi_split * S(u) via sub-stepping
                for i_react in range(N_react):
                    u = self.step(
                        dtC * 0.5 / N_react,
                        u,
                        mode="masked_split",
                        solve_opts=solve_opts,
                        chi_split=chi_split,
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
