"""Common utilities for test scripts.

Provides FV class selection, evaluator factory, lazy solver construction
with standard DITR variants, method registry building, run loop, error
norm printing, spatial profile plotting, and probe time series plotting.
"""

import numpy as np
import matplotlib.pyplot as plt
from Solver.AdvReactUni import AdvReactUni1DSolver, AdvReactUni1DEval
from Solver.FVUni2nd import FVUni2nd1D
from Solver.FVUniWENO5Z import FVUniWENO5Z1D
from Solver.ODE import ESDIRK, DITRExp
import PlotEnv


# ── FV class selection ─────────────────────────────────────────────

FV_CLASSES = {
    "muscl2": FVUni2nd1D,
    "weno5z": FVUniWENO5Z1D,
}


def make_fv(rec_scheme: str, nx: int):
    return FV_CLASSES[rec_scheme](nx=nx)


# ── Evaluator factory ──────────────────────────────────────────────

def make_ev(fv, model: str = "", params: dict = {}, nVars: int = 1,
            source_quadrature: int = 0, ax: float = 1.0):
    """Create an AdvReactUni1DEval with the given parameters."""
    ev = AdvReactUni1DEval(fv=fv, model=model, params=params,
                           nVars=nVars, source_quadrature=source_quadrature)
    ev.ax = ax
    return ev


# ── Lazy solver set ────────────────────────────────────────────────

# ODE constructor specs: (name_key, constructor_callable)
_ODE_SPECS = [
    ("esdirk4",    lambda: ESDIRK("ESDIRK4")),
    ("esdirk3",    lambda: ESDIRK("ESDIRK3")),
    ("DITR U2R2",  lambda: DITRExp()),
    ("DITR U2R1",  lambda: DITRExp(c2=1./3, method="U2R1")),
]

DITR_KEYS = [name for name, _ in _ODE_SPECS if name.startswith("DITR")]


class SolverSet:
    """Lazily-created collection of solvers sharing one evaluator.

    Solvers are instantiated on first access via __getitem__.
    Probe locations are applied to each solver as it is created.
    """

    def __init__(self, ev: AdvReactUni1DEval, probe_locations: list = []):
        self._ev = ev
        self._probe_locations = list(probe_locations)
        self._solvers = {}

    def __getitem__(self, key: str) -> AdvReactUni1DSolver:
        if key not in self._solvers:
            ode_ctor = None
            for name, ctor in _ODE_SPECS:
                if name == key:
                    ode_ctor = ctor
                    break
            if ode_ctor is None:
                raise KeyError(f"Unknown solver key: {key!r}")
            s = AdvReactUni1DSolver(eval=self._ev, ode=ode_ctor())
            if self._probe_locations:
                s.set_probes(self._probe_locations)
            self._solvers[key] = s
        return self._solvers[key]

    def keys_created(self):
        return list(self._solvers.keys())


# ── Method registry building ───────────────────────────────────────

def _add_runners_for_solver_set(runners, ss: SolverSet, suffix: str,
                                dt, dtRef, u0, tEnd,
                                CFL_ref, CFL_coarse, rel_tol,
                                max_iter_exp, rp):
    """Populate runners dict from one SolverSet.

    Method names are base names + suffix (e.g. "" or " p-source").
    """
    runners["ref" + suffix] = (
        lambda _ss=ss: _ss["esdirk4"].stepInterval(
            dtRef, u0, 0.0, tEnd, mode="full",
            solve_opts={"CFL": CFL_ref}, record_probes=rp,
        ),
        lambda _ss=ss: _ss["esdirk4"],
    )
    runners["ESDIRK3" + suffix] = (
        lambda _ss=ss: _ss["esdirk3"].stepInterval(
            dt, u0, 0.0, tEnd, mode="full",
            solve_opts={"rel_tol": rel_tol, "CFL": CFL_coarse},
            record_probes=rp,
        ),
        lambda _ss=ss: _ss["esdirk3"],
    )
    runners["Strang ESDIRK3" + suffix] = (
        lambda _ss=ss: _ss["esdirk3"].stepInterval(
            dt, u0, 0.0, tEnd, mode="strang",
            solve_opts={"CFL": CFL_coarse}, record_probes=rp,
        ),
        lambda _ss=ss: _ss["esdirk3"],
    )
    runners["Embed ESDIRK3" + suffix] = (
        lambda _ss=ss: _ss["esdirk3"].stepInterval(
            dt, u0, 0.0, tEnd, mode="embed",
            solve_opts={"CFL": CFL_coarse}, record_probes=rp,
        ),
        lambda _ss=ss: _ss["esdirk3"],
    )

    for dk in DITR_KEYS:
        runners[dk + suffix] = (
            lambda _ss=ss, _dk=dk: _ss[_dk].stepInterval(
                dt, u0, 0.0, tEnd, mode="full",
                solve_opts={"rel_tol": rel_tol, "CFL": CFL_coarse},
                use_exp=False, record_probes=rp,
            ),
            lambda _ss=ss, _dk=dk: _ss[_dk],
        )
        runners["Strang " + dk + suffix] = (
            lambda _ss=ss, _dk=dk: _ss[_dk].stepInterval(
                dt, u0, 0.0, tEnd, mode="strang",
                solve_opts={"rel_tol": rel_tol, "CFL": CFL_coarse},
                record_probes=rp,
            ),
            lambda _ss=ss, _dk=dk: _ss[_dk],
        )
        runners["Exp " + dk + suffix] = (
            lambda _ss=ss, _dk=dk: _ss[_dk].stepInterval(
                dt, u0, 0.0, tEnd, mode="full",
                solve_opts={"rel_tol": rel_tol, "CFL": CFL_coarse,
                            "max_iter": max_iter_exp},
                use_exp=True, record_probes=rp,
            ),
            lambda _ss=ss, _dk=dk: _ss[_dk],
        )
        runners["Embed " + dk + suffix] = (
            lambda _ss=ss, _dk=dk: _ss[_dk].stepInterval(
                dt, u0, 0.0, tEnd, mode="embed",
                solve_opts={"rel_tol": rel_tol, "CFL": CFL_coarse},
                use_exp=False, record_probes=rp,
            ),
            lambda _ss=ss, _dk=dk: _ss[_dk],
        )


def build_method_runners(solver_sets: dict, dt, dtRef, u0, tEnd,
                         CFL_ref, CFL_coarse, rel_tol,
                         max_iter_exp=50, record_probes=True,
                         ref_suffix=""):
    """Build the method runner registry from one or more SolverSets.

    Args:
        solver_sets: dict mapping suffix -> SolverSet.
            Use {"": solver_set_base, " p-source": solver_set_psrc}
            or just {"": solver_set_base} for no p-source variants.
        dt, dtRef, u0, tEnd, ...: time-stepping and solver parameters.
        ref_suffix: which solver set suffix to use for the "ref" method.
            Default "" uses the base evaluator.  Set to " p-source" to
            make the reference use source quadrature.

    Returns:
        dict mapping method name -> (runner_callable, solver_getter).
        solver_getter is a callable returning the solver instance
        (lazy -- the solver is created on first call).
    """
    rp = record_probes
    runners = {}
    for suffix, ss in solver_sets.items():
        _add_runners_for_solver_set(
            runners, ss, suffix,
            dt, dtRef, u0, tEnd,
            CFL_ref, CFL_coarse, rel_tol, max_iter_exp, rp,
        )

    # Override "ref" to use the specified solver set
    if ref_suffix in solver_sets:
        ref_ss = solver_sets[ref_suffix]
        runners["ref"] = (
            lambda _ss=ref_ss: _ss["esdirk4"].stepInterval(
                dtRef, u0, 0.0, tEnd, mode="full",
                solve_opts={"CFL": CFL_ref}, record_probes=rp,
            ),
            lambda _ss=ref_ss: _ss["esdirk4"],
        )

    return runners


# ── Run loop ───────────────────────────────────────────────────────

def run_methods(method_runners: dict, enabled_methods: list):
    """Run enabled methods, return (results, probe_results) dicts.

    method_runners values are (runner, solver_getter) where solver_getter
    is a callable returning the solver instance.
    """
    results = {}
    probe_results = {}

    for name in enabled_methods:
        entry = method_runners.get(name)
        if entry is None:
            print(f"WARNING: unknown method '{name}', skipping")
            continue
        runner, solver_getter = entry
        solver_inst = solver_getter()
        solver_inst.clear_probes()
        print("=" * 60)
        print(name)
        print("=" * 60)
        try:
            sol = runner()
            results[name] = sol
            probe_results[name] = solver_inst.get_probe_data()
            print(f"  >> {name} completed, uNorm = {np.linalg.norm(sol):.6e}")
        except Exception as e:
            print(f"  >> {name} FAILED: {e}")
            import traceback
            traceback.print_exc()
            results[name] = None
            probe_results[name] = None

    return results, probe_results


# ── Error norm printing ────────────────────────────────────────────

def print_errors(results: dict, enabled_methods: list, header: str = ""):
    """Print L2 errors vs the 'ref' solution."""
    u1_ref = results.get("ref")
    if u1_ref is None:
        return
    print("\n" + "=" * 60)
    if header:
        print(f"L2 errors vs reference  ({header}):")
    else:
        print("L2 errors vs reference:")
    for name in enabled_methods:
        if name == "ref":
            continue
        sol = results.get(name)
        if sol is not None:
            err = np.linalg.norm(sol - u1_ref) / np.linalg.norm(u1_ref)
            print(f"  {name:30s}: {err:.6e}")
        else:
            print(f"  {name:30s}: FAILED")


# ── Spatial profile plotting ───────────────────────────────────────

def plot_profiles(fv, results, enabled_methods, var_names, title_base,
                  tag, pic_dir, fmt_fig, rec_scheme,
                  plotEnv=None, fig_start=201):
    """Plot spatial profiles for each variable."""
    if plotEnv is None:
        plotEnv = PlotEnv.PlotEnv(dpi=180, markEvery=max(1, fv.nx // 20))

    for iVar, vname in enumerate(var_names):
        fig = plotEnv.figure(fig_start + iVar, figsize=(6, 4))
        for i, name in enumerate(enabled_methods):
            sol = results.get(name)
            if sol is not None:
                plotEnv.plot(fv.xcs, sol[iVar], plotIndex=i, label=name)
        plt.legend(fontsize=7)
        plt.title(title_base + f" {vname}"
                  + (" WENO5" if rec_scheme == "weno5z" else ""))
        plt.xlabel("x")
        plt.ylabel(vname)
        plt.savefig(pic_dir / f"{tag}_{vname}.{fmt_fig}",
                    dpi=180, bbox_inches="tight")
        plt.show()


# ── Probe time series plotting ─────────────────────────────────────

def plot_probes(probe_results, probe_locations, enabled_methods,
                var_names, tag, pic_dir, fmt_fig, rec_scheme,
                plotEnv=None, fig_start=300):
    """Plot probe time series for each variable at each probe location."""
    if plotEnv is None:
        plotEnv = PlotEnv.PlotEnv(dpi=180, markEvery=0)

    for x_probe in probe_locations:
        for iVar, vname in enumerate(var_names):
            fignum = fig_start + iVar * 100 + int(x_probe * 100)
            fig = plotEnv.figure(fignum, figsize=(6, 4))
            for i, name in enumerate(enabled_methods):
                pdata = probe_results.get(name)
                if pdata is not None and x_probe in pdata:
                    t_arr = np.array(pdata[x_probe]["t"])
                    u_arr = np.array(pdata[x_probe]["u"])
                    if u_arr.ndim < 2:
                        continue
                    plotEnv.plot(t_arr, u_arr[:, iVar],
                                plotIndex=i, label=name)
            plt.legend(fontsize=7)
            plt.title(f"{vname} at x={x_probe:.2f}"
                      + (" WENO5" if rec_scheme == "weno5z" else ""))
            plt.xlabel("t")
            plt.ylabel(vname)
            plt.savefig(
                pic_dir / f"{tag}_{vname}_x{x_probe}.{fmt_fig}",
                dpi=180, bbox_inches="tight",
            )
            plt.show()
