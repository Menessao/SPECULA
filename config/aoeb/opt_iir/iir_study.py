"""
IIR Wiener study: all comparisons and figures in one script.

  python iir_study.py compare   # table: LI vs IIR vs bound (CONAN) + optical-gain test
  python iir_study.py order     # residual_vs_radial_order.png             (CONAN)
  python iir_study.py tf        # psd_rtf_ntf.png                          (CONAN)
  python iir_study.py sweep     # sr_vs_noise_aliasing.png,
                                # residual_total_vs_noise_aliasing.png     (VK)
  python iir_study.py all       # everything (default)

Two configurations (edit below):
  CONAN: Conan-like PSD per radial order, residuals in rad^2.
  VK:    multi-layer frozen-flow von Karman Zernike PSDs, residuals in rad^2 at VK.lam.
         Aliasing on radial order n: variance alpha*(n+1)/(n_max+1) times that of one
         radial-order n_max+1 mode, same temporal shape. Same per-frame noise for every
         mode. sigma^2 = sum_n (n+1) sigma_n^2 (PCHIP in log-log between the design
         orders), SR = exp(-sigma^2), no fitting error.
Vibrations: {Noll index j: [(f0 [Hz], variance [rad^2], damping zeta), ...]}.
  A vibrated mode j gets the turbulence PSD of its radial order (the average over that
  order: no wind-direction dependence) plus its lines, and is designed separately; all
  other modes are turbulence only. Vibrations are not aliased.
Every controller is optimised mode by mode. Optical gain (order, tf): controllers are
designed for the assumed gain g_hat and evaluated with the true gain g; the Wiener
bound is the perfect-knowledge reference (designed and evaluated with g).
Results are cached in files named after a hash of the configuration and of the code
used to compute them, so a cached result is never stale.
"""
import os, json, argparse
from types import SimpleNamespace
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import wiener_ao_iir, ao_tools
from wiener_ao_iir import ModeSpectra, wiener_iir
from ao_tools import (ZernikePSD, r0_from_seeing, conan_psd, with_vibrations,
                      noll_radial_order, closed_loop_residual, optimal_integrator,
                      optimal_leaky_integrator, transfer_functions, cache_file,
                      total_variance)

# ================================================================== configuration
CONAN = SimpleNamespace(
    fs=1000.0, D=8.0, V=15.0, r0=0.15,
    orders=(2, 3, 4),                     # IIR orders shown (bound always added)
    alias_n=40, alias_frac=1.0,           # aliasing PSD = alias_frac x PSD of order alias_n
    radial=[1, 2, 3, 4, 6, 8, 10, 13, 16, 20, 25, 30],
    noises=(1e-4, 1e-2),                  # per-frame noise variance [rad^2]
    delays=(1.5, 2.5),                    # total delay [frames], Males convention (T + tau)
    vibrations={},                        # e.g. {2: [(20.0, 1e-2, 0.005)], 3: [(55.0, 3e-3, 0.002)]}
)
VK = SimpleNamespace(
    D=8.0, L0=25.0, seeing=1.0, lam=750e-9,             # seeing at 500 nm, residuals at lam
    layers=[(0.60, 7.0), (0.25, 20.0), (0.15, 35.0)],   # (Cn2 fraction, wind speed [m/s])
    fs=1000.0, delay=2.0, M=2**14,                      # total delay [frames]
    n_max=30, n_design=[1, 2, 3, 4, 6, 10, 16, 23, 30],
    noises=np.geomspace(1e-5, 1e-1, 7).tolist(),       # rad^2/frame/mode @ lam
    alphas=np.geomspace(1e-2, 1e1, 7).tolist(),        # aliasing level
    noise_fixed=1e-3, alpha_fixed=1.0,                  # value of the other sweep
    vibrations={},                                      # variances in rad^2 @ lam
)
C_LABELS = (["Leaky integrator"] + [f"IIR Wiener, N={N}" for N in CONAN.orders]
            + ["Wiener bound (N = ∞)"])
C_COLORS = ["k", "tab:orange", "tab:green", "tab:blue", "tab:red"]
C_MARKERS = ["o", "s", "D", "x", "^"]
V_LABELS = ["Integrator (leak = 1)", "Leaky integrator", "IIR Wiener, N=2",
            "IIR Wiener, N=3", "Wiener bound (N = ∞)"]
V_COLORS = ["0.5", "k", "tab:orange", "tab:green", "tab:red"]
V_MARKERS = ["v", "o", "s", "D", "^"]


def vib_modes(vibrations, n_max=None):
    """[(n, j)] of the vibrated modes, sorted by Noll index."""
    modes = [(noll_radial_order(j), j) for j in sorted(vibrations)]
    if n_max is not None and any(n > n_max for n, _ in modes):
        raise ValueError(f"vibrated mode beyond radial order {n_max}")
    return modes


def mode_label(n, j):
    return f"n={n}" if j is None else f"j={j}*"


# ======================================================== CONAN: design / compute
def conan_spec(n, noise, total, j=None, M=None):
    """Spectra of mode j (vibrations) or of the average mode of radial order n.
    Default grid: fine (2^16) for low cut-offs or vibrations, 2^15 otherwise."""
    c = CONAN
    S = with_vibrations(conan_psd(n, c.D, c.r0, c.V), c.vibrations.get(j, ()))
    S_al = conan_psd(c.alias_n, c.D, c.r0, c.V)
    M = M or (2**16 if n < 4 or j is not None else 2**15)
    return ModeSpectra(S, c.fs, total - 1.0, lambda f: c.alias_frac * S_al(f),
                       noise=noise, M=M)


def conan_design(spec, g=1.0, g_hat=1.0):
    """Controllers [(structure, b, a, C_grid, residual, g, g_hat), ...]: leaky
    integrator, IIR orders CONAN.orders (designed for g_hat, evaluated with g), bound."""
    (G, l), _ = optimal_leaky_integrator(spec, g_hat, g_hat)
    info = wiener_iir(spec, max(CONAN.orders), g=g_hat)[2]
    info_g = info if g == g_hat else wiener_iir(spec, max(CONAN.orders), g=g)[2]
    ctrls = [("residual", [G], [1.0, -l], None,
              closed_loop_residual(spec, [G], [1.0, -l], "residual", g, g_hat), g, g_hat)]
    for N in CONAN.orders:
        b, a, r = info["lower"][N - 1]
        ev = closed_loop_residual(spec, b, a, "pol", g, g_hat)
        assert g != g_hat or abs(ev - r) < 1e-6 * ev        # design vs closed-loop eval
        ctrls.append(("pol", b, a, None, ev, g, g_hat))
    ctrls.append(("pol", None, None, info_g["F_bound"], info_g["bound"], g, g))
    return ctrls


def conan_compute(g=1.0, g_hat=1.0):
    """res[delay, noise, mode, controller], modes = radial orders then vibrated modes."""
    c = CONAN
    modes = [(n, None) for n in c.radial] + vib_modes(c.vibrations)
    cfg = dict(vars(c), g=g, g_hat=g_hat)
    cache = cache_file("results_iir", cfg, wiener_ao_iir, ao_tools, conan_spec,
                       conan_design, ext="npz")
    shape = (len(c.delays), len(c.noises), len(modes), len(c.orders) + 2)
    res = np.load(cache)["res"] if os.path.exists(cache) else np.full(shape, np.nan)
    for i, total in enumerate(c.delays):
        for k, noise in enumerate(c.noises):
            for m, (n, j) in enumerate(modes):
                if np.all(np.isfinite(res[i, k, m])):
                    continue
                spec = conan_spec(n, noise, total, j)
                res[i, k, m] = [x[4] for x in conan_design(spec, g, g_hat)]
                np.savez(cache, res=res)
                print(total, noise, mode_label(n, j), flush=True)
    return res


def conan_compare():
    c = CONAN
    print(f"fs={c.fs:.0f} Hz, D={c.D} m, V={c.V} m/s, D/r0={c.D/c.r0:.0f}; residuals in rad^2"
          + (f"; * = with vibrations" if c.vibrations else "") + "\n")
    modes = [(n, None) for n in (1, 4, 10, 20)] + vib_modes(c.vibrations)
    for total in c.delays:
        print(f"===== total delay {total} frames =====")
        print(f"{'mode':>6} {'noise':>6} {'LI':>9} "
              + " ".join(f"{'IIR' + str(N):>9}" for N in c.orders) + f" {'bound':>9}")
        for noise in c.noises:
            for n, j in modes:
                row = [x[4] for x in conan_design(conan_spec(n, noise, total, j, M=2**16))]
                print(f"{mode_label(n, j):>6} {noise:6.0e} " + " ".join(f"{x:9.3e}" for x in row))
        print()

    n, noise, total, g = 10, 1e-4, 2.5, 0.5
    print(f"===== optical gain: n={n}, noise {noise:.0e}, total delay {total}, true g={g} =====")
    spec = conan_spec(n, noise, total, M=2**16)
    ok = wiener_iir(spec, max(c.orders), g=g)[2]["lower"]      # designed knowing g
    no = wiener_iir(spec, max(c.orders), g=1.0)[2]["lower"]    # designed assuming g = 1
    for N in c.orders:
        (b, a, _), (b1, a1, _) = ok[N - 1], no[N - 1]
        print(f"IIR{N}: compensated {closed_loop_residual(spec, b, a, 'pol', g, g):.3e} | "
              f"uncompensated {closed_loop_residual(spec, b1, a1, 'pol', g, 1.0):.3e}")
    print(f"LI   : {optimal_leaky_integrator(spec, g, g)[1]:.3e} (independent of g_hat)")


# =========================================================== VK: design / compute
SWEEPS = {"noise": (VK.noises, lambda x: (x, VK.alpha_fixed), VK.noise_fixed),
          "alias": (VK.alphas, lambda x: (VK.noise_fixed, x), VK.alpha_fixed)}


def vk_modes():
    return [(n, None) for n in VK.n_design] + vib_modes(VK.vibrations, VK.n_max)


def vk_design(turb, n, j, noise, alpha, init=None):
    """Residuals [integrator, leaky, IIR2, IIR3, bound] for one mode; also IIR state."""
    S_al = turb.psd(VK.n_max + 1)
    k_al = alpha * (n + 1) / (VK.n_max + 1)
    spec = ModeSpectra(with_vibrations(turb.psd(n), VK.vibrations.get(j, ())), VK.fs,
                       VK.delay - 1.0, lambda f: k_al * S_al(f), noise=noise, M=VK.M)
    _, ji = optimal_integrator(spec)
    _, jl = optimal_leaky_integrator(spec)
    info = wiener_iir(spec, 3, init=init)[2]
    return [ji, jl, info["lower"][1][2], info["lower"][2][2], info["bound"]], info["v"]


def vk_key(n, j, noise, alpha):
    return f"{n if j is None else f'j{j}'}|{noise:.6e}|{alpha:.6e}"


def vk_compute():
    """{'noise': array[len(noises), len(modes), 5], 'alias': ...}, modes = vk_modes()."""
    fn = cache_file("sr_cache", vars(VK), wiener_ao_iir, ao_tools, vk_design)
    done = json.load(open(fn)) if os.path.exists(fn) else {}
    turb = ZernikePSD(VK.D, r0_from_seeing(VK.seeing), VK.L0, VK.layers, lam=VK.lam)
    for n, j in vk_modes():
        for name, (xs, pt, x0) in SWEEPS.items():
            # continuation from the shared point outwards along each sweep
            i0 = int(np.argmin(np.abs(np.log(xs) - np.log(x0))))
            state = {}
            for i in [i0, *range(i0 + 1, len(xs)), *range(i0 - 1, -1, -1)]:
                k = vk_key(n, j, *pt(xs[i]))
                if k in done:
                    state[i] = {int(N): np.array(v) for N, v in done[k]["v"].items()}
                    continue
                init = state.get(i - 1 if i > i0 else i + 1, state.get(i0))
                res, state[i] = vk_design(turb, n, j, *pt(xs[i]), init=init)
                done[k] = dict(res=res, v={N: v.tolist() for N, v in state[i].items()})
                with open(fn, "w") as fh:
                    json.dump(done, fh)
                print(mode_label(n, j), name, f"{xs[i]:.1e}", np.round(res, 6).tolist(),
                      flush=True)
    return {name: np.array([[done[vk_key(n, j, *pt(x))]["res"] for n, j in vk_modes()]
                            for x in xs]) for name, (xs, pt, _) in SWEEPS.items()}


def vk_total(res_modes):
    """res_modes[len(modes), 5] -> total residual per controller."""
    nd = len(VK.n_design)
    return total_variance(res_modes[:nd], VK.n_design, VK.n_max, res_modes[nd:],
                          [n for n, _ in vk_modes()[nd:]])


# ======================================================================= plotting
def vibration_text(vib):
    """Short description of {j: lines} for figure titles ('' if none)."""
    return "; ".join(f"j={j}: " + ", ".join(f"{f0:g} Hz/{v:.1e} rad²/ζ={z:g}" for f0, v, z in L)
                     for j, L in sorted(vib.items()) if L)


def _finish(fig, title, fname, fontsize=10):
    fig.suptitle(title, fontsize=fontsize)
    fig.tight_layout()
    fig.savefig(fname, dpi=150)
    plt.close(fig)
    return fname


def conan_title(g=1.0, g_hat=1.0, extra=""):
    c, vib = CONAN, vibration_text(CONAN.vibrations)
    return (f"fs = {c.fs:g} Hz, D = {c.D:g} m, V = {c.V:g} m/s, D/r0 = {c.D / c.r0:.0f}, "
            f"Conan-like PSDs, aliasing = {c.alias_frac:g} × radial order {c.alias_n}"
            + (f"\nvibrations {vib}" if vib else "") + extra
            + ("" if g == g_hat == 1 else f", true/assumed optical gain {g:g}/{g_hat:g}"))


def plot_order(res, g=1.0, g_hat=1.0, fname="residual_vs_radial_order.png"):
    """Residual vs radial order; vibrated modes as filled markers at their order."""
    c, styles = CONAN, ["-", "--", "-.", ":"]
    nr, vm = len(c.radial), vib_modes(c.vibrations)
    xv = [n * 1.05 ** (1 + [m for m, _ in vm[:i]].count(n)) for i, (n, _) in enumerate(vm)]
    fig, axes = plt.subplots(1, len(c.delays), figsize=(6 * len(c.delays), 5), sharey=True)
    axes = np.atleast_1d(axes)
    for i, (ax, total) in enumerate(zip(axes, c.delays)):
        for k in range(len(c.noises)):
            for m in range(len(C_LABELS)):
                ax.plot(c.radial, res[i, k, :nr, m], styles[k], color=C_COLORS[m],
                        marker=C_MARKERS[m], ms=4, lw=1.3, mfc="none")
                ax.plot(xv, res[i, k, nr:, m], ls="none", color=C_COLORS[m],
                        marker=C_MARKERS[m], ms=6)
        for x, (_, j), r in zip(xv, vm, res[i, 0, nr:].min(1)):   # label vibrated modes
            ax.annotate(f"j={j}", (x, r), xytext=(0, -12), textcoords="offset points",
                        ha="center", fontsize=7)
        ax.set_xscale("log"); ax.set_yscale("log")
        ax.set_xticks(c.radial); ax.set_xticklabels(c.radial); ax.minorticks_off()
        ax.set(xlabel="Radial order n", title=f"Total delay {total} frames")
        ax.grid(True, which="both", alpha=0.3)
    axes[0].set_ylabel("Residual variance per mode [rad²]")
    h = [Line2D([], [], color=col, lw=1.5, marker=mk, mfc="none")
         for col, mk in zip(C_COLORS, C_MARKERS)]
    h += [Line2D([], [], color="gray", ls=s) for s in styles[:len(c.noises)]]
    lab = C_LABELS + [f"noise {x:.0e} rad²/frame" for x in c.noises]
    if vm:
        h.append(Line2D([], [], color="gray", ls="none", marker="o"))
        lab.append("filled: modes with vibrations (" + ", ".join(f"j={j}" for _, j in vm) + ")")
    axes[-1].legend(h, lab, fontsize=9)
    return _finish(fig, conan_title(g, g_hat), fname)


def plot_tf(n=2, j=None, noise=1e-2, total=1.5, g=1.0, g_hat=1.0, fname="psd_rtf_ntf.png",
            n_pts=3000):
    """Left: disturbance PSD and |RTF|^2. Right: noise and (folded, WFS-averaged)
    aliasing PSDs and |NTF|^2 (aliasing propagates as H*NTF, ~NTF in band).
    Mode j (Noll, with its vibrations) if given, else the average mode of order n."""
    c = CONAN
    n = n if j is None else noll_radial_order(j)
    lines = c.vibrations.get(j, ())
    spec = conan_spec(n, noise, total, j)
    ctrls = conan_design(spec, g, g_hat)
    nu = np.fft.fftfreq(spec.M, 1 / c.fs)
    idx = np.nonzero(nu > 0)[0]
    idx = np.unique(idx[np.round(np.geomspace(1, len(idx), n_pts)).astype(int) - 1])
    f = nu[idx]
    fd = np.unique(np.r_[f, *[f0 * (1 + z * np.linspace(-20, 20, 201)) for f0, _, z in lines]])
    fig, (axL, axR) = plt.subplots(1, 2, figsize=(13, 5.2))

    def area(ax, x, y, color, label):                    # PSDs as shaded areas
        ax.fill_between(x, y, 1e-300, color=color, alpha=0.18, lw=0, label=label)
        ax.loglog(x, y, color=color, lw=0.8, alpha=0.6)
        return y

    S = with_vibrations(conan_psd(n, c.D, c.r0, c.V), lines)
    St = area(axL, fd, S(fd), "tab:gray", f"Disturbance PSD ({mode_label(n, j)})")
    axL.set_ylim(St.min() / 10, St.max() * 10)
    axL.set(xlabel="Frequency [Hz]", ylabel="PSD [rad²/Hz]", title="Disturbance PSD and |RTF|²")
    Sn = area(axR, f, 2 * spec.Pnn[idx] / c.fs, "tab:gray", "Noise PSD")
    Sa = area(axR, f, 2 * spec.Paa[idx] / c.fs, "tab:purple",
              "Aliasing PSD (folded, WFS-averaged)")
    axR.set_ylim(min(Sn.min(), Sa.min()) / 10, max(Sn.max(), Sa.max()) * 10)
    axR.set(xlabel="Frequency [Hz]", ylabel="PSD [rad²/Hz]",
            title="Noise & aliasing PSD and |NTF|²")
    axL2, axR2 = axL.twinx(), axR.twinx()
    for m, (st, b, a, C, r, gg, gh) in enumerate(ctrls):
        E, Nt, _ = transfer_functions(b, a, f, c.fs, spec.delay, st, gg, gh,
                                      C=None if C is None else C[idx])
        kw = dict(color=C_COLORS[m], lw=1.3, ls=":" if m == len(ctrls) - 1 else "-")
        axL2.loglog(f, np.abs(E) ** 2, label=f"{C_LABELS[m]}  (σ² = {r:.3e})", **kw)
        axR2.loglog(f, np.abs(Nt) ** 2, label=C_LABELS[m], **kw)
    axL2.set_ylabel("|RTF|²"); axR2.set_ylabel("|NTF|²")
    for ax, ax2, loc in ((axL, axL2, "lower right"), (axR, axR2, "lower left")):
        ax.grid(True, which="both", alpha=0.3)
        h1, l1 = ax.get_legend_handles_labels(); h2, l2 = ax2.get_legend_handles_labels()
        ax2.legend(h1 + h2, l1 + l2, fontsize=8, loc=loc)
    title = conan_title(g, g_hat, f"\n{'radial order ' + str(n) if j is None else 'mode j = ' + str(j)}"
                                  f", noise {noise:.0e} rad²/frame, total delay {total} frames")
    return _finish(fig, title, fname)


def plot_sweep(out, kind="sr", fname=None):
    """kind 'sr': Strehl ratio; 'residual': total residual variance (log scale)."""
    v, sr = VK, kind == "sr"
    tot = {k: np.array([vk_total(r) for r in x]) for k, x in out.items()}
    y = {k: np.exp(-t) if sr else t for k, t in tot.items()}
    what, lam = ("SR" if sr else "Residual"), f"{v.lam * 1e9:.0f} nm"
    panels = [(v.noises, y["noise"], f"Noise variance per mode [rad²/frame @ {lam}]",
               f"{what} vs noise (aliasing level α = {v.alpha_fixed:g})"),
              (v.alphas, y["alias"], "Aliasing level α",
               f"{what} vs aliasing (noise {v.noise_fixed:.0e} rad²/frame)")]
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharey=True)
    for ax, (x, yy, xl, t) in zip(axes, panels):
        for m in range(len(V_LABELS)):
            ax.plot(x, yy[:, m], color=V_COLORS[m], marker=V_MARKERS[m], mfc="none",
                    ls=":" if m == len(V_LABELS) - 1 else "-", lw=1.4, label=V_LABELS[m])
        ax.set_xscale("log")
        ax.set_yscale("linear" if sr else "log")
        ax.set(xlabel=xl, title=t)
        ax.grid(True, which="both", alpha=0.3)
    axes[0].set_ylabel(f"Strehl ratio @ {lam} (Maréchal, no fitting error)" if sr
                       else f"Total residual variance [rad² @ {lam}]")
    axes[1].legend(fontsize=9)
    lay = ", ".join(f"{w:.0%}@{s:g} m/s" for w, s in v.layers)
    n_modes = sum(n + 1 for n in range(1, v.n_max + 1))
    vib = vibration_text(v.vibrations)
    title = (f"D = {v.D:g} m, seeing {v.seeing:g}\" @ 500 nm, L0 = {v.L0:g} m, layers {lay}, "
             f"fs = {v.fs:g} Hz, total delay {v.delay:g} frames"
             + (f", radial orders 1–{v.n_max}\naliasing variance per mode = α·(n+1)/{v.n_max + 1}"
                f" × variance of a radial-order-{v.n_max + 1} mode (same temporal shape)" if sr
                else f"\nquadrature sum over the {n_modes} modes of radial orders 1–{v.n_max},"
                     " no fitting error")
             + (f"\nvibrations {vib}" if vib else ""))
    fname = fname or ("sr_vs_noise_aliasing.png" if sr
                      else "residual_total_vs_noise_aliasing.png")
    return _finish(fig, title, fname, fontsize=9)


# =========================================================================== main
if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("task", nargs="?", default="all",
                    choices=["compare", "order", "tf", "sweep", "all"])
    task = ap.parse_args().task
    if task in ("compare", "all"):
        conan_compare()
    if task in ("order", "all"):
        print(plot_order(conan_compute()))
    if task in ("tf", "all"):                 # first vibrated mode if any, else n = 2
        j = min(CONAN.vibrations) if CONAN.vibrations else None
        print(plot_tf(j=j))
    if task in ("sweep", "all"):
        out = vk_compute()
        print(plot_sweep(out, "sr"))
        print(plot_sweep(out, "residual"))
