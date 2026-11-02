"""
IIR Wiener study: all comparisons, sweeps and figures in one script.

  python iir_study.py compare   # table: LI vs IIR vs bound per mode + optical-gain test
  python iir_study.py order     # residual_vs_radial_order.png
  python iir_study.py tf        # psd_rtf_ntf.png
  python iir_study.py sweep     # residual_total_vs_noise_aliasing.png
  python iir_study.py gain      # residual_vs_optical_gain.png
  python iir_study.py coeffs    # filter_coefficients.png (optimal filters of the sweep)
  python iir_study.py all       # everything (default)

Turbulence (TURB): multi-layer frozen-flow von Karman, Zernike temporal PSDs averaged
over the n+1 modes of each radial order; residuals in rad^2 at TURB.lam.
Aliasing on radial order n: variance alpha*(n+1)/(n_max+1) times that of one radial-order
n_max+1 mode, with that mode's temporal shape. Noise: per-frame variance, same for every mode.
Vibrations: {Noll index j: [(f0 [Hz], variance [rad^2 @ lam], damping zeta), ...]}.
  A vibrated mode j gets the turbulence PSD of its radial order plus its lines and is
  designed separately; all other modes are turbulence only. Vibrations are not aliased.
Every controller is optimised mode by mode. Optical gain (order, tf): controllers are
designed for the assumed gain g_hat and evaluated with the true gain g; the Wiener
bound is the perfect-knowledge reference (designed and evaluated with g).
gain: controllers designed for g_hat = 1, evaluated with true optical gain g (POL built
with g_hat, so IIR loops see a gain error); reference: Wiener bound with g known.
sweep: sigma^2 = sum_n (n+1) sigma_n^2 over n = 1..n_max (PCHIP in log-log between
the design orders), vibrated modes replacing one mode of their order.
Results are cached in files named after a hash of the configuration and of the code
used to compute them, so a cached result is never stale.
"""
import os, json, argparse
from functools import lru_cache
from types import SimpleNamespace
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import wiener_ao_iir, ao_tools
from wiener_ao_iir import ModeSpectra, wiener_iir, _step_up, KMAX
from ao_tools import (ZernikePSD, r0_from_seeing, with_vibrations, noll_radial_order,
                      closed_loop_residual, optimal_integrator, optimal_leaky_integrator,
                      transfer_functions, cache_file, total_variance)

# ================================================================== configuration
TURB = SimpleNamespace(                                 # shared by all tasks
    D=8.0, L0=25.0, seeing=1.0, lam=750e-9,             # seeing at 500 nm, residuals at lam
    layers=[(0.60, 7.0), (0.25, 20.0), (0.15, 35.0)],   # (Cn2 fraction, wind speed [m/s])
    fs=1000.0,
    n_max=30,                                           # radial orders 1..n_max (aliasing model)
    vibrations={},                      # e.g. {2: [(20.0, 1e-2, 0.005)], 3: [(55.0, 3e-3, 0.002)]}
)
ORDER = SimpleNamespace(                                # compare, order, tf
    orders=(2, 3),                                      # IIR orders shown (bound always added)
    radial=[1, 2, 3, 4, 6, 8, 10, 13, 16, 20, 25, 30],
    noises=(1e-4, 1e-3),                                # rad^2/frame/mode @ lam
    delays=(1.5, 2.5),                                  # total delay [frames] (T + tau)
    alpha=0.1,                                          # aliasing level (0.1: pyramid-like)
)
SWEEP = SimpleNamespace(                                # sweep
    delay=2.0, M=2**14,                                 # total delay [frames], grid size
    n_design=[1, 2, 3, 4, 6, 10, 16, 23, 30],
    noises=np.geomspace(1e-5, 1e-1, 7).tolist(),       # rad^2/frame/mode @ lam
    alphas=np.geomspace(1e-2, 1e1, 7).tolist(),        # aliasing level
    noise_fixed=1e-3, alpha_fixed=0.1,                  # value of the other sweep
)


def styles(orders):
    """Labels, colors, markers: integrator, leaky integrator, IIR orders, bound."""
    k = len(orders)
    return (["Integrator (leak = 1)", "Leaky integrator"]
            + [f"IIR Wiener, N={N}" for N in orders] + ["Wiener bound (N = ∞)"],
            ["0.5", "k"] + ["tab:orange", "tab:green", "tab:blue", "tab:purple"][:k] + ["tab:red"],
            ["v", "o"] + ["s", "D", "x", "P"][:k] + ["^"])


GAIN = SimpleNamespace(                                 # gain: robustness to optical gain
    modes=[1, 4, 10, 20],                               # radial orders (one panel each)
    ratios=np.geomspace(0.25, 2.0, 10).tolist(),        # true / assumed optical gain
    noise=1e-3, delay=2.0,                              # rad^2/frame @ lam, total delay [frames]
)
O_LABELS, O_COLORS, O_MARKERS = styles(ORDER.orders)
S_LABELS, S_COLORS, S_MARKERS = styles((2, 3))              # sweep: IIR orders 2 and 3


# ===================================================================== spectra
@lru_cache(None)
def turbulence():
    t = TURB
    return ZernikePSD(t.D, r0_from_seeing(t.seeing), t.L0, t.layers, lam=t.lam)


def vib_modes():
    """[(n, j)] of the vibrated modes, sorted by Noll index."""
    modes = [(noll_radial_order(j), j) for j in sorted(TURB.vibrations)]
    if any(n > TURB.n_max for n, _ in modes):
        raise ValueError(f"vibrated mode beyond radial order {TURB.n_max}")
    return modes


def mode_label(n, j):
    return f"n={n}" if j is None else f"j={j}*"


def disturbance_psd(n, j=None):
    """Turbulence PSD of radial order n, + the vibrations of mode j if given."""
    return with_vibrations(turbulence().psd(n), TURB.vibrations.get(j, ()))


def make_spec(n, noise, total, alpha, j=None, M=None):
    """Spectra of mode j (with vibrations) or of the average mode of radial order n.
    Default grid: 2^16 for low cut-offs or vibrations, 2^15 otherwise."""
    S_al = turbulence().psd(TURB.n_max + 1)
    k_al = alpha * (n + 1) / (TURB.n_max + 1)
    M = M or (2**16 if n < 4 or j is not None else 2**15)
    return ModeSpectra(disturbance_psd(n, j), TURB.fs, total - 1.0, lambda f: k_al * S_al(f),
                       noise=noise, M=M)


# ================================================ per-mode comparison (compare/order/tf)
def design(spec, g=1.0, g_hat=1.0):
    """Controllers [(structure, b, a, C_grid, residual, g, g_hat), ...]: integrator,
    leaky integrator, IIR orders ORDER.orders (designed for g_hat, evaluated with g), bound."""
    Gi, _ = optimal_integrator(spec, g_hat, g_hat)
    (G, l), _ = optimal_leaky_integrator(spec, g_hat, g_hat)
    info = wiener_iir(spec, max(ORDER.orders), g=g_hat)[2]
    info_g = info if g == g_hat else wiener_iir(spec, max(ORDER.orders), g=g)[2]
    ctrls = [("residual", [Gk], [1.0, -lk], None,
              closed_loop_residual(spec, [Gk], [1.0, -lk], "residual", g, g_hat), g, g_hat)
             for Gk, lk in ((Gi, 1.0), (G, l))]
    for N in ORDER.orders:
        b, a, r = info["lower"][N - 1]
        ev = closed_loop_residual(spec, b, a, "pol", g, g_hat)
        assert g != g_hat or abs(ev - r) < 1e-6 * ev        # design vs closed-loop eval
        ctrls.append(("pol", b, a, None, ev, g, g_hat))
    ctrls.append(("pol", None, None, info_g["F_bound"], info_g["bound"], g, g))
    return ctrls


def compute_order(g=1.0, g_hat=1.0):
    """res[delay, noise, mode, controller], modes = ORDER.radial then vibrated modes."""
    o = ORDER
    modes = [(n, None) for n in o.radial] + vib_modes()
    cfg = dict(TURB=vars(TURB), ORDER=vars(o), g=g, g_hat=g_hat)
    cache = cache_file("results_iir", cfg, wiener_ao_iir, ao_tools, make_spec, design,
                       ext="npz")
    shape = (len(o.delays), len(o.noises), len(modes), len(O_LABELS))
    res = np.load(cache)["res"] if os.path.exists(cache) else np.full(shape, np.nan)
    for i, total in enumerate(o.delays):
        for k, noise in enumerate(o.noises):
            for m, (n, j) in enumerate(modes):
                if np.all(np.isfinite(res[i, k, m])):
                    continue
                spec = make_spec(n, noise, total, o.alpha, j)
                res[i, k, m] = [x[4] for x in design(spec, g, g_hat)]
                np.savez(cache, res=res)
                print(total, noise, mode_label(n, j), flush=True)
    return res


def compare():
    o = ORDER
    print(turb_text().replace("\n", "; ") + f"; aliasing α = {o.alpha:g}; residuals in rad²"
          + ("; * = with vibrations" if TURB.vibrations else "") + "\n")
    modes = [(n, None) for n in (1, 4, 10, 20)] + vib_modes()
    for total in o.delays:
        print(f"===== total delay {total} frames =====")
        print(f"{'mode':>6} {'noise':>6} {'INT':>9} {'LI':>9} "
              + " ".join(f"{'IIR' + str(N):>9}" for N in o.orders) + f" {'bound':>9}")
        for noise in o.noises:
            for n, j in modes:
                row = [x[4] for x in design(make_spec(n, noise, total, o.alpha, j))]
                print(f"{mode_label(n, j):>6} {noise:6.0e} " + " ".join(f"{x:9.3e}" for x in row))
        print()

    n, noise, total, g = 10, 1e-4, 2.5, 0.5
    print(f"===== optical gain: n={n}, noise {noise:.0e}, total delay {total}, true g={g} =====")
    spec = make_spec(n, noise, total, o.alpha)
    ok = wiener_iir(spec, max(o.orders), g=g)[2]["lower"]      # designed knowing g
    no = wiener_iir(spec, max(o.orders), g=1.0)[2]["lower"]    # designed assuming g = 1
    for N in o.orders:
        (b, a, _), (b1, a1, _) = ok[N - 1], no[N - 1]
        print(f"IIR{N}: compensated {closed_loop_residual(spec, b, a, 'pol', g, g):.3e} | "
              f"uncompensated {closed_loop_residual(spec, b1, a1, 'pol', g, 1.0):.3e}")
    print(f"LI   : {optimal_leaky_integrator(spec, g, g)[1]:.3e} (independent of g_hat)")


# ============================================================== optical-gain robustness
def compute_gain():
    """res[mode, ratio, controller]: leaky integrator, IIR2, IIR3 designed for g_hat = 1
    and evaluated with g = ratio (inf if unstable); Wiener bound designed with g known."""
    gn = GAIN
    cfg = dict(TURB=vars(TURB), GAIN=vars(gn), alpha=ORDER.alpha)
    cache = cache_file("gain_cache", cfg, wiener_ao_iir, ao_tools, make_spec, ext="npz")
    if os.path.exists(cache):
        return np.load(cache)["res"]
    res = np.full((len(gn.modes), len(gn.ratios), 4), np.nan)
    for m, n in enumerate(gn.modes):
        spec = make_spec(n, gn.noise, gn.delay, ORDER.alpha)
        (G, l), _ = optimal_leaky_integrator(spec)
        lower = wiener_iir(spec, 3)[2]["lower"]
        init = None
        for r, g in enumerate(gn.ratios):
            info = wiener_iir(spec, 3, g=g, init=init)[2]       # bound, g known
            init = info["v"]
            res[m, r] = [closed_loop_residual(spec, [G], [1.0, -l], "residual", g, 1.0),
                         *(closed_loop_residual(spec, *lower[N - 1][:2], "pol", g, 1.0)
                           for N in (2, 3)), info["bound"]]
            print(f"n={n} g={g:.2f}", np.round(res[m, r], 6).tolist(), flush=True)
    np.savez(cache, res=res)
    return res


# ============================================================ noise/aliasing sweep
SWEEPS = {"noise": (SWEEP.noises, lambda x: (x, SWEEP.alpha_fixed), SWEEP.noise_fixed),
          "alias": (SWEEP.alphas, lambda x: (SWEEP.noise_fixed, x), SWEEP.alpha_fixed)}


def sweep_modes():
    return [(n, None) for n in SWEEP.n_design] + vib_modes()


def sweep_design(n, j, noise, alpha, init=None):
    """Residuals [integrator, leaky, IIR2, IIR3, bound] for one mode; also IIR state."""
    spec = make_spec(n, noise, SWEEP.delay, alpha, j, M=SWEEP.M)
    _, ji = optimal_integrator(spec)
    _, jl = optimal_leaky_integrator(spec)
    info = wiener_iir(spec, 3, init=init)[2]
    return [ji, jl, info["lower"][1][2], info["lower"][2][2], info["bound"]], info["v"]


def sweep_key(n, j, noise, alpha):
    return f"{n if j is None else f'j{j}'}|{noise:.6e}|{alpha:.6e}"


def sweep_cache():
    return cache_file("sweep_cache", dict(TURB=vars(TURB), SWEEP=vars(SWEEP)), wiener_ao_iir,
                      ao_tools, make_spec, sweep_design)


def compute_sweep():
    """{'noise': array[len(noises), len(modes), 5], 'alias': ...}, modes = sweep_modes()."""
    fn = sweep_cache()
    done = json.load(open(fn)) if os.path.exists(fn) else {}
    for n, j in sweep_modes():
        for name, (xs, pt, x0) in SWEEPS.items():
            # continuation from the shared point outwards along each sweep
            i0 = int(np.argmin(np.abs(np.log(xs) - np.log(x0))))
            state = {}
            for i in [i0, *range(i0 + 1, len(xs)), *range(i0 - 1, -1, -1)]:
                k = sweep_key(n, j, *pt(xs[i]))
                if k in done:
                    state[i] = {int(N): np.array(v) for N, v in done[k]["v"].items()}
                    continue
                init = state.get(i - 1 if i > i0 else i + 1, state.get(i0))
                res, state[i] = sweep_design(n, j, *pt(xs[i]), init=init)
                done[k] = dict(res=res, v={N: v.tolist() for N, v in state[i].items()})
                with open(fn, "w") as fh:
                    json.dump(done, fh)
                print(mode_label(n, j), name, f"{xs[i]:.1e}", np.round(res, 6).tolist(),
                      flush=True)
    return {name: np.array([[done[sweep_key(n, j, *pt(x))]["res"] for n, j in sweep_modes()]
                            for x in xs]) for name, (xs, pt, _) in SWEEPS.items()}


def sweep_total(res_modes):
    """res_modes[len(modes), 5] -> total residual per controller."""
    nd = len(SWEEP.n_design)
    return total_variance(res_modes[:nd], SWEEP.n_design, TURB.n_max, res_modes[nd:],
                          [n for n, _ in sweep_modes()[nd:]])


# ======================================================================= plotting
def turb_text():
    t = TURB
    lay = ", ".join(f"{w:.0%}@{v:g} m/s" for w, v in t.layers)
    vib = "; ".join(f"j={j}: " + ", ".join(f"{f0:g} Hz/{v:.1e} rad²/ζ={z:g}" for f0, v, z in L)
                    for j, L in sorted(t.vibrations.items()) if L)
    return (f"D = {t.D:g} m, seeing {t.seeing:g}\" @ 500 nm, L0 = {t.L0:g} m, layers {lay}, "
            f"fs = {t.fs:g} Hz, residuals @ {t.lam * 1e9:.0f} nm"
            + (f"\nvibrations {vib}" if vib else ""))


def _finish(fig, title, fname, fontsize=9):
    fig.suptitle(title, fontsize=fontsize)
    fig.tight_layout()
    fig.savefig(fname, dpi=150)
    plt.close(fig)
    return fname


def gain_text(g, g_hat):
    return "" if g == g_hat == 1 else f", true/assumed optical gain {g:g}/{g_hat:g}"


def plot_order(res, g=1.0, g_hat=1.0, fname="residual_vs_radial_order.png"):
    """Residual vs radial order; vibrated modes as filled markers next to their order."""
    o, styles = ORDER, ["-", "--", "-.", ":"]
    nr, vm = len(o.radial), vib_modes()
    xv = [n * 1.05 ** (1 + [m for m, _ in vm[:i]].count(n)) for i, (n, _) in enumerate(vm)]
    fig, axes = plt.subplots(1, len(o.delays), figsize=(6 * len(o.delays), 5), sharey=True)
    axes = np.atleast_1d(axes)
    for i, (ax, total) in enumerate(zip(axes, o.delays)):
        for k in range(len(o.noises)):
            for m in range(len(O_LABELS)):
                ax.plot(o.radial, res[i, k, :nr, m], styles[k], color=O_COLORS[m],
                        marker=O_MARKERS[m], ms=4, lw=1.3, mfc="none")
                ax.plot(xv, res[i, k, nr:, m], ls="none", color=O_COLORS[m],
                        marker=O_MARKERS[m], ms=6)
        for x, (_, j), r in zip(xv, vm, res[i, 0, nr:].min(1)):   # label vibrated modes
            ax.annotate(f"j={j}", (x, r), xytext=(0, -12), textcoords="offset points",
                        ha="center", fontsize=7)
        ax.set_xscale("log"); ax.set_yscale("log")
        ax.set_xticks(o.radial); ax.set_xticklabels(o.radial); ax.minorticks_off()
        ax.set(xlabel="Radial order n", title=f"Total delay {total} frames")
        ax.grid(True, which="both", alpha=0.3)
    axes[0].set_ylabel("Residual variance per mode [rad²]")
    h = [Line2D([], [], color=c, lw=1.5, marker=mk, mfc="none")
         for c, mk in zip(O_COLORS, O_MARKERS)]
    h += [Line2D([], [], color="gray", ls=s) for s in styles[:len(o.noises)]]
    lab = O_LABELS + [f"noise {x:.0e} rad²/frame" for x in o.noises]
    if vm:
        h.append(Line2D([], [], color="gray", ls="none", marker="o"))
        lab.append("filled: modes with vibrations (" + ", ".join(f"j={j}" for _, j in vm) + ")")
    axes[-1].legend(h, lab, fontsize=9)
    return _finish(fig, turb_text() + f", aliasing α = {o.alpha:g}" + gain_text(g, g_hat), fname)


def plot_tf(n=2, j=None, noise=1e-2, total=1.5, g=1.0, g_hat=1.0, fname="psd_rtf_ntf.png",
            n_pts=3000):
    """Left: disturbance PSD and |RTF|^2. Right: noise and (folded, WFS-averaged)
    aliasing PSDs and |NTF|^2 (aliasing propagates as H*NTF, ~NTF in band).
    Mode j (Noll, with its vibrations) if given, else the average mode of order n."""
    fs = TURB.fs
    n = n if j is None else noll_radial_order(j)
    lines = TURB.vibrations.get(j, ())
    spec = make_spec(n, noise, total, ORDER.alpha, j)
    ctrls = design(spec, g, g_hat)
    nu = np.fft.fftfreq(spec.M, 1 / fs)
    idx = np.nonzero(nu > 0)[0]
    idx = np.unique(idx[np.round(np.geomspace(1, len(idx), n_pts)).astype(int) - 1])
    f = nu[idx]
    fd = np.unique(np.r_[f, *[f0 * (1 + z * np.linspace(-20, 20, 201)) for f0, _, z in lines]])
    fig, (axL, axR) = plt.subplots(1, 2, figsize=(13, 5.2))

    def area(ax, x, y, color, label):                    # PSDs as shaded areas
        ax.fill_between(x, y, 1e-300, color=color, alpha=0.18, lw=0, label=label)
        ax.loglog(x, y, color=color, lw=0.8, alpha=0.6)
        return y

    St = area(axL, fd, disturbance_psd(n, j)(fd), "tab:gray",
              f"Disturbance PSD ({mode_label(n, j)})")
    axL.set_ylim(St.min() / 10, St.max() * 10)
    axL.set(xlabel="Frequency [Hz]", ylabel="PSD [rad²/Hz]", title="Disturbance PSD and |RTF|²")
    Sn = area(axR, f, 2 * spec.Pnn[idx] / fs, "tab:gray", "Noise PSD")
    Sa = area(axR, f, 2 * spec.Paa[idx] / fs, "tab:purple", "Aliasing PSD (folded, WFS-averaged)")
    axR.set_ylim(min(Sn.min(), Sa.min()) / 10, max(Sn.max(), Sa.max()) * 10)
    axR.set(xlabel="Frequency [Hz]", ylabel="PSD [rad²/Hz]",
            title="Noise & aliasing PSD and |NTF|²")
    axL2, axR2 = axL.twinx(), axR.twinx()
    for m, (st, b, a, C, r, gg, gh) in enumerate(ctrls):
        E, Nt, _ = transfer_functions(b, a, f, fs, spec.delay, st, gg, gh,
                                      C=None if C is None else C[idx])
        kw = dict(color=O_COLORS[m], lw=1.3, ls=":" if m == len(ctrls) - 1 else "-")
        axL2.loglog(f, np.abs(E) ** 2, label=f"{O_LABELS[m]}  (σ² = {r:.3e})", **kw)
        axR2.loglog(f, np.abs(Nt) ** 2, label=O_LABELS[m], **kw)
    axL2.set_ylabel("|RTF|²"); axR2.set_ylabel("|NTF|²")
    for ax, ax2, loc in ((axL, axL2, "lower right"), (axR, axR2, "lower left")):
        ax.grid(True, which="both", alpha=0.3)
        h1, l1 = ax.get_legend_handles_labels(); h2, l2 = ax2.get_legend_handles_labels()
        ax2.legend(h1 + h2, l1 + l2, fontsize=8, loc=loc)
    mode = f"radial order {n}" if j is None else f"mode j = {j}"
    return _finish(fig, turb_text() + f"\n{mode}, noise {noise:.0e} rad²/frame, aliasing "
                   f"α = {ORDER.alpha:g}, total delay {total} frames" + gain_text(g, g_hat), fname)


def plot_sweep(out, fname="residual_total_vs_noise_aliasing.png"):
    """Total residual variance vs noise (left) and vs aliasing level (right)."""
    s = SWEEP
    tot = {k: np.array([sweep_total(r) for r in x]) for k, x in out.items()}
    panels = [(s.noises, tot["noise"], "Noise variance per mode [rad²/frame]",
               f"Residual vs noise (aliasing level α = {s.alpha_fixed:g})"),
              (s.alphas, tot["alias"], "Aliasing level α",
               f"Residual vs aliasing (noise {s.noise_fixed:.0e} rad²/frame)")]
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharey=True)
    for ax, (x, y, xl, t) in zip(axes, panels):
        for m in range(len(S_LABELS)):
            ax.loglog(x, y[:, m], color=S_COLORS[m], marker=S_MARKERS[m], mfc="none",
                      ls=":" if m == len(S_LABELS) - 1 else "-", lw=1.4, label=S_LABELS[m])
        ax.set(xlabel=xl, title=t)
        ax.grid(True, which="both", alpha=0.3)
    axes[0].set_ylabel("Total residual variance [rad²]")
    axes[1].legend(fontsize=9)
    n_modes = sum(n + 1 for n in range(1, TURB.n_max + 1))
    return _finish(fig, turb_text() + f", total delay {s.delay:g} frames\nsum over the "
                   f"{n_modes} modes of radial orders 1–{TURB.n_max}; aliasing variance per mode "
                   f"= α·(n+1)/{TURB.n_max + 1} × variance of a radial-order-{TURB.n_max + 1} mode",
                   fname)


def plot_gain(res, fname="residual_vs_optical_gain.png"):
    """Residual vs true/assumed optical gain (top) and its ratio to the Wiener bound with
    the gain known (bottom), one column per radial order."""
    gn = GAIN
    lab, col, mk = (np.array(x, dtype=object)[[1, 2, 3, 4]] for x in styles((2, 3)))
    fig, axes = plt.subplots(2, len(gn.modes), figsize=(4.2 * len(gn.modes), 7.5), sharex=True)
    for top, bot, n, r in zip(axes[0], axes[1], gn.modes, res):
        for c in range(4):
            kw = dict(color=col[c], marker=mk[c], ms=4, mfc="none", lw=1.3,
                      ls=":" if c == 3 else "-", label=lab[c] + (" (g known)" if c == 3 else ""))
            top.loglog(gn.ratios, r[:, c], **kw)
            bot.loglog(gn.ratios, r[:, c] / r[:, 3], **kw)
        top.set(title=f"Radial order {n}")
        bot.set(xlabel="True / assumed optical gain", xticks=[0.25, 0.5, 1, 2])
        bot.set_xticklabels(["0.25", "0.5", "1", "2"]); bot.xaxis.set_minor_formatter("")
        for ax in (top, bot):
            ax.axvline(1.0, color="0.6", lw=0.8)
            ax.grid(True, which="both", alpha=0.3)
    q = res / res[..., 3:]                               # same y range for all ratio panels
    lo, hi = 0.9, 1.15 * q[np.isfinite(q)].max()
    for bot in axes[1]:
        bot.set_ylim(lo, hi)
        bot.set_yticks([t for t in (1, 1.5, 2, 3, 5, 10, 20, 50) if lo <= t <= hi])
        bot.set_yticklabels([f"{t:g}" for t in bot.get_yticks()])
        bot.yaxis.set_minor_formatter("")
    axes[0, 0].set_ylabel("Residual variance per mode [rad²]")
    axes[1, 0].set_ylabel("Residual / Wiener bound (g known)")
    axes[0, 0].legend(fontsize=8)
    return _finish(fig, turb_text() + f"\ncontrollers designed for optical gain 1, noise "
                   f"{gn.noise:.0e} rad²/frame, aliasing α = {ORDER.alpha:g}, total delay "
                   f"{gn.delay:g} frames; missing points = unstable", fname)


def plot_coeffs(noises=(1e-5, 1e-3, 1e-1), fname="filter_coefficients.png"):
    """Optimal filters of the sweep (total delay SWEEP.delay, aliasing SWEEP.alpha_fixed)
    vs radial order, one row per noise level (must be in SWEEP.noises): leaky integrator
    gain G and leak (recomputed), IIR2 and IIR3 b_i (solid) and a_i (dashed), powers of
    z^-1, a_0 = 1 (from the sweep cache). LI acts on the residual measurement, the IIRs on
    POL, so their coefficients are not directly comparable."""
    compute_sweep()
    with open(sweep_cache()) as fh:
        done = json.load(fh)
    ns = SWEEP.n_design
    cols = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    fig, axes = plt.subplots(len(noises), 3, figsize=(14, 3.3 * len(noises)), sharex=True,
                             squeeze=False)
    for row, noise in zip(axes, noises):
        li = []
        for n in ns:
            spec = make_spec(n, noise, SWEEP.delay, SWEEP.alpha_fixed, M=SWEEP.M)
            li.append(optimal_leaky_integrator(spec)[0])
        li = np.array(li)
        row[0].plot(ns, li[:, 0], "o-", color=cols[0], label="gain G")
        row[0].plot(ns, li[:, 1], "s--", color=cols[1], label="leak")
        for ax, N in zip(row[1:], (2, 3)):
            v = np.array([done[sweep_key(n, None, noise, SWEEP.alpha_fixed)]["v"][str(N)]
                          for n in ns])
            b = v[:, :N + 1]
            a = np.array([_step_up(KMAX * np.tanh(t)) for t in v[:, N + 1:]])
            for i in range(N + 1):
                ax.plot(ns, b[:, i], "o-", color=cols[i], ms=4, label=f"b{i}")
            for i in range(1, N + 1):
                ax.plot(ns, a[:, i], "s--", color=cols[i], ms=4, mfc="none", label=f"a{i}")
        for ax, t in zip(row, ("Leaky integrator", "IIR Wiener, N=2", "IIR Wiener, N=3")):
            ax.set_xscale("log"); ax.set_xticks(ns); ax.set_xticklabels(ns); ax.minorticks_off()
            ax.axhline(0, color="0.6", lw=0.8)
            ax.set_title(f"{t}, noise {noise:.0e} rad²/frame", fontsize=10)
            ax.grid(True, alpha=0.3)
        row[0].set_ylabel("Coefficient")
    for ax in axes[0]:
        ax.legend(fontsize=8, ncol=2)
    for ax in axes[-1]:
        ax.set_xlabel("Radial order n")
    return _finish(fig, turb_text() + f"\noptimal filters, total delay {SWEEP.delay:g} frames, "
                   f"aliasing α = {SWEEP.alpha_fixed:g}; IIR: u = B(z)/A(z)·POL, "
                   "b solid, a dashed; LI: u_k = leak·u_k-1 + G·y_k", fname)


# =========================================================================== main
if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("task", nargs="?", default="all",
                    choices=["compare", "order", "tf", "sweep", "gain", "coeffs", "all"])
    task = ap.parse_args().task
    if task in ("compare", "all"):
        compare()
    if task in ("order", "all"):
        print(plot_order(compute_order()))
    if task in ("tf", "all"):                 # first vibrated mode if any, else n = 2
        print(plot_tf(j=min(TURB.vibrations) if TURB.vibrations else None))
    if task in ("sweep", "all"):
        print(plot_sweep(compute_sweep()))
    if task in ("gain", "all"):
        print(plot_gain(compute_gain()))
    if task in ("coeffs", "all"):
        print(plot_coeffs())
