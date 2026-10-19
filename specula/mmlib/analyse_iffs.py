import numpy as np
from scipy.ndimage import shift, affine_transform, distance_transform_edt, zoom
from scipy.optimize import minimize
import matplotlib.pyplot as plt
from astropy.io import fits

def compute_actuator_pitch(mask_shape: tuple, n_acts: int) -> float:
    """Estimates actuator pitch for a circular grid: dim / sqrt(4 * n_acts / pi)."""
    dim = max(mask_shape)
    return dim / np.sqrt(4.0 * n_acts / np.pi)


def estimate_actuator_peaks(ifs: np.ndarray, mask: np.ndarray = None, subpixel: bool = True) -> np.ndarray:
    """Estimates (y, x) peak coordinates for each influence function in its native grid."""
    n_acts, h, w = ifs.shape
    coords = np.zeros((n_acts, 2))  # Stored as [y, x]
    
    for i in range(n_acts):
        img = np.abs(ifs[i]) * (mask if mask is not None else 1.0)
        max_val = np.max(img)
        
        if max_val < 1e-12:
            coords[i] = [np.nan, np.nan]
            continue
            
        max_idx = np.unravel_index(np.argmax(img), img.shape)
        py, px = max_idx[0], max_idx[1]
        
        if subpixel and 1 <= py < h - 1 and 1 <= px < w - 1:
            dx = (img[py, px + 1] - img[py, px - 1]) / (2.0 * (2.0 * img[py, px] - img[py, px + 1] - img[py, px - 1] + 1e-12))
            dy = (img[py + 1, px] - img[py - 1, px]) / (2.0 * (2.0 * img[py, px] - img[py + 1, px] - img[py - 1, px] + 1e-12))
            coords[i] = [py + np.clip(dy, -0.5, 0.5), px + np.clip(dx, -0.5, 0.5)]
        else:
            coords[i] = [float(py), float(px)]
            
    return coords


def identify_inner_actuators_radial(coords: np.ndarray, mask_shape: tuple, pitch: float, n_outer_rings: int = 2) -> np.ndarray:
    """Identifies inner actuators using radial distance from grid center."""
    valid_mask = ~np.isnan(coords[:, 0])
    cy, cx = mask_shape[0] / 2.0, mask_shape[1] / 2.0
    
    radii = np.full(len(coords), np.inf)
    radii[valid_mask] = np.sqrt((coords[valid_mask, 0] - cy)**2 + (coords[valid_mask, 1] - cx)**2)
    
    max_radius = np.max(radii[valid_mask])
    threshold_radius = max_radius - (n_outer_rings - 0.5) * pitch
    
    return (radii < threshold_radius) & valid_mask

def fit_similarity_transform(src_pts: np.ndarray, dst_pts: np.ndarray):
    """Computes global similarity transformation (Scale, Rotation, Shift) mapping src -> dst."""
    src_centroid = np.mean(src_pts, axis=0)  # [cy, cx]
    dst_centroid = np.mean(dst_pts, axis=0)  # [cy, cx]
    
    src_centered = src_pts - src_centroid
    dst_centered = dst_pts - dst_centroid
    
    H = src_centered.T @ dst_centered
    U, S, Vt = np.linalg.svd(H)
    R = Vt.T @ U.T
    
    if np.linalg.det(R) < 0:
        Vt[-1, :] *= -1
        R = Vt.T @ U.T
        
    scale = np.sum(S) / np.sum(src_centered**2)
    t_y_x = dst_centroid - scale * (R @ src_centroid)
    
    return scale, R, t_y_x


def apply_global_registration_to_ifs(sim_ifs: np.ndarray, scale: float, R: np.ndarray, t_y_x: np.ndarray, target_shape: tuple) -> np.ndarray:
    """Warps simulated IFs into the measured grid frame [y, x]."""
    n_acts = len(sim_ifs)
    reg_sim_ifs = np.zeros((n_acts, *target_shape))
    
    inv_R = R.T / scale
    inv_t = -inv_R @ t_y_x
    
    for i in range(n_acts):
        reg_sim_ifs[i] = affine_transform(
            sim_ifs[i], 
            matrix=inv_R, 
            offset=inv_t,
            output_shape=target_shape, 
            order=3
        )
    return reg_sim_ifs


def transform_coordinates(coords: np.ndarray, scale: float, R: np.ndarray, t_y_x: np.ndarray) -> np.ndarray:
    """Applies global similarity transformation directly to [y, x] coordinate array."""
    return scale * (coords @ R.T) + t_y_x


def _refine_actuator_shift(
    meas_img: np.ndarray,
    sim_img: np.ndarray,
    py: int,
    px: int,
    window: int,
    max_shift: float,
    pad: int = 4,
):
    """
    Refines the local (dy, dx) sub-pixel shift of one simulated IF against its
    measured counterpart, using a small padded crop instead of shifting the
    full-frame image on every optimizer evaluation. Amplitude is solved in
    closed form (linear least squares) for each trial (dy, dx), so the
    optimizer only searches a 2D space.

    Returns (dy_opt, dx_opt, amp_opt) or None if the crop has no signal.
    """
    h, w = sim_img.shape
    y_min, y_max = max(0, py - window), min(h, py + window)
    x_min, x_max = max(0, px - window), min(w, px + window)

    meas_crop = meas_img[y_min:y_max, x_min:x_max]

    # Padded region so the shifted crop doesn't need samples outside it
    y0, y1 = max(0, y_min - pad), min(h, y_max + pad)
    x0, x1 = max(0, x_min - pad), min(w, x_max + pad)
    sim_pad = sim_img[y0:y1, x0:x1]
    sim_crop0 = sim_img[y_min:y_max, x_min:x_max]

    if np.max(np.abs(meas_crop)) <= 1e-12 or np.max(np.abs(sim_crop0)) <= 1e-12:
        return None

    # Local slice of the padded crop that corresponds to (y_min:y_max, x_min:x_max)
    sl_y = slice(y_min - y0, y_max - y0)
    sl_x = slice(x_min - x0, x_max - x0)

    def shifted_crop(dy, dx):
        return shift(sim_pad, shift=[dy, dx], order=3)[sl_y, sl_x]

    def loss(params):
        dy, dx = params
        crop = shifted_crop(dy, dx)
        denom = np.sum(crop**2) + 1e-12
        amp = np.sum(meas_crop * crop) / denom  # closed-form optimal amplitude
        return np.sum((meas_crop - amp * crop)**2)

    res = minimize(
        loss,
        x0=[0.0, 0.0],
        bounds=[(-max_shift, max_shift), (-max_shift, max_shift)],
        method='L-BFGS-B'
    )

    dy_opt, dx_opt = res.x
    final_crop = shifted_crop(dy_opt, dx_opt)
    denom = np.sum(final_crop**2) + 1e-12
    amp_opt = np.sum(meas_crop * final_crop) / denom

    return dy_opt, dx_opt, amp_opt


def _nearest_neighbour_pitch(coords: np.ndarray) -> float:
    """Median nearest-neighbour distance between actuator coordinates [y, x]."""
    c = coords[~np.isnan(coords[:, 0])]
    d = np.hypot(c[:, None, 0] - c[None, :, 0], c[:, None, 1] - c[None, :, 1])
    np.fill_diagonal(d, np.inf)
    return float(np.median(d.min(axis=1)))


def _subpixel_patch(img: np.ndarray, c, half: int, margin: int = 3) -> np.ndarray:
    """(2*half+1)^2 patch of img centred on the sub-pixel position c = [y, x] (cubic interpolation,
    zero padded outside the frame)."""
    h, w = img.shape
    cy, cx = int(round(c[0])), int(round(c[1]))
    r = half + margin
    y0, x0 = cy - r, cx - r
    crop = np.zeros((2 * r + 1, 2 * r + 1))
    ya, yb = max(0, y0), min(h, y0 + 2 * r + 1)
    xa, xb = max(0, x0), min(w, x0 + 2 * r + 1)
    if ya < yb and xa < xb:
        crop[ya - y0:yb - y0, xa - x0:xb - x0] = img[ya:yb, xa:xb]
    sh = shift(crop, [-(c[0] - cy), -(c[1] - cx)], order=3, mode='constant')
    return sh[margin:margin + 2 * half + 1, margin:margin + 2 * half + 1]


def _place_template(template: np.ndarray, coords: np.ndarray, shape: tuple) -> np.ndarray:
    """Builds one IF per actuator by placing the (square, odd-sized) template at the sub-pixel
    position coords[j] = [y, x]. Actuators with NaN coordinates get a zero IF."""
    h, w = shape
    half = template.shape[0] // 2
    pad = 3
    tpl = np.pad(template, pad)
    out = np.zeros((len(coords), h, w))
    for j, c in enumerate(coords):
        if np.isnan(c[0]):
            continue
        cy, cx = int(round(c[0])), int(round(c[1]))
        sh = shift(tpl, [c[0] - cy, c[1] - cx], order=3, mode='constant')
        y0, x0 = cy - half - pad, cx - half - pad
        ya, yb = max(0, y0), min(h, y0 + sh.shape[0])
        xa, xb = max(0, x0), min(w, x0 + sh.shape[1])
        if ya < yb and xa < xb:
            out[j, ya:yb, xa:xb] = sh[ya - y0:yb - y0, xa - x0:xb - x0]
    return out


def _template_peak(template: np.ndarray) -> float:
    """Sub-pixel peak value of a template (cubic upsampling of its central region)."""
    half = template.shape[0] // 2
    r = min(half, 4)
    z = zoom(template[half - r:half + r + 1, half - r:half + r + 1], 8, order=3)
    return float(z.flat[np.argmax(np.abs(z))])


def _build_if_template(
    meas_ifs: np.ndarray,
    valid_mask: np.ndarray,
    sim_ifs_registered: np.ndarray,
    coords: np.ndarray,
    pitch_est: float,
    halfwidth_pitches: float = 2.5,
    min_snr_frac: float = 0.5,
    n_iter: int = 3,
):
    """
    Estimates the average measured IF shape from the high-SNR actuators whose whole template
    window lies inside the pupil (edge actuators are excluded).

    Per selected actuator a patch of the measured IF is cut around its registered position
    (cubic sub-pixel interpolation), the tip/tilt of a plane fitted on the outer ring is removed (drift),
    and the patches are aligned to the evolving template with a few Lucas-Kanade shift updates
    (so that residual mis-registration does not blur the template). The template is the
    amplitude-weighted average (rank-1 model patch_i = a_i * template), after rejecting outlier
    actuators, tapered smoothly to zero at the window edge, and scaled so that its peak equals
    the median peak of the simulated IFs of the same actuators (gain convention: peak, relative
    to the simulation).

    Returns None if fewer than 5 usable actuators are found, else a dict with 'template'
    (P x P), 'used' (actuator indices), 'centroid' ([dy, dx] of the template centre of mass
    from the window centre: a global offset between the registered coordinates and the measured
    IFs shows up here; it is baked into the template, so placing it at the registered coordinates
    stays consistent), 'half', 'rel_residual' and 'n_candidates'.
    """
    n_acts, h, w = meas_ifs.shape
    half = max(3, int(round(halfwidth_pitches * pitch_est)))
    P = 2 * half + 1
    yy, xx = np.mgrid[-half:half + 1, -half:half + 1]
    rn = np.hypot(yy, xx) / half
    ring = ((rn >= 0.75) & (rn <= 1.0)).ravel()
    G = np.stack([np.ones(P * P), yy.ravel().astype(float), xx.ravel().astype(float)], axis=1)
    G_ring_pinv = np.linalg.pinv(G[ring])
    taper = np.where(rn <= 0.7, 1.0, np.where(rn >= 1.0, 0.0, 0.5 * (1.0 + np.cos(np.pi * (rn - 0.7) / 0.3))))

    med = np.median(meas_ifs[:, valid_mask], axis=1)
    dist = distance_transform_edt(valid_mask)

    # Candidates: whole window (plus interpolation margin) inside the pupil
    cand = []
    for i in range(n_acts):
        if np.isnan(coords[i, 0]):
            continue
        cy, cx = int(round(coords[i, 0])), int(round(coords[i, 1]))
        if 0 <= cy < h and 0 <= cx < w and dist[cy, cx] >= half + 3:
            cand.append(i)
    cand = np.array(cand, dtype=int)
    if len(cand) < 5:
        return None

    def extract(ids, offsets):
        patches = np.empty((len(ids), P * P))
        for k, i in enumerate(ids):
            patches[k] = _subpixel_patch(meas_ifs[i] - med[i], coords[i] + offsets[k], half).ravel()
        coef = patches[:, ring] @ G_ring_pinv.T          # plane fitted on the outer ring
        return patches - coef[:, 1:] @ G[:, 1:].T        # remove only its tip/tilt (the pedestal holds the real IF tails)

    # SNR selection: peak over robust noise (MAD of the whole IF over the pupil)
    P0 = extract(cand, np.zeros((len(cand), 2)))
    sigma = np.array([1.4826 * np.median(np.abs(meas_ifs[i][valid_mask] - med[i])) for i in cand])
    snr = np.max(np.abs(P0), axis=1) / np.maximum(sigma, 1e-12)
    keep = snr >= min_snr_frac * np.median(snr)
    ids, Pm = cand[keep], P0[keep]
    if len(ids) < 5:
        return None

    # Initial template: median of unit-peak patches
    pk = Pm[np.arange(len(ids)), np.argmax(np.abs(Pm), axis=1)]
    t = np.median(Pm / pk[:, None], axis=0)
    t = t / t[np.argmax(np.abs(t))]

    # Alternate Lucas-Kanade alignment of every patch to the template and amplitude-weighted averaging
    d = np.zeros((len(ids), 2))
    max_off = 0.5 * pitch_est
    for _ in range(n_iter):
        gy, gx = np.gradient(t.reshape(P, P))
        D = np.stack([t, gy.ravel(), gx.ravel()], axis=1)
        coef = Pm @ np.linalg.pinv(D).T                   # [a, -a*dy, -a*dx] per patch
        a = coef[:, 0]
        step = np.zeros_like(d)
        ok = np.abs(a) > 1e-12
        step[ok] = -coef[ok, 1:] / a[ok, None]
        d = np.clip(d + np.clip(step, -1.0, 1.0), -max_off, max_off)
        Pm = extract(ids, d)
        a = Pm @ t / (t @ t)
        t = (a @ Pm) / (a @ a)
        t = t / t[np.argmax(np.abs(t))]

    # Reject actuators whose patch deviates from a_i * template, then final average
    a = Pm @ t / (t @ t)
    rel = np.linalg.norm(Pm - a[:, None] * t[None, :], axis=1) / np.maximum(np.abs(a) * np.linalg.norm(t), 1e-12)
    mad = 1.4826 * np.median(np.abs(rel - np.median(rel)))
    good = rel <= np.median(rel) + 3.0 * max(mad, 1e-12)
    if np.sum(good) < 5:
        good[:] = True
    ids, Pm, a, d, rel = ids[good], Pm[good], a[good], d[good], rel[good]
    t = (a @ Pm) / (a @ a)
    t = t / t[np.argmax(np.abs(t))]

    # Peak convention: sub-pixel (interpolated) peak value, for the simulated IFs at their registered
    # coordinates and for the template, so the pixel sampling of the peak does not bias the gains.
    sim_pk = np.array([_subpixel_patch(sim_ifs_registered[i], coords[i], 1)[1, 1] for i in ids])
    t_tapered = t.reshape(P, P) * taper
    template = t_tapered * (np.median(sim_pk) / _template_peak(t_tapered))

    tp = np.clip(template * np.sign(np.median(sim_pk)), 0.0, None)
    centroid = np.array([np.sum(tp * yy), np.sum(tp * xx)]) / np.sum(tp)

    return dict(template=template, used=ids, centroid=centroid, half=half,
                rel_residual=float(np.median(rel)), n_candidates=len(cand))


def _tilt_basis(valid_mask: np.ndarray):
    """Orthonormal basis Q (n_pix x 2) and raw columns T of tip/tilt over the valid pixels
    (coordinates centred on the pupil and normalised by its radius)."""
    py, px = np.nonzero(valid_mask)
    py = py - py.mean()
    px = px - px.mean()
    radius = max(np.max(np.hypot(py, px)), 1e-12)
    T = np.stack([py / radius, px / radius], axis=1)
    Q, _ = np.linalg.qr(T)
    return Q, T


def _solve_amplitudes(S_pix: np.ndarray, M_proj: np.ndarray, Q, reg_param: float):
    """
    Ridge least-squares fit of every (already median-subtracted and tilt-projected)
    measured IF M_proj (n_acts x n_pix) onto the simulated basis S_pix (n_acts x n_pix).

    The simulated basis is median-subtracted per IF here (the measured IFs are median-free)
    and, if Q is given, tip/tilt is projected out of it. Because tip/tilt is an unpenalised
    nuisance term shared by all IFs, projecting it out of both sides is exactly equivalent to
    fitting it jointly with the amplitudes (Frisch-Waugh-Lovell).

    Returns amplitudes (row i = measured IF i), the total squared data residual, the per-IF
    medians removed from the basis, and the median-subtracted (unprojected) basis.
    """
    n_acts = S_pix.shape[0]
    med = np.median(S_pix, axis=1)
    S0 = S_pix - med[:, None]
    Sp = S0 - (S0 @ Q) @ Q.T if Q is not None else S0

    StS = Sp @ Sp.T
    rhs = Sp @ M_proj.T                                   # column i = Sp @ M_proj[i]
    reg_I = reg_param * np.eye(n_acts) * (np.trace(StS) / n_acts)
    A = np.linalg.solve(StS + reg_I, rhs).T

    # total ||M - A Sp||^2 without forming the (n_acts x n_pix) product
    resid = np.sum(M_proj**2) - 2.0 * np.sum(A * rhs.T) + np.sum((A @ StS) * A)
    return A, resid, med, S0


def fit_amplitudes_to_sim_ifs(
    meas_ifs: np.ndarray, 
    meas_mask: np.ndarray, 
    sim_ifs_registered: np.ndarray, 
    reg_param: float = 1e-4,
    optimize_coords: bool = True,
    fit_if_template: bool = True,
    template_halfwidth_pitches: float = 2.5,
    template_min_snr_frac: float = 0.5,
    fit_tilt: bool = True,
    return_diagnostics: bool = False
):
    """
    Fits measured IFs to registered simulated IFs (dense fit over all actuators).

    Steps:
      0. (fit_if_template) the basis shape is replaced by the AVERAGE MEASURED IF of the
         high-SNR, fully-inside-the-pupil actuators (see _build_if_template), placed at every
         actuator's registered position. Since this shape already contains the average coupling
         of an actuator to its surroundings, the amplitude matrix now describes per-actuator gain
         (diagonal) and deviations from that average (off-diagonal), not the absolute coupling.
      1. (optimize_coords) per-actuator sub-pixel shift refinement.
      2. Median is subtracted from every measured and simulated IF over the valid pixels,
         tip/tilt (shared columns, unpenalised) is fitted out (fit_tilt), then a single
         batched ridge solve gives all amplitudes. Finally the median of each amplitude row
         (the poorly constrained common-mode offset, robust to the actuator peaks) is removed.

    Returns:
        amplitude_matrix: (N_acts, N_acts) array of coupling weights.
        scaled_sim_ifs: (N_acts, H, W) array of the final reconstructed IFs (median-subtracted basis).
        diagnostics (only if return_diagnostics): dict with 'gain' (diagonal), and, if the template
            was used, 'template', 'template_actuators', 'template_centroid';
            'row_median' (offset removed from each row), 'row_noise_mad' (robust rms of each
            row's entries, a coefficient-noise estimate) and, if fit_tilt, 'tilt' (n_acts x 2,
            fitted [tip, tilt] per measured IF in data units per pupil radius).
    """
    n_acts, h, w = meas_ifs.shape
    valid_mask = meas_mask > 0

    updated_sim_ifs = np.copy(sim_ifs_registered)

    # Estimate actuator pitch for shift constraints
    pitch = max(h, w) / np.sqrt(4.0 * n_acts / np.pi)
    max_shift = 0.5 * pitch
    window = int(pitch)

    # Measured data: median-subtracted over valid pixels, tilt projected out
    Q, T = _tilt_basis(valid_mask) if fit_tilt else (None, None)
    M0 = meas_ifs[:, valid_mask]
    M0 = M0 - np.median(M0, axis=1, keepdims=True)
    M_proj = M0 - (M0 @ Q) @ Q.T if Q is not None else M0

    # --- Phase 0: replace the simulated IF shape by the average measured IF ---
    tpl_info = None
    if fit_if_template:
        print("Building average measured IF template...")
        coords_reg = estimate_actuator_peaks(sim_ifs_registered)   # unmasked: sim IFs live on the full frame
        pitch_est = _nearest_neighbour_pitch(coords_reg)
        tpl_info = _build_if_template(meas_ifs, valid_mask, sim_ifs_registered, coords_reg, pitch_est,
                                      halfwidth_pitches=template_halfwidth_pitches,
                                      min_snr_frac=template_min_snr_frac)
        if tpl_info is None:
            print("   WARNING: too few usable actuators for a template; keeping the simulated IF shapes.")
        else:
            cen = tpl_info['centroid']
            print(f"-> template {2 * tpl_info['half'] + 1}x{2 * tpl_info['half'] + 1} pix from "
                  f"{len(tpl_info['used'])}/{tpl_info['n_candidates']} inner actuators "
                  f"(pitch ~ {pitch_est:.2f} pix); median patch misfit {tpl_info['rel_residual']:.3f}; "
                  f"template centroid offset [dy,dx] = [{cen[0]:.2f}, {cen[1]:.2f}] pix")
            updated_sim_ifs = _place_template(tpl_info['template'], coords_reg, (h, w))

    # --- Phase 1: per-actuator sub-pixel shift refinement (independent, no
    # global solve needed here) ---
    if optimize_coords:
        print("Starting per-actuator coordinate refinement...")
        for idx in range(n_acts):
            meas_img = meas_ifs[idx]
            sim_img = updated_sim_ifs[idx]

            py, px = np.unravel_index(np.argmax(np.abs(sim_img)), (h, w))
            result = _refine_actuator_shift(meas_img, sim_img, py, px, window, max_shift)
            if result is None:
                continue

            dy_opt, dx_opt, amp_opt = result
            # dy/dx are bounded independently, so compare each axis to
            # max_shift rather than the combined magnitude (a diagonal shift
            # can legitimately reach up to sqrt(2) * max_shift).
            if max(abs(dy_opt), abs(dx_opt)) > 1e-3:
                print(f'Act {idx}: [{dx_opt:.3f}, {dy_opt:.3f}] pix')
                updated_sim_ifs[idx] = shift(sim_img, shift=[dy_opt, dx_opt], order=3)

    # --- Phase 2: amplitude/coupling fit ---
    print("Solving for amplitude/coupling matrix...")
    amplitude_matrix, _, sim_med, S0 = _solve_amplitudes(updated_sim_ifs[:, valid_mask], M_proj, Q, reg_param)

    # Remove the common-mode offset of each row (median: robust to the actuator peaks)
    row_median = np.median(amplitude_matrix, axis=1)
    amplitude_matrix = amplitude_matrix - row_median[:, None]
    row_noise_mad = 1.4826 * np.median(np.abs(amplitude_matrix), axis=1)

    gain = np.diag(amplitude_matrix)
    med_g = np.median(gain)
    print(f"Gain (diag): median = {med_g:.4g}, robust spread (MAD/median) = "
          f"{np.median(np.abs(gain - med_g)) / abs(med_g):.3f}, "
          f"median row noise (MAD) = {np.median(row_noise_mad):.3g}")

    # Reconstruct the final scaled IFs across the FULL domain using the updated, median-subtracted basis
    scaled_sim_ifs = np.einsum('ij,jhw->ihw', amplitude_matrix, updated_sim_ifs - sim_med[:, None, None])

    if return_diagnostics:
        diagnostics = dict(gain=gain, row_median=row_median, row_noise_mad=row_noise_mad)
        if tpl_info is not None:
            diagnostics.update(template=tpl_info['template'], template_actuators=tpl_info['used'],
                               template_centroid=tpl_info['centroid'])
        if Q is not None:
            resid = M0 - amplitude_matrix @ S0
            diagnostics['tilt'] = resid @ T @ np.linalg.inv(T.T @ T)
        return amplitude_matrix, scaled_sim_ifs, diagnostics
    return amplitude_matrix, scaled_sim_ifs


def plot_results(meas_ifs: np.ndarray, sim_ifs: np.ndarray, pupil_center: tuple, pupil_radius: float, coords: np.ndarray):
    """Displays measured IFs, registered simulated IFs, and the Nact x Nact TPS amplitude matrix."""
    
    norm_meas = (meas_ifs - np.mean(meas_ifs[abs(meas_ifs)>0])) / np.max(np.abs(meas_ifs))
    norm_sim = (sim_ifs - np.mean(sim_ifs[abs(sim_ifs)>0]))/ np.max(np.abs(sim_ifs))
    diff = np.sqrt(np.sum((norm_meas-norm_sim)**2,axis=0))
    
    sum_meas = np.sum(np.abs(norm_meas)**3, axis=0)
    sum_sim = np.sum(np.abs(norm_sim)**3, axis=0)
    
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
        
    im0 = axes[0].imshow(sum_meas, cmap='magma', origin='lower',vmin=0,vmax=1.5)
    axes[0].set_title("Sum of cubed MEASURED iffs")
    axes[0].axis('off')
    fig.colorbar(im0, ax=axes[0], fraction=0.046, pad=0.04)
    
    im1 = axes[1].imshow(sum_sim, cmap='magma', origin='lower',vmin=0,vmax=1.5)
    axes[1].set_title("Sum of cubed SIMULATED iffs")
    axes[1].axis('off')
    fig.colorbar(im1, ax=axes[1], fraction=0.046, pad=0.04)

    im2 = axes[2].imshow(diff * (np.sum(abs(meas_ifs),axis=0)>0), cmap='RdBu', origin='lower')
    axes[2].plot(coords[:,1],coords[:,0],'.',markersize=2,c='gray',alpha=0.5,label='act coords')
    axes[2].set_title("Difference RSS")
    axes[2].axis('off')
    fig.colorbar(im2, ax=axes[2], fraction=0.046, pad=0.04)
    
    # Add pupil rims
    for ax in axes:
        circle = plt.Circle(pupil_center, pupil_radius, color='red', fill=False, linestyle='--', linewidth=1.5, alpha=0.8, label='Pupil Rim')
        ax.add_patch(circle)
        ax.legend(loc='upper right')
    
    plt.tight_layout()
    plt.show()


def register_ao_ifs(
    meas_ifs: np.ndarray, 
    meas_mask: np.ndarray, 
    sim_ifs: np.ndarray, 
    sim_mask: np.ndarray,
    n_outer_rings: int = 2,
    show_plots: bool = True,
    optimize_coords: bool = False,
    fit_if_template: bool = True,
    fit_tilt: bool = True
):
    print("\n--- Starting Simplified AO Influence Function Calibration ---")
    
    n_acts = sim_ifs.shape[0]
    pitch_sim = compute_actuator_pitch(sim_mask.shape, n_acts)
    
    meas_cy, meas_cx = meas_mask.shape[0] / 2.0, meas_mask.shape[1] / 2.0
    orig_pupil_center = (meas_cx, meas_cy)
    y_idx, x_idx = np.where(meas_mask > 0)
    orig_pupil_radius = np.max(np.sqrt((x_idx - meas_cx)**2 + (y_idx - meas_cy)**2)) if len(y_idx) > 0 else 0.0
    
    print("1. Estimating actuator peak positions on native grids [y, x]...")
    coords_meas = estimate_actuator_peaks(meas_ifs, meas_mask)
    coords_sim = estimate_actuator_peaks(sim_ifs, sim_mask)
    
    print(f"2. Identifying inner actuators (excluding outer {n_outer_rings} ring(s))...")
    inner_mask = identify_inner_actuators_radial(coords_sim, mask_shape=sim_mask.shape, pitch=pitch_sim, n_outer_rings=n_outer_rings)
    
    print("3. Computing global registration parameters (Scale, Rotation, Shift)...")
    scale, R, t_y_x = fit_similarity_transform(coords_sim[inner_mask], coords_meas[inner_mask])
    rotInDeg = np.arctan2(-R[0,1],R[0,0])*180/np.pi
    print(f'-> Scale: {scale:1.4f}, Rotation: {rotInDeg:1.1f}°, Shift [dx,dy]: [{t_y_x[0]:1.2f},{t_y_x[1]:1.2f}] pix')
    
    print("4. Registering simulated IFs and center coordinates...")
    target_shape = meas_ifs.shape[1:]
    sim_ifs_registered = apply_global_registration_to_ifs(sim_ifs, scale, R, t_y_x, target_shape)
    reg_coords_sim = transform_coordinates(coords_sim, scale, R, t_y_x)

    if show_plots:
        plot_results(meas_ifs, sim_ifs_registered, orig_pupil_center, orig_pupil_radius, reg_coords_sim)
    
    print("5. Computing TPS amplitude matrix across all degrees of freedom...")
    # amplitude_matrix, sim_ifs_registered_rescaled = fit_tps_amplitude_matrix(meas_ifs, meas_mask, reg_coords_sim)
    amplitude_matrix, scaled_sim_ifs = fit_amplitudes_to_sim_ifs(meas_ifs, meas_mask, sim_ifs_registered, optimize_coords=optimize_coords, fit_if_template=fit_if_template, fit_tilt=fit_tilt)
    
    print("--- Calibration Complete! ---\n")
    
    if show_plots:
        plot_results(meas_ifs, scaled_sim_ifs, orig_pupil_center, orig_pupil_radius, reg_coords_sim)
        plt.figure()
        plt.imshow(amplitude_matrix,cmap='magma',origin='upper')
        plt.colorbar()
        plt.title('Actuator displacements matrix')
        
    return scaled_sim_ifs, amplitude_matrix, reg_coords_sim

def reshape3d(ifs,mask):
    ifs2use = ifs.copy() if ifs.shape[0]<ifs.shape[1] else ifs.T
    mask2use = mask.astype(bool) if np.sum(mask) == ifs2use.shape[1] else (1-mask).astype(bool)

    ifs2use -= np.median(ifs2use,axis=1)[:,None]

    nActs = np.shape(ifs2use)[0]
    ifs2d = np.zeros([nActs,mask.shape[0],mask.shape[1]])
    for j in range(nActs):
        img = np.zeros(mask.shape).flatten()
        img[mask.astype(bool).flatten()] = ifs2use[j]
        ifs2d[j] = img.reshape(mask.shape)
    return ifs2d, mask2use