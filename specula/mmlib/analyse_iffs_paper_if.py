import numpy as np
from scipy.ndimage import shift, affine_transform, map_coordinates, spline_filter, distance_transform_edt
from scipy.optimize import minimize, least_squares
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


def _grid_axes_angle(coords: np.ndarray) -> float:
    """Orientation (rad, modulo 90 deg) of a square actuator grid from its nearest-neighbour vectors."""
    c = coords[~np.isnan(coords[:, 0])]
    d = np.hypot(c[:, None, 0] - c[None, :, 0], c[:, None, 1] - c[None, :, 1])
    np.fill_diagonal(d, np.inf)
    v = c[np.argmin(d, axis=1)] - c
    ang = np.arctan2(v[:, 0], v[:, 1])
    return float(np.angle(np.mean(np.exp(4j * ang))) / 4.0)


def _fit_regular_grid(coords_ref: np.ndarray, meas_pos: np.ndarray, use: np.ndarray, n_iter: int = 5):
    """
    Fits a regular Cartesian actuator grid (pitch, angle, offset) to measured actuator positions,
    as in Berdeu et al. (2026), Sec. 2.1. Integer grid indices are assigned from coords_ref (the
    registered simulated positions, which carry the grid topology); the grid is then fitted to the
    measured positions meas_pos of the actuators flagged in `use`, rejecting outliers (> 0.3 pitch)
    and refitting n_iter times. Returns a dict with 'positions' (grid position of EVERY actuator,
    [y, x]), 'indices' (integer grid indices [i, j], NaN if unknown), 'pitch', 'R' (2x2, index
    frame -> image frame), 't', 'angle_deg' (direction of the
    grid's x axis in the image, counter-clockwise from +x), 'rms' (pix), 'n_used'.
    """
    ok = ~np.isnan(coords_ref[:, 0])
    theta = _grid_axes_angle(coords_ref)
    c, s_ = np.cos(theta), np.sin(theta)
    xr = (c * coords_ref[:, 1] + s_ * coords_ref[:, 0])[ok]
    yr = (-s_ * coords_ref[:, 1] + c * coords_ref[:, 0])[ok]

    # Pitch and grid phase from the coherence of the de-rotated coordinates: a nearest-neighbour
    # distance is biased low by position noise, and a small pitch error accumulates across the DM
    # into wrong integer indices.
    pitch0 = _nearest_neighbour_pitch(coords_ref)
    ps = np.linspace(0.85 * pitch0, 1.2 * pitch0, 1401)
    ex = np.exp(2j * np.pi * xr[None, :] / ps[:, None]).mean(axis=1)
    ey = np.exp(2j * np.pi * yr[None, :] / ps[:, None]).mean(axis=1)
    k = np.argmax(np.abs(ex)**2 + np.abs(ey)**2)
    pitch_c = ps[k]
    off_x = np.angle(ex[k]) / (2 * np.pi) * pitch_c
    off_y = np.angle(ey[k]) / (2 * np.pi) * pitch_c
    idx = np.full((len(coords_ref), 2), np.nan)
    idx[ok, 0] = np.round((yr - off_y) / pitch_c)
    idx[ok, 1] = np.round((xr - off_x) / pitch_c)

    sel = use & ok & ~np.isnan(meas_pos[:, 0])
    for _ in range(n_iter):
        scale, R, t = fit_similarity_transform(idx[sel], meas_pos[sel])
        positions = np.full_like(coords_ref, np.nan)
        positions[ok] = transform_coordinates(idx[ok], scale, R, t)
        res = np.full(len(coords_ref), np.inf)
        both = ok & ~np.isnan(meas_pos[:, 0])
        res[both] = np.hypot(*(positions[both] - meas_pos[both]).T)
        new_sel = use & both & (res < 0.3 * scale)
        if new_sel.sum() < 10 or np.array_equal(new_sel, sel):
            break
        sel = new_sel
    rms = float(np.sqrt(np.mean(res[sel]**2)))
    return dict(positions=positions, indices=idx, pitch=float(scale), R=R, t=t,
                angle_deg=float(np.degrees(np.arctan2(R[0, 1], R[1, 1]))), rms=rms, n_used=int(sel.sum()))


def _sample_frame(img: np.ndarray, Y: np.ndarray, X: np.ndarray, order: int = 3, margin: int = 6) -> np.ndarray:
    """Interpolates img at frame coordinates (Y, X), working on a crop around them (zero outside the frame)."""
    h, w = img.shape
    y0, y1 = int(np.floor(Y.min())) - margin, int(np.ceil(Y.max())) + margin + 1
    x0, x1 = int(np.floor(X.min())) - margin, int(np.ceil(X.max())) + margin + 1
    ya, yb, xa, xb = max(0, y0), min(h, y1), max(0, x0), min(w, x1)
    if ya >= yb or xa >= xb:
        return np.zeros(Y.shape)
    return map_coordinates(img[ya:yb, xa:xb], [Y - ya, X - xa], order=order, mode='constant', cval=0.0)


def _eval_fine_template(tf_coeff: np.ndarray, dy, dx, pitch: float, R: np.ndarray, superres: int) -> np.ndarray:
    """Evaluates the super-resolved template (cubic-spline coefficients, actuator-grid frame, `superres`
    samples per pitch, centred) at image-frame offsets (dy, dx) from the actuator position."""
    half_f = tf_coeff.shape[0] // 2
    uy = (R[0, 0] * dy + R[1, 0] * dx) / pitch          # u = R^T d / pitch
    ux = (R[0, 1] * dy + R[1, 1] * dx) / pitch
    return map_coordinates(tf_coeff, [uy * superres + half_f, ux * superres + half_f],
                           order=3, mode='constant', cval=0.0, prefilter=False)


def _fit_super_gaussian(Yp, Xp, vals, x0, pitch, prev=None, fix_pos=False):
    """
    Eq. 1 of Berdeu et al. (2026): vals ~ rho * exp(-(r / alpha)**beta) + gamma, with r the distance
    (in actuator pitches) to the actuator position, fitted as well unless fix_pos (then x0 is used).
    The position is bounded to +/- 0.5 pitch around x0. Returns [rho, alpha, beta, gamma, y, x].
    """
    if prev is not None and np.all(np.isfinite(prev)):
        p0 = np.array(prev, dtype=float)
    else:
        k = np.argmin((Yp - x0[0])**2 + (Xp - x0[1])**2)
        g0 = np.median(vals)
        p0 = np.array([vals[k] - g0, 1.0, 2.0, g0, x0[0], x0[1]])
    lo_pos, hi_pos = np.asarray(x0) - 0.5 * pitch, np.asarray(x0) + 0.5 * pitch
    p0[4:6] = np.clip(p0[4:6], lo_pos + 1e-6, hi_pos - 1e-6) if not fix_pos else x0
    p0[1] = np.clip(p0[1], 0.051, 4.99)
    p0[2] = np.clip(p0[2], 0.51, 3.99)

    def model(q, cy, cx):
        r = np.hypot(Yp - cy, Xp - cx) / pitch
        return q[0] * np.exp(-(r / q[1])**q[2]) + q[3]

    if fix_pos:
        res = least_squares(lambda q: model(q, x0[0], x0[1]) - vals, p0[:4],
                            bounds=([-np.inf, 0.05, 0.5, -np.inf], [np.inf, 5.0, 4.0, np.inf]),
                            x_scale='jac', max_nfev=200)
        return np.concatenate([res.x, x0])
    res = least_squares(lambda q: model(q, q[4], q[5]) - vals, p0,
                        bounds=([-np.inf, 0.05, 0.5, -np.inf, lo_pos[0], lo_pos[1]],
                                [np.inf, 5.0, 4.0, np.inf, hi_pos[0], hi_pos[1]]),
                        x_scale='jac', max_nfev=300)
    return res.x


def _build_empirical_2d_if(
    meas_ifs: np.ndarray,
    valid_mask: np.ndarray,
    coords_reg: np.ndarray,
    radius_pitches: float = 7.5,
    superres: int = 10,
    symmetrize: bool = True,
    n_grid_iter: int = 5,
    sim_ifs: np.ndarray = None,
):
    """
    Empirical 2D influence function, following Berdeu et al. (2026), Sec. 2.1, as closely as the
    data allow:

      1. Eq. 1: super-Gaussian fit rho*exp(-(r/alpha)^beta) + gamma of every actuator over the disc
         Omega of radius_pitches (valid pixels only) centred on its position on a regular Cartesian
         grid; actuators whose grid position falls outside the measured pupil have their position
         forced to the grid. The grid (pitch, angle, offset) is then refitted to the super-Gaussian
         positions of the actuators inside the pupil, and the two steps are repeated n_grid_iter
         times (5 in the paper). The initial grid comes from the registered simulated positions.
      2. Eq. 2: every measured IF, offset gamma_a removed and normalised by rho_a, is re-interpolated
         on a super-resolved grid (superres samples per pitch) in the de-rotated actuator-grid
         frame, centred on its grid position, and the cube is AVERAGED over the actuators (samples
         outside the pupil are ignored).
      3. (symmetrize) the 8 flipped/transposed versions are averaged (square-grid symmetry).
      4. Eq. 3: amplitude rho_2D and offset gamma_2D of every actuator are fitted over Omega with
         this single 2D IF placed at its grid position.

    The 2D IF has the normalisation of the paper (unit super-Gaussian amplitude), so the gains are in
    the units of the measured IFs. Only actuators whose grid position lies inside the measured pupil
    enter the average (a safeguard: a partially seen IF has a poorly constrained rho_a), and
    actuators whose rho_a has the wrong sign or is below 20 % of the median are skipped.

    meas_ifs are used as given (the per-actuator offset gamma absorbs the local pedestal, as in the
    paper). If sim_ifs (registered simulated / TPS IFs) is given, their average is built on the same
    grid for comparison: each is centred on its own position coords_reg, normalised by its peak,
    sampled only where the measured pupil is valid, averaged over the same actuators, symmetrised
    like the 2D IF, and finally scaled by the least-squares factor that best matches the 2D IF.

    Returns None if fewer than 10 actuators are usable, else a dict with 'basis' (n_acts, H, W: the
    2D IF at every grid position), 'template' (the super-resolved 2D IF, actuator-grid frame),
    'superres', 'radius_pitches', 'positions' (grid), 'grid' (fit summary), 'rho_2d', 'gamma_2d'
    (Eq. 3), 'sg' (dict of the Eq. 1 maps: rho, alpha, beta, gamma, positions), 'used' (stacked
    actuators) and, with sim_ifs, 'template_tps' (scaled TPS average) and 'tps_scale'.
    """
    n_acts, h, w = meas_ifs.shape
    ok = ~np.isnan(coords_reg[:, 0])
    if ok.sum() < 10:
        return None

    def in_fov(p_):
        cy, cx = int(round(p_[0])), int(round(p_[1]))
        return 0 <= cy < h and 0 <= cx < w and bool(valid_mask[cy, cx])

    def disc_pixels(c, r_pix):
        rb = int(np.ceil(r_pix)) + 1
        cy, cx = int(round(c[0])), int(round(c[1]))
        ya, yb, xa, xb = max(0, cy - rb), min(h, cy + rb + 1), max(0, cx - rb), min(w, cx + rb + 1)
        if ya >= yb or xa >= xb:
            return None, None
        Yp, Xp = np.mgrid[ya:yb, xa:xb]
        m = valid_mask[ya:yb, xa:xb] & ((Yp - c[0])**2 + (Xp - c[1])**2 <= r_pix**2)
        return Yp[m], Xp[m]

    # Initial regular grid from the registered simulated positions
    grid = _fit_regular_grid(coords_reg, coords_reg, ok)

    # 1. Super-Gaussian fits (Eq. 1) alternated with regular-grid refits
    sg = np.full((n_acts, 6), np.nan)
    for _ in range(n_grid_iter):
        pos, pitch = grid['positions'], grid['pitch']
        r_pix = radius_pitches * pitch
        fov = np.array([ok[a] and in_fov(pos[a]) for a in range(n_acts)])
        for a in np.where(ok)[0]:
            Yp, Xp = disc_pixels(pos[a], r_pix)
            if Yp is None or len(Yp) < 20:
                sg[a] = np.nan
                continue
            prev = sg[a] if np.all(np.isfinite(sg[a])) else None
            sg[a] = _fit_super_gaussian(Yp, Xp, meas_ifs[a][Yp, Xp], pos[a], pitch, prev, fix_pos=not fov[a])
        fitted = fov & np.isfinite(sg[:, 0])
        if fitted.sum() < 10:
            return None
        sg_pos = np.where(fitted[:, None], sg[:, 4:6], np.nan)
        grid = _fit_regular_grid(coords_reg, sg_pos, fitted)
    pos, pitch, R = grid['positions'], grid['pitch'], grid['R']
    r_pix = radius_pitches * pitch
    fov = np.array([ok[a] and in_fov(pos[a]) for a in range(n_acts)])

    # 2. Normalised IFs on the super-resolved, de-rotated grid, averaged (Eq. 2)
    half_f = int(np.ceil(radius_pitches * superres))
    k = np.arange(-half_f, half_f + 1) / superres
    uy, ux = np.meshgrid(k, k, indexing='ij')
    in_disc = np.hypot(uy, ux) <= radius_pitches
    dyf = pitch * (R[0, 0] * uy + R[0, 1] * ux)
    dxf = pitch * (R[1, 0] * uy + R[1, 1] * ux)
    vm = valid_mask.astype(float)

    rho_sg = sg[:, 0]
    med_rho = np.nanmedian(rho_sg[fov])
    used = np.where(fov & np.isfinite(rho_sg) & (rho_sg * np.sign(med_rho) > 0.2 * abs(med_rho)))[0]
    if len(used) < 10:
        return None
    acc = np.zeros(uy.shape)
    cnt = np.zeros(uy.shape)
    for a in used:
        Y, X = pos[a, 0] + dyf, pos[a, 1] + dxf
        vals = (_sample_frame(meas_ifs[a], Y, X, order=3) - sg[a, 3]) / rho_sg[a]
        ok_s = in_disc & (_sample_frame(vm, Y, X, order=1) >= 0.999)
        acc[ok_s] += vals[ok_s]
        cnt[ok_s] += 1
    tf = np.where(cnt > 0, acc / np.maximum(cnt, 1), 0.0)

    # 3. Symmetrisation: average of the 8 flipped/transposed versions
    if symmetrize:
        v8 = [tf, tf[::-1], tf[:, ::-1], tf[::-1, ::-1]]
        tf = np.mean(v8 + [v.T for v in v8], axis=0)
    tf[~in_disc] = 0.0
    coeff = spline_filter(tf, order=3)

    # Average simulated (TPS) IF on the same grid, for comparison only
    tps, tps_scale = None, np.nan
    if sim_ifs is not None:
        acc_t = np.zeros(uy.shape)
        cnt_t = np.zeros(uy.shape)
        for a in used:
            c = coords_reg[a]
            pk = _sample_frame(sim_ifs[a], np.array([c[0]]), np.array([c[1]]), order=3)[0]
            if abs(pk) < 1e-12:
                continue
            Y, X = c[0] + dyf, c[1] + dxf
            vals = _sample_frame(sim_ifs[a], Y, X, order=3) / pk
            ok_s = in_disc & (_sample_frame(vm, Y, X, order=1) >= 0.999)
            acc_t[ok_s] += vals[ok_s]
            cnt_t[ok_s] += 1
        tps = np.where(cnt_t > 0, acc_t / np.maximum(cnt_t, 1), 0.0)
        if symmetrize:
            v8 = [tps, tps[::-1], tps[:, ::-1], tps[::-1, ::-1]]
            tps = np.mean(v8 + [v.T for v in v8], axis=0)
        tps[~in_disc] = 0.0
        tps_scale = float(np.sum(tps * tf) / max(np.sum(tps * tps), 1e-30))
        tps = tps * tps_scale

    # 4. Eq. 3: amplitude and offset of every actuator with the single 2D IF
    rho_2d = np.full(n_acts, np.nan)
    gamma_2d = np.full(n_acts, np.nan)
    for a in np.where(ok)[0]:
        Yp, Xp = disc_pixels(pos[a], r_pix)
        if Yp is None or len(Yp) < 10:
            continue
        m = _eval_fine_template(coeff, Yp - pos[a, 0], Xp - pos[a, 1], pitch, R, superres)
        sol, *_ = np.linalg.lstsq(np.stack([m, np.ones_like(m)], axis=1), meas_ifs[a][Yp, Xp], rcond=None)
        rho_2d[a], gamma_2d[a] = sol

    # Basis: the 2D IF at every actuator's grid position (zero beyond Omega)
    basis = np.zeros((n_acts, h, w))
    rb = int(np.ceil(r_pix)) + 1
    for a in np.where(ok)[0]:
        cy, cx = int(round(pos[a, 0])), int(round(pos[a, 1]))
        ya, yb, xa, xb = max(0, cy - rb), min(h, cy + rb + 1), max(0, cx - rb), min(w, cx + rb + 1)
        if ya >= yb or xa >= xb:
            continue
        Yp, Xp = np.mgrid[ya:yb, xa:xb]
        basis[a, ya:yb, xa:xb] = _eval_fine_template(coeff, Yp - pos[a, 0], Xp - pos[a, 1], pitch, R, superres)

    sg_maps = dict(rho=sg[:, 0], alpha=sg[:, 1], beta=sg[:, 2], gamma=sg[:, 3], positions=sg[:, 4:6])
    return dict(basis=basis, template=tf, superres=superres, radius_pitches=radius_pitches,
                positions=pos, grid=grid, rho_2d=rho_2d, gamma_2d=gamma_2d, sg=sg_maps, used=used,
                template_tps=tps, tps_scale=tps_scale)


def _actuator_guard(indices: np.ndarray, guard_width: int = 2) -> np.ndarray:
    """Guard zone of Berdeu et al. (2026), Sec. 2.2: actuators lying within guard_width actuators of
    the edge of the DM actuator layout (distance, in grid steps, to the nearest empty grid site)."""
    ok = ~np.isnan(indices[:, 0])
    guard = np.ones(len(indices), dtype=bool)
    ij = indices[ok].astype(int)
    pad = guard_width + 1
    lo = ij.min(axis=0)
    occ = np.zeros(tuple(ij.max(axis=0) - lo + 1 + 2 * pad), dtype=bool)
    occ[ij[:, 0] - lo[0] + pad, ij[:, 1] - lo[1] + pad] = True
    d = distance_transform_edt(occ)
    guard[ok] = d[ij[:, 0] - lo[0] + pad, ij[:, 1] - lo[1] + pad] <= guard_width
    return guard


def _fit_paper_coupling(
    meas_pix: np.ndarray,
    basis_pix: np.ndarray,
    indices: np.ndarray,
    guard: np.ndarray,
    nuisance: np.ndarray,
    radius: float = 2.5,
    method: str = 'decoupling',
    tol: float = 1e-4,
    max_iter: int = 100,
):
    """
    (De)coupling matrix of Berdeu et al. (2026), Sec. 2.2, fitted directly on measured IFs (pixels of
    the valid pupil) instead of slopes: the measured IFs play the role of the IM I and the basis IFs
    (empirical 2D IF or TPS, placed at every actuator) that of the synthetic IM I0^2D.

      - Neighbours N_a: actuators within `radius` grid steps of a (a included); 21 for 2.5.
      - Guard (boolean per actuator): if a is in the guard, all its coefficients follow the current
        median (de)coupling map with one free factor mu0; otherwise its neighbours in the guard follow
        the map (with mu0) and the other neighbours are free (Eqs. 10-13).
      - Pixels S_a: valid pixels where at least one neighbour's basis IF (coupling) or measured IF
        (decoupling) is non-zero (Eq. 9).
      - 'coupling', Eqs. 10-11: measured IF a ~ sum_a' C[a, a'] basis a'.
        'decoupling', Eqs. 12-13: basis a ~ sum_a' D[a, a'] measured IF a', then C = D^-1 (Eq. 8).
      - The median map starts as a centred Dirac; after the first pass over all actuators it is
        recomputed as the median of the fitted coefficient maps (pairs of actuators both outside
        the guard; guard coefficients are only copies of the map), and the actuators having guard
        neighbours (or being in the guard) are refitted, until the map changes by less than tol
        (relative to its centre value).
    One addition to the paper's equations: `nuisance` columns (an offset, plus tip/tilt if requested)
    are fitted jointly in every actuator fit and discarded. They play the role of the slope operator,
    which is blind to piston: here the data are phase maps whose piston is not measured.

    Returns a dict with 'A' (coupling matrix, row a = measured IF a in terms of the basis), 'W' (the
    fitted matrix: C for 'coupling', D for 'decoupling'), 'map' ((2K+1)^2 median map of W used for
    the guard, NaN outside the radius), 'map_C' (median map of the final coupling matrix C, as in
    Fig. 8), 'n_iter', 'n_refit' (actuators refitted in each iteration).
    """
    n = len(indices)
    ok = ~np.isnan(indices[:, 0])
    K = int(np.floor(radius))
    offs = [(di, dj) for di in range(-K, K + 1) for dj in range(-K, K + 1) if di * di + dj * dj <= radius**2 + 1e-9]
    lut = {(int(i), int(j)): a for a, (i, j) in enumerate(indices) if ok[a]}
    nbr = []
    for a in range(n):
        if not ok[a]:
            nbr.append([(a, (0, 0))])
            continue
        i, j = int(indices[a, 0]), int(indices[a, 1])
        nbr.append([(lut[(i + di, j + dj)], (di, dj)) for (di, dj) in offs if (i + di, j + dj) in lut])
    # Pairs (a, a') per offset used for the median map: only coefficients that are actually fitted
    # (a and a' outside the guard). Guard coefficients are copies of the current map; including
    # them would make the map partly self-referential (it would stick to its initial Dirac).
    pairs = {d: ([], []) for d in offs}
    for a in range(n):
        if ok[a] and not guard[a]:
            for b, d in nbr[a]:
                if not guard[b]:
                    pairs[d][0].append(a)
                    pairs[d][1].append(b)
    pairs = {d: (np.array(v[0], dtype=int), np.array(v[1], dtype=int)) for d, v in pairs.items()}

    if method == 'coupling':
        regress, target = basis_pix, meas_pix
        support = np.abs(basis_pix) > 1e-6 * np.max(np.abs(basis_pix), axis=1, keepdims=True)
    elif method == 'decoupling':
        regress, target = meas_pix, basis_pix
        support = meas_pix != 0
    else:
        raise ValueError("method must be 'coupling' or 'decoupling'")

    W = np.zeros((n, n))
    cmap = {d: (1.0 if d == (0, 0) else 0.0) for d in offs}

    def fit_one(a):
        ids = [b for b, _ in nbr[a]]
        if guard[a]:
            free, bound = [], nbr[a]
        else:
            free = [(b, d) for b, d in nbr[a] if not guard[b]]
            bound = [(b, d) for b, d in nbr[a] if guard[b]]
        S = np.any(support[ids], axis=0)
        cols = [regress[b][S] for b, _ in free]
        if bound:
            cols.append(sum(cmap[d] * regress[b][S] for b, d in bound))
        X = np.column_stack(cols + [nuisance[S]])
        sol, *_ = np.linalg.lstsq(X, target[a][S], rcond=None)
        W[a, :] = 0.0
        for k, (b, _) in enumerate(free):
            W[a, b] = sol[k]
        if bound:
            mu0 = sol[len(free)]
            for b, d in bound:
                W[a, b] = mu0 * cmap[d]

    def median_map():
        return {d: (float(np.median(W[pairs[d][0], pairs[d][1]])) if len(pairs[d][0]) else 0.0) for d in offs}

    for a in range(n):
        fit_one(a)
    needs = [a for a in range(n) if guard[a] or any(guard[b] for b, _ in nbr[a])]
    n_iter, n_refit = 0, []
    for n_iter in range(1, max_iter + 1):
        new = median_map()
        change = max(abs(new[d] - cmap[d]) for d in offs) / max(abs(new[(0, 0)]), 1e-30)
        cmap = new
        if change < tol:
            break
        for a in needs:
            fit_one(a)
        n_refit.append(len(needs))

    A = W if method == 'coupling' else np.linalg.inv(W)

    def to_arr(dct):
        arr = np.full((2 * K + 1, 2 * K + 1), np.nan)
        for (di, dj), v in dct.items():
            arr[di + K, dj + K] = v
        return arr
    # Median map of the final coupling matrix C (= A), as displayed in Fig. 8 for both methods
    map_C = {d: (float(np.median(A[pairs[d][0], pairs[d][1]])) if len(pairs[d][0]) else np.nan) for d in offs}
    return dict(A=A, W=W, map=to_arr(cmap), map_C=to_arr(map_C), n_iter=n_iter, n_refit=n_refit)


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
    template_radius_pitches: float = 7.5,
    template_superres: int = 10,
    template_symmetrize: bool = True,
    template_grid_iter: int = 5,
    fit_tilt: bool = True,
    coupling_method: str = 'decoupling',
    coupling_radius_pitches: float = 2.5,
    guard_width: int = 2,
    return_diagnostics: bool = False
):
    """
    Fits measured IFs to registered simulated IFs.

    Steps:
      0. (fit_if_template) the simulated IF shape is replaced by the empirical 2D IF of Berdeu et
         al. (2026), Sec. 2.1 (see _build_empirical_2d_if): super-Gaussian fits alternated with
         a regular-grid fit (template_grid_iter times), MEAN of the gain-normalised measured IFs on
         a super-resolved de-rotated grid (template_superres samples per pitch, disc of
         template_radius_pitches), symmetrised (template_symmetrize), and placed at the fitted grid
         position of every actuator. As in the paper, the 2D IF has unit super-Gaussian amplitude,
         so the gains (diagonal) are in the units of the measured IFs, and the off-diagonal terms
         are coupling/shape deviations relative to that common IF.
      1. (optimize_coords) per-actuator sub-pixel shift refinement.
      2. Amplitude / coupling matrix, with the same procedure whatever the basis (empirical 2D IF
         or TPS / simulated IFs), so that both models can be compared:
         - coupling_method='decoupling' (default, the paper's preferred solution) or 'coupling':
           neighbour-limited (de)coupling matrix of Berdeu et al. (2026), Sec. 2.2 (Eqs. 8-13), see
           _fit_paper_coupling: coupling_radius_pitches (r^n; 0 = gains only), guard of guard_width
           actuators at the edge of the actuator layout tied to the median (de)coupling map.
           Actuators whose grid position falls outside the measured pupil are also put in the guard
           (they cannot be fitted freely). An offset (and tip/tilt if fit_tilt) is fitted jointly in
           every actuator fit and discarded.
         - coupling_method='dense': previous global ridge fit over all actuators (median and, if
           fit_tilt, tip/tilt projected out, row median removed afterwards).
      The output IFs are the complete basis combined with the fitted matrix; only their piston (median
      over the valid pixels) is removed, since it is not measured. Tip/tilt are kept.

    Returns:
        amplitude_matrix: (N_acts, N_acts) array of coupling weights.
        scaled_sim_ifs: (N_acts, H, W) array of the final reconstructed IFs, built from the complete
            basis, median (piston) over the valid pixels removed; tip/tilt kept.
        diagnostics (only if return_diagnostics): dict with 'gain' (diagonal), and, if the empirical
            2D IF was used, 'template' (super-resolved IF, actuator-grid frame), 'template_superres',
            'template_positions' (fitted grid positions), 'template_grid' (pitch, angle_deg, rms, ...),
            'template_rho_2d' / 'template_gamma_2d' (Eq. 3: gain and offset of each actuator with that
            IF, no coupling), 'template_sg' (Eq. 1 maps: rho, alpha, beta, gamma, positions),
            'template_actuators' (actuators in the average), 'template_tps' (average registered
            simulated/TPS IF on the same grid, scaled to best match the 2D IF) and
            'template_tps_scale' (that scale factor);
            for the paper methods: 'coupling_method', 'coupling_map' (median map of the coupling
            matrix C, (2K+1)^2 in grid steps, as in Fig. 8), 'guard' (bool per actuator),
            'grid_indices', 'coupling_iterations', and for decoupling 'decoupling_matrix' (D) and
            'decoupling_map' (median map of D used for the guard);
            for 'dense': 'row_median' (offset removed from each row), 'row_noise_mad' (robust rms of
            each row's entries) and, if fit_tilt, 'tilt' (n_acts x 2, fitted [tip, tilt] per measured
            IF in data units per pupil radius).
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

    # --- Phase 0: replace the simulated IF shape by the empirical 2D IF of the paper (optional) ---
    tpl_info = None
    if fit_if_template:
        print(f"Building empirical 2D IF (Berdeu et al. 2026): {template_grid_iter} super-Gaussian/grid "
              f"iterations, mean stack, {template_superres} samples/pitch, radius {template_radius_pitches} pitches...")
        # The 2D IF is built from the measured IFs as given (per-actuator offset gamma, as in the paper):
        # piston/tip/tilt are removed only inside the amplitude fit, never from the IF model or the output.
        coords_reg = estimate_actuator_peaks(sim_ifs_registered)   # registered simulated positions (initial grid)
        tpl_info = _build_empirical_2d_if(meas_ifs, valid_mask, coords_reg,
                                          radius_pitches=template_radius_pitches, superres=template_superres,
                                          symmetrize=template_symmetrize, n_grid_iter=template_grid_iter,
                                          sim_ifs=sim_ifs_registered)
        if tpl_info is None:
            print("   WARNING: too few usable actuators for the 2D IF; keeping the simulated IF shapes.")
        else:
            gr, sgm = tpl_info['grid'], tpl_info['sg']
            print(f"-> grid: pitch {gr['pitch']:.3f} pix, angle {gr['angle_deg']:.2f} deg, rms residual of the "
                  f"super-Gaussian positions {gr['rms']:.3f} pix ({gr['n_used']} actuators); super-Gaussian "
                  f"alpha {np.nanmedian(sgm['alpha']):.3f}, beta {np.nanmedian(sgm['beta']):.3f} (medians); "
                  f"average of {len(tpl_info['used'])} actuators")
            if gr['rms'] > 0.25 * gr['pitch']:
                print("   WARNING: large grid residual; the actuators may not lie on a regular square grid.")
            updated_sim_ifs = tpl_info['basis']

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
    diagnostics = {}
    if coupling_method == 'dense':
        print("Solving for amplitude/coupling matrix (dense)...")
        amplitude_matrix, _, sim_med, S0 = _solve_amplitudes(updated_sim_ifs[:, valid_mask], M_proj, Q, reg_param)
        # Remove the common-mode offset of each row (median: robust to the actuator peaks)
        row_median = np.median(amplitude_matrix, axis=1)
        amplitude_matrix = amplitude_matrix - row_median[:, None]
        diagnostics.update(row_median=row_median,
                           row_noise_mad=1.4826 * np.median(np.abs(amplitude_matrix), axis=1))
        if Q is not None:
            resid = M0 - amplitude_matrix @ S0
            diagnostics['tilt'] = resid @ T @ np.linalg.inv(T.T @ T)
    else:
        # Actuator grid (indices and positions): the 2D IF's grid, or one fitted to the registered TPS
        if tpl_info is not None:
            gr = tpl_info['grid']
        else:
            coords_reg = estimate_actuator_peaks(sim_ifs_registered)
            gr = _fit_regular_grid(coords_reg, coords_reg, ~np.isnan(coords_reg[:, 0]))
        indices, positions = gr['indices'], gr['positions']
        guard = _actuator_guard(indices, guard_width)
        outside = np.array([np.isnan(p_[0]) or not (0 <= int(round(p_[0])) < h and 0 <= int(round(p_[1])) < w
                                                   and valid_mask[int(round(p_[0])), int(round(p_[1]))])
                            for p_ in positions])
        guard = guard | outside
        nuis = [np.ones(int(valid_mask.sum()))]
        if fit_tilt:
            nuis += [T[:, 0], T[:, 1]] if T is not None else []
        nuisance = np.stack(nuis, axis=1)
        print(f"Fitting the {coupling_method} matrix (Berdeu et al. 2026, Sec. 2.2): r^n = "
              f"{coupling_radius_pitches}, guard {guard_width} actuators ({guard.sum()} of {n_acts} actuators in "
              f"the guard, {outside.sum()} of them outside the measured pupil)...")
        cpl = _fit_paper_coupling(meas_ifs[:, valid_mask].astype(float), updated_sim_ifs[:, valid_mask],
                                  indices, guard, nuisance, radius=coupling_radius_pitches, method=coupling_method)
        amplitude_matrix = cpl['A']
        print(f"-> median map converged in {cpl['n_iter']} iteration(s)")
        diagnostics.update(coupling_method=coupling_method, coupling_map=cpl['map_C'], guard=guard,
                           grid_indices=indices, coupling_iterations=cpl['n_iter'])
        if coupling_method == 'decoupling':
            diagnostics.update(decoupling_matrix=cpl['W'], decoupling_map=cpl['map'])

    gain = np.diag(amplitude_matrix)
    med_g = np.median(gain)
    print(f"Gain (diag): median = {med_g:.4g}, robust spread (MAD/median) = "
          f"{np.median(np.abs(gain - med_g)) / abs(med_g):.3f}")

    # Output IFs: complete basis combined with the fitted matrix (piston/tip/tilt were removed only
    # inside the fit); then only the piston (median over the valid pixels) is removed, as it is not
    # measured. Tip/tilt are kept.
    scaled_sim_ifs = np.einsum('ij,jhw->ihw', amplitude_matrix, updated_sim_ifs)
    scaled_sim_ifs -= np.median(scaled_sim_ifs[:, valid_mask], axis=1)[:, None, None]

    if return_diagnostics:
        diagnostics['gain'] = gain
        if tpl_info is not None:
            diagnostics.update(template=tpl_info['template'], template_superres=tpl_info['superres'],
                               template_positions=tpl_info['positions'], template_grid=tpl_info['grid'],
                               template_rho_2d=tpl_info['rho_2d'], template_gamma_2d=tpl_info['gamma_2d'],
                               template_sg=tpl_info['sg'], template_actuators=tpl_info['used'],
                               template_tps=tpl_info['template_tps'], template_tps_scale=tpl_info['tps_scale'])
        return amplitude_matrix, scaled_sim_ifs, diagnostics
    return amplitude_matrix, scaled_sim_ifs


def plot_coupling_map(cmap: np.ndarray, title: str = 'Median coupling map'):
    """Shows a median coupling map (diagnostics['coupling_map']) normalised by its centre value,
    in %, with the values written in each cell (as in Fig. 8 of Berdeu et al. 2026)."""
    K = cmap.shape[0] // 2
    m = 100.0 * cmap / cmap[K, K]
    off = m.copy()
    off[K, K] = np.nan
    lim = max(np.nanmax(np.abs(off)), 1e-6)
    fig, ax = plt.subplots(figsize=(5, 4.5))
    im = ax.imshow(np.where(np.isnan(m), np.nan, np.clip(m, -lim, lim)), cmap='RdBu_r', vmin=-lim, vmax=lim,
                   origin='lower', extent=[-K - 0.5, K + 0.5, -K - 0.5, K + 0.5])
    for i in range(2 * K + 1):
        for j in range(2 * K + 1):
            if np.isfinite(m[i, j]):
                ax.text(j - K, i - K, f'{m[i, j]:.1f}', ha='center', va='center', fontsize=8)
    ax.set_xlabel('x [grid steps]')
    ax.set_ylabel('y [grid steps]')
    ax.set_title(title)
    fig.colorbar(im, ax=ax, label='% of centre')
    plt.tight_layout()
    plt.show()


def plot_average_if(template: np.ndarray, superres: int, template_tps: np.ndarray = None,
                    title: str = 'Average fitted influence function'):
    """
    Shows the super-resolved average (empirical 2D) influence function and, if template_tps is given,
    the average TPS / simulated IF on the same grid and their difference: 2D maps (top row) and cuts
    through the actuator centre along the actuator-grid x and y axes (bottom row). Axes are in
    actuator pitches. Both inputs are those returned in the diagnostics ('template',
    'template_superres', 'template_tps'). Note: with template_symmetrize=True the x and y cuts are
    identical by construction.
    """
    hf = template.shape[0] // 2
    k = np.arange(-hf, hf + 1) / superres
    ext = [k[0], k[-1], k[0], k[-1]]
    has_tps = template_tps is not None
    fig, axes = plt.subplots(2, 3 if has_tps else 2, figsize=(17 if has_tps else 11, 9))
    vmax = np.max(np.abs(template))

    im = axes[0, 0].imshow(template, origin='lower', extent=ext, cmap='magma', vmin=min(template.min(), 0), vmax=vmax)
    axes[0, 0].set_title('Fitted (empirical 2D) IF')
    fig.colorbar(im, ax=axes[0, 0], fraction=0.046, pad=0.04)
    if has_tps:
        diff = template - template_tps
        im = axes[0, 1].imshow(template_tps, origin='lower', extent=ext, cmap='magma', vmin=min(template.min(), 0), vmax=vmax)
        axes[0, 1].set_title('TPS / simulated IF (scaled)')
        fig.colorbar(im, ax=axes[0, 1], fraction=0.046, pad=0.04)
        dl = max(np.max(np.abs(diff)), 1e-30)
        im = axes[0, 2].imshow(diff, origin='lower', extent=ext, cmap='RdBu_r', vmin=-dl, vmax=dl)
        rms = np.sqrt(np.mean(diff[np.hypot(*np.meshgrid(k, k)) <= k[-1]]**2))
        axes[0, 2].set_title(f'Fitted - TPS (rms {rms:.3g}, {100 * rms / vmax:.2f}% of peak)')
        fig.colorbar(im, ax=axes[0, 2], fraction=0.046, pad=0.04)
    for ax in axes[0]:
        ax.set_xlabel('x [pitch]')
        ax.set_ylabel('y [pitch]')

    for col, (name, cut) in enumerate([('x', lambda m: m[hf, :]), ('y', lambda m: m[:, hf])]):
        ax = axes[1, col]
        ax.plot(k, cut(template), label='fitted')
        if has_tps:
            ax.plot(k, cut(template_tps), '--', label='TPS (scaled)')
        ax.axhline(0, color='gray', lw=0.5)
        for g in range(-int(k[-1]), int(k[-1]) + 1):
            ax.axvline(g, color='gray', lw=0.3, alpha=0.5)
        ax.set_title(f'Profile along {name} through the actuator')
        ax.set_xlabel(f'{name} [pitch]')
        ax.legend()
    if has_tps:
        ax = axes[1, 2]
        ax.plot(k, diff[hf, :], label='x')
        ax.plot(k, diff[:, hf], '--', label='y')
        ax.axhline(0, color='gray', lw=0.5)
        ax.set_title('Fitted - TPS profiles')
        ax.set_xlabel('[pitch]')
        ax.legend()
    fig.suptitle(title)
    plt.tight_layout()
    plt.show()


def _remove_piston_tilt_for_display(ifs: np.ndarray, valid_mask: np.ndarray, remove_tilt: bool = True) -> np.ndarray:
    """Display only: removes the median (and tip/tilt) of every IF over the valid pixels, so that a
    comparison with the measured IFs is made in the subspace that was actually fitted."""
    out = np.zeros(ifs.shape)
    X = ifs[:, valid_mask].astype(float)
    X = X - np.median(X, axis=1, keepdims=True)
    if remove_tilt:
        Q, _ = _tilt_basis(valid_mask)
        X = X - (X @ Q) @ Q.T
    out[:, valid_mask] = X
    return out


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
    fit_tilt: bool = True,
    coupling_method: str = 'decoupling',
    coupling_radius_pitches: float = 2.5
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
    amplitude_matrix, scaled_sim_ifs, diag = fit_amplitudes_to_sim_ifs(meas_ifs, meas_mask, sim_ifs_registered, optimize_coords=optimize_coords, fit_if_template=fit_if_template, fit_tilt=fit_tilt, coupling_method=coupling_method, coupling_radius_pitches=coupling_radius_pitches, return_diagnostics=True)
    
    print("--- Calibration Complete! ---\n")
    
    if show_plots:
        # Compare in the fitted subspace (piston/tip-tilt removed for display only; outputs are complete)
        valid = meas_mask > 0
        plot_results(_remove_piston_tilt_for_display(meas_ifs, valid, fit_tilt),
                     _remove_piston_tilt_for_display(scaled_sim_ifs, valid, fit_tilt),
                     orig_pupil_center, orig_pupil_radius, reg_coords_sim)
        if 'template' in diag:
            plot_average_if(diag['template'], diag['template_superres'], diag.get('template_tps'))
        if 'coupling_map' in diag:
            plot_coupling_map(diag['coupling_map'], f"Median coupling map C ({diag['coupling_method']}, "
                              f"{'empirical 2D IF' if 'template' in diag else 'TPS'} basis)")
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