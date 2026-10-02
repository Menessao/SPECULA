import numpy as np
from scipy.ndimage import shift, affine_transform
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


def fit_amplitudes_to_sim_ifs(
    meas_ifs: np.ndarray, 
    meas_mask: np.ndarray, 
    sim_ifs_registered: np.ndarray, 
    reg_param: float = 1e-4,
    optimize_coords: bool = True
):
    """
    Fits measured IFs to registered simulated IFs, with optional per-actuator
    sub-pixel coordinate refinement.

    The coordinate refinement (phase 1) is independent across actuators -
    each simulated IF is only ever compared to its own measured counterpart,
    so it's decoupled from a global solve rather than being interleaved with
    one. The amplitude/coupling fit (phase 2) is then done once, for all
    actuators simultaneously, via a single batched linear solve instead of
    an explicit-inverse solver rebuilt on every iteration.

    Returns:
        amplitude_matrix: (N_acts, N_acts) array of coupling weights.
        scaled_sim_ifs: (N_acts, H, W) array of the final reconstructed IFs.
    """
    n_acts, h, w = meas_ifs.shape
    valid_mask = meas_mask > 0

    updated_sim_ifs = np.copy(sim_ifs_registered)

    # Estimate actuator pitch for shift constraints
    pitch = max(h, w) / np.sqrt(4.0 * n_acts / np.pi)
    max_shift = 0.5 * pitch
    window = int(pitch)

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

    # --- Phase 2: single batched linear solve for all actuators' amplitudes ---
    print("Solving for amplitude/coupling matrix...")
    S_fit = updated_sim_ifs[:, valid_mask]           # (n_acts, n_pixels)
    meas_fit = meas_ifs[:, valid_mask]                # (n_acts, n_pixels)

    StS = S_fit @ S_fit.T                             # (n_acts, n_acts)
    reg_I = reg_param * np.eye(n_acts) * (np.trace(StS) / n_acts)
    rhs = S_fit @ meas_fit.T                          # (n_acts, n_acts)

    # solve (StS + reg_I) @ X = rhs for X, then transpose so row i holds the
    # amplitude vector for measured actuator i (avoids forming an explicit
    # inverse, both faster and better conditioned)
    amplitude_matrix = np.linalg.solve(StS + reg_I, rhs).T
    amplitude_matrix -= amplitude_matrix.mean(axis=1, keepdims=True)

    # Reconstruct the final scaled IFs across the FULL domain using the updated basis
    scaled_sim_ifs = np.einsum('ij,jhw->ihw', amplitude_matrix, updated_sim_ifs)

    return amplitude_matrix, scaled_sim_ifs


def plot_results(meas_ifs: np.ndarray, sim_ifs: np.ndarray, pupil_center: tuple, pupil_radius: float, coords: np.ndarray):
    """Displays measured IFs, registered simulated IFs, and the Nact x Nact TPS amplitude matrix."""
    
    norm_meas = (meas_ifs - np.mean(meas_ifs[abs(meas_ifs)>0])) / np.max(np.abs(meas_ifs))
    norm_sim = (sim_ifs - np.mean(sim_ifs[abs(sim_ifs)>0]))/ np.max(np.abs(sim_ifs))
    diff = np.sqrt(np.sum((norm_meas-norm_sim)**2,axis=0))
    
    sum_meas = np.sum(np.abs(norm_meas)**3, axis=0)
    sum_sim = np.sum(np.abs(norm_sim)**3, axis=0)
    
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
        
    im0 = axes[0].imshow(sum_meas, cmap='viridis', origin='lower',vmin=0,vmax=1.5)
    axes[0].set_title("Sum of cubed MEASURED iffs")
    axes[0].axis('off')
    fig.colorbar(im0, ax=axes[0], fraction=0.046, pad=0.04)
    
    im1 = axes[1].imshow(sum_sim, cmap='viridis', origin='lower',vmin=0,vmax=1.5)
    axes[1].set_title("Sum of cubed SIMULATED iffs")
    axes[1].axis('off')
    fig.colorbar(im1, ax=axes[1], fraction=0.046, pad=0.04)

    im2 = axes[2].imshow(diff * (np.sum(abs(meas_ifs),axis=0)>0), cmap='RdBu', origin='lower')
    axes[2].plot(coords[:,1],coords[:,0],'x',c='gray',alpha=0.5,label='act coords')
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
    optimize_coords: bool = False
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
    amplitude_matrix, scaled_sim_ifs = fit_amplitudes_to_sim_ifs(meas_ifs, meas_mask, sim_ifs_registered, optimize_coords=optimize_coords)
    
    print("--- Calibration Complete! ---\n")
    
    if show_plots:
        plot_results(meas_ifs, scaled_sim_ifs, orig_pupil_center, orig_pupil_radius, reg_coords_sim)
        plt.figure()
        plt.imshow(amplitude_matrix,cmap='magma')
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