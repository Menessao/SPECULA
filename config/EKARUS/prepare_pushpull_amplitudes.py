import os
import numpy as np
from astropy.io import fits

import specula
specula.init(0)  # Use GPU device 0 (or -1 for CPU)
from specula.calib_manager import CalibManager

def create_scaled_amplitudes(n_actuators, base_amplitude=50):
    """
    Create amplitude vector with scaling pattern:
    [1, 1, 1/sqrt(2), 1/sqrt(2), 1/sqrt(2), 1/sqrt(3), 1/sqrt(3), 1/sqrt(3), 1/sqrt(3), ...]

    Parameters:
    -----------
    n_actuators : int
        Total number of actuators
    base_amplitude : float
        Base amplitude in nm (default: 50nm)

    Returns:
    --------
    amplitudes : ndarray
        Scaled amplitude vector
    """
    amplitudes = np.zeros(n_actuators)

    # Pattern: n repetitions of 1/sqrt(n)
    # Group 1: 2 actuators with factor 1 (1/sqrt(1))
    # Group 2: 3 actuators with factor 1/sqrt(2)
    # Group 3: 4 actuators with factor 1/sqrt(3)
    # etc.

    idx = 0
    group = 1

    while idx < n_actuators:
        # Number of actuators in this group
        group_size = group + 1

        # Scale factor for this group
        scale_factor = 1.0 / np.sqrt(group)

        # Fill the group (up to remaining actuators)
        end_idx = min(idx + group_size, n_actuators)
        amplitudes[idx:end_idx] = scale_factor

        print(f"Group {group}: actuators {idx:4d}-{end_idx-1:4d} (size={end_idx-idx:2d}), factor=1/√{group} = {scale_factor:.4f}")

        idx = end_idx
        group += 1

    # Apply base amplitude
    amplitudes *= base_amplitude

    return amplitudes

def main():
    root_dir = './calib'

    # Initialize calibration manager
    calib_manager = CalibManager(root_dir)

    # Create scaled amplitudes for all valid actuators
    n_actuators = 443  # Number of valid actuators -1 (from influence functions)
    base_amplitude = 50  # 50nm

    # tags
    data_filename = f'pushpull_{n_actuators}modes_amp50'
    data_uniform_filename = f'pushpull_{n_actuators}modes_amp50_uniform'

    print(f"Creating scaled amplitude vector for {n_actuators} actuators")
    print(f"Base amplitude: {base_amplitude:.1f} nm")
    print("")

    amplitudes = create_scaled_amplitudes(n_actuators, base_amplitude)

    # Print statistics
    print(f"\nAmplitude statistics:")
    print(f"  Minimum: {np.min(amplitudes):.2f} nm")
    print(f"  Maximum: {np.max(amplitudes):.2f} nm")
    print(f"  Mean:    {np.mean(amplitudes):.2f} nm")
    print(f"  Std:     {np.std(amplitudes):.2f} nm")

    # Show first and last few values
    print(f"\nFirst 10 amplitudes [nm]: {amplitudes[:10]}")
    print(f"Last 10 amplitudes [nm]:  {amplitudes[-10:]}")

    # Save amplitude vector
    os.makedirs(os.path.join(root_dir, 'data'), exist_ok=True)

    output_file = calib_manager.filename('data', data_filename)

    fits.writeto(output_file, amplitudes, overwrite=True)
    print(f"\nOK: Saved scaled amplitude vector: {output_file}")

    # Create comparison with uniform amplitudes
    uniform_amplitudes = np.full(n_actuators, base_amplitude)
    uniform_file = calib_manager.filename('data', data_uniform_filename)
    fits.writeto(uniform_file, uniform_amplitudes, overwrite=True)
    print(f"OK: Saved uniform amplitude vector: {uniform_file}")

    return amplitudes

if __name__ == "__main__":
    amplitudes = main()
