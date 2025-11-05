import os
import glob
import yaml
import numpy as np
import matplotlib.pyplot as plt
from astropy.io import fits

output_base = "./output/magnitude"
dirs = sorted(glob.glob(os.path.join(output_base, "mag*/2*/")))

mean_sr = {}

for d in dirs:
    # Find the YAML file to get the magnitude value
    params_file = os.path.join(d, "params.yml")
    with open(params_file, 'r') as f:
        yml_data = yaml.safe_load(f)
    mag = float(yml_data['on_axis_source']['magnitude'])
    sr_file = os.path.join(d, "sr.fits")
    if os.path.exists(sr_file):
        with fits.open(sr_file) as hdul:
            sr = hdul[0].data
        mean_sr[mag] = sr[50:].mean()  # Ignore initial transient
        print(f"Magnitude {mag:.1f}: mean SR = {sr[50:].mean():.4f}")
    else:
        print(f"Warning: {sr_file} not found.")

# Plot
mag_to_plot = sorted(mean_sr.keys())
sr_to_plot = [mean_sr[mag] for mag in mag_to_plot]
plt.figure()
plt.plot(mag_to_plot, sr_to_plot, marker='o')
plt.xlabel("Guide Star Magnitude")
plt.ylabel("Mean Strehl Ratio")
plt.title("SR vs Guide Star Magnitude")
plt.grid(True)
plt.show()
