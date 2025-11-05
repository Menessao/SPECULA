import os
import glob
import yaml
import numpy as np
import matplotlib.pyplot as plt
from astropy.io import fits

output_base = "./output/gain_opt"
dirs = sorted(glob.glob(os.path.join(output_base, "gain_*/2*/")))

gains = []
mean_sr = []

for d in dirs:
    # Find the YAML file to get the gain value
    yml_files = glob.glob(os.path.join(d, "*.yml"))
    gain = None
    for yml in yml_files:
        with open(yml, "r") as f:
            yml_data = yaml.safe_load(f)
            if "integrator" in yml_data:
                gain = float(yml_data["integrator"]["int_gain"][0])
                break
    if gain is None:
        # Fallback: parse from directory name
        gain = float(d.split("_")[-2].replace("/", ""))
    # Load sr.fits
    sr_file = os.path.join(d, "sr.fits")
    if os.path.exists(sr_file):
        with fits.open(sr_file) as hdul:
            sr = hdul[0].data
        mean_sr.append(sr[50:].mean())  # Ignore initial transient
        gains.append(gain)
        print(f"Gain {gain:.2f}: mean SR = {sr[50:].mean():.4f}")
    else:
        print(f"Warning: {sr_file} not found.")

# Plot
plt.figure()
plt.plot(gains, mean_sr, marker='o')
plt.xlabel("Integrator Gain")
plt.ylabel("Mean Strehl Ratio")
plt.title("Loop Gain Optimization")
plt.grid(True)
plt.show()

# output_base = "./output/gain_opt"
# dirs = sorted(glob.glob(os.path.join(output_base, "ttgain_*/2*/")))

# ttgains = []
# fagains = []
# mean_sr = []

# for d in dirs:
#     # Find the YAML file to get the gain value
#     yml_files = glob.glob(os.path.join(d, "*.yml"))
#     ttgain = None
#     fagain = None
#     for yml in yml_files:
#         with open(yml, "r") as f:
#             yml_data = yaml.safe_load(f)
#             if "integrator" in yml_data:
#                 ttgain = float(yml_data["control"]["int_gain"][0])
#                 fagain = float(yml_data["control"]["int_gain"][1])
#                 break
#     if ttgain is None:
#         # Fallback: parse from directory name
#         ttgain = float(d.split("_")[-4].replace("/", ""))
#         fagain = float(d.split("_")[-2].replace("/", ""))
#     # Load sr.fits
#     sr_file = os.path.join(d, "sr.fits")
#     if os.path.exists(sr_file):
#         with fits.open(sr_file) as hdul:
#             sr = hdul[0].data
#         mean_sr.append(sr[50:].mean())  # Ignore initial transient
#         ttgains.append(ttgain)
#         fagains.append(fagain)
#         print(f"TT gain {ttgain:.2f}, FA gain {fagain:.2f}: mean SR = {sr[50:].mean():.4f}")
#     else:
#         print(f"Warning: {sr_file} not found.")

# # Plot
# N = int(np.sqrt(len(fagains)))
# sr_mat = np.reshape(mean_sr,[N,N])

# print(fagains[:N])

# plt.figure()
# plt.imshow(sr_mat,cmap='jet')
# plt.colorbar()
# plt.yticks(np.arange(N),labels=[f'{g1:.2f}' for g1 in ttgains[::N]])
# plt.xticks(np.arange(N),labels=[f'{g2:.2f}' for g2 in fagains[:N]])
# plt.xlabel("Integrator FA Gain")
# plt.ylabel("Integrator TT Gain")
# plt.show()
