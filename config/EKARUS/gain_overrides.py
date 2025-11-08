import specula
import numpy as np

# Range of gains to test
gains = np.linspace(0.1, 1.0, 10)
output_dir = "gain_overrides"
base_config = "config/main_params.yaml"

for gain in gains:
    overrides = ("{"
                f"integrator.int_gain: [{gain:.2f}], "
                f"data_store.store_dir: ./output/gain_opt/gain_{gain:.2f}, "
                f"main.total_time: 1.0"
                "}")

    specula.main_simul(yml_files=[base_config], overrides=overrides)

# tt_gains = np.linspace(0.2, 0.8, 6)
# fa_gains = np.linspace(0.2, 0.8, 6)
# for tt_gain in tt_gains:
#     for fa_gain in fa_gains:
#         overrides = ("{"
#                     f"control.int_gain: [{tt_gain:.2f},{fa_gain:.2f},0.5], "
#                     f"data_store.store_dir: ./output/gain_opt/ttgain_{tt_gain:.2f}_fagain_{fa_gain:.2f}"
#                     # f"main.total_time: 1.0"
#                     "}")

#         specula.main_simul(yml_files=[base_config], overrides=overrides)
