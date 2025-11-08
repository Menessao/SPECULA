import specula
import numpy as np

magnitudes = np.arange(0, 13)
output_dir = "magnitude_overrides"
base_config = "config/main_params.yaml"

for mag in magnitudes:
    overrides = ("{"
                f"on_axis_source.magnitude: {mag}, "
                f"data_store.store_dir: ./output/magnitude/mag{mag}"
                "}")

    specula.main_simul(yml_files=[base_config], overrides=overrides)
