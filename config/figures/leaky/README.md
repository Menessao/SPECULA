# WFS Workshop 2026 talk figures (leaky)

Notebook: config/wfs_talk_figures.ipynb, commit 70048ab (uncommitted changes in config/). EB controller: leaky; z = 30 deg, seeing 0.9" along the line of sight. EB selection criterion: contrast (every single-sensor reference is the best binning 48/36/24/14 and frame rate, every fusion at its best frame rate, ranked as the e2e runs: 2-5 lambda/D contrast among SR >= 10 %, then SR; slide 17 uses the older Strehl-criterion c4a sweeps).

| slide | file (.pdf, .png) | data and code | quick EB evaluation |
| --- | --- | --- | --- |
| 12 | slide12_eb_psd_NIR_mag5_leaky | results/multiwfs_cache: single-stage high flux nir (leaky); PsfEngine.psd_terms / psf_terms | - |
| 12 | slide12_eb_psf2d_NIR_mag5_leaky | results/multiwfs_cache: single-stage high flux nir (leaky); PsfEngine.psd_terms / psf_terms; image total, psf (no coronagraph, DL peak = 1); dashed circles: 2 and 5 lambda/D | - |
| 12 | slide12_eb_contrast_profile_NIR_mag5_leaky | results/multiwfs_cache: single-stage high flux nir (leaky); PsfEngine.psd_terms / psf_terms; azimuthal mean, perfect coronagraph | - |
| 12 | slide12_eb_coro2d_NIR_mag5_leaky | results/multiwfs_cache: single-stage high flux nir (leaky); PsfEngine.psf_terms; image total, coro (perfect coronagraph, Cavarroc et al. 2006); dashed circles: 2 and 5 lambda/D | - |
| 12 | slide12_eb_coro2d_NIR_mag5_wiener | results/multiwfs_cache: single-stage high flux nir (wiener); PsfEngine.psf_terms; image total, coro (perfect coronagraph, Cavarroc et al. 2006); dashed circles: 2 and 5 lambda/D | - |
| 12 | slide12_eb_contrast_profile_NIR_mag5_wiener_vs_leaky | results/multiwfs_cache: single-stage high flux nir (wiener terms; dashed: total of leaky); azimuthal mean, perfect coronagraph | - |
| 13 | slide13_contrast_terms_grid_leaky | results/multiwfs_cache: single-stage high/low flux vis/nir (leaky) | - |
| 14 | slide14_case1_VIS_leaky | results/multiwfs_cache: c1 single/double rMod2 900 (leaky) | - |
| 14 | slide14_case1_profiles_VIS_leaky | Case 1, VIS (550 nm), z = 30°: budgets rebuilt from the sweep rows (contrast reproduced to 1e-6); PsfEngine.psf_terms, perfect coronagraph, azimuthal mean | budgets of the cached configurations (sweep rows), rebuilt for the PSF: not in the EB cache |
| 14 | slide14_case1_NIR_leaky | results/multiwfs_cache: c1 single/double rMod0 1300 (leaky) | - |
| 14 | slide14_case1_profiles_NIR_leaky | Case 1, NIR (1650 nm), z = 30°: budgets rebuilt from the sweep rows (contrast reproduced to 1e-6); PsfEngine.psf_terms, perfect coronagraph, azimuthal mean | budgets of the cached configurations (sweep rows), rebuilt for the PSF: not in the EB cache |
| 15 | slide15_case2_NIR_leaky | results/multiwfs_cache: c2 alone/double/split rMod0 1300 N100 (leaky) | - |
| 15 | slide15_case2_profiles_NIR_leaky | Case 2, NIR (1650 nm), z = 30°: budgets rebuilt from the sweep rows (contrast reproduced to 1e-6); PsfEngine.psf_terms, perfect coronagraph, azimuthal mean | budgets of the cached configurations (sweep rows), rebuilt for the PSF: not in the EB cache |
| 15 | slide15_case2_split_NIR_leaky | results/multiwfs_cache: c2 split rMod0 1300 N100 (column f_LO) | - |
| 16 | slide16_case3_VIS_leaky | results/multiwfs_cache: c1 single rMod2, c3 split rMod2+0 900, c1 single rMod0 900 (leaky) | - |
| 16 | slide16_case3_profiles_VIS_leaky | Case 3, VIS (550 nm), z = 30°: budgets rebuilt from the sweep rows (contrast reproduced to 1e-6); PsfEngine.psf_terms, perfect coronagraph, azimuthal mean | budgets of the cached configurations (sweep rows), rebuilt for the PSF: not in the EB cache |
| 16 | slide16_case3_NIR_leaky | results/multiwfs_cache: c1 single rMod2, c3 split rMod2+0 1300, c1 single rMod0 1300 (leaky) | - |
| 16 | slide16_case3_profiles_NIR_leaky | Case 3, NIR (1650 nm), z = 30°: budgets rebuilt from the sweep rows (contrast reproduced to 1e-6); PsfEngine.psf_terms, perfect coronagraph, azimuthal mean | budgets of the cached configurations (sweep rows), rebuilt for the PSF: not in the EB cache |
| 17 | slide17_chromatic_VIS_leaky | results/multiwfs_cache: c4a aoc/chrom rMod2 900 (leaky) | - |
| 17 | slide17_chromatic_NIR_leaky | results/multiwfs_cache: c4a aoc/chrom rMod0 1300 (leaky) | - |
| 18 | slide18_case4_VIS_leaky | results/multiwfs_cache: c4b * rMod2 (z = 30) (leaky) | - |
| 18 | slide18_case4_profiles_VIS_leaky | Case 4, VIS (550 nm), z = 30°: budgets rebuilt from the sweep rows (contrast reproduced to 1e-6); PsfEngine.psf_terms, perfect coronagraph, azimuthal mean | budgets of the cached configurations (sweep rows), rebuilt for the PSF: not in the EB cache |
| 19 | slide19_summary_leaky | results/multiwfs_cache: sweeps of cases 1, 2, 3 and 5 (leaky); aoeb.plotting.fusion_gain | - |
| 21 | slide21_kalman_block_diagram | specula/processing_objects/kalman_filter.py; aoeb/e2e/ristretto_main.yml (kalman, kpred, dm) | - |
| 22 | slide22_dynamic_range_seeing | results/WFSW/sweep_seeing_20261110: sr_vis.fits of every run listed in results.csv (sweep_seeing.read_rows / stats) | - |
| 23 | slide23_e2e_profiles_mag7 | results/WFSW/sweep_final_20261108 (replays; final_sweep_results.profile / _best_case / load_perfect_coro) | - |

slide19_ratios_leaky.csv: contrast and SR ratios (fused / single) at mags 5, 8, 12.
