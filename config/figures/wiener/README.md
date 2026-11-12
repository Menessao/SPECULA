# WFS Workshop 2026 talk figures (wiener)

Notebook: config/wfs_talk_figures.ipynb, commit fc5ac43 (uncommitted changes in config/). EB controller: wiener; z = 30 deg, seeing 0.9" along the line of sight.

| slide | file (.pdf, .png) | data and code | quick EB evaluation |
| --- | --- | --- | --- |
| 12 | slide12_eb_psd_VIS_mag5_wiener | results/multiwfs_cache: single-stage high flux vis (wiener); PsfEngine.psd_terms / psf_terms | - |
| 12 | slide12_eb_psf2d_VIS_mag5_wiener | results/multiwfs_cache: single-stage high flux vis (wiener); PsfEngine.psd_terms / psf_terms; image total, psf (no coronagraph, DL peak = 1); dashed circles: 2 and 5 lambda/D | - |
| 12 | slide12_eb_contrast_profile_VIS_mag5_wiener | results/multiwfs_cache: single-stage high flux vis (wiener); PsfEngine.psd_terms / psf_terms; azimuthal mean, perfect coronagraph | - |
| 13 | slide13_contrast_terms_grid_wiener | results/multiwfs_cache: single-stage high/low flux vis/nir (wiener) | - |
| 14 | slide14_case1_VIS_wiener | results/multiwfs_cache: c1 single/double rMod2 900 (wiener) | - |
| 14 | slide14_case1_NIR_wiener | results/multiwfs_cache: c1 single/double rMod0 1300 (wiener) | - |
| 19 | slide19_summary_wiener | results/multiwfs_cache: sweeps of cases 1, 2, 3 and 5 (wiener); aoeb.plotting.fusion_gain | - |
| 21 | slide21_kalman_block_diagram | specula/processing_objects/kalman_filter.py; aoeb/e2e/ristretto_main.yml (kalman, kpred, dm) | - |
| 22 | slide22_dynamic_range_seeing | results/WFSW/sweep_seeing_20261110: sr_vis.fits of every run listed in results.csv (sweep_seeing.read_rows / stats) | - |
| 23 | slide23_e2e_profiles_mag7 | results/WFSW/sweep_final_20261108 (replays; final_sweep_results.profile / _best_case / load_perfect_coro) | - |

slide19_ratios_wiener.csv: contrast and SR ratios (fused / single) at mags 5, 8, 12.
