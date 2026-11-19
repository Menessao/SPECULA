# WFS Workshop 2026 talk figures (leaky)

Notebook: config/wfs_talk_figures.ipynb, commit 94eeb64 (uncommitted changes in config/). EB controller: leaky; z = 30 deg, seeing 0.9" along the line of sight. EB selection criterion: contrast (every single-sensor reference is the best binning 48/36/24/14 and frame rate, every fusion at its best frame rate, ranked as the e2e runs: 2-5 lambda/D contrast among SR >= 10 %, then SR; slide 17 uses the older Strehl-criterion c4a sweeps).

| slide | file (.pdf, .png) | data and code | quick EB evaluation |
| --- | --- | --- | --- |
| 16 | slide16_case23_e2e | E2E sweep values (nominal realization): sweep_final_20261108 single_wfs*_chrom_r2, dual_eq*_chrom_u*; talk_cases_20261112 c2s_*, c3_* | - |
| 19 | slide19_reference_single_leaky | results/multiwfs_cache: c1 single rMod2 (chromatic), c5 single rMod2 (achromatic) | - |
| 19 | slide19_reference_single_rmod1_leaky | results/multiwfs_cache: c1 single rMod1 (chromatic), c5 single rMod1 (achromatic) | - |
| 21 | slide21_kalman_block_diagram | specula/processing_objects/kalman_filter.py; aoeb/e2e/ristretto_main.yml (kalman, kpred, dm) | - |
| 22 | slide22_dynamic_range_seeing | results/WFSW/sweep_seeing_20261110: sr_vis.fits of every run listed in results.csv (sweep_seeing.read_rows / stats) | - |
| 23 | slide23_e2e_profiles_mag7 | results/WFSW/sweep_final_20261108 (replays; final_sweep_results.profile / _best_case / load_perfect_coro) | - |
| 23 | slide23_e2e_profiles_eqflux_mag5 | results/WFSW/sweep_final_20261108: paired single realization (sweep runs at mags 7-12, replay realization 0 at mag 5); final_sweep_results.best_eq_split / _paired_run / load_coro_profile / load_perfect_coro | - |
| 23 | slide23_e2e_profiles_eqflux_mag7 | results/WFSW/sweep_final_20261108: paired single realization (sweep runs at mags 7-12, replay realization 0 at mag 5); final_sweep_results.best_eq_split / _paired_run / load_coro_profile / load_perfect_coro | - |
| 23 | slide23_e2e_profiles_eqflux_mag10 | results/WFSW/sweep_final_20261108: paired single realization (sweep runs at mags 7-12, replay realization 0 at mag 5); final_sweep_results.best_eq_split / _paired_run / load_coro_profile / load_perfect_coro | - |
| 24 | slide24_eb_vs_e2e_profiles_VIS_leaky | E2E: coro_psf_*_int.fits of the selected runs (sweep_final_20261108, talk_cases_20261112); EB: budgets at those configurations (leaky), PsfEngine.psf_terms, AO-peak normalised | - |
| 24 | slide24_eb_vs_e2e_profiles_zoom_VIS_leaky | E2E: coro_psf_*_int.fits of the selected runs (sweep_final_20261108, talk_cases_20261112); EB: budgets at those configurations (leaky), PsfEngine.psf_terms, AO-peak normalised | - |
| 24 | slide24_eb_vs_e2e_profiles_NIR_leaky | E2E: coro_psf_*_int.fits of the selected runs (sweep_final_20261108, talk_cases_20261112); EB: budgets at those configurations (leaky), PsfEngine.psf_terms, AO-peak normalised | - |
| 24 | slide24_eb_vs_e2e_profiles_zoom_NIR_leaky | E2E: coro_psf_*_int.fits of the selected runs (sweep_final_20261108, talk_cases_20261112); EB: budgets at those configurations (leaky), PsfEngine.psf_terms, AO-peak normalised | - |

slide19_ratios_leaky.csv: contrast and SR ratios (fused / single) at mags 5, 8, 12.
