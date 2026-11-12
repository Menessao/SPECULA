# Missing data (wiener)

### Slide 14 – case 1, panel (c) per-term bars
- Status: partially produced
- Missing: per-term contrast at the lowest-ratio magnitude (not cached)
- Needed to produce it: skipped by decision; would need one budget + PsfEngine.psf_terms per curve at that magnitude
- Existing code that would generate it: Study.single / fused, PsfEngine.psf_terms
- Discrepancies found: none found

### Slide 15 – case 2 NIR
- Status: not produced
- Missing: c2 ['alone', 'double', 'split'] rMod0 1300 N100 (wiener)
- Needed to produce it: EB sweep (analyse_errorbudget.ipynb, case 2)
- Existing code that would generate it: Study.best_single / best_dual / best_split
- Discrepancies found: none found

### Slide 15 – case 2, panel (c) per-term bars
- Status: partially produced
- Missing: per-term contrast at the lowest-ratio magnitude (not cached)
- Needed to produce it: skipped by decision
- Existing code that would generate it: Study.best_dual / best_split, PsfEngine.psf_terms
- Discrepancies found: none found

### Slide 16 – case 3 VIS
- Status: not produced
- Missing: c1 single rMod2 / c3 split rMod2+0 900 (wiener)
- Needed to produce it: EB sweep (analyse_errorbudget.ipynb, case 3)
- Existing code that would generate it: Study.single / Study.fused
- Discrepancies found: none found

### Slide 16 – case 3 NIR
- Status: not produced
- Missing: c1 single rMod2 / c3 split rMod2+0 1300 (wiener)
- Needed to produce it: EB sweep (analyse_errorbudget.ipynb, case 3)
- Existing code that would generate it: Study.single / Study.fused
- Discrepancies found: none found

### Slide 16 – case 3, panel (c) per-term bars
- Status: partially produced
- Missing: per-term contrast at the lowest-ratio magnitude (not cached)
- Needed to produce it: skipped by decision
- Existing code that would generate it: Study.fused, PsfEngine.psf_terms
- Discrepancies found: none found

### Slide 17 – chromatic shear VIS
- Status: not produced
- Missing: c4a aoc / chrom rMod2 900 (wiener)
- Needed to produce it: EB sweep (cell 19 of analyse_errorbudget.ipynb at e3b9efe)
- Existing code that would generate it: Study.best_single, Study.chromatic_contrast
- Discrepancies found: none found

### Slide 17 – chromatic shear NIR
- Status: not produced
- Missing: c4a aoc / chrom rMod0 1300 (wiener)
- Needed to produce it: EB sweep (cell 19 of analyse_errorbudget.ipynb at e3b9efe)
- Existing code that would generate it: Study.best_single, Study.chromatic_contrast
- Discrepancies found: none found

### Slide 18 – case 4 VIS
- Status: not produced
- Missing: c4b ['900 nm, all the light', 'fusion: 750 / 950 nm, 25/75 flux', 'split modes: 25/75 flux'] rMod2, z = 30 (wiener)
- Needed to produce it: EB sweep (analyse_errorbudget.ipynb, case 4)
- Existing code that would generate it: Study.best_rate, budget_fused, combine_stages
- Discrepancies found: none found

### Slide 18 – case 4, panel (c) per-term bars
- Status: partially produced
- Missing: per-term contrast at the lowest-ratio magnitude, z = 30 (the cached Case_4 PSFs are at z = 50)
- Needed to produce it: skipped by decision
- Existing code that would generate it: budget_fused / combine_stages, PsfEngine.psf_terms
- Discrepancies found: none found

### Slide 19 – summary
- Status: partially produced
- Missing: case 2: optimal split VIS, case 2: optimal split NIR, case 3: rMod 2 + rMod 0, 50/50 VIS, case 3: rMod 2 + rMod 0, 50/50 NIR, case 2: double flux VIS, case 2: double flux NIR, combined: mod 900 + unmod 1300 VIS, combined: mod 900 + unmod 1300 NIR, combined: unmod 900 + mod 1300 VIS, combined: unmod 900 + mod 1300 NIR (wiener)
- Needed to produce it: the EB sweeps of those cases
- Existing code that would generate it: see cases 1-5
- Discrepancies found: none found

### Slide 24 – EB vs E2E validation
- Status: not produced
- Missing: EB runs for the E2E configurations of slide 23 (Kalman controller, chromatic, z = 30, selected rates and sizes)
- Needed to produce it: skipped by decision: no EB controller is a Kalman filter; the closest EB case (combined, case 5) has no chromatic term and optimised rates
- Existing code that would generate it: ErrorBudgetMachine.budget_dual / budget_fused with the E2E rates and sizes, PsfEngine.psf_terms
- Discrepancies found: none found

### Slide 28-39 – backup slides
- Status: not produced
- Missing: -
- Needed to produce it: skipped by decision: several were made from other data sets
- Existing code that would generate it: -
- Discrepancies found: none found
