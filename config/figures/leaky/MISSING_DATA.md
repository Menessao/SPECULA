# Missing data (leaky)

### Slide 14 – case 1, panel (c) per-term bars
- Status: partially produced
- Missing: per-term contrast at the lowest-ratio magnitude (not cached)
- Needed to produce it: skipped by decision; would need one budget + PsfEngine.psf_terms per curve at that magnitude
- Existing code that would generate it: Study.single / fused, PsfEngine.psf_terms
- Discrepancies found: none found

### Slide 15 – case 2, panel (c) per-term bars
- Status: partially produced
- Missing: per-term contrast at the lowest-ratio magnitude (not cached)
- Needed to produce it: skipped by decision
- Existing code that would generate it: Study.best_dual / best_split, PsfEngine.psf_terms
- Discrepancies found: none found

### Slide 16 – case 3, panel (c) per-term bars
- Status: partially produced
- Missing: per-term contrast at the lowest-ratio magnitude (not cached)
- Needed to produce it: skipped by decision
- Existing code that would generate it: Study.fused, PsfEngine.psf_terms
- Discrepancies found: none found

### Slide 18 – case 4, panel (c) per-term bars
- Status: partially produced
- Missing: per-term contrast at the lowest-ratio magnitude, z = 30 (the cached Case_4 PSFs are at z = 50)
- Needed to produce it: skipped by decision
- Existing code that would generate it: budget_fused / combine_stages, PsfEngine.psf_terms
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
