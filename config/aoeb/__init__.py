"""aoeb: semi-analytical error budget for pyramid-WFS AO.

budget.py            ErrorBudgetMachine   - fast per-mode budget from SPECULA calibration products
generation.py        ErrorBudgetGenerator - runs the SPECULA calibrations and post-processes them
filter_optimizers.py vectorised loop model and controller optimiser
reconstructors.py    SVD / ML / MAP reconstructors
noise.py             detector and pyramid slope-noise model
utils.py             r0/seeing, KL modal variances, pupil geometry, FITS helpers
naming.py            file layout shared by generator and budget
plotting.py          budget and calibration plots
"""
