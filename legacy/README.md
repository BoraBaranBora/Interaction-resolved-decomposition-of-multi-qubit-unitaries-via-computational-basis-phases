# Legacy optimization experiments

The historical `optimize_pulse_multi_RWA_2C.py` and `objective_functions.py`
files should be retained only for provenance, preferably in this directory.
They must not be imported by the production workflow: the objective file
contains repeated definitions under identical function names, so the effective
runtime objective depends on whichever definition happens to occur last.

The production gradient workflow is `control_optimization/` with the entry point
`optimize_control.py`.
