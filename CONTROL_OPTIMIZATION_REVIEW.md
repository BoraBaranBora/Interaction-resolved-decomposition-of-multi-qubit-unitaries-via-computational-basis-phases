# Review of the historical control code

The production optimizer was rebuilt instead of editing the historical files in
place. The audit found several issues that make those files unsuitable as an
imported library:

- `objective_functions.py` defines `FoM_gate_transformation` 19 times,
  `FoM_multi_state_preparation` 9 times, and `calculate_primal` 3 times. Python
  retains only the final definition of each name.
- Several regularizers return `.item()`, which converts a tensor to a Python
  scalar and removes that term from the gradient graph.
- The XZZ driver contains mutually inconsistent target definitions, including a
  zero tripartite coordinate while constructing a nonzero XZZ target unitary.
- Multiple model imports overwrite `set_active_carbons` and `get_precomp`.
- Gradient-free and gradient-based algorithms, target construction, plotting,
  checkpointing, and experimental alternatives are mixed in one executable.
- Legacy parameter semantics and constraint enforcement are not documented
  reliably enough to trust blindly. The replacement attempts the physical
  amplitude/frequency/phase vector, verifies it against the saved waveform, and
  falls back to a waveform fit when the reconstruction is inconsistent.

The replacement has one objective implementation, one pulse implementation, and
one optimizer runner. Every optimization term remains a tensor until logging.
