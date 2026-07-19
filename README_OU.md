# Ornstein–Uhlenbeck dephasing ensemble

This addition replaces the excited-population heuristic with explicit stochastic propagation under colored longitudinal noise.

## Files

- `src/ou_dephasing.py`: exact discrete OU sampler, Ramsey/Hahn calibration, and ensemble gate metrics.
- `reproduce_dephasing.py`: propagates the saved ZZZ or XZZ pulse over an ensemble of noise realizations.
- `quantum_model_NV_ou.patch`: minimal patch allowing `get_U_RWA` to receive an additive stochastic Hamiltonian.
- `tests/test_ou_dephasing.py`: analytic and Monte-Carlo checks of the OU implementation.
- `paper/replacement_sections.tex`: compact cost-function formulation and dephasing-model text for the revised NV application paper.

## Install

Copy `src/ou_dephasing.py` into the repository's `src/` directory and `reproduce_dephasing.py` into the repository root. Apply the model patch from the repository root:

```bash
patch -p1 < quantum_model_NV_ou.patch
```

Run the tests:

```bash
PYTHONPATH=src pytest -q tests/test_ou_dephasing.py
```

## Run an ensemble

Times in `--channel` specifications are in microseconds.

```bash
python reproduce_dephasing.py \
  --gate diagonal \
  --channel electron:ramsey:10:50 \
  --n-realizations 512 \
  --seed 7
```

The example above chooses the OU width so that Ramsey coherence is `exp(-1)` at `T2* = 10 us`, with correlation time `tau_c = 50 us`.

To calibrate to a Hahn-echo time instead:

```bash
python reproduce_dephasing.py \
  --gate nondiagonal \
  --channel electron:echo:100:20 \
  --n-realizations 512 \
  --seed 7
```

Independent channels can be combined:

```bash
python reproduce_dephasing.py \
  --gate diagonal \
  --channel electron:ramsey:10:50 \
  --channel c1:echo:5000:500 \
  --channel c2:echo:5000:500
```

The local correction is fitted once from the noiseless pulse and then held fixed for every realization. Refitting a local correction separately for each stochastic trajectory would unrealistically assume knowledge of the noise realization and would overestimate the surviving fidelity.
