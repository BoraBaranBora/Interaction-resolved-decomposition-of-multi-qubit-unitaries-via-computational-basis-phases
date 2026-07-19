# Part II — Noise-robustness evaluation

This part does not optimize controls. It evaluates immutable reference and
selected robust checkpoints under electron-only OU detuning.

The Cartesian sensitivity map is

- `T2* = {2, 5, 10, 20, 50} us`
- `tau_c = {0.1, 0.3, 1, 3, 15, 30} us`

The Bar-Gill, Hayashi, and Bauch settings are evaluated separately with their
source-native parameters and overlaid on the Cartesian plots.

Preliminary figures:

```powershell
& .\noise_robustness_evaluation\SMOKE.ps1
```

Longer evaluation:

```powershell
& .\noise_robustness_evaluation\OVERNIGHT.ps1
```

Publication evaluation:

```powershell
& .\noise_robustness_evaluation\PUBLICATION.ps1
```
