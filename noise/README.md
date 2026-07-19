# Electron Ramsey OU noise workflow

The production noise model is

\[
H_r(t)=H_{\mathrm{NV}}(t)+\beta_r(t)\,\widetilde Z_A/2.
\]

`beta_r(t)` is a stationary Ornstein--Uhlenbeck detuning in rad/s. It is
inserted into every short-time Hamiltonian before the matrix exponential is
evaluated. Only the electron channel is active. Calibration is exclusively to
Ramsey `T2*`; the workflow inserts no Hahn-echo, refocusing, or dynamical-
decoupling pulse.

The fixed local correction is recalibrated from the noiseless version of each
loaded pulse and then held constant across all stochastic realizations.

The primary grid uses tau_c = 0.1, 0.3, 1, 3, 15, and 30 us to resolve the crossover from fast fluctuations to the quasi-static gate limit.
