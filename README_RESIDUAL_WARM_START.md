# Exact-reference residual warm start

This patch replaces lossy waveform fitting for historical checkpoints with

\[
\Omega(t)=\Omega_{\rm ref}(t)+h(t)\tanh[\Delta\Omega(t)/\Omega_{\max}],
\qquad
h(t)=\Omega_{\max}-|\Omega_{\rm ref}(t)|.
\]

At zero residual parameters, the control is exactly the saved published
waveform. The Fourier residual is differentiable and cannot push the total
field outside the configured amplitude bound.

After installation, `--gradient-check` should print

```
Selected warm start: exact published waveform + bounded Fourier residual
(waveform_error=0.000e+00)
```

The finite-difference check may show zero derivatives for frequency or phase
entries at the exact zero-residual point; those parameters become active after
Adam moves the residual amplitudes away from zero.
