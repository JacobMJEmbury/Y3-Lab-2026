# Figures

Every script saves its figures here through `save_figure.py`. Each run overwrites the previous PNGs, so this folder always matches the latest analysis. `fits_copy.py` (and `chi_squared.py`, which imports it) regenerates the `fits_*` figures. Each calibration script draws its `calib_*` figure only when you run it directly.

| File | Made by | What it shows |
|---|---|---|
| `calib_1_deadtime.png` | `deadtime.py` | GM-tube count rate against tube current, with the non-paralysable and paralysable dead-time fits (left) and the linearised 1/R vs 1/I straight line (right). Gives τ. |
| `calib_2_zero_offset.png` | `angle_offset.py` | Direct-beam scan with no crystal, raw and dead-time corrected, with Gaussian fits on linear and log scales. The fitted centre is the goniometer zero offset β₀. |
| `calib_3_alignment_error.png` | `alignment_error.py` | The crystal turned 180° about the beam axis, overlaid on the normal scan: the 1st-order region before and after sliding the normal scan by the best shift (left), the Kα n=1 fit window with residuals (middle), and χ² against trial shift both ways round, with Kβ n=1 as a cross-check (right). Half the Kα n=1 shift is σ_align. |
| `calib_4_temperature_error.png` | `temperature_error.py` | The heated scan overlaid on the unheated one, in the same three panels as `calib_3`. The whole Kα n=1 shift is σ_heat. |
| `fits_1_full_spectrum.png` | `fits_copy.py` | The full Cu/NaCl Bragg spectrum with the dedicated 1st, 2nd and 3rd order Kα scans overlaid. All are dead-time and zero-offset corrected. |
| `fits_5_two_gaussians_free.png` | `fits_copy.py` | Two Gaussians (Kα₁ + Kα₂) fitted to each peak with every parameter free. |
| `fits_6_two_gaussians_fixed_ratio.png` | `fits_copy.py` | Two Gaussians with I₂/I₁ fixed at 0.518 and a shared σ, with the Bragg's-law positions of Kα₁ and Kα₂ marked. This is the main fit used for the wavelength. Each panel also lists the error budget of the K-alpha1 centre. |
| `fits_8_pseudo_voigt.png` | `fits_copy.py` | One and two pseudo-Voigts (fixed ratio, shared FWHM and η), with χ²_ν compared against the Voigt fits. |
