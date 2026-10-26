from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from deadtime import tau, tau_err

# Zero-angle offset of the goniometer from a direct-beam scan (Cu anode, no NaCl crystal).
# With no crystal in place the undiffracted beam goes straight into the GM tube, so the count rate
# peaks where the angle reads true zero. Any shift of that peak from 0 is the offset of the angle scale.
# Columns: angle beta in deg, count rate in 1/s, its Poisson uncertainty in 1/s, raw counts (10 s per step).
# Importing this file (as fits_copy.py does) gives zero_offset and zero_offset_err without printing or plotting.
beta, rate, rate_err, counts = np.genfromtxt(Path(__file__).parent / "Test_run_long_run4.csv", delimiter=",",
                                             skip_header=1, unpack=True)

# The peak reaches ~2900/s, where the GM tube loses almost half its counts. Dead time flattens the top
# symmetrically, so it barely moves the centre, but without it a Gaussian cannot describe the shape.
# Dead time and its error come from the non-paralysable fit in deadtime.py.


# Non-paralysable correction N = R / (1 - R tau). The counting error propagates through dN/dR = 1 / (1 - R tau)^2.
def dead_time_correct(rate, rate_err, tau):
    return rate / (1 - rate * tau), rate_err / (1 - rate * tau) ** 2


def gaussian(beta, height, centre, width, background):
    return height * np.exp(-0.5 * ((beta - centre) / width) ** 2) + background


def fit_peak(y, sigma):
    # Floor of one count in 10 s so a zero-count point still carries weight
    sigma = np.maximum(sigma, 0.1)
    p0 = [y.max(), beta[np.argmax(y)], 0.2, np.median(y)]
    popt, pcov = curve_fit(gaussian, beta, y, p0=p0, sigma=sigma)
    popt[2] = abs(popt[2])
    return sigma, popt, np.sqrt(np.diag(pcov))


fits = {}
for label, (y, sigma) in [("Raw", (rate, rate_err)),
                          ("Dead-time corrected", dead_time_correct(rate, rate_err, tau))]:
    sigma, popt, perr = fit_peak(y, sigma)
    chi2 = np.sum(((y - gaussian(beta, *popt)) / sigma) ** 2)
    fits[label] = (y, sigma, popt, perr, chi2 / (len(y) - len(popt)))

# The error on tau is shared by every point, so adding it to each point's error bar would wrongly treat it as
# independent scatter. Instead refit with tau shifted by -/+ 1 sigma and take half the spread of the results
# as a systematic error on each parameter, added in quadrature to the statistical error from the fit.
y, sigma, popt, perr, _ = fits["Dead-time corrected"]
popt_low, popt_high = [fit_peak(*dead_time_correct(rate, rate_err, tau + shift))[1] for shift in (-tau_err, tau_err)]
tau_sys = np.abs(popt_high - popt_low) / 2
total_err = np.hypot(perr, tau_sys)

# Model-free cross-check: background-subtracted centroid of the points within the peak.
# chi2/dof ~ 26 shows the beam profile is not really a Gaussian, so the centre depends on the shape assumed.
# The difference between the Gaussian centre and the centroid is taken as a shape error on the offset.
in_peak = np.abs(beta - popt[1]) < 3 * popt[2]
signal = y[in_peak] - popt[3]
centroid = np.sum(beta[in_peak] * signal) / np.sum(signal)
shape_err = abs(popt[1] - centroid)

# The values used by fits_copy.py
zero_offset = popt[1]
zero_offset_err = np.hypot(total_err[1], shape_err)

if __name__ == "__main__":
    for label, (_, _, popt_m, perr_m, chi2_nu) in fits.items():
        print(f"{label}: centre = {popt_m[1]:+.4f} +/- {perr_m[1]:.4f} deg, "
              f"FWHM = {2.3548 * popt_m[2]:.3f} deg, chi2/dof = {chi2_nu:.1f}")
    print(f"Dead-time corrected fit, tau = {tau * 1e6:.1f} +/- {tau_err * 1e6:.1f} us "
          f"(errors: statistical, from tau, total):")
    for name, value, stat, sys, total in zip(["Height (1/s)", "Centre (deg)", "Width (deg)", "Background (1/s)"],
                                             popt, perr, tau_sys, total_err):
        print(f"    {name:17s} {value:+10.4f}  +/- {stat:.4f}  +/- {sys:.4f}  =  +/- {total:.4f}")
    print(f"Centroid of corrected peak: {centroid:+.4f} deg -> shape error |Gaussian - centroid| = {shape_err:.4f} deg")

    step = np.min(np.diff(beta))
    print(f"Zero offset: beta_0 = {zero_offset:+.4f} +/- {zero_offset_err:.4f} deg (statistical, dead time and shape); "
          f"the {step:.1f} deg step size limits how well the angle itself is set")
    print(f"Correct measured angles with beta_true = beta_measured - ({zero_offset:+.4f} deg)")

    beta_fine = np.linspace(beta.min(), beta.max(), 1000)
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    for ax, yscale in zip(axes, ["linear", "log"]):
        for (label, (y_m, sigma_m, popt_m, _, _)), colour in zip(fits.items(), ["grey", "k"]):
            ax.errorbar(beta, y_m, yerr=sigma_m, fmt=".", color=colour, markersize=5, capsize=2, label=label)
            ax.plot(beta_fine, gaussian(beta_fine, *popt_m), "-", color="r" if colour == "k" else "b", linewidth=1.2,
                    label=f"{label} fit, centre = {popt_m[1]:+.3f}$\\degree$")
        ax.axvline(0, color="k", linestyle=":", linewidth=1)
        ax.axvline(zero_offset, color="r", linestyle="--", linewidth=1)
        ax.set_yscale(yscale)
        ax.set_xlabel("Angle $\\beta$ (deg)")
        ax.set_ylabel("Count rate (1/s)")
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3)
    axes[0].set_title("Direct beam")
    axes[1].set_title("Direct beam, log scale")
    fig.suptitle(f"Goniometer zero offset, Cu anode, no crystal: $\\beta_0$ = {zero_offset:+.3f} $\\pm$ "
                 f"{zero_offset_err:.3f}$\\degree$")
    fig.tight_layout()
    plt.show()
