from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

# Dead time of the GM tube from a tube-current scan (Cu anode, NaCl crystal, goniometer held at one angle).
# The true photon rate is proportional to the emission current, N = k * I. A GM tube is blind for a time tau
# after each count, so the measured rate R falls below N and saturates at high current.
# Columns: emission current in mA, measured count rate in 1/s. The 0 mA row has no reading and is dropped.
# Importing this file (as fits_copy.py does) gives tau and tau_err without printing or plotting anything.
current, rate = np.genfromtxt(Path(__file__).parent / "current_characterisation.csv", delimiter=",", skip_header=1,
                              unpack=True)
has_rate = ~np.isnan(rate)
current, rate = current[has_rate], rate[has_rate]

# Poisson uncertainty from the raw counts, sigma_R = sqrt(R / t). absolute_sigma=False below, so only the
# relative weights matter and the gate time t cancels; the residual scatter sets the uncertainty scale.
gate_time = 10  # s, matches the one-decimal rates; does not affect tau
sigma = np.sqrt(rate / gate_time)


# Non-paralysable: each count is followed by a fixed dead time, R = N / (1 + N tau)  <=>  N = R / (1 - R tau).
# This is the standard GM-tube model and the correction quoted in fits_documentation.txt.
def non_paralysable(current, k, tau):
    true_rate = k * current
    return true_rate / (1 + true_rate * tau)


# Paralysable: every photon, counted or not, restarts the dead time, R = N exp(-N tau). For comparison only.
def paralysable(current, k, tau):
    true_rate = k * current
    return true_rate * np.exp(-true_rate * tau)


fits = {}
for label, model in [("Non-paralysable", non_paralysable), ("Paralysable", paralysable)]:
    popt, pcov = curve_fit(model, current, rate, p0=[7000, 1e-4], sigma=sigma)
    chi2 = np.sum(((rate - model(current, *popt)) / sigma) ** 2)
    fits[label] = (model, popt, np.sqrt(np.diag(pcov)), chi2 / (len(rate) - len(popt)))

# The value used by every other script
k, tau = fits["Non-paralysable"][1]
tau_err = fits["Non-paralysable"][2][1]

# Cross-check: the non-paralysable model is a straight line in 1/R against 1/I, 1/R = (1/k)(1/I) + tau
slope, intercept = np.polyfit(1 / current, 1 / rate, 1)

if __name__ == "__main__":
    for label, (model, popt, perr, chi2_nu) in fits.items():
        print(f"{label}: k = {popt[0]:.0f} +/- {perr[0]:.0f} 1/s per mA, tau = {popt[1] * 1e6:.1f} +/- "
              f"{perr[1] * 1e6:.1f} us, chi2/dof (t = {gate_time} s) = {chi2_nu:.2f}")
    print(f"Linearised 1/R vs 1/I: tau = {intercept * 1e6:.1f} us, k = {1 / slope:.0f} 1/s per mA")
    print(f"Size of the correction R_true / R = 1 / (1 - R tau) with tau = {tau * 1e6:.0f} us:")
    for measured in [100, 1090, rate.max()]:
        print(f"    R = {measured:6.0f} 1/s  ->  x{1 / (1 - measured * tau):.3f}")

    current_fine = np.linspace(0, current.max() * 1.05, 500)
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    axes[0].errorbar(current, rate, yerr=sigma, fmt="k.", markersize=6, capsize=2, label="Measured")
    axes[0].plot(current_fine, k * current_fine, "k:", linewidth=1, label=f"True rate, N = kI (k = {k:.0f} 1/s per mA)")
    for (label, (model, popt_m, perr_m, _)), style in zip(fits.items(), ["r-", "b--"]):
        axes[0].plot(current_fine, model(current_fine, *popt_m), style, linewidth=1.5,
                     label=f"{label}, $\\tau$ = {popt_m[1] * 1e6:.0f} $\\pm$ {perr_m[1] * 1e6:.0f} $\\mu$s")
    axes[0].set_ylim(0, 1.3 * rate.max())
    axes[0].set_xlabel("Emission current (mA)")
    axes[0].set_ylabel("Count rate (1/s)")
    axes[0].set_title("Count rate against tube current")
    axes[0].legend(fontsize=8)
    axes[0].grid(alpha=0.3)

    inv_current_fine = np.linspace(0, 1.05 / current.min(), 500)
    axes[1].plot(1 / current, 1 / rate, "k.", markersize=6, label="Measured")
    axes[1].plot(inv_current_fine, slope * inv_current_fine + intercept, "r-", linewidth=1.5,
                 label=f"Straight line, intercept $\\tau$ = {intercept * 1e6:.0f} $\\mu$s")
    axes[1].set_xlabel("1 / current (1/mA)")
    axes[1].set_ylabel("1 / count rate (s)")
    axes[1].set_title("Linearised non-paralysable model")
    axes[1].legend(fontsize=8)
    axes[1].grid(alpha=0.3)
    fig.suptitle("GM tube dead time, Cu anode, NaCl crystal")
    fig.tight_layout()
    plt.show()
