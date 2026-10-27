from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from save_figure import save_figure, report_style

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

    # Only the non-paralysable fit is drawn: the GM tube is self-quenching, so it is non-paralysable.
    # The paralysable fit and the linearised line are still printed above as cross-checks.
    plt.rcParams.update(report_style)
    current_fine = np.linspace(0, current.max() * 1.05, 500)
    model, popt, perr, _ = fits["Non-paralysable"]
    fig, ax = plt.subplots(figsize=(9, 6.5))
    ax.errorbar(current, rate, yerr=sigma, fmt="ko", markersize=4, capsize=2, label="Data")
    ax.plot(current_fine, k * current_fine, "k:", linewidth=1, label=f"N = kI (k = {k:.0f} /s per mA)")
    ax.plot(current_fine, model(current_fine, *popt), "r-", linewidth=1.5,
            label=f"Non-paralysable, $\\tau$ = {popt[1] * 1e6:.0f} $\\pm$ {perr[1] * 1e6:.0f} $\\mu$s")
    ax.set_ylim(0, 1.3 * rate.max())
    ax.set_xlabel("Current (mA)")
    ax.set_ylabel("Count rate (/s)")
    ax.set_title("Detector deadtime")
    ax.legend(loc="lower right")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    save_figure("calib_1_deadtime", fig)
    plt.show()
