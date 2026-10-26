import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from scipy.interpolate import Akima1DInterpolator
from scipy.special import voigt_profile

# Calibrations and systematic errors, each worked out in its own file (run that file to see its fits and plots):
#   deadtime.py           tau, tau_err                 GM-tube dead time (tube-current scan)
#   angle_offset.py       zero_offset, zero_offset_err goniometer zero (direct-beam scan)
#   alignment_error.py    sigma_align_constant         crystal alignment (roll-flipped crystal), deg
#                         sigma_repeat_pooled          run-to-run repeatability of a peak centre, deg
#   temperature_error.py  thermal_d_over_d             fractional uncertainty of d from the lab temperature
#                         sigma_stability              heated-unheated angle shift, a cross-check of sigma_repeat
from deadtime import tau, tau_err
from angle_offset import zero_offset, zero_offset_err
from alignment_error import sigma_align_constant, sigma_repeat_pooled
from temperature_error import thermal_d_over_d, sigma_stability

# Load Cu X-ray spectrum off NaCl crystal (columns: Bragg angle in deg, count rate in 1/s)
measured_full = np.loadtxt("Cu_NaCl_long_1.2KPeak.csv", delimiter=",", skiprows=1, unpack=True)

# Dedicated scans over the 1st, 2nd and 3rd order K-alpha peaks
measured_n1 = np.loadtxt("1st_order_long.csv", delimiter=",", skiprows=1, unpack=True)
measured_n2 = np.loadtxt("2nd_Order_Good.csv", delimiter=",", skiprows=1, unpack=True)
# This file also repeats the n=2 scan in column 1; only rows with an n=3 reading are kept
angle_n3, rate_n3 = np.genfromtxt("3rd_Order_Long_Good.csv", delimiter=",", skip_header=1, usecols=(0, 2), unpack=True)
has_n3 = ~np.isnan(rate_n3)
measured_n3 = angle_n3[has_n3], rate_n3[has_n3]

peak_keys = ["n1", "n2", "n3"]
order_labels = ["1st order", "2nd order", "3rd order"]
colours = ["C0", "C1", "C2"]

# Corrections from the two calibration measurements, applied to every scan before anything else:
#   dead time of the GM tube, from the tube-current scan in deadtime.py (non-paralysable model)
#   zero offset of the angle scale, from the direct-beam scan in angle_offset.py: theta_true = theta_measured - offset

# Counting time per angle step, which sets the size of the Poisson error bars. Read from the settings stored in the
# .xry files: the full scan is 1 s per step, the dedicated n=1, 2, 3 scans 10 s per step.
gate_full, gate_peaks = 1, 10  # s


# Dead-time and zero-offset correction of one scan. Returns the measured angle (kept for selecting points: the
# peak and flank windows below were read off plots of the measured angle), the true angle, the true rate
# N = R / (1 - R tau), and its Poisson error.
# The error comes from the raw counts R t, with a floor of one count so a zero-count point still carries weight,
# and propagates through dN/dR = 1 / (1 - R tau)^2. It is taken before any offset or background is removed:
# subtracting a smooth curve does not remove counting noise, so a flank point that measured ~20/s still has the
# scatter of ~20/s.
def correct(scan, gate_time, tau, offset):
    measured_angle, measured_rate = scan
    sigma = np.sqrt(np.maximum(measured_rate * gate_time, 1)) / gate_time
    loss = 1 - measured_rate * tau
    return measured_angle, measured_angle - offset, measured_rate / loss, sigma / loss ** 2


# The systematic errors are shared by every point, so adding them to each error bar would wrongly treat them as
# independent scatter. Instead the whole analysis is repeated with each one shifted by -/+ 1 sigma (the others held
# at their central values), as in angle_offset.py. Half the spread of a result is its systematic error, added in
# quadrature to the statistical error from the nominal fit.
# Alignment and repeatability are both errors on where the angle scale really is, so like the zero offset they are
# applied as a shift of every angle. A shift common to all angles cancels in a doublet separation.
# Each source: (name printed by report, (tau, angle shift) at -1 sigma, at +1 sigma).
error_sources = {"tau": ("dead time", (tau - tau_err, zero_offset), (tau + tau_err, zero_offset)),
                 "offset": ("zero offset", (tau, zero_offset - zero_offset_err), (tau, zero_offset + zero_offset_err)),
                 "align": ("alignment", (tau, zero_offset - sigma_align_constant),
                           (tau, zero_offset + sigma_align_constant)),
                 "repeat": ("repeatability", (tau, zero_offset - sigma_repeat_pooled),
                            (tau, zero_offset + sigma_repeat_pooled))}
variants = {"nominal": (tau, zero_offset)}
for key, (_, low, high) in error_sources.items():
    variants[f"{key} low"], variants[f"{key} high"] = low, high


# Print fitted values with their error budget and return the total errors.
# results maps each variant to an array of fitted values; stat holds their statistical errors from the nominal fit.
# extra optionally maps the name of a further systematic error, not from the variants, to its array of errors.
def report(names, results, stat, units, fmts, extra={}):
    sys = {name: np.abs(results[f"{key} high"] - results[f"{key} low"]) / 2
           for key, (name, _, _) in error_sources.items()}
    sys.update(extra)
    total = np.sqrt(stat ** 2 + sum(errors ** 2 for errors in sys.values()))
    for i, (name, value, unit, fmt) in enumerate(zip(names, results["nominal"], units, fmts)):
        budget = ", ".join([f"fit {stat[i]:{fmt}}"] + [f"{source} {errors[i]:{fmt}}" for source, errors in sys.items()])
        print(f"    {name:10s} = {value:{fmt}} +/- {total[i]:{fmt}} {unit:3s}  ({budget})")
    return total


# curve_fit with the Poisson error bars taken as absolute (the gate times are known). If chi2/dof > 1 the model
# misses something the error bars do not cover, so the parameter errors are scaled up by sqrt(chi2/dof); they are
# never scaled down below the Poisson level. Returns the parameters, their covariance and chi2/dof.
def weighted_fit(model, x, y, sigma, **kwargs):
    popt, pcov = curve_fit(model, x, y, sigma=sigma, absolute_sigma=True, **kwargs)
    chi2_nu = np.sum(((y - model(x, *popt)) / sigma) ** 2) / (len(y) - len(popt))
    return popt, pcov * max(chi2_nu, 1), chi2_nu


scans = {variant: {"full": correct(measured_full, gate_full, *shift),
                   "n1": correct(measured_n1, gate_peaks, *shift),
                   "n2": correct(measured_n2, gate_peaks, *shift),
                   "n3": correct(measured_n3, gate_peaks, *shift)}
         for variant, shift in variants.items()}
_, angle, rate, sigma = scans["nominal"]["full"]

print(f"Corrections: dead time tau = {tau * 1e6:.1f} +/- {tau_err * 1e6:.1f} us, "
      f"zero offset = {zero_offset:+.4f} +/- {zero_offset_err:.4f} deg")
print(f"Angle errors: alignment +/- {sigma_align_constant:.4f} deg, repeatability +/- {sigma_repeat_pooled:.4f} deg "
      f"(cross-check: heated-unheated stability {sigma_stability:.4f} deg, not counted again); "
      f"thermal sigma_d/d = {thermal_d_over_d:.1e}")
for label, (_, measured_rate) in zip(["Full scan"] + order_labels, [measured_full, measured_n1, measured_n2, measured_n3]):
    print(f"    {label:9s}: highest rate {measured_rate.max():6.0f} 1/s, "
          f"dead-time correction there x{1 / (1 - measured_rate.max() * tau):.3f}")


# Remove the vertical offset of each dedicated scan relative to the full scan, for Figure 1 only.
# The offset is the mean difference in the off-peak region, where both scans should
# be measuring the same background; the peak window itself is excluded.
# The peak fits below fit their own baseline under each peak, which absorbs any constant,
# so they use the scans without this shift.
def remove_offset(scan, peak_low, peak_high):
    measured_angle, scan_angle, scan_rate, _ = scan
    off_peak = (measured_angle < peak_low) | (measured_angle > peak_high)
    full_rate = np.interp(scan_angle, angle, rate)
    offset = np.mean(scan_rate[off_peak] - full_rate[off_peak])
    return scan_rate - offset, offset


rate_n1, offset_n1 = remove_offset(scans["nominal"]["n1"], 15.8, 16.4)
rate_n2, offset_n2 = remove_offset(scans["nominal"]["n2"], 33.0, 33.7)
rate_n3, offset_n3 = remove_offset(scans["nominal"]["n3"], 54.8, 55.5)
print(f"Offset removed: 1st order {offset_n1:.2f} 1/s, 2nd order {offset_n2:.2f} 1/s, 3rd order {offset_n3:.2f} 1/s")

plt.figure(figsize=(10, 5))
plt.errorbar(angle, rate, yerr=sigma, linewidth=1, elinewidth=0.5, label="Full scan")
for key, label, shifted_rate in zip(peak_keys, order_labels, [rate_n1, rate_n2, rate_n3]):
    _, scan_angle, _, scan_sigma = scans["nominal"][key]
    plt.errorbar(scan_angle, shifted_rate, yerr=scan_sigma, linewidth=1, elinewidth=0.5, label=f"{label} scan")
plt.xlabel("Bragg angle, zero offset removed (deg)")
plt.ylabel("Count rate, dead-time corrected (1/s)")
plt.title("Cu anode, NaCl crystal")
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()
plt.show()

# Checkmark 1
# Remove the characteristic Cu peaks to leave only the bremsstrahlung background.
# Each window (measured angle, deg) covers one peak from where it leaves the background to where it returns.
peak_windows = [
    (14.2, 14.9),  # K-beta, n=1
    (15.2, 15.6),  # small line between K-beta and K-alpha, n=1
    (15.8, 16.4),  # K-alpha, n=1
    (29.6, 30.1),  # K-beta, n=2
    (33.0, 33.7),  # K-alpha, n=2
    (47.7, 48.0),  # K-beta, n=3
    (54.8, 55.5),  # K-alpha, n=3
]

background = np.ones_like(measured_full[0], dtype=bool)
for low, high in peak_windows:
    background &= ~((measured_full[0] >= low) & (measured_full[0] <= high))


# Fit the bremsstrahlung background with Kramers' law, including an absorption term:
#   I(lambda) = A * (lambda/lambda_min - 1) / lambda^2 * exp(-B * (lambda / 100 pm)^3)
# Kramers' law alone over-predicts the long-wavelength tail; the exp(-B lambda^3) factor
# models absorption in the tube window, air and detector, which grows roughly as lambda^3.
d_NaCl = 282.01  # pm, NaCl lattice plane spacing


def kramers(wavelength, A, lambda_min, B):
    shape = (wavelength / lambda_min - 1) / wavelength**2 * np.exp(-B * (wavelength / 100) ** 3)
    return np.where(wavelength > lambda_min, A * shape, 0.0)


# The error bars are Poisson for 1 s per angle step (gate_full); see weighted_fit for chi2/dof > 1.
# angle_max limits the fit to the background below that angle (used for the fit-range check below).
def fit_background(scan, angle_max=np.inf):
    _, scan_angle, scan_rate, scan_sigma = (x[background] for x in scan)
    keep = scan_angle < angle_max
    wavelength = 2 * d_NaCl * np.sin(np.radians(scan_angle[keep]))  # Bragg's law, n=1
    popt, pcov, chi2_nu = weighted_fit(kramers, wavelength, scan_rate[keep], scan_sigma[keep], p0=[1e5, 34, 1])
    return popt, np.sqrt(np.diag(pcov)), chi2_nu


background_fits = {variant: fit_background(scans[variant]["full"]) for variant in variants}
popt, perr, chi2_nu = background_fits["nominal"]
A, lambda_min, B = popt
_, angle_bg, rate_bg, sigma_bg = (x[background] for x in scans["nominal"]["full"])
dof = len(rate_bg) - len(popt)

# Fit-range check: the same fit to the background below 10 deg only, near the onset, where the simplifications of
# the model (the empirical absorption term, and no 2nd-order bremsstrahlung, which starts at ~7 deg) matter least.
# The whole change in each result is taken as a systematic error.
range_max = 10  # deg
popt_range = fit_background(scans["nominal"]["full"], range_max)[0]

# Duane-Hunt law: lambda_min = hc / (eU)  ->  U [kV] = 1239.84 / lambda_min [pm]
voltage = 1239.84 / lambda_min
voltage_stat = voltage * perr[1] / lambda_min

print("Bremsstrahlung fit:")
A_err, lambda_min_err, B_err, voltage_err = report(
    ["A", "lambda_min", "B", "U"],
    {variant: np.append(fit[0], 1239.84 / fit[0][1]) for variant, fit in background_fits.items()},
    np.append(perr, voltage_stat), ["", "pm", "", "kV"], [".4g", ".2f", ".3f", ".2f"],
    {"fit range": np.abs(np.append(popt_range, 1239.84 / popt_range[1]) - np.append(popt, voltage))})
print(f"    chi2/dof   = {chi2_nu:.2f} ({dof} dof), so the fit errors are scaled up by {np.sqrt(max(chi2_nu, 1)):.2f}")
print(f"    fit below {range_max} deg only: lambda_min = {popt_range[1]:.2f} pm, U = {1239.84 / popt_range[1]:.2f} kV")
print(f"    thermal error on d adds {voltage * thermal_d_over_d:.3f} kV to U (negligible, not included)")

angle_fine = np.linspace(angle_bg.min(), angle_bg.max(), 2000)
wavelength_fine = 2 * d_NaCl * np.sin(np.radians(angle_fine))

plt.figure(figsize=(10, 5))
plt.errorbar(angle_bg, rate_bg, yerr=sigma_bg, fmt=".", markersize=3, elinewidth=0.5,
             label="Background data (peaks removed)")
plt.plot(angle_fine, kramers(wavelength_fine, *popt), "r-", linewidth=1.5,
         label=f"Kramers fit, $\\lambda_{{min}}$ = {lambda_min:.2f} $\\pm$ {lambda_min_err:.2f} pm, "
               f"U = {voltage:.2f} $\\pm$ {voltage_err:.2f} kV ($\\chi^2_\\nu$ = {chi2_nu:.2f})")
plt.xlabel("Bragg angle, zero offset removed (deg)")
plt.ylabel("Count rate, dead-time corrected (1/s)")
plt.title("Cu anode, NaCl crystal: bremsstrahlung fit")
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()
plt.show()

# Checkmark 2
# Isolate the 1st, 2nd and 3rd order K-alpha peaks on a flat, zero baseline:
#   1. subtract the Kramers bremsstrahlung fit (only significant for the 1st order;
#      by 33 deg and 55 deg it has fallen to ~0)
#   2. fit a straight line through the flat regions either side of the peak and subtract it,
#      which removes any remaining tilt in the base (higher on one side than the other)
# The line is fitted only to the flanks, so the peak shape itself is untouched.
# Each scan is cropped to start/end at its flanks, which excludes the small neighbouring line
# at ~15.4 deg (1st order) and its 2nd order image at ~32.1 deg.
# The flank windows are in measured angle, so every variant uses the same points.
flank_windows = {"n1": ((15.7, 15.8), (16.4, 16.5)),
                 "n2": ((32.3, 32.9), (33.8, 35.5)),
                 "n3": ((53.5, 54.5), (55.7, 56.5))}


def subtract_background(scan, popt_kramers, left_flank, right_flank):
    measured_angle = scan[0]
    in_range = (measured_angle >= left_flank[0]) & (measured_angle <= right_flank[1])
    measured_angle, scan_angle, scan_rate, scan_sigma = (x[in_range] for x in scan)

    residual = scan_rate - kramers(2 * d_NaCl * np.sin(np.radians(scan_angle)), *popt_kramers)

    flanks = (((measured_angle >= left_flank[0]) & (measured_angle <= left_flank[1]))
              | ((measured_angle >= right_flank[0]) & (measured_angle <= right_flank[1])))
    slope, intercept = np.polyfit(scan_angle[flanks], residual[flanks], 1)
    return scan_angle, residual - (slope * scan_angle + intercept), scan_sigma, slope


# peaks[variant] holds (angle, rate, error, base tilt) for each order
peaks = {variant: [subtract_background(scans[variant][key], background_fits[variant][0], *flank_windows[key])
                   for key in peak_keys]
         for variant in variants}
slope_n1, slope_n2, slope_n3 = [peak[3] for peak in peaks["nominal"]]
print(f"Base tilt removed (1/s per deg): 1st order {slope_n1:.2f}, 2nd order {slope_n2:.2f}, 3rd order {slope_n3:.2f}")

# Smooth curve through each peak: modified Akima (makima) spline.
# It passes through every data point but, unlike a cubic spline, does not overshoot/ring
# at the sharp corners where the peak meets the flat base (the 1st order has only 11 points),
# and unlike PCHIP its maximum is not pinned to a data point.
splines = [Akima1DInterpolator(peak_angle, peak_rate, method="makima")
           for peak_angle, peak_rate, _, _ in peaks["nominal"]]

fig, axes = plt.subplots(1, 3, figsize=(14, 5))
for ax, colour, label, (peak_angle, peak_rate, peak_sigma, _), spline in zip(
        axes, colours, order_labels, peaks["nominal"], splines):
    angle_smooth = np.linspace(peak_angle.min(), peak_angle.max(), 2000)
    ax.errorbar(peak_angle, peak_rate, yerr=peak_sigma, fmt=".", color=colour, markersize=5, capsize=2, label="Data")
    ax.plot(angle_smooth, spline(angle_smooth), color=colour, linewidth=1.5, label="Makima spline")
    ax.axhline(0, color="k", linewidth=0.5)
    ax.legend()
    ax.set_title(f"{label} scan")
    ax.set_xlabel("Bragg angle, zero offset removed (deg)")
    ax.grid(alpha=0.3)
axes[0].set_ylabel("Count rate, background subtracted (1/s)")
fig.suptitle("Cu K-alpha peaks, dead-time corrected, bremsstrahlung and base tilt removed")
fig.tight_layout()
plt.show()


# Copy 2 of the peak subplots: Gaussian fit to each peak
def gaussian(x, amplitude, mean, sigma):
    return amplitude * np.exp(-(x - mean) ** 2 / (2 * sigma ** 2))


# Poisson weights from the raw counts (10 s per step), errors scaled up if chi2/dof > 1 (see weighted_fit).
# Returns the parameters, their statistical errors and chi2/dof.
def fit_gaussian(peak):
    peak_angle, peak_rate, peak_sigma, _ = peak
    p0 = [peak_rate.max(), peak_angle[np.argmax(peak_rate)], 0.1]
    popt_g, pcov_g, chi2_nu = weighted_fit(gaussian, peak_angle, peak_rate, peak_sigma, p0=p0)
    popt_g[2] = abs(popt_g[2])
    return popt_g, np.sqrt(np.diag(pcov_g)), chi2_nu


gaussian_fits = {variant: [fit_gaussian(peak) for peak in peaks[variant]] for variant in variants}
gaussian_errors = []  # total errors on [amplitude, mean, sigma], one row per order
for i, label in enumerate(order_labels):
    print(f"{label} Gaussian (chi2/dof = {gaussian_fits['nominal'][i][2]:.2f}):")
    gaussian_errors.append(report(["amplitude", "mean", "sigma"],
                                  {variant: fits[i][0] for variant, fits in gaussian_fits.items()},
                                  gaussian_fits["nominal"][i][1], ["1/s", "deg", "deg"], [".1f", ".4f", ".4f"]))

fig, axes = plt.subplots(1, 3, figsize=(14, 5))
for ax, colour, label, (peak_angle, peak_rate, peak_sigma, _), spline, (popt_g, _, _), err_g in zip(
        axes, colours, order_labels, peaks["nominal"], splines, gaussian_fits["nominal"], gaussian_errors):
    angle_smooth = np.linspace(peak_angle.min(), peak_angle.max(), 2000)
    ax.errorbar(peak_angle, peak_rate, yerr=peak_sigma, fmt=".", color=colour, markersize=5, capsize=2, label="Data")
    ax.plot(angle_smooth, spline(angle_smooth), color=colour, linewidth=1.5, label="Makima spline")
    ax.plot(angle_smooth, gaussian(angle_smooth, *popt_g), "k--", linewidth=1.2,
            label=f"Gaussian: $\\mu$ = {popt_g[1]:.4f} $\\pm$ {err_g[1]:.4f}$^\\circ$,\n"
                  f"$\\sigma$ = {popt_g[2]:.4f} $\\pm$ {err_g[2]:.4f}$^\\circ$")
    ax.axhline(0, color="k", linewidth=0.5)
    ax.set_ylim(top=1.5 * peak_rate.max())  # headroom so the legend clears the peak
    ax.legend(loc="upper left", fontsize=8)
    ax.set_title(f"{label} scan")
    ax.set_xlabel("Bragg angle, zero offset removed (deg)")
    ax.grid(alpha=0.3)
axes[0].set_ylabel("Count rate, background subtracted (1/s)")
fig.suptitle("Cu K-alpha peaks, dead-time corrected, bremsstrahlung and base tilt removed")
fig.tight_layout()
plt.show()


# Copy 3 of the peak subplots: sum of two Gaussians fitted to each peak
# All six parameters (amplitude, mean, sigma for each Gaussian) are free. The fit is to the measured
# points with the same Poisson weights as the single Gaussian, not to the spline: the spline's
# interpolated values carry no information beyond the data points they pass through.
# Starting guesses treat the peak as the Cu K-alpha1/K-alpha2 doublet: K-alpha2 at ~half the
# height, shifted by the expected splitting n * (154.44 - 154.06 pm) / (2d cos(theta)).
def two_gaussians(x, amplitude_1, mean_1, sigma_1, amplitude_2, mean_2, sigma_2):
    return gaussian(x, amplitude_1, mean_1, sigma_1) + gaussian(x, amplitude_2, mean_2, sigma_2)


def doublet_split(n, mean):
    return np.degrees(n * (154.44 - 154.06) / (2 * d_NaCl * np.cos(np.radians(mean))))


def fit_two_gaussians(n, peak, popt_g):
    peak_angle, peak_rate, peak_sigma, _ = peak
    split = doublet_split(n, popt_g[1])
    p0 = [0.67 * popt_g[0], popt_g[1] - split / 3, 0.8 * popt_g[2],
          0.33 * popt_g[0], popt_g[1] + 2 * split / 3, 0.8 * popt_g[2]]
    lower = [0, peak_angle.min(), 0, 0, peak_angle.min(), 0]
    upper = [np.inf, peak_angle.max(), np.inf, np.inf, peak_angle.max(), np.inf]
    popt_2g, pcov_2g, chi2_nu = weighted_fit(two_gaussians, peak_angle, peak_rate, peak_sigma, p0=p0,
                                             bounds=(lower, upper))
    return popt_2g, np.sqrt(np.diag(pcov_2g)), chi2_nu


pair_names = ["A1", "mean 1", "sigma 1", "A2", "mean 2", "sigma 2"]
pair_units = ["1/s", "deg", "deg"] * 2
pair_fmts = [".1f", ".4f", ".4f"] * 2


# Text box listing both Gaussians of a two-Gaussian fit with their total errors
def pair_text(popt_2g, err_2g):
    return "\n".join(f"Gaussian {i // 3 + 1}:\n"
                     f"  A = {popt_2g[i]:.1f} $\\pm$ {err_2g[i]:.1f} 1/s\n"
                     f"  $\\mu$ = {popt_2g[i + 1]:.4f} $\\pm$ {err_2g[i + 1]:.4f}$^\\circ$\n"
                     f"  $\\sigma$ = {popt_2g[i + 2]:.4f} $\\pm$ {err_2g[i + 2]:.4f}$^\\circ$" for i in (0, 3))


two_gaussian_fits = {variant: [fit_two_gaussians(n, peak, popt_g)
                               for n, peak, (popt_g, _, _) in zip([1, 2, 3], peaks[variant], gaussian_fits[variant])]
                     for variant in variants}
two_gaussian_errors = []
for i, label in enumerate(order_labels):
    print(f"{label} two-Gaussian fit (chi2/dof = {two_gaussian_fits['nominal'][i][2]:.2f}):")
    two_gaussian_errors.append(report(pair_names, {variant: fits[i][0] for variant, fits in two_gaussian_fits.items()},
                                      two_gaussian_fits["nominal"][i][1], pair_units, pair_fmts))

fig, axes = plt.subplots(1, 3, figsize=(15, 6))
for ax, colour, label, (peak_angle, peak_rate, peak_sigma, _), spline, (popt_2g, _, _), err_2g in zip(
        axes, colours, order_labels, peaks["nominal"], splines, two_gaussian_fits["nominal"], two_gaussian_errors):
    angle_smooth = np.linspace(peak_angle.min(), peak_angle.max(), 2000)
    ax.errorbar(peak_angle, peak_rate, yerr=peak_sigma, fmt=".", color=colour, markersize=5, capsize=2, label="Data")
    ax.plot(angle_smooth, spline(angle_smooth), color=colour, linewidth=1.5, label="Makima spline")
    ax.plot(angle_smooth, gaussian(angle_smooth, *popt_2g[:3]), ":", color="tab:red", linewidth=1.2, label="Gaussian 1")
    ax.plot(angle_smooth, gaussian(angle_smooth, *popt_2g[3:]), ":", color="tab:purple", linewidth=1.2, label="Gaussian 2")
    ax.plot(angle_smooth, two_gaussians(angle_smooth, *popt_2g), "k--", linewidth=1.2, label="Sum of Gaussians")
    ax.axhline(0, color="k", linewidth=0.5)
    ax.set_ylim(top=1.9 * peak_rate.max())  # headroom for the legend and annotation
    ax.legend(loc="upper left", fontsize=8)
    ax.text(0.98, 0.98, pair_text(popt_2g, err_2g),
            transform=ax.transAxes, ha="right", va="top", multialignment="left", fontsize=7,
            bbox=dict(boxstyle="round", facecolor="white", edgecolor="0.7"))
    ax.set_title(f"{label} scan")
    ax.set_xlabel("Bragg angle, zero offset removed (deg)")
    ax.grid(alpha=0.3)
axes[0].set_ylabel("Count rate, background subtracted (1/s)")
fig.suptitle("Cu K-alpha peaks, dead-time corrected: two-Gaussian fit to data")
fig.tight_layout()
plt.show()

# Copy 4 of the peak subplots: two-Gaussian fit to data with the intensity ratio fixed
# Gaussian 1 is K-alpha1 and Gaussian 2 is K-alpha2, with intensity ratio I2 / I1 = 0.518.
# The intensity of a line is the area under its Gaussian, A * sigma * sqrt(2 pi), not its height A.
# Both lines share one sigma: at these angles the width is set by the instrument (slits, divergence,
# crystal mosaic), which broadens two lines 0.04-0.2 deg apart identically, and the natural line
# widths (~0.005 deg) are negligible. With a shared sigma the area ratio equals the amplitude ratio,
# so fixing A2 = 0.518 * A1 fixes the intensity ratio. Free parameters: A1, both means, sigma.
amplitude_ratio = 0.518


def two_gaussians_fixed_ratio(x, amplitude_1, mean_1, mean_2, sigma):
    return two_gaussians(x, amplitude_1, mean_1, sigma, amplitude_ratio * amplitude_1, mean_2, sigma)


# Returns the six parameters (expanded so it plots the same way as the free fit), their statistical errors,
# the statistical error on the separation mean_2 - mean_1, which includes the correlation of the two means,
# and chi2/dof
def fit_fixed_ratio(n, peak, popt_g):
    peak_angle, peak_rate, peak_sigma, _ = peak
    split = doublet_split(n, popt_g[1])
    p0 = [0.67 * popt_g[0], popt_g[1] - split / 3, popt_g[1] + 2 * split / 3, 0.8 * popt_g[2]]
    lower = [0, peak_angle.min(), peak_angle.min(), 0]
    upper = [np.inf, peak_angle.max(), peak_angle.max(), np.inf]
    popt_r, pcov_r, chi2_nu = weighted_fit(two_gaussians_fixed_ratio, peak_angle, peak_rate, peak_sigma, p0=p0,
                                           bounds=(lower, upper))
    perr_r = np.sqrt(np.diag(pcov_r))
    popt_2g = np.array([popt_r[0], popt_r[1], popt_r[3], amplitude_ratio * popt_r[0], popt_r[2], popt_r[3]])
    perr_2g = np.array([perr_r[0], perr_r[1], perr_r[3], amplitude_ratio * perr_r[0], perr_r[2], perr_r[3]])
    separation_err = np.sqrt(pcov_r[1, 1] + pcov_r[2, 2] - 2 * pcov_r[1, 2])
    return popt_2g, perr_2g, separation_err, chi2_nu


ratio_fits = {variant: [fit_fixed_ratio(n, peak, popt_g)
                        for n, peak, (popt_g, _, _) in zip([1, 2, 3], peaks[variant], gaussian_fits[variant])]
              for variant in variants}
kalpha_lines = [("K-alpha1", 154.06, "tab:red"), ("K-alpha2", 154.44, "tab:purple")]  # pm


# Angle at which Bragg's law puts a line of this wavelength in order n
def bragg_angle(n, wavelength):
    return np.degrees(np.arcsin(n * wavelength / (2 * d_NaCl)))


# Mark where K-alpha1 and K-alpha2 are EXPECTED from Bragg's law (dotted vertical lines) and join each expected angle
# to the top of the matching fitted line (arrow). fitted - expected goes in the legend: at n=2 and n=3 the arrows are
# too short to carry a readable label. means and heights are those of the fitted K-alpha1 and K-alpha2 lines.
def mark_expected(ax, n, means, heights):
    for (name, wavelength, colour), mean, height in zip(kalpha_lines, means, heights):
        expected = bragg_angle(n, wavelength)
        ax.axvline(expected, linestyle=":", color=colour, linewidth=1.5,
                   label=f"Bragg's law, {name}: fit {mean - expected:+.3f}$^\\circ$")
        ax.annotate("", xy=(mean, height), xytext=(expected, height),
                    arrowprops=dict(arrowstyle="->", color=colour, linewidth=1.2, mutation_scale=10))


ratio_errors = []
for i, label in enumerate(order_labels):
    print(f"{label} two-Gaussian fit, I2/I1 = {amplitude_ratio}, shared sigma "
          f"(chi2/dof = {ratio_fits['nominal'][i][3]:.2f}):")
    ratio_errors.append(report(pair_names, {variant: fits[i][0] for variant, fits in ratio_fits.items()},
                               ratio_fits["nominal"][i][1], pair_units, pair_fmts))

fig, axes = plt.subplots(1, 3, figsize=(15, 6))
for ax, n, colour, label, (peak_angle, peak_rate, peak_sigma, _), spline, (popt_2g, _, _, _), err_2g in zip(
        axes, [1, 2, 3], colours, order_labels, peaks["nominal"], splines, ratio_fits["nominal"], ratio_errors):
    angle_smooth = np.linspace(peak_angle.min(), peak_angle.max(), 2000)
    ax.errorbar(peak_angle, peak_rate, yerr=peak_sigma, fmt=".", color=colour, markersize=5, capsize=2, label="Data")
    ax.plot(angle_smooth, spline(angle_smooth), color=colour, linewidth=1.5, label="Makima spline")
    ax.plot(angle_smooth, gaussian(angle_smooth, *popt_2g[:3]), ":", color="tab:red", linewidth=1.2, label="Gaussian 1 (K-alpha1)")
    ax.plot(angle_smooth, gaussian(angle_smooth, *popt_2g[3:]), ":", color="tab:purple", linewidth=1.2, label="Gaussian 2 (K-alpha2)")
    ax.plot(angle_smooth, two_gaussians(angle_smooth, *popt_2g), "k--", linewidth=1.2, label="Sum of Gaussians")
    mark_expected(ax, n, popt_2g[[1, 4]], popt_2g[[0, 3]])
    ax.axhline(0, color="k", linewidth=0.5)
    ax.set_ylim(top=1.9 * peak_rate.max())  # headroom for the legend and annotation
    ax.legend(loc="upper left", fontsize=8)
    ax.text(0.98, 0.98, pair_text(popt_2g, err_2g),
            transform=ax.transAxes, ha="right", va="top", multialignment="left", fontsize=7,
            bbox=dict(boxstyle="round", facecolor="white", edgecolor="0.7"))
    ax.set_title(f"{label} scan")
    ax.set_xlabel("Bragg angle, zero offset removed (deg)")
    ax.grid(alpha=0.3)
axes[0].set_ylabel("Count rate, background subtracted (1/s)")
fig.suptitle(f"Cu K-alpha peaks, dead-time corrected: two-Gaussian fit, I2/I1 = {amplitude_ratio}, shared $\\sigma$")
fig.tight_layout()
plt.show()

# Separation of the two summed Gaussians (K-alpha2 mean - K-alpha1 mean) in the ratio-constrained fit,
# against the Bragg's-law prediction. A constant zero offset moves both means equally, so it cancels here.
print("K-alpha doublet separation, ratio-constrained fit:")
for i, n in enumerate([1, 2, 3]):
    popt_2g, _, separation_err, _ = ratio_fits["nominal"][i]
    report([f"n={n}"], {variant: np.array([fits[i][0][4] - fits[i][0][1]]) for variant, fits in ratio_fits.items()},
           np.array([separation_err]), ["deg"], [".4f"])
    print(f"               predicted {doublet_split(n, popt_2g[1]):.4f} deg")


# K-alpha1 wavelength from each order, lambda = 2 d sin(mean_1) / n, against the reference value 154.06 pm.
# Every angle error moves it; in addition d is uncertain by thermal_d_over_d (lab temperature, temperature_error.py).
def kalpha1_wavelength(n, mean_1):
    return 2 * d_NaCl * np.sin(np.radians(mean_1)) / n


print("K-alpha1 wavelength from the ratio-constrained fit, lambda = 2 d sin(mean 1) / n:")
for i, n in enumerate([1, 2, 3]):
    popt_2g, perr_2g, _, _ = ratio_fits["nominal"][i]
    wavelength = kalpha1_wavelength(n, popt_2g[1])
    wavelength_stat = 2 * d_NaCl * np.cos(np.radians(popt_2g[1])) / n * np.radians(perr_2g[1])
    wavelength_err = report([f"n={n}"], {variant: np.array([kalpha1_wavelength(n, fits[i][0][1])])
                                         for variant, fits in ratio_fits.items()},
                            np.array([wavelength_stat]), ["pm"], [".2f"],
                            {"thermal": np.array([wavelength * thermal_d_over_d])})[0]
    print(f"               reference 154.06 pm (Bragg angle {np.degrees(np.arcsin(n * 154.06 / (2 * d_NaCl))):.3f} "
          f"deg): {(wavelength - 154.06) / wavelength_err:+.1f} sigma")


# Copy 5 of the peak subplots: Voigt line shape, one Voigt and two Voigts
# The Gaussian fits have chi2/dof well above 1 because the real peaks have slightly heavier tails than a Gaussian.
# A Voigt profile is a Gaussian (width sigma: the instrument - slits, divergence, crystal mosaic) convolved with a
# Lorentzian (half width gamma: the natural line shape and crystal effects), the standard line shape for X-ray
# diffraction peaks. gamma = 0 gives back the Gaussian. The two-Voigt fit has the same constraints as the
# two-Gaussian one: I2/I1 = 0.518 and both lines share sigma and gamma, so the height ratio is still the area ratio.
# Nominal variant only: this checks the line shape; the quoted results stay those of the Gaussian fits above.
def voigt(x, amplitude, mean, sigma, gamma):
    return amplitude * voigt_profile(x - mean, sigma, gamma) / voigt_profile(0, sigma, gamma)


def two_voigts_fixed_ratio(x, amplitude_1, mean_1, mean_2, sigma, gamma):
    return voigt(x, amplitude_1, mean_1, sigma, gamma) + voigt(x, amplitude_ratio * amplitude_1, mean_2, sigma, gamma)


# Starting guesses from the Gaussian fits, with a small Lorentzian part. Returns the parameters and fit errors of
# the single Voigt [amplitude, mean, sigma, gamma] and of the two Voigts [A1, mean_1, mean_2, sigma, gamma].
def fit_voigts(peak, popt_g, popt_2g):
    peak_angle, peak_rate, peak_sigma, _ = peak
    popt_1, pcov_1, _ = weighted_fit(voigt, peak_angle, peak_rate, peak_sigma,
                                     p0=[popt_g[0], popt_g[1], popt_g[2], 0.01],
                                     bounds=([0, peak_angle.min(), 1e-4, 0], [np.inf, peak_angle.max(), 1, 1]))
    popt_2, pcov_2, _ = weighted_fit(two_voigts_fixed_ratio, peak_angle, peak_rate, peak_sigma,
                                     p0=[popt_2g[0], popt_2g[1], popt_2g[4], popt_2g[2], 0.01],
                                     bounds=([0, peak_angle.min(), peak_angle.min(), 1e-4, 0],
                                             [np.inf, peak_angle.max(), peak_angle.max(), 1, 1]))
    return popt_1, np.sqrt(np.diag(pcov_1)), popt_2, np.sqrt(np.diag(pcov_2))


# chi^2 against the measured points and its dof, which also counts the 2 parameters of the flank baseline fitted to
# the same points (as in chi_squared.py). If the model were right, chi2/dof would scatter about 1 by sqrt(2/dof).
# tex=True writes chi and +/- as symbols for the figure.
def chi2_text(name, model, params, peak, tex=False):
    peak_angle, peak_rate, peak_sigma, _ = peak
    chi2 = np.sum(((peak_rate - model(peak_angle, *params)) / peak_sigma) ** 2)
    dof = len(peak_rate) - len(params) - 2
    chi, pm = ("$\\chi^2$", "$\\pm$") if tex else ("chi2", "+/-")
    return f"{name}: {chi} = {chi2:.1f}/{dof} = {chi2 / dof:.2f} {pm} {np.sqrt(2 / dof):.2f}"


voigt_fits = [fit_voigts(peak, popt_g, popt_2g) for peak, (popt_g, _, _), (popt_2g, _, _, _)
              in zip(peaks["nominal"], gaussian_fits["nominal"], ratio_fits["nominal"])]
for label, peak, (popt_1, perr_1, popt_2, perr_2), (popt_g, _, _), (popt_2g, _, _, _) in zip(
        order_labels, peaks["nominal"], voigt_fits, gaussian_fits["nominal"], ratio_fits["nominal"]):
    print(f"{label} Voigt fits (nominal variant, fit errors only):")
    print(f"    one Voigt:  mean = {popt_1[1]:.4f} +/- {perr_1[1]:.4f} deg, sigma = {popt_1[2]:.4f} deg, "
          f"gamma = {popt_1[3]:.4f} deg")
    print(f"    two Voigts: mean 1 = {popt_2[1]:.4f} +/- {perr_2[1]:.4f} deg, mean 2 = {popt_2[2]:.4f} +/- "
          f"{perr_2[2]:.4f} deg, sigma = {popt_2[3]:.4f} deg, gamma = {popt_2[4]:.4f} deg")
    for text in [chi2_text("one Voigt    ", voigt, popt_1, peak),
                 chi2_text("two Voigts   ", two_voigts_fixed_ratio, popt_2, peak),
                 chi2_text("one Gaussian ", gaussian, popt_g, peak),
                 chi2_text("two Gaussians", two_gaussians_fixed_ratio, popt_2g[[0, 1, 4, 2]], peak)]:
        print(f"    {text}")

fig, axes = plt.subplots(1, 3, figsize=(15, 6))
for ax, n, colour, label, peak, (popt_1, _, popt_2, _), (popt_g, _, _), (popt_2g, _, _, _) in zip(
        axes, [1, 2, 3], colours, order_labels, peaks["nominal"], voigt_fits, gaussian_fits["nominal"],
        ratio_fits["nominal"]):
    peak_angle, peak_rate, peak_sigma, _ = peak
    angle_smooth = np.linspace(peak_angle.min(), peak_angle.max(), 2000)
    ax.errorbar(peak_angle, peak_rate, yerr=peak_sigma, fmt=".", color=colour, markersize=5, capsize=2, label="Data")
    ax.plot(angle_smooth, voigt(angle_smooth, *popt_1), "-.", color="0.4", linewidth=1.2, label="One Voigt")
    ax.plot(angle_smooth, voigt(angle_smooth, *popt_2[[0, 1, 3, 4]]), ":", color="tab:red", linewidth=1.2,
            label="Voigt 1 (K-alpha1)")
    ax.plot(angle_smooth, voigt(angle_smooth, amplitude_ratio * popt_2[0], *popt_2[[2, 3, 4]]), ":",
            color="tab:purple", linewidth=1.2, label="Voigt 2 (K-alpha2)")
    ax.plot(angle_smooth, two_voigts_fixed_ratio(angle_smooth, *popt_2), "k--", linewidth=1.2, label="Sum of Voigts")
    mark_expected(ax, n, popt_2[[1, 2]], [popt_2[0], amplitude_ratio * popt_2[0]])
    ax.axhline(0, color="k", linewidth=0.5)
    ax.set_ylim(top=1.9 * peak_rate.max())  # headroom for the legend and annotation
    ax.legend(loc="upper left", fontsize=7)  # smaller than the other figures so it clears the chi2 box
    ax.text(0.98, 0.98, "\n".join([chi2_text("One Voigt", voigt, popt_1, peak, tex=True),
                                   chi2_text("Two Voigts", two_voigts_fixed_ratio, popt_2, peak, tex=True),
                                   "Gaussians, for comparison:",
                                   chi2_text("  one", gaussian, popt_g, peak, tex=True),
                                   chi2_text("  two", two_gaussians_fixed_ratio, popt_2g[[0, 1, 4, 2]], peak,
                                             tex=True)]),
            transform=ax.transAxes, ha="right", va="top", multialignment="left", fontsize=7,
            bbox=dict(boxstyle="round", facecolor="white", edgecolor="0.7"))
    ax.set_title(f"{label} scan")
    ax.set_xlabel("Bragg angle, zero offset removed (deg)")
    ax.grid(alpha=0.3)
axes[0].set_ylabel("Count rate, background subtracted (1/s)")
fig.suptitle(f"Cu K-alpha peaks, dead-time corrected: one Voigt and two Voigts (I2/I1 = {amplitude_ratio}, "
             f"shared $\\sigma$ and $\\gamma$)")
fig.tight_layout()
plt.show()

# Copy 6 of the peak subplots: pseudo-Voigt line shape, one pseudo-Voigt and two pseudo-Voigts
# A pseudo-Voigt approximates the Voigt by a weighted sum, not a convolution, of a Gaussian and a Lorentzian of the
# same FWHM: eta = 0 is a pure Gaussian, eta = 1 a pure Lorentzian. It is within ~1% of a true Voigt but has a single
# width, so unlike the Voigt it cannot split the broadening into instrument (sigma) and natural line (gamma).
# The two-pseudo-Voigt fit has the same constraints as the two-Voigt one: I2/I1 = 0.518 and both lines share fwhm and
# eta, so the height ratio is still the area ratio. Nominal variant only, as for the Voigts: this checks whether the
# choice of line shape moves the fitted peak positions.
def pseudo_voigt(x, amplitude, mean, fwhm, eta):
    u = (x - mean) / fwhm
    return amplitude * (eta / (1 + 4 * u ** 2) + (1 - eta) * np.exp(-4 * np.log(2) * u ** 2))


def two_pseudo_voigts_fixed_ratio(x, amplitude_1, mean_1, mean_2, fwhm, eta):
    return (pseudo_voigt(x, amplitude_1, mean_1, fwhm, eta)
            + pseudo_voigt(x, amplitude_ratio * amplitude_1, mean_2, fwhm, eta))


# FWHM of a Voigt from its sigma and gamma, and the eta of the pseudo-Voigt that best matches it
# (Thompson, Cox and Hastings 1987). Used as starting guesses and to compare the fitted eta with the Voigt fits.
def voigt_fwhm_eta(sigma, gamma):
    f_g, f_l = 2 * np.sqrt(2 * np.log(2)) * sigma, 2 * gamma
    fwhm = (f_g ** 5 + 2.69269 * f_g ** 4 * f_l + 2.42843 * f_g ** 3 * f_l ** 2 + 4.47163 * f_g ** 2 * f_l ** 3
            + 0.07842 * f_g * f_l ** 4 + f_l ** 5) ** (1 / 5)
    ratio = f_l / fwhm
    return fwhm, 1.36603 * ratio - 0.47719 * ratio ** 2 + 0.11116 * ratio ** 3


# Starting guesses from the Voigt fits. Returns the parameters and fit errors of the single pseudo-Voigt
# [amplitude, mean, fwhm, eta] and of the two pseudo-Voigts [A1, mean_1, mean_2, fwhm, eta], and the fit error on
# the separation mean_2 - mean_1, which includes the correlation of the two means.
def fit_pseudo_voigts(peak, popt_1v, popt_2v):
    peak_angle, peak_rate, peak_sigma, _ = peak
    popt_1, pcov_1, _ = weighted_fit(pseudo_voigt, peak_angle, peak_rate, peak_sigma,
                                     p0=[popt_1v[0], popt_1v[1], *voigt_fwhm_eta(*popt_1v[2:])],
                                     bounds=([0, peak_angle.min(), 1e-4, 0], [np.inf, peak_angle.max(), 2, 1]))
    popt_2, pcov_2, _ = weighted_fit(two_pseudo_voigts_fixed_ratio, peak_angle, peak_rate, peak_sigma,
                                     p0=[popt_2v[0], popt_2v[1], popt_2v[2], *voigt_fwhm_eta(*popt_2v[3:])],
                                     bounds=([0, peak_angle.min(), peak_angle.min(), 1e-4, 0],
                                             [np.inf, peak_angle.max(), peak_angle.max(), 2, 1]))
    separation_err = np.sqrt(pcov_2[1, 1] + pcov_2[2, 2] - 2 * pcov_2[1, 2])
    return popt_1, np.sqrt(np.diag(pcov_1)), popt_2, np.sqrt(np.diag(pcov_2)), separation_err


pseudo_voigt_fits = [fit_pseudo_voigts(peak, popt_1v, popt_2v)
                     for peak, (popt_1v, _, popt_2v, _) in zip(peaks["nominal"], voigt_fits)]
for n, label, peak, (popt_1, perr_1, popt_2, perr_2, separation_err), (popt_1v, _, popt_2v, _) in zip(
        [1, 2, 3], order_labels, peaks["nominal"], pseudo_voigt_fits, voigt_fits):
    fwhm_1v, eta_1v = voigt_fwhm_eta(*popt_1v[2:])
    fwhm_2v, eta_2v = voigt_fwhm_eta(*popt_2v[3:])
    print(f"{label} pseudo-Voigt fits (nominal variant, fit errors only; Voigt equivalent in brackets):")
    print(f"    one pseudo-Voigt:  mean = {popt_1[1]:.4f} +/- {perr_1[1]:.4f} deg ({popt_1v[1]:.4f}), "
          f"fwhm = {popt_1[2]:.4f} deg ({fwhm_1v:.4f}), eta = {popt_1[3]:.3f} +/- {perr_1[3]:.3f} ({eta_1v:.3f})")
    print(f"    two pseudo-Voigts: mean 1 = {popt_2[1]:.4f} +/- {perr_2[1]:.4f} deg ({popt_2v[1]:.4f}), "
          f"mean 2 = {popt_2[2]:.4f} +/- {perr_2[2]:.4f} deg ({popt_2v[2]:.4f}),")
    print(f"                       fwhm = {popt_2[3]:.4f} deg ({fwhm_2v:.4f}), "
          f"eta = {popt_2[4]:.3f} +/- {perr_2[4]:.3f} ({eta_2v:.3f})")
    for text in [chi2_text("one pseudo-Voigt ", pseudo_voigt, popt_1, peak),
                 chi2_text("two pseudo-Voigts", two_pseudo_voigts_fixed_ratio, popt_2, peak),
                 chi2_text("one Voigt        ", voigt, popt_1v, peak),
                 chi2_text("two Voigts       ", two_voigts_fixed_ratio, popt_2v, peak)]:
        print(f"    {text}")
    # Change from the Voigt to the pseudo-Voigt, in units of the pseudo-Voigt fit error
    separation, separation_v = popt_2[2] - popt_2[1], popt_2v[2] - popt_2v[1]
    wavelength, wavelength_v = kalpha1_wavelength(n, popt_2[1]), kalpha1_wavelength(n, popt_2v[1])
    wavelength_stat = 2 * d_NaCl * np.cos(np.radians(popt_2[1])) / n * np.radians(perr_2[1])
    print(f"    pseudo-Voigt - Voigt: mean 1 {popt_2[1] - popt_2v[1]:+.4f} deg "
          f"({(popt_2[1] - popt_2v[1]) / perr_2[1]:+.2f} fit sigma), separation {separation:.4f} +/- "
          f"{separation_err:.4f} deg vs {separation_v:.4f} ({(separation - separation_v) / separation_err:+.2f} fit sigma; "
          f"predicted {doublet_split(n, popt_2[1]):.4f})")
    print(f"                          K-alpha1 wavelength {wavelength:.2f} pm vs {wavelength_v:.2f} pm "
          f"({(wavelength - wavelength_v) / wavelength_stat:+.2f} fit sigma)")

fig, axes = plt.subplots(1, 3, figsize=(15, 6))
for ax, n, colour, label, peak, (popt_1, _, popt_2, _, _), (popt_1v, _, popt_2v, _) in zip(
        axes, [1, 2, 3], colours, order_labels, peaks["nominal"], pseudo_voigt_fits, voigt_fits):
    peak_angle, peak_rate, peak_sigma, _ = peak
    angle_smooth = np.linspace(peak_angle.min(), peak_angle.max(), 2000)
    ax.errorbar(peak_angle, peak_rate, yerr=peak_sigma, fmt=".", color=colour, markersize=5, capsize=2, label="Data")
    ax.plot(angle_smooth, pseudo_voigt(angle_smooth, *popt_1), "-.", color="0.4", linewidth=1.2,
            label="One pseudo-Voigt")
    ax.plot(angle_smooth, pseudo_voigt(angle_smooth, *popt_2[[0, 1, 3, 4]]), ":", color="tab:red", linewidth=1.2,
            label="Pseudo-Voigt 1 (K-alpha1)")
    ax.plot(angle_smooth, pseudo_voigt(angle_smooth, amplitude_ratio * popt_2[0], *popt_2[[2, 3, 4]]), ":",
            color="tab:purple", linewidth=1.2, label="Pseudo-Voigt 2 (K-alpha2)")
    ax.plot(angle_smooth, two_pseudo_voigts_fixed_ratio(angle_smooth, *popt_2), "k--", linewidth=1.2,
            label="Sum of pseudo-Voigts")
    mark_expected(ax, n, popt_2[[1, 2]], [popt_2[0], amplitude_ratio * popt_2[0]])
    ax.axhline(0, color="k", linewidth=0.5)
    ax.set_ylim(top=1.9 * peak_rate.max())  # headroom for the legend and annotation
    ax.legend(loc="upper left", fontsize=7)
    ax.text(0.98, 0.98, "\n".join([chi2_text("One pV", pseudo_voigt, popt_1, peak, tex=True),
                                   chi2_text("Two pV", two_pseudo_voigts_fixed_ratio, popt_2, peak, tex=True),
                                   f"$\\eta$ = {popt_2[4]:.3f}",
                                   "Voigts, for comparison:",
                                   chi2_text("  one", voigt, popt_1v, peak, tex=True),
                                   chi2_text("  two", two_voigts_fixed_ratio, popt_2v, peak, tex=True)]),
            transform=ax.transAxes, ha="right", va="top", multialignment="left", fontsize=7,
            bbox=dict(boxstyle="round", facecolor="white", edgecolor="0.7"))
    ax.set_title(f"{label} scan")
    ax.set_xlabel("Bragg angle, zero offset removed (deg)")
    ax.grid(alpha=0.3)
axes[0].set_ylabel("Count rate, background subtracted (1/s)")
fig.suptitle(f"Cu K-alpha peaks, dead-time corrected: one and two pseudo-Voigts (I2/I1 = {amplitude_ratio}, "
             f"shared FWHM and $\\eta$)")
fig.tight_layout()
plt.show()
