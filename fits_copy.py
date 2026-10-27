import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from scipy.interpolate import Akima1DInterpolator
from scipy.special import voigt_profile
from save_figure import save_figure, report_style, report_text_size

# Calibrations and systematic errors, each worked out in its own file (run that file to see its fits and plots):
#   deadtime.py           tau, tau_err                 GM-tube dead time (tube-current scan)
#   angle_offset.py       zero_offset, zero_offset_err goniometer zero (direct-beam scan)
#   alignment_error.py    sigma_align                  crystal alignment: half the Kα n=1 shift when the crystal is
#                                                      turned 180 deg about the beam axis, deg
#   temperature_error.py  sigma_heat                   crystal heating: Kα n=1 shift between the unheated and heated
#                                                      scans, deg (also covers run-to-run stability)
from deadtime import tau, tau_err
from angle_offset import zero_offset, zero_offset_err
from alignment_error import sigma_align
from temperature_error import sigma_heat

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
# Alignment and heating are both errors on where the angle scale really is, so like the zero offset they are
# applied as a shift of every angle. A shift common to all angles cancels in a doublet separation.
# Each source: (name used in the error budget, (tau, angle shift) at -1 sigma, at +1 sigma).
error_sources = {"tau": ("dead time", (tau - tau_err, zero_offset), (tau + tau_err, zero_offset)),
                 "offset": ("zero offset", (tau, zero_offset - zero_offset_err), (tau, zero_offset + zero_offset_err)),
                 "align": ("alignment", (tau, zero_offset - sigma_align), (tau, zero_offset + sigma_align)),
                 "heat": ("heating", (tau, zero_offset - sigma_heat), (tau, zero_offset + sigma_heat))}
variants = {"nominal": (tau, zero_offset)}
for key, (_, low, high) in error_sources.items():
    variants[f"{key} low"], variants[f"{key} high"] = low, high


# Error budget of fitted values: the total errors and each systematic error.
# results maps each variant to an array of fitted values; stat holds their statistical errors from the nominal fit.
# extra optionally maps the name of a further systematic error, not from the variants, to its array of errors.
def error_budget(results, stat, extra={}):
    sys = {name: np.abs(results[f"{key} high"] - results[f"{key} low"]) / 2
           for key, (name, _, _) in error_sources.items()}
    sys.update(extra)
    return np.sqrt(stat ** 2 + sum(errors ** 2 for errors in sys.values())), sys


# curve_fit with the Poisson error bars taken as absolute (the gate times are known). If chi2/dof > 1 the model
# misses something the error bars do not cover, so the parameter errors are scaled up by sqrt(chi2/dof); they are
# never scaled down below the Poisson level. extra_params counts parameters already fitted to the same points (the
# flank baseline under a peak), which use up degrees of freedom too. Returns the parameters, their covariance and
# chi2/dof.
def weighted_fit(model, x, y, sigma, extra_params=0, **kwargs):
    popt, pcov = curve_fit(model, x, y, sigma=sigma, absolute_sigma=True, **kwargs)
    chi2_nu = np.sum(((y - model(x, *popt)) / sigma) ** 2) / (len(y) - len(popt) - extra_params)
    return popt, pcov * max(chi2_nu, 1), chi2_nu


scans = {variant: {"full": correct(measured_full, gate_full, *shift),
                   "n1": correct(measured_n1, gate_peaks, *shift),
                   "n2": correct(measured_n2, gate_peaks, *shift),
                   "n3": correct(measured_n3, gate_peaks, *shift)}
         for variant, shift in variants.items()}
_, angle, rate, sigma = scans["nominal"]["full"]

# Highest measured rate of each scan and the dead-time correction factor there
highest_rates = {label: measured_rate.max() for label, (_, measured_rate)
                 in zip(["Full scan"] + order_labels, [measured_full, measured_n1, measured_n2, measured_n3])}
deadtime_factors = {label: 1 / (1 - highest_rate * tau) for label, highest_rate in highest_rates.items()}


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

# Tube current of each scan, from its .xry file (see the run table in fits_documentation.txt), for the legend
currents = {"full": 0.12, "n1": 0.12, "n2": 1.0, "n3": 1.0}  # mA

# The full scan is left out where the 2nd and 3rd order scans cover it, so each of those peaks is drawn once
full_rate_shown = rate.copy()
for key in ["n2", "n3"]:
    scan_angle = scans["nominal"][key][1]
    full_rate_shown[(angle >= scan_angle.min()) & (angle <= scan_angle.max())] = np.nan

# Report text size on this figure only (the later figures keep their own sizes for now)
with plt.rc_context(report_style):
    plt.figure(figsize=(10, 5))
    plt.errorbar(angle, full_rate_shown, yerr=sigma, linewidth=1, elinewidth=0.5,
                 label=f"Full scan ({currents['full']} mA)")
    for key, label, shifted_rate in zip(peak_keys, order_labels, [rate_n1, rate_n2, rate_n3]):
        _, scan_angle, _, scan_sigma = scans["nominal"][key]
        plt.errorbar(scan_angle, shifted_rate, yerr=scan_sigma, linewidth=1, elinewidth=0.5,
                     label=f"{label} scan ({currents[key]} mA)")
    plt.xlabel("Bragg angle (deg)")
    plt.ylabel("Count rate (/s)")
    plt.title("Full spectrum scan")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    save_figure("fits_1_full_spectrum")
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
background_dof = np.count_nonzero(background) - len(popt)

# Fit-range check: the same fit to the background below 10 deg only, near the onset, where the simplifications of
# the model (the empirical absorption term, and no 2nd-order bremsstrahlung, which starts at ~7 deg) matter least.
# The whole change in each result is taken as a systematic error.
range_max = 10  # deg
popt_range = fit_background(scans["nominal"]["full"], range_max)[0]

# Duane-Hunt law: lambda_min = hc / (eU)  ->  U [kV] = 1239.84 / lambda_min [pm]
voltage = 1239.84 / lambda_min
voltage_stat = voltage * perr[1] / lambda_min
voltage_range = 1239.84 / popt_range[1]

# Error budget of [A, lambda_min, B, U]: (total errors, each systematic error)
background_budget = error_budget(
    {variant: np.append(fit[0], 1239.84 / fit[0][1]) for variant, fit in background_fits.items()},
    np.append(perr, voltage_stat),
    {"fit range": np.abs(np.append(popt_range, voltage_range) - np.append(popt, voltage))})
A_err, lambda_min_err, B_err, voltage_err = background_budget[0]

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


# The baseline's slope and intercept were fitted to these same points, so every peak fit below has
# dof = points - its own parameters - baseline_params (also in chi2_dof and chi_squared.py).
baseline_params = 2


# peaks[variant] holds (angle, rate, error, base tilt) for each order
peaks = {variant: [subtract_background(scans[variant][key], background_fits[variant][0], *flank_windows[key])
                   for key in peak_keys]
         for variant in variants}
slope_n1, slope_n2, slope_n3 = [peak[3] for peak in peaks["nominal"]]

# Smooth curve through each peak: modified Akima (makima) spline.
# It passes through every data point but, unlike a cubic spline, does not overshoot/ring
# at the sharp corners where the peak meets the flat base (the 1st order has only 11 points),
# and unlike PCHIP its maximum is not pinned to a data point.
splines = [Akima1DInterpolator(peak_angle, peak_rate, method="makima")
           for peak_angle, peak_rate, _, _ in peaks["nominal"]]


# Gaussian fit to each peak
def gaussian(x, amplitude, mean, sigma):
    return amplitude * np.exp(-(x - mean) ** 2 / (2 * sigma ** 2))


# Poisson weights from the raw counts (10 s per step), errors scaled up if chi2/dof > 1 (see weighted_fit).
# Returns the parameters, their statistical errors and chi2/dof.
def fit_gaussian(peak):
    peak_angle, peak_rate, peak_sigma, _ = peak
    p0 = [peak_rate.max(), peak_angle[np.argmax(peak_rate)], 0.1]
    popt_g, pcov_g, chi2_nu = weighted_fit(gaussian, peak_angle, peak_rate, peak_sigma, baseline_params, p0=p0)
    popt_g[2] = abs(popt_g[2])
    return popt_g, np.sqrt(np.diag(pcov_g)), chi2_nu


gaussian_fits = {variant: [fit_gaussian(peak) for peak in peaks[variant]] for variant in variants}
# Error budget of [amplitude, mean, sigma] for each order: (total errors, each systematic error)
gaussian_budgets = [error_budget({variant: fits[i][0] for variant, fits in gaussian_fits.items()},
                                 gaussian_fits["nominal"][i][1])
                    for i in range(len(order_labels))]


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
    popt_2g, pcov_2g, chi2_nu = weighted_fit(two_gaussians, peak_angle, peak_rate, peak_sigma, baseline_params,
                                             p0=p0, bounds=(lower, upper))
    return popt_2g, np.sqrt(np.diag(pcov_2g)), chi2_nu


two_gaussian_fits = {variant: [fit_two_gaussians(n, peak, popt_g)
                               for n, peak, (popt_g, _, _) in zip([1, 2, 3], peaks[variant], gaussian_fits[variant])]
                     for variant in variants}
# Error budget of [A1, mean 1, sigma 1, A2, mean 2, sigma 2] for each order: (total errors, each systematic error)
two_gaussian_budgets = [error_budget({variant: fits[i][0] for variant, fits in two_gaussian_fits.items()},
                                     two_gaussian_fits["nominal"][i][1])
                        for i in range(len(order_labels))]
two_gaussian_errors = [total for total, _ in two_gaussian_budgets]

# Helpers shared by the three-panel peak figures 5, 6 and 8
# (name, wavelength in pm, colour, name for figure labels)
kalpha_lines = [("K-alpha1", 154.06, "tab:red", "K$\\alpha_1$"), ("K-alpha2", 154.44, "tab:purple", "K$\\alpha_2$")]


# Angle at which Bragg's law puts a line of this wavelength in order n
def bragg_angle(n, wavelength):
    return np.degrees(np.arcsin(n * wavelength / (2 * d_NaCl)))


# Mark where K-alpha1 and K-alpha2 are EXPECTED from Bragg's law (dash-dot vertical lines, so their legend entries
# differ from the dotted fitted lines) and join each expected angle to the top of the matching fitted line (arrow).
# means and heights are those of the fitted K-alpha1 and K-alpha2 lines. fitted - expected is printed, not drawn: at
# n=2 and n=3 the arrows are too short to carry a readable label.
def mark_expected(ax, n, means, heights):
    for (name, wavelength, colour, tex_name), mean, height in zip(kalpha_lines, means, heights):
        expected = bragg_angle(n, wavelength)
        ax.axvline(expected, linestyle="-.", color=colour, linewidth=1.5, label=f"Bragg's law {tex_name}")
        ax.annotate("", xy=(mean, height), xytext=(expected, height),
                    arrowprops=dict(arrowstyle="->", color=colour, linewidth=1.2, mutation_scale=10))


# The suptitles of figures 5, 6 and 8 are centred over the gap from the right edge of the first panel to the left edge
# of the third. Without fontsize the title is sized to span that gap exactly (figure 5); figures 6 and 8 pass figure
# 5's size. Lays the figure out and returns the font size.
def place_peak_title(fig, axes, title, fontsize=None):
    text = fig.suptitle(title, fontsize=fontsize)
    for _ in range(3):  # tight_layout moves the panels a little each time the title changes size
        fig.tight_layout()
        left, right = axes[0].get_window_extent().x1, axes[2].get_window_extent().x0
        if fontsize is None:
            text.set_fontsize(text.get_fontsize() * (right - left) / text.get_window_extent().width)
        text.set_x((left + right) / 2 / fig.bbox.width)
    return text.get_fontsize()


# x-range of each order's panel: centred on that order's single-Gaussian centre and as many of its single-Gaussian
# sigmas wide on each side as the 1st-order scan reaches (about 4.9), plus 5%, so the 2nd and 3rd order peaks fill
# their panels the way the 1st-order peak does
first_angle, (first_centre, first_sigma) = peaks["nominal"][0][0], gaussian_fits["nominal"][0][0][1:]
half_width_sigmas = 1.05 * max(first_centre - first_angle.min(), first_angle.max() - first_centre) / first_sigma
peak_xlims = [(popt_g[1] - half_width_sigmas * popt_g[2], popt_g[1] + half_width_sigmas * popt_g[2])
              for popt_g, _, _ in gaussian_fits["nominal"]]
peak_legend_size = 0.75 * report_text_size  # asked for: these legends 25% smaller than the report text


# Raise the top of each panel's y-axis until its legend covers no data point or fitted curve (the Bragg's-law lines
# run the full height, so they are left out). Call after the figure is laid out.
def clear_legends(axes):
    for ax in axes:
        legend_box = ax.get_legend().get_window_extent().padded(12)  # a clear gap, not just no overlap
        curves = [line for line in ax.lines if line.get_transform() == ax.transData]
        for _ in range(40):
            points = np.concatenate([ax.transData.transform(line.get_xydata()) for line in curves])
            if not legend_box.count_contains(points):
                break
            ax.set_ylim(top=1.05 * ax.get_ylim()[1])


# The fit does not order the two Gaussians, so K-alpha1 (the shorter wavelength) is whichever sits at the lower angle.
# For each order: where K-alpha1's and then K-alpha2's (A, mu, sigma) start in popt_2g.
kalpha_starts = [sorted((0, 3), key=lambda i: popt_2g[i + 1]) for popt_2g, _, _ in two_gaussian_fits["nominal"]]

# The fitted parameters are printed rather than drawn on the figure
for label, (popt_2g, _, _), err_2g, starts in zip(order_labels, two_gaussian_fits["nominal"], two_gaussian_errors,
                                                  kalpha_starts):
    for name, i in zip(["K-alpha1", "K-alpha2"], starts):
        print(f"{label}, two Gaussians free, {name}: A = {popt_2g[i]:.1f} +/- {err_2g[i]:.1f} /s, "
              f"mu = {popt_2g[i + 1]:.4f} +/- {err_2g[i + 1]:.4f} deg, sigma = {popt_2g[i + 2]:.4f} +/- "
              f"{err_2g[i + 2]:.4f} deg")

with plt.rc_context(report_style):
    fig, axes = plt.subplots(1, 3, figsize=(18, 6.5))
    for ax, n, label, (peak_angle, peak_rate, peak_sigma, _), (popt_2g, _, _), starts in zip(
            axes, [1, 2, 3], order_labels, peaks["nominal"], two_gaussian_fits["nominal"], kalpha_starts):
        kalpha1, kalpha2 = [popt_2g[i:i + 3] for i in starts]
        angle_smooth = np.linspace(peak_angle.min(), peak_angle.max(), 2000)
        ax.errorbar(peak_angle, peak_rate, yerr=peak_sigma, fmt="o", color="grey", markersize=4, capsize=2,
                    label="Data")
        ax.plot(angle_smooth, gaussian(angle_smooth, *kalpha1), ":", color="tab:red", linewidth=1.5,
                label="K$\\alpha_1$")
        ax.plot(angle_smooth, gaussian(angle_smooth, *kalpha2), ":", color="tab:purple", linewidth=1.5,
                label="K$\\alpha_2$")
        ax.plot(angle_smooth, two_gaussians(angle_smooth, *popt_2g), "k-", linewidth=1.5,
                label="Superposed double Gaussian")
        mark_expected(ax, n, (kalpha1[1], kalpha2[1]), (kalpha1[0], kalpha2[0]))
        ax.axhline(0, color="k", linewidth=0.5)
        ax.set_xlim(*peak_xlims[n - 1])
        ax.set_ylim(top=1.15 * peak_rate.max())  # raised by clear_legends if the legend would cover the data
        ax.legend(loc="upper left", fontsize=peak_legend_size)
        ax.set_title(f"{label} scan")
        ax.set_xlabel("Bragg angle (deg)")
        ax.grid(alpha=0.3)
    axes[0].set_ylabel("Count rate (/s)")
    peak_title_size = place_peak_title(fig, axes, "Superposed double Gaussian fit (parameters: all free)")
    clear_legends(axes)
    save_figure("fits_5_two_gaussians_free", fig)
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
    popt_r, pcov_r, chi2_nu = weighted_fit(two_gaussians_fixed_ratio, peak_angle, peak_rate, peak_sigma,
                                           baseline_params, p0=p0, bounds=(lower, upper))
    perr_r = np.sqrt(np.diag(pcov_r))
    popt_2g = np.array([popt_r[0], popt_r[1], popt_r[3], amplitude_ratio * popt_r[0], popt_r[2], popt_r[3]])
    perr_2g = np.array([perr_r[0], perr_r[1], perr_r[3], amplitude_ratio * perr_r[0], perr_r[2], perr_r[3]])
    separation_err = np.sqrt(pcov_r[1, 1] + pcov_r[2, 2] - 2 * pcov_r[1, 2])
    return popt_2g, perr_2g, separation_err, chi2_nu


ratio_fits = {variant: [fit_fixed_ratio(n, peak, popt_g)
                        for n, peak, (popt_g, _, _) in zip([1, 2, 3], peaks[variant], gaussian_fits[variant])]
              for variant in variants}
ratio_budgets, mean_1_budgets = [], []
for i in range(len(order_labels)):
    mean_1_budgets.append(error_budget({variant: fits[i][0][[1]] for variant, fits in ratio_fits.items()},
                                       ratio_fits["nominal"][i][1][[1]]))
    ratio_budgets.append(error_budget({variant: fits[i][0] for variant, fits in ratio_fits.items()},
                                      ratio_fits["nominal"][i][1]))
ratio_errors = [total for total, _ in ratio_budgets]

# The fitted parameters, their offsets from Bragg's law and the error budget of the K-alpha1 centre are printed rather
# than drawn on the figure. Gaussian 1 is K-alpha1 (the larger line, A2 = 0.518 A1).
for n, label, (popt_2g, perr_2g, _, _), err_2g, (total, sys) in zip([1, 2, 3], order_labels, ratio_fits["nominal"],
                                                                     ratio_errors, mean_1_budgets):
    for i, (name, wavelength, _, _) in zip((0, 3), kalpha_lines):
        print(f"{label}, fixed ratio, {name}: A = {popt_2g[i]:.1f} +/- {err_2g[i]:.1f} /s, mu = {popt_2g[i + 1]:.4f} "
              f"+/- {err_2g[i + 1]:.4f} deg (fit - Bragg's law {popt_2g[i + 1] - bragg_angle(n, wavelength):+.3f}), "
              f"sigma = {popt_2g[i + 2]:.4f} +/- {err_2g[i + 2]:.4f} deg")
    print(f"{label}, fixed ratio, K-alpha1 centre error budget (deg): fit {perr_2g[1]:.4f}, "
          + ", ".join(f"{source} {errors[0]:.4f}" for source, errors in sys.items()) + f", total {total[0]:.4f}")

with plt.rc_context(report_style):
    fig, axes = plt.subplots(1, 3, figsize=(18, 6.5))
    for ax, n, label, (peak_angle, peak_rate, peak_sigma, _), (popt_2g, _, _, _) in zip(
            axes, [1, 2, 3], order_labels, peaks["nominal"], ratio_fits["nominal"]):
        angle_smooth = np.linspace(peak_angle.min(), peak_angle.max(), 2000)
        ax.errorbar(peak_angle, peak_rate, yerr=peak_sigma, fmt="o", color="grey", markersize=4, capsize=2,
                    label="Data")
        ax.plot(angle_smooth, gaussian(angle_smooth, *popt_2g[:3]), ":", color="tab:red", linewidth=1.5,
                label="K$\\alpha_1$")
        ax.plot(angle_smooth, gaussian(angle_smooth, *popt_2g[3:]), ":", color="tab:purple", linewidth=1.5,
                label="K$\\alpha_2$")
        ax.plot(angle_smooth, two_gaussians(angle_smooth, *popt_2g), "k-", linewidth=1.5,
                label="Superposed double Gaussian")
        mark_expected(ax, n, popt_2g[[1, 4]], popt_2g[[0, 3]])
        ax.axhline(0, color="k", linewidth=0.5)
        ax.set_xlim(*peak_xlims[n - 1])
        ax.set_ylim(top=1.15 * peak_rate.max())  # raised by clear_legends if the legend would cover the data
        ax.legend(loc="upper left", fontsize=peak_legend_size)
        ax.set_title(f"{label} scan")
        ax.set_xlabel("Bragg angle (deg)")
        ax.grid(alpha=0.3)
    axes[0].set_ylabel("Count rate (/s)")
    place_peak_title(fig, axes, "Superposed double Gaussian fit (parameters: fixed intensity ratio and shared "
                                "$\\sigma$)", peak_title_size)
    clear_legends(axes)
    save_figure("fits_6_two_gaussians_fixed_ratio", fig)
    plt.show()

# Separation of the two summed Gaussians (K-alpha2 mean - K-alpha1 mean) in the ratio-constrained fit,
# against the Bragg's-law prediction. A constant zero offset moves both means equally, so it cancels here.
# One entry per order, in deg; each budget is (total error, each systematic error).
separations, separation_budgets, predicted_separations = [], [], []
for i, n in enumerate([1, 2, 3]):
    popt_2g, _, separation_err, _ = ratio_fits["nominal"][i]
    separations.append(popt_2g[4] - popt_2g[1])
    separation_budgets.append(error_budget({variant: np.array([fits[i][0][4] - fits[i][0][1]])
                                            for variant, fits in ratio_fits.items()}, np.array([separation_err])))
    predicted_separations.append(doublet_split(n, popt_2g[1]))
separation_errors = [total[0] for total, _ in separation_budgets]


# K-alpha1 wavelength from each order, lambda = 2 d sin(mean_1) / n, against the reference value 154.06 pm.
# Every angle error moves it, the heating error included, which stands in for any change of d with temperature.
def kalpha1_wavelength(n, mean_1):
    return 2 * d_NaCl * np.sin(np.radians(mean_1)) / n


# K-alpha1 wavelength from the ratio-constrained fit, one entry per order, in pm; each budget is (total error, each
# systematic error). wavelength_pulls is (wavelength - 154.06 pm) in units of its total error.
wavelengths, wavelength_budgets = [], []
for i, n in enumerate([1, 2, 3]):
    popt_2g, perr_2g, _, _ = ratio_fits["nominal"][i]
    wavelengths.append(kalpha1_wavelength(n, popt_2g[1]))
    wavelength_stat = 2 * d_NaCl * np.cos(np.radians(popt_2g[1])) / n * np.radians(perr_2g[1])
    wavelength_budgets.append(error_budget({variant: np.array([kalpha1_wavelength(n, fits[i][0][1])])
                                            for variant, fits in ratio_fits.items()}, np.array([wavelength_stat])))
wavelength_errors = [total[0] for total, _ in wavelength_budgets]
wavelength_pulls = [(wavelength - 154.06) / error for wavelength, error in zip(wavelengths, wavelength_errors)]


# Voigt line shape, one Voigt and two Voigts
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
    popt_1, pcov_1, _ = weighted_fit(voigt, peak_angle, peak_rate, peak_sigma, baseline_params,
                                     p0=[popt_g[0], popt_g[1], popt_g[2], 0.01],
                                     bounds=([0, peak_angle.min(), 1e-4, 0], [np.inf, peak_angle.max(), 1, 1]))
    popt_2, pcov_2, _ = weighted_fit(two_voigts_fixed_ratio, peak_angle, peak_rate, peak_sigma, baseline_params,
                                     p0=[popt_2g[0], popt_2g[1], popt_2g[4], popt_2g[2], 0.01],
                                     bounds=([0, peak_angle.min(), peak_angle.min(), 1e-4, 0],
                                             [np.inf, peak_angle.max(), peak_angle.max(), 1, 1]))
    return popt_1, np.sqrt(np.diag(pcov_1)), popt_2, np.sqrt(np.diag(pcov_2))


# chi^2 against the measured points and its dof, which also counts the 2 parameters of the flank baseline fitted to
# the same points (baseline_params, as in weighted_fit and chi_squared.py). If the model were right, chi2/dof would
# scatter about 1 by sqrt(2/dof).
# chi2_dof returns (chi2, dof); chi2_text writes them as one line for printing.
def chi2_dof(model, params, peak):
    peak_angle, peak_rate, peak_sigma, _ = peak
    chi2 = np.sum(((peak_rate - model(peak_angle, *params)) / peak_sigma) ** 2)
    return chi2, len(peak_rate) - len(params) - baseline_params


def chi2_text(name, model, params, peak):
    chi2, dof = chi2_dof(model, params, peak)
    return f"{name}: chi2 = {chi2:.1f}/{dof} = {chi2 / dof:.2f} +/- {np.sqrt(2 / dof):.2f}"


voigt_fits = [fit_voigts(peak, popt_g, popt_2g) for peak, (popt_g, _, _), (popt_2g, _, _, _)
              in zip(peaks["nominal"], gaussian_fits["nominal"], ratio_fits["nominal"])]
# (chi2, dof) of each line shape, one dict per order
voigt_chi2 = [{"one Voigt": chi2_dof(voigt, popt_1, peak),
               "two Voigts": chi2_dof(two_voigts_fixed_ratio, popt_2, peak),
               "one Gaussian": chi2_dof(gaussian, popt_g, peak),
               "two Gaussians": chi2_dof(two_gaussians_fixed_ratio, popt_2g[[0, 1, 4, 2]], peak)}
              for peak, (popt_1, _, popt_2, _), (popt_g, _, _), (popt_2g, _, _, _)
              in zip(peaks["nominal"], voigt_fits, gaussian_fits["nominal"], ratio_fits["nominal"])]
# Reduced chi^2 of the two-line fits in the order they are plotted (all free, then I2/I1 fixed with a shared width),
# with its expected scatter sqrt(2/dof)
for label, peak, (popt_free, _, _), chi2s in zip(order_labels, peaks["nominal"], two_gaussian_fits["nominal"],
                                                 voigt_chi2):
    for name, (chi2, dof) in [("two Gaussians, all free", chi2_dof(two_gaussians, popt_free, peak)),
                              ("two Gaussians, I2/I1 fixed", chi2s["two Gaussians"]),
                              ("two Voigts, I2/I1 fixed", chi2s["two Voigts"])]:
        print(f"{label}, {name}: chi2/dof = {chi2:.1f}/{dof} = {chi2 / dof:.2f} +/- {np.sqrt(2 / dof):.2f}")

# Copy 6 of the peak subplots: pseudo-Voigt line shape, one pseudo-Voigt and two pseudo-Voigts
# A pseudo-Voigt approximates the Voigt by a weighted sum, not a convolution, of a Gaussian and a Lorentzian of the
# same FWHM: eta = 0 is a pure Gaussian, eta = 1 a pure Lorentzian. It is within ~1% of a true Voigt but has a single
# width, so unlike the Voigt it cannot split the broadening into instrument (sigma) and natural line (gamma).
# As in Thompson, Cox and Hastings, eta is the Lorentzian's share of the AREA: both parts are normalised to unit area
# (centre heights 2/pi and 2 sqrt(ln 2 / pi), per FWHM) before mixing, and the sum is divided by its centre height so
# amplitude is still the peak height.
# The two-pseudo-Voigt fit has the same constraints as the two-Voigt one: I2/I1 = 0.518 and both lines share fwhm and
# eta, so the height ratio is still the area ratio. Nominal variant only, as for the Voigts: this checks whether the
# choice of line shape moves the fitted peak positions.
def pseudo_voigt(x, amplitude, mean, fwhm, eta):
    u = (x - mean) / fwhm
    height_l, height_g = eta * 2 / np.pi, (1 - eta) * 2 * np.sqrt(np.log(2) / np.pi)
    shape = height_l / (1 + 4 * u ** 2) + height_g * np.exp(-4 * np.log(2) * u ** 2)
    return amplitude * shape / (height_l + height_g)


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
    popt_1, pcov_1, _ = weighted_fit(pseudo_voigt, peak_angle, peak_rate, peak_sigma, baseline_params,
                                     p0=[popt_1v[0], popt_1v[1], *voigt_fwhm_eta(*popt_1v[2:])],
                                     bounds=([0, peak_angle.min(), 1e-4, 0], [np.inf, peak_angle.max(), 2, 1]))
    popt_2, pcov_2, _ = weighted_fit(two_pseudo_voigts_fixed_ratio, peak_angle, peak_rate, peak_sigma,
                                     baseline_params,
                                     p0=[popt_2v[0], popt_2v[1], popt_2v[2], *voigt_fwhm_eta(*popt_2v[3:])],
                                     bounds=([0, peak_angle.min(), peak_angle.min(), 1e-4, 0],
                                             [np.inf, peak_angle.max(), peak_angle.max(), 2, 1]))
    separation_err = np.sqrt(pcov_2[1, 1] + pcov_2[2, 2] - 2 * pcov_2[1, 2])
    return popt_1, np.sqrt(np.diag(pcov_1)), popt_2, np.sqrt(np.diag(pcov_2)), separation_err


pseudo_voigt_fits = [fit_pseudo_voigts(peak, popt_1v, popt_2v)
                     for peak, (popt_1v, _, popt_2v, _) in zip(peaks["nominal"], voigt_fits)]
# Comparison of the pseudo-Voigt fits with the Voigt fits, one dict per order (nominal variant, fit errors only).
# The *_shift_sigma values are the change from the Voigt to the pseudo-Voigt in units of the pseudo-Voigt fit error.
pseudo_voigt_comparison = []
for n, peak, (popt_1, perr_1, popt_2, perr_2, separation_err), (popt_1v, _, popt_2v, _) in zip(
        [1, 2, 3], peaks["nominal"], pseudo_voigt_fits, voigt_fits):
    separation, separation_v = popt_2[2] - popt_2[1], popt_2v[2] - popt_2v[1]
    wavelength, wavelength_v = kalpha1_wavelength(n, popt_2[1]), kalpha1_wavelength(n, popt_2v[1])
    wavelength_stat = 2 * d_NaCl * np.cos(np.radians(popt_2[1])) / n * np.radians(perr_2[1])
    pseudo_voigt_comparison.append(dict(
        voigt_fwhm_eta_one=voigt_fwhm_eta(*popt_1v[2:]),
        voigt_fwhm_eta_two=voigt_fwhm_eta(*popt_2v[3:]),
        chi2={"one pseudo-Voigt": chi2_dof(pseudo_voigt, popt_1, peak),
              "two pseudo-Voigts": chi2_dof(two_pseudo_voigts_fixed_ratio, popt_2, peak),
              "one Voigt": chi2_dof(voigt, popt_1v, peak),
              "two Voigts": chi2_dof(two_voigts_fixed_ratio, popt_2v, peak)},
        mean_1_shift=popt_2[1] - popt_2v[1],
        mean_1_shift_sigma=(popt_2[1] - popt_2v[1]) / perr_2[1],
        separation=separation, separation_err=separation_err, separation_voigt=separation_v,
        separation_shift_sigma=(separation - separation_v) / separation_err,
        predicted_separation=doublet_split(n, popt_2[1]),
        wavelength=wavelength, wavelength_voigt=wavelength_v,
        wavelength_shift_sigma=(wavelength - wavelength_v) / wavelength_stat))

# The chi2 of each line shape, eta and the offsets from Bragg's law are printed rather than drawn on the figure
for n, label, peak, (popt_1, _, popt_2, _, _), (popt_1v, _, popt_2v, _) in zip(
        [1, 2, 3], order_labels, peaks["nominal"], pseudo_voigt_fits, voigt_fits):
    print(f"{label}, " + "; ".join([chi2_text("one pseudo-Voigt", pseudo_voigt, popt_1, peak),
                                    chi2_text("two pseudo-Voigts", two_pseudo_voigts_fixed_ratio, popt_2, peak),
                                    f"eta = {popt_2[4]:.3f}",
                                    chi2_text("one Voigt", voigt, popt_1v, peak),
                                    chi2_text("two Voigts", two_voigts_fixed_ratio, popt_2v, peak)]))
    for mean, (name, wavelength, _, _) in zip(popt_2[[1, 2]], kalpha_lines):
        print(f"{label}, two pseudo-Voigts, {name}: fit - Bragg's law {mean - bragg_angle(n, wavelength):+.3f} deg")

with plt.rc_context(report_style):
    fig, axes = plt.subplots(1, 3, figsize=(18, 6.5))
    for ax, n, label, (peak_angle, peak_rate, peak_sigma, _), (_, _, popt_2, _, _) in zip(
            axes, [1, 2, 3], order_labels, peaks["nominal"], pseudo_voigt_fits):
        angle_smooth = np.linspace(peak_angle.min(), peak_angle.max(), 2000)
        ax.errorbar(peak_angle, peak_rate, yerr=peak_sigma, fmt="o", color="grey", markersize=4, capsize=2,
                    label="Data")
        ax.plot(angle_smooth, pseudo_voigt(angle_smooth, *popt_2[[0, 1, 3, 4]]), ":", color="tab:red",
                linewidth=1.5, label="K$\\alpha_1$")
        ax.plot(angle_smooth, pseudo_voigt(angle_smooth, amplitude_ratio * popt_2[0], *popt_2[[2, 3, 4]]), ":",
                color="tab:purple", linewidth=1.5, label="K$\\alpha_2$")
        ax.plot(angle_smooth, two_pseudo_voigts_fixed_ratio(angle_smooth, *popt_2), "k-", linewidth=1.5,
                label="Superposition of pseudo-Voigts")
        mark_expected(ax, n, popt_2[[1, 2]], [popt_2[0], amplitude_ratio * popt_2[0]])
        ax.axhline(0, color="k", linewidth=0.5)
        ax.set_xlim(*peak_xlims[n - 1])
        ax.set_ylim(top=1.15 * peak_rate.max())  # raised by clear_legends if the legend would cover the data
        ax.legend(loc="upper left", fontsize=peak_legend_size)
        ax.set_title(f"{label} scan")
        ax.set_xlabel("Bragg angle (deg)")
        ax.grid(alpha=0.3)
    axes[0].set_ylabel("Count rate (/s)")
    place_peak_title(fig, axes, "Superposed double pseudo-Voigt fit (parameters: fixed intensity ratio and shared FWHM "
                                "and $\\eta$)", peak_title_size)
    clear_legends(axes)
    save_figure("fits_8_pseudo_voigt", fig)
    plt.show()
