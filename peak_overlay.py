import csv
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import CubicSpline
from scipy.optimize import curve_fit
from deadtime import tau

# Peak shift between two scans found by overlaying them, shared by alignment_error.py and temperature_error.py.
# The reference scan is interpolated with a cubic spline and slid across the other scan; at each trial shift delta
# the other scan is fitted as
#     y(beta) = A * reference(beta - delta) + c0 + c1 * (beta - beta_mean)
# (A absorbs a different tube current or crystal reflectivity, c0 and c1 a different background) and chi2(delta) is
# minimised. No peak shape is assumed, so the Kα1/Kα2 doublet that a single Gaussian fits badly is used as it is.
# Errors on the shift:
#   statistical: from chi2_min + 1 (a parabola fitted to chi2 near its minimum), scaled up by sqrt(chi2_red) if > 1
#   method: the overlay is done both ways (scan onto reference and reference onto scan); if the two peaks have the
#   same shape the two agree, so half their difference measures how much a change of shape biases the shift.
# On synthetic doublets sampled at the same 0.1 deg step, a known shift of 0.03-0.13 deg is recovered to < 0.0005 deg,
# so interpolating between the steps does not bias it.
# Both rates are dead-time corrected first (tau from deadtime.py), since the correction changes the peak shape and
# differs between currents. Changing tau by its error moves the shifts by < 0.001 deg.
csv_dir = Path(__file__).parent / "Error analysis CSVs"
trial_shifts = np.arange(-0.25, 0.25, 0.0005)  # deg

# Fit windows (deg, on Thursday's angle scale). Tuesday's scans sit ~0.12 deg higher, so their windows are moved up
# by tuesday_offset. The Kα window holds the whole Kα1/Kα2 doublet.
windows = {"Kα n=1": (15.7, 16.6), "Kβ n=1": (14.2, 14.9)}
first_order_region = (14.1, 16.6)
figure_start = 15.7  # deg, left edge of the calib_3 and calib_4 figures (each on its own angle scale)
tuesday_offset = 0.12  # deg


# Dead-time corrected rate and its Poisson error for one run in one of the full-scan CSVs (dwell in s)
def load(file_name, run, dwell):
    with open(csv_dir / file_name, encoding="utf-8") as f:
        rows = [row for row in csv.DictReader(f) if row[f"{run}_rate_per_s"]]
    beta = np.array([float(row["beta_deg"]) for row in rows])
    rate = np.array([float(row[f"{run}_rate_per_s"]) for row in rows])
    loss = 1 - rate * tau
    return beta, rate / loss, np.sqrt(np.maximum(rate * dwell, 1)) / dwell / loss ** 2


# chi2 of scan against the shifted reference at each trial shift, the dof, and the fitted overlay at the minimum.
# The reference's own error (interpolated, times A) is added in quadrature; A is iterated since the weights need it.
def chi2_curve(reference, scan, window):
    spline, spline_err = CubicSpline(reference[0], reference[1]), CubicSpline(reference[0], reference[2])
    beta, y, err = scan
    # Only points whose shifted position stays inside the reference scan for every trial shift
    keep = ((beta >= window[0]) & (beta <= window[1]) & (beta - trial_shifts.min() <= reference[0].max())
            & (beta - trial_shifts.max() >= reference[0].min()))
    beta, y, err = beta[keep], y[keep], err[keep]
    chi2, fitted = [], []
    for delta in trial_shifts:
        template, template_err = spline(beta - delta), np.abs(spline_err(beta - delta))
        scale = 1
        for _ in range(3):
            weight = 1 / np.sqrt(err ** 2 + (scale * template_err) ** 2)
            design = np.column_stack([template, np.ones_like(beta), beta - beta.mean()]) * weight[:, None]
            params = np.linalg.lstsq(design, y * weight, rcond=None)[0]
            scale = params[0]
        chi2.append(np.sum((design @ params - y * weight) ** 2))
        fitted.append(design @ params / weight)
    chi2 = np.array(chi2)
    return chi2, len(beta) - 4, beta, fitted[np.argmin(chi2)]


# Shift of scan relative to reference (positive = scan at higher angle) with its statistical error and chi2_red
def one_way_shift(reference, scan, window):
    chi2, dof, _, _ = chi2_curve(reference, scan, window)
    near = slice(max(np.argmin(chi2) - 20, 0), np.argmin(chi2) + 21)
    curvature, slope, _ = np.polyfit(trial_shifts[near], chi2[near], 2)
    chi2_red = chi2.min() / dof
    return -slope / (2 * curvature), np.sqrt(max(chi2_red, 1) / curvature), chi2_red


# Both ways round: the shift is their mean, the method error half their difference. The backward overlay uses the
# window moved by the forward shift, so it covers the same peak.
def overlay_shift(reference, scan, window):
    forward, forward_err, forward_chi2 = one_way_shift(reference, scan, window)
    moved = (window[0] - forward, window[1] - forward)
    backward, backward_err, backward_chi2 = one_way_shift(scan, reference, moved)
    shift, stat_err, method_err = (forward - backward) / 2, (forward_err + backward_err) / 2, abs(forward + backward) / 2
    return dict(shift=shift, stat_err=stat_err, method_err=method_err, err=np.hypot(stat_err, method_err),
                forward=forward, backward=-backward, chi2_red=(forward_chi2, backward_chi2))


def shift_text(name, result):
    return (f"{name}: shift = {result['shift']:+.4f} +/- {result['err']:.4f} deg (stat {result['stat_err']:.4f}, "
            f"method {result['method_err']:.4f}; one way {result['forward']:+.4f}, other way "
            f"{result['backward']:+.4f}, chi2_red {result['chi2_red'][0]:.1f}/{result['chi2_red'][1]:.1f})")


# Text for plot_pair's box: the overlay shift as "Offset (scan − reference)" with true minus signs, so the number's
# sign matches the one in the label and on the tick labels
def shift_box_text(scan_name, reference_name, result):
    shift = f"{result['shift']:+.4f}".replace("-", "−")
    return f"Offset ({scan_name} − {reference_name})\n= {shift} $\\pm$ {result['err']:.4f}°"


# Single Gaussian on a flat background, fitted to one scan over span. For the figures only: the Kα1/Kα2 doublet is not
# really one Gaussian, and the shifts and errors above come from the overlay, not from these centres.
def gaussian(beta, height, centre, width, background):
    return height * np.exp(-0.5 * ((beta - centre) / width) ** 2) + background


def fit_gaussian(scan, span):
    beta, rate, rate_err = scan
    inside = (beta >= span[0]) & (beta <= span[1])
    p0 = [np.ptp(rate[inside]), beta[inside][np.argmax(rate[inside])], 0.1, rate[inside].min()]
    popt, pcov = curve_fit(gaussian, beta[inside], rate[inside], p0=p0, sigma=rate_err[inside])
    popt[2] = abs(popt[2])
    return popt, np.sqrt(np.diag(pcov))


# Single-panel figure for one pair of scans over span: each scan's points with its Gaussian fit (reference blue, the
# other scan red), a dotted line through each fitted centre with the gap between them shaded red (as in calib_2), and
# a text box with offset_text. With scale_reference the reference (points and fit) is scaled to the other scan's
# fitted peak height, for scans of different intensity; only the peak positions matter here.
# Text sizes come from calib_style in save_figure.py. Returns the figure and each scan's (popt, perr).
def plot_pair(reference, scan, labels, span, offset_text, scale_reference=False, title="1st order region"):
    fits = [fit_gaussian(one_scan, span) for one_scan in (reference, scan)]
    peak_heights = [gaussian(popt[1], *popt) for popt, _ in fits]
    scales = [peak_heights[1] / peak_heights[0] if scale_reference else 1, 1]
    beta_fine = np.linspace(*span, 500)
    fig, ax = plt.subplots(figsize=(9, 6.5))
    handles = []
    for (beta, rate, rate_err), (popt, _), scale, colour in zip((reference, scan), fits, scales, ("C0", "C3")):
        inside = (beta >= span[0]) & (beta <= span[1])
        points = ax.errorbar(beta[inside], scale * rate[inside], yerr=scale * rate_err[inside], fmt="o", color=colour,
                             markersize=4, capsize=2)
        line, = ax.plot(beta_fine, scale * gaussian(beta_fine, *popt), "-", color=colour, linewidth=2)
        ax.axvline(popt[1], color=colour, linestyle=":", linewidth=1.5)
        handles.append((points, line))
    ax.axvspan(*sorted(popt[1] for popt, _ in fits), color="r", alpha=0.3, linewidth=0)
    ax.legend(handles, labels, loc="upper left")
    ax.text(0.03, 0.8, offset_text, transform=ax.transAxes, va="top",
            bbox=dict(boxstyle="round", facecolor="white", edgecolor="0.7"))
    ax.set_xlim(span[0] - 0.02, span[1] + 0.02)  # small margin so the end points aren't cut in half
    ax.set_ylim(top=1.5 * max(peak_heights[1], scales[0] * peak_heights[0]))
    ax.set_xlabel("Angle (deg)")
    ax.set_ylabel("Count rate (/s)")
    ax.set_title(title)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    return fig, fits
