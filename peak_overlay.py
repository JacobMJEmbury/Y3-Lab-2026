import csv
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import CubicSpline
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


# Figure for one pair of scans: (left) the 1st-order region with the reference overlaid before and after the shift,
# (middle) the Kα n=1 fit window close up with residuals, (right) chi2 against trial shift both ways round for the
# Kα n=1 window, with the Kβ n=1 one-way curve as a cross-check. Each chi2 curve is divided by its chi2_red when > 1,
# so the +1 line gives the statistical error. results maps each window name to its overlay_shift result.
def plot_overlay(reference, scan, labels, results, offset, result_text, title):
    reference_label, scan_label = labels
    fig = plt.figure(figsize=(15, 5.5))
    grid = fig.add_gridspec(2, 3, height_ratios=[3, 1], hspace=0.05)
    region_ax, peak_ax = fig.add_subplot(grid[:, 0]), fig.add_subplot(grid[0, 1])
    residual_ax, chi2_ax = fig.add_subplot(grid[1, 1], sharex=peak_ax), fig.add_subplot(grid[:, 2])
    region = (first_order_region[0] + offset, first_order_region[1] + offset)
    window = (windows["Kα n=1"][0] + offset, windows["Kα n=1"][1] + offset)

    for ax, span in [(region_ax, region), (peak_ax, window)]:
        shown = (scan[0] >= span[0]) & (scan[0] <= span[1])
        ax.errorbar(scan[0][shown], scan[1][shown], yerr=scan[2][shown], fmt="o", color="k", markersize=3,
                    capsize=2, label=scan_label, zorder=3)
        _, _, beta, fitted = chi2_curve(reference, scan, span)
        ax.plot(beta, fitted, "-", color="C3",
                label=f"{reference_label}, shifted {one_way_shift(reference, scan, span)[0]:+.3f}$^\\circ$")
        in_span = (reference[0] >= span[0]) & (reference[0] <= span[1])
        ax.plot(reference[0][in_span], reference[1][in_span] * fitted.max() / reference[1][in_span].max(), ":",
                color="C0", label=f"{reference_label}, unshifted (scaled)")
        ax.grid(alpha=0.3)
    region_ax.axvspan(*window, color="C3", alpha=0.08, label="Kα n=1 fit window")
    region_ax.set_xlabel("$\\beta$ (deg)")
    region_ax.set_ylabel("Count rate, dead-time corrected (1/s)")
    region_ax.set_title("1st-order region")
    region_ax.legend(fontsize=8)

    _, _, beta, fitted = chi2_curve(reference, scan, window)
    in_window = np.isin(scan[0], beta)
    residual_ax.errorbar(beta, scan[1][in_window] - fitted, yerr=scan[2][in_window], fmt="o", color="k",
                         markersize=3, capsize=2)
    residual_ax.axhline(0, color="C3", linewidth=1)
    residual_ax.set_xlabel("$\\beta$ (deg)")
    residual_ax.set_ylabel("Residual (1/s)")
    residual_ax.grid(alpha=0.3)
    plt.setp(peak_ax.get_xticklabels(), visible=False)
    peak_ax.set_title("Kα n=1 fit window, shifted overlay")
    peak_ax.legend(fontsize=8)

    curves = [("Kα n=1", reference, scan, window, "C3", "-", "one way"),
              ("Kα n=1", scan, reference, (window[0] - results["Kα n=1"]["forward"],
                                           window[1] - results["Kα n=1"]["forward"]), "C1", "--", "other way"),
              ("Kβ n=1", reference, scan, (windows["Kβ n=1"][0] + offset, windows["Kβ n=1"][1] + offset), "C0", ":",
               "one way, cross-check")]
    for name, first, second, span, colour, style, how in curves:
        chi2, dof, _, _ = chi2_curve(first, second, span)
        sign = 1 if first is reference else -1  # the other-way overlay finds -shift
        chi2_ax.plot(sign * trial_shifts, (chi2 - chi2.min()) / max(chi2.min() / dof, 1), style, color=colour,
                     label=f"{name}, {how}")
    main = results["Kα n=1"]
    chi2_ax.axvspan(main["shift"] - main["err"], main["shift"] + main["err"], color="C3", alpha=0.15,
                    label=f"Kα n=1: {main['shift']:+.4f} $\\pm$ {main['err']:.4f}$^\\circ$")
    chi2_ax.axvline(results["Kβ n=1"]["shift"], color="C0", linewidth=0.8,
                    label=f"Kβ n=1: {results['Kβ n=1']['shift']:+.4f} $\\pm$ {results['Kβ n=1']['err']:.4f}$^\\circ$")
    chi2_ax.axhline(1, color="grey", linestyle="--", linewidth=0.8)
    shifts = [result["shift"] for result in results.values()]
    chi2_ax.set_xlim(min(shifts) - 0.03, max(shifts) + 0.03)
    chi2_ax.set_ylim(0, 10)
    chi2_ax.set_xlabel("Trial shift (deg)")
    chi2_ax.set_ylabel("$(\\chi^2 - \\chi^2_{min})$ / max($\\chi^2_{red}$, 1)")
    chi2_ax.set_title("$\\chi^2$ against shift")
    chi2_ax.legend(fontsize=8, loc="upper left")
    chi2_ax.text(0.98, 0.03, result_text, transform=chi2_ax.transAxes, ha="right", va="bottom", fontsize=9,
                 bbox=dict(boxstyle="round", facecolor="white", edgecolor="0.7"))
    chi2_ax.grid(alpha=0.3)
    fig.suptitle(title)
    fig.subplots_adjust(left=0.05, right=0.98, top=0.86, bottom=0.11, wspace=0.25)
    return fig
