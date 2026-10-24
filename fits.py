import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from scipy.interpolate import Akima1DInterpolator

# Load Cu X-ray spectrum off NaCl crystal (columns: Bragg angle in deg, count rate in 1/s)
angle, rate = np.loadtxt("Cu_NaCl_long_1.2KPeak.csv", delimiter=",", skiprows=1, unpack=True)

# Dedicated scans over the 1st, 2nd and 3rd order K-alpha peaks
angle_n1, rate_n1 = np.loadtxt("1st_order_long.csv", delimiter=",", skiprows=1, unpack=True)
angle_n2, rate_n2 = np.loadtxt("2nd_Order_Good.csv", delimiter=",", skiprows=1, unpack=True)
# This file also repeats the n=2 scan in column 1; only rows with an n=3 reading are kept
angle_n3, rate_n3 = np.genfromtxt("3rd_Order_Long_Good.csv", delimiter=",", skip_header=1, usecols=(0, 2), unpack=True)
has_n3 = ~np.isnan(rate_n3)
angle_n3, rate_n3 = angle_n3[has_n3], rate_n3[has_n3]


# Remove the vertical offset of each dedicated scan relative to the full scan.
# The offset is the mean difference in the off-peak region, where both scans should
# be measuring the same background; the peak window itself is excluded.
def remove_offset(scan_angle, scan_rate, peak_low, peak_high):
    off_peak = (scan_angle < peak_low) | (scan_angle > peak_high)
    full_rate = np.interp(scan_angle, angle, rate)
    offset = np.mean(scan_rate[off_peak] - full_rate[off_peak])
    return scan_rate - offset, offset


rate_n1, offset_n1 = remove_offset(angle_n1, rate_n1, 15.8, 16.4)
rate_n2, offset_n2 = remove_offset(angle_n2, rate_n2, 33.0, 33.7)
rate_n3, offset_n3 = remove_offset(angle_n3, rate_n3, 54.8, 55.5)
print(f"Offset removed: 1st order {offset_n1:.2f} 1/s, 2nd order {offset_n2:.2f} 1/s, 3rd order {offset_n3:.2f} 1/s")

plt.figure(figsize=(10, 5))
plt.plot(angle, rate, linewidth=1, label="Full scan")
plt.plot(angle_n1, rate_n1, linewidth=1, label="1st order scan")
plt.plot(angle_n2, rate_n2, linewidth=1, label="2nd order scan")
plt.plot(angle_n3, rate_n3, linewidth=1, label="3rd order scan")
plt.xlabel("Bragg angle (deg)")
plt.ylabel("Count rate (1/s)")
plt.title("Cu anode, NaCl crystal")
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()
plt.show()

# Checkmark 1
# Remove the characteristic Cu peaks to leave only the bremsstrahlung background.
# Each window (deg) covers one peak from where it leaves the background to where it returns.
peak_windows = [
    (14.2, 14.9),  # K-beta, n=1
    (15.2, 15.6),  # small line between K-beta and K-alpha, n=1
    (15.8, 16.4),  # K-alpha, n=1
    (29.6, 30.1),  # K-beta, n=2
    (33.0, 33.7),  # K-alpha, n=2
    (47.7, 48.0),  # K-beta, n=3
    (54.8, 55.5),  # K-alpha, n=3
]

background = np.ones_like(angle, dtype=bool)
for low, high in peak_windows:
    background &= ~((angle >= low) & (angle <= high))

angle_bg, rate_bg = angle[background], rate[background]


# Fit the bremsstrahlung background with Kramers' law, including an absorption term:
#   I(lambda) = A * (lambda/lambda_min - 1) / lambda^2 * exp(-B * (lambda / 100 pm)^3)
# Kramers' law alone over-predicts the long-wavelength tail; the exp(-B lambda^3) factor
# models absorption in the tube window, air and detector, which grows roughly as lambda^3.
d_NaCl = 282.01  # pm, NaCl lattice plane spacing
wavelength_bg = 2 * d_NaCl * np.sin(np.radians(angle_bg))  # Bragg's law, n=1


def kramers(wavelength, A, lambda_min, B):
    shape = (wavelength / lambda_min - 1) / wavelength**2 * np.exp(-B * (wavelength / 100) ** 3)
    return np.where(wavelength > lambda_min, A * shape, 0.0)


# Poisson uncertainty on each point (assumes 1 s per angle step); floor of 1 so zero-count points still count
sigma_bg = np.sqrt(np.maximum(rate_bg, 1))
popt, pcov = curve_fit(kramers, wavelength_bg, rate_bg, p0=[1e5, 34, 1], sigma=sigma_bg, absolute_sigma=True)
perr = np.sqrt(np.diag(pcov))
A, lambda_min, B = popt

chi2 = np.sum(((rate_bg - kramers(wavelength_bg, *popt)) / sigma_bg) ** 2)
dof = len(rate_bg) - len(popt)

# Duane-Hunt law: lambda_min = hc / (eU)  ->  U [kV] = 1239.84 / lambda_min [pm]
voltage = 1239.84 / lambda_min
voltage_err = voltage * perr[1] / lambda_min

print(f"A          = {A:.4g} +/- {perr[0]:.2g}")
print(f"lambda_min = {lambda_min:.2f} +/- {perr[1]:.2f} pm")
print(f"B          = {B:.3f} +/- {perr[2]:.3f}")
print(f"chi2/dof   = {chi2:.1f}/{dof} = {chi2 / dof:.2f}")
print(f"Tube voltage from Duane-Hunt: {voltage:.2f} +/- {voltage_err:.2f} kV")

angle_fine = np.linspace(angle_bg.min(), angle_bg.max(), 2000)
wavelength_fine = 2 * d_NaCl * np.sin(np.radians(angle_fine))

plt.figure(figsize=(10, 5))
plt.plot(angle_bg, rate_bg, ".", markersize=3, label="Background data (peaks removed)")
plt.plot(angle_fine, kramers(wavelength_fine, *popt), "r-", linewidth=1.5,
         label=f"Kramers fit, $\\lambda_{{min}}$ = {lambda_min:.1f} pm ($\\chi^2_\\nu$ = {chi2 / dof:.2f})")
plt.xlabel("Bragg angle (deg)")
plt.ylabel("Count rate (1/s)")
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
def subtract_background(scan_angle, scan_rate, left_flank, right_flank):
    in_range = (scan_angle >= left_flank[0]) & (scan_angle <= right_flank[1])
    scan_angle, scan_rate = scan_angle[in_range], scan_rate[in_range]

    residual = scan_rate - kramers(2 * d_NaCl * np.sin(np.radians(scan_angle)), *popt)

    flanks = (((scan_angle >= left_flank[0]) & (scan_angle <= left_flank[1]))
              | ((scan_angle >= right_flank[0]) & (scan_angle <= right_flank[1])))
    slope, intercept = np.polyfit(scan_angle[flanks], residual[flanks], 1)
    return scan_angle, residual - (slope * scan_angle + intercept), slope


peak_angle_n1, peak_rate_n1, slope_n1 = subtract_background(angle_n1, rate_n1, (15.7, 15.8), (16.4, 16.5))
peak_angle_n2, peak_rate_n2, slope_n2 = subtract_background(angle_n2, rate_n2, (32.3, 32.9), (33.8, 35.5))
peak_angle_n3, peak_rate_n3, slope_n3 = subtract_background(angle_n3, rate_n3, (53.5, 54.5), (55.7, 56.5))
print(f"Base tilt removed (1/s per deg): 1st order {slope_n1:.2f}, 2nd order {slope_n2:.2f}, 3rd order {slope_n3:.2f}")

# Smooth curve through each peak: modified Akima (makima) spline.
# It passes through every data point but, unlike a cubic spline, does not overshoot/ring
# at the sharp corners where the peak meets the flat base (the 1st order has only 11 points),
# and unlike PCHIP its maximum is not pinned to a data point.
spline_n1 = Akima1DInterpolator(peak_angle_n1, peak_rate_n1, method="makima")
spline_n2 = Akima1DInterpolator(peak_angle_n2, peak_rate_n2, method="makima")
spline_n3 = Akima1DInterpolator(peak_angle_n3, peak_rate_n3, method="makima")

fig, axes = plt.subplots(1, 3, figsize=(14, 5))
for ax, colour, label, peak_angle, peak_rate, spline in zip(
        axes, ["C0", "C1", "C2"], ["1st order scan", "2nd order scan", "3rd order scan"],
        [peak_angle_n1, peak_angle_n2, peak_angle_n3], [peak_rate_n1, peak_rate_n2, peak_rate_n3],
        [spline_n1, spline_n2, spline_n3]):
    angle_smooth = np.linspace(peak_angle.min(), peak_angle.max(), 2000)
    ax.plot(peak_angle, peak_rate, ".", color=colour, markersize=5, label="Data")
    ax.plot(angle_smooth, spline(angle_smooth), color=colour, linewidth=1.5, label="Makima spline")
    ax.axhline(0, color="k", linewidth=0.5)
    ax.legend()
    ax.set_title(label)
    ax.set_xlabel("Bragg angle (deg)")
    ax.grid(alpha=0.3)
axes[0].set_ylabel("Count rate, background subtracted (1/s)")
fig.suptitle("Cu K-alpha peaks, bremsstrahlung and base tilt removed")
fig.tight_layout()
plt.show()


# Copy 2 of the peak subplots: Gaussian fit to each peak
def gaussian(x, amplitude, mean, sigma):
    return amplitude * np.exp(-(x - mean) ** 2 / (2 * sigma ** 2))


# Poisson weights; absolute_sigma=False lets the scatter set the uncertainty scale,
# since the counting time per point (and so the absolute error) isn't known
gaussian_fits = []
for label, peak_angle, peak_rate in zip(["1st order", "2nd order", "3rd order"],
                                        [peak_angle_n1, peak_angle_n2, peak_angle_n3],
                                        [peak_rate_n1, peak_rate_n2, peak_rate_n3]):
    p0 = [peak_rate.max(), peak_angle[np.argmax(peak_rate)], 0.1]
    popt_g, pcov_g = curve_fit(gaussian, peak_angle, peak_rate, p0=p0, sigma=np.sqrt(np.maximum(peak_rate, 1)))
    perr_g = np.sqrt(np.diag(pcov_g))
    gaussian_fits.append((popt_g, perr_g))
    print(f"{label} Gaussian: mean = {popt_g[1]:.4f} +/- {perr_g[1]:.4f} deg, "
          f"sigma = {abs(popt_g[2]):.4f} +/- {perr_g[2]:.4f} deg")

fig, axes = plt.subplots(1, 3, figsize=(14, 5))
for ax, colour, label, peak_angle, peak_rate, spline, (popt_g, perr_g) in zip(
        axes, ["C0", "C1", "C2"], ["1st order scan", "2nd order scan", "3rd order scan"],
        [peak_angle_n1, peak_angle_n2, peak_angle_n3], [peak_rate_n1, peak_rate_n2, peak_rate_n3],
        [spline_n1, spline_n2, spline_n3], gaussian_fits):
    angle_smooth = np.linspace(peak_angle.min(), peak_angle.max(), 2000)
    ax.plot(peak_angle, peak_rate, ".", color=colour, markersize=5, label="Data")
    ax.plot(angle_smooth, spline(angle_smooth), color=colour, linewidth=1.5, label="Makima spline")
    ax.plot(angle_smooth, gaussian(angle_smooth, *popt_g), "k--", linewidth=1.2,
            label=f"Gaussian: $\\mu$ = {popt_g[1]:.3f}$^\\circ$, $\\sigma$ = {abs(popt_g[2]):.3f}$^\\circ$")
    ax.axhline(0, color="k", linewidth=0.5)
    ax.set_ylim(top=1.45 * peak_rate.max())  # headroom so the legend clears the peak
    ax.legend(loc="upper left", fontsize=8)
    ax.set_title(label)
    ax.set_xlabel("Bragg angle (deg)")
    ax.grid(alpha=0.3)
axes[0].set_ylabel("Count rate, background subtracted (1/s)")
fig.suptitle("Cu K-alpha peaks, bremsstrahlung and base tilt removed")
fig.tight_layout()
plt.show()

# Copy 3 of the peak subplots: sum of two Gaussians fitted to each spline
# All six parameters (amplitude, mean, sigma for each Gaussian) are free. The fit target is the
# spline on a fine grid, so the two-Gaussian sum is made to reproduce the spline shape.
# Starting guesses treat the peak as the Cu K-alpha1/K-alpha2 doublet: K-alpha2 at ~half the
# height, shifted by the expected splitting n * (154.44 - 154.06 pm) / (2d cos(theta)).
def two_gaussians(x, amplitude_1, mean_1, sigma_1, amplitude_2, mean_2, sigma_2):
    return gaussian(x, amplitude_1, mean_1, sigma_1) + gaussian(x, amplitude_2, mean_2, sigma_2)


two_gaussian_fits = []
for n, label, peak_angle, spline, (popt_g, perr_g) in zip(
        [1, 2, 3], ["1st order", "2nd order", "3rd order"],
        [peak_angle_n1, peak_angle_n2, peak_angle_n3], [spline_n1, spline_n2, spline_n3], gaussian_fits):
    angle_smooth = np.linspace(peak_angle.min(), peak_angle.max(), 2000)
    split = np.degrees(n * (154.44 - 154.06) / (2 * d_NaCl * np.cos(np.radians(popt_g[1]))))
    p0 = [0.67 * popt_g[0], popt_g[1] - split / 3, 0.8 * abs(popt_g[2]),
          0.33 * popt_g[0], popt_g[1] + 2 * split / 3, 0.8 * abs(popt_g[2])]
    lower = [0, peak_angle.min(), 0, 0, peak_angle.min(), 0]
    upper = [np.inf, peak_angle.max(), np.inf, np.inf, peak_angle.max(), np.inf]
    popt_2g, _ = curve_fit(two_gaussians, angle_smooth, spline(angle_smooth), p0=p0, bounds=(lower, upper))
    two_gaussian_fits.append(popt_2g)
    print(f"{label} two-Gaussian fit:")
    for i in (0, 3):
        print(f"    Gaussian {i // 3 + 1}: amplitude = {popt_2g[i]:.1f} 1/s, "
              f"mean = {popt_2g[i + 1]:.4f} deg, sigma = {popt_2g[i + 2]:.4f} deg")

fig, axes = plt.subplots(1, 3, figsize=(15, 6))
for ax, colour, label, peak_angle, peak_rate, spline, popt_2g in zip(
        axes, ["C0", "C1", "C2"], ["1st order scan", "2nd order scan", "3rd order scan"],
        [peak_angle_n1, peak_angle_n2, peak_angle_n3], [peak_rate_n1, peak_rate_n2, peak_rate_n3],
        [spline_n1, spline_n2, spline_n3], two_gaussian_fits):
    angle_smooth = np.linspace(peak_angle.min(), peak_angle.max(), 2000)
    ax.plot(peak_angle, peak_rate, ".", color=colour, markersize=5, label="Data")
    ax.plot(angle_smooth, spline(angle_smooth), color=colour, linewidth=1.5, label="Makima spline")
    ax.plot(angle_smooth, gaussian(angle_smooth, *popt_2g[:3]), ":", color="tab:red", linewidth=1.2, label="Gaussian 1")
    ax.plot(angle_smooth, gaussian(angle_smooth, *popt_2g[3:]), ":", color="tab:purple", linewidth=1.2, label="Gaussian 2")
    ax.plot(angle_smooth, two_gaussians(angle_smooth, *popt_2g), "k--", linewidth=1.2, label="Sum of Gaussians")
    ax.axhline(0, color="k", linewidth=0.5)
    ax.set_ylim(top=1.9 * peak_rate.max())  # headroom for the legend and annotation
    ax.legend(loc="upper left", fontsize=8)
    ax.text(0.98, 0.98,
            f"Gaussian 1:\n  A = {popt_2g[0]:.1f} 1/s\n  $\\mu$ = {popt_2g[1]:.4f}$^\\circ$\n  $\\sigma$ = {popt_2g[2]:.4f}$^\\circ$\n"
            f"Gaussian 2:\n  A = {popt_2g[3]:.1f} 1/s\n  $\\mu$ = {popt_2g[4]:.4f}$^\\circ$\n  $\\sigma$ = {popt_2g[5]:.4f}$^\\circ$",
            transform=ax.transAxes, ha="right", va="top", multialignment="left", fontsize=8,
            bbox=dict(boxstyle="round", facecolor="white", edgecolor="0.7"))
    ax.set_title(label)
    ax.set_xlabel("Bragg angle (deg)")
    ax.grid(alpha=0.3)
axes[0].set_ylabel("Count rate, background subtracted (1/s)")
fig.suptitle("Cu K-alpha peaks: two-Gaussian fit to spline")
fig.tight_layout()
plt.show()

# Copy 4 of the peak subplots: two-Gaussian fit to spline with the amplitude ratio fixed
# Gaussian 1 is K-alpha1 and Gaussian 2 is K-alpha2, with A2 / A1 = 0.518 (K-alpha2/K-alpha1).
# A2 is therefore not a free parameter; the remaining five (A1, both means, both sigmas) are.
amplitude_ratio = 0.518


def two_gaussians_fixed_ratio(x, amplitude_1, mean_1, sigma_1, mean_2, sigma_2):
    return two_gaussians(x, amplitude_1, mean_1, sigma_1, amplitude_ratio * amplitude_1, mean_2, sigma_2)


ratio_fits = []
for n, label, peak_angle, spline, (popt_g, perr_g) in zip(
        [1, 2, 3], ["1st order", "2nd order", "3rd order"],
        [peak_angle_n1, peak_angle_n2, peak_angle_n3], [spline_n1, spline_n2, spline_n3], gaussian_fits):
    angle_smooth = np.linspace(peak_angle.min(), peak_angle.max(), 2000)
    split = np.degrees(n * (154.44 - 154.06) / (2 * d_NaCl * np.cos(np.radians(popt_g[1]))))
    p0 = [0.67 * popt_g[0], popt_g[1] - split / 3, 0.8 * abs(popt_g[2]),
          popt_g[1] + 2 * split / 3, 0.8 * abs(popt_g[2])]
    lower = [0, peak_angle.min(), 0, peak_angle.min(), 0]
    upper = [np.inf, peak_angle.max(), np.inf, peak_angle.max(), np.inf]
    popt_r, _ = curve_fit(two_gaussians_fixed_ratio, angle_smooth, spline(angle_smooth), p0=p0, bounds=(lower, upper))
    # Expand to the full six parameters so it plots the same way as the free fit
    popt_2g = np.array([popt_r[0], popt_r[1], popt_r[2], amplitude_ratio * popt_r[0], popt_r[3], popt_r[4]])
    ratio_fits.append(popt_2g)
    print(f"{label} two-Gaussian fit, A2/A1 = {amplitude_ratio}:")
    for i in (0, 3):
        print(f"    Gaussian {i // 3 + 1}: amplitude = {popt_2g[i]:.1f} 1/s, "
              f"mean = {popt_2g[i + 1]:.4f} deg, sigma = {popt_2g[i + 2]:.4f} deg")

fig, axes = plt.subplots(1, 3, figsize=(15, 6))
for ax, colour, label, peak_angle, peak_rate, spline, popt_2g in zip(
        axes, ["C0", "C1", "C2"], ["1st order scan", "2nd order scan", "3rd order scan"],
        [peak_angle_n1, peak_angle_n2, peak_angle_n3], [peak_rate_n1, peak_rate_n2, peak_rate_n3],
        [spline_n1, spline_n2, spline_n3], ratio_fits):
    angle_smooth = np.linspace(peak_angle.min(), peak_angle.max(), 2000)
    ax.plot(peak_angle, peak_rate, ".", color=colour, markersize=5, label="Data")
    ax.plot(angle_smooth, spline(angle_smooth), color=colour, linewidth=1.5, label="Makima spline")
    ax.plot(angle_smooth, gaussian(angle_smooth, *popt_2g[:3]), ":", color="tab:red", linewidth=1.2, label="Gaussian 1 (K-alpha1)")
    ax.plot(angle_smooth, gaussian(angle_smooth, *popt_2g[3:]), ":", color="tab:purple", linewidth=1.2, label="Gaussian 2 (K-alpha2)")
    ax.plot(angle_smooth, two_gaussians(angle_smooth, *popt_2g), "k--", linewidth=1.2, label="Sum of Gaussians")
    ax.axhline(0, color="k", linewidth=0.5)
    ax.set_ylim(top=1.9 * peak_rate.max())  # headroom for the legend and annotation
    ax.legend(loc="upper left", fontsize=8)
    ax.text(0.98, 0.98,
            f"Gaussian 1:\n  A = {popt_2g[0]:.1f} 1/s\n  $\\mu$ = {popt_2g[1]:.4f}$^\\circ$\n  $\\sigma$ = {popt_2g[2]:.4f}$^\\circ$\n"
            f"Gaussian 2:\n  A = {popt_2g[3]:.1f} 1/s\n  $\\mu$ = {popt_2g[4]:.4f}$^\\circ$\n  $\\sigma$ = {popt_2g[5]:.4f}$^\\circ$",
            transform=ax.transAxes, ha="right", va="top", multialignment="left", fontsize=8,
            bbox=dict(boxstyle="round", facecolor="white", edgecolor="0.7"))
    ax.set_title(label)
    ax.set_xlabel("Bragg angle (deg)")
    ax.grid(alpha=0.3)
axes[0].set_ylabel("Count rate, background subtracted (1/s)")
fig.suptitle(f"Cu K-alpha peaks: two-Gaussian fit to spline, amplitude ratio A2/A1 = {amplitude_ratio}")
fig.tight_layout()
plt.show()

# Separation of the two summed Gaussians (K-alpha2 mean - K-alpha1 mean) in the ratio-constrained fit
for n, popt_2g in zip([1, 2, 3], ratio_fits):
    print(f"n={n}: delta mu = {popt_2g[4] - popt_2g[1]:.4f} deg")
