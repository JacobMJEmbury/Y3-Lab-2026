import numpy as np
import matplotlib.pyplot as plt
from save_figure import save_figure, report_style
from peak_overlay import (load, overlay_shift, shift_text, windows, first_order_region, figure_start,
                          plot_pair, shift_box_text)

# Crystal-alignment systematic on the Bragg angle. The only misalignment that moves the peaks to first order is a
# tilt epsilon of the lattice planes about the VERTICAL axis (the goniometer axis): it adds straight onto the angle,
# theta_measured = theta_Bragg + epsilon, the same at every order. A tilt about the beam axis (out of the scattering
# plane) moves the peaks only by (psi^2 / 2) tan(theta), < 0.003 deg at 1st order for psi = 1 deg, and mostly costs
# intensity, so it is left out.
# The crystal was turned 180 deg about the beam axis in its holder (Friday Task 3, the "flipped_roll" scan). That
# reverses the tilt about the vertical axis, so the peaks move by 2 epsilon between the normal and flipped scans and
#     sigma_align = |shift| / 2.
# The shift is found by overlaying the 1st-order Kα peak of the two scans (peak_overlay.py). The normal scan is
# Thursday's after the angle fix, with the same 1 s dwell and 0.1 deg step as the flipped one; only the current
# differs (0.6 vs 1 mA), which the overlay's free scale factor absorbs. The 1st-order Kβ peak is overlaid as a
# cross-check but not used.
# Importing this file (as fits_copy.py does) gives sigma_align without printing or plotting anything.
normal_run, flipped_run = "normal_thu_0.6mA", "flipped_roll"
normal = load("alignment_full_scans.csv", normal_run, 1)
flipped = load("alignment_full_scans.csv", flipped_run, 1)
results = {name: overlay_shift(normal, flipped, window) for name, window in windows.items()}

sigma_align = abs(results["Kα n=1"]["shift"]) / 2  # deg
sigma_align_err = results["Kα n=1"]["err"] / 2

if __name__ == "__main__":
    print(f"Crystal turned 180 deg about the beam axis ({flipped_run}) against the normal orientation ({normal_run}):")
    for name, result in results.items():
        print(f"    {shift_text(name, result)}{'' if name == 'Kα n=1' else '  [cross-check only]'}")
    print(f"Alignment error: sigma_align = |Kα n=1 shift| / 2 = {sigma_align:.4f} +/- {sigma_align_err:.4f} deg")
    print(f"    Kβ n=1 cross-check gives {abs(results['Kβ n=1']['shift']) / 2:.4f} deg")
    for n, theta in [(1, 16.0), (2, 33.2), (3, 55.0)]:
        print(f"    Kα n={n} at ~{theta:.0f} deg: sigma_d/d = cot(theta) * sigma_align = "
              f"{np.radians(sigma_align) / np.tan(np.radians(theta)):.1e}")

    # The normal scan and the flipped one from 15.7 deg up, each with a Gaussian fit and its centre marked, neither
    # shifted, and the Kα n=1 overlay shift between them. The flipped crystal reflects much less (its corrected peak
    # is ~2.3x lower despite 1 vs 0.6 mA), so the normal scan is scaled to the flipped one's fitted peak height.
    plt.rcParams.update(report_style)
    fig, gaussian_fits = plot_pair(normal, flipped, ["Normal (scaled to the same height)",
                                                     "Flipped 180° about beam axis"],
                                   (figure_start, first_order_region[1]),
                                   shift_box_text("flipped", "normal", results["Kα n=1"]), scale_reference=True)
    (normal_popt, normal_perr), (flipped_popt, flipped_perr) = gaussian_fits
    print(f"Gaussian centres in the figure (one Gaussian each, display only): normal {normal_popt[1]:.4f} +/- "
          f"{normal_perr[1]:.4f}, flipped {flipped_popt[1]:.4f} +/- {flipped_perr[1]:.4f}, difference "
          f"{flipped_popt[1] - normal_popt[1]:+.4f} deg")
    save_figure("calib_3_alignment_error", fig)
    plt.show()
