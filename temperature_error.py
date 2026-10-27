import matplotlib.pyplot as plt
from save_figure import save_figure, calib_style
from peak_overlay import (load, overlay_shift, shift_text, windows, first_order_region, figure_start, tuesday_offset,
                          plot_pair, shift_box_text)

# Heating systematic on the Bragg angle. Two full scans on Tuesday with identical settings (1 mA, 1 s, 0.1 deg step),
# one with the NaCl crystal at room temperature and one with it heated by the lamp. The shift of the 1st-order Kα
# peak between them is found by overlaying the two scans (peak_overlay.py) and taken whole as the error:
#     sigma_heat = |shift|.
# It covers everything that changed between two runs taken the same day with nothing touched in the goniometer:
# thermal expansion of the crystal and any drift of the angle scale. That makes it the run-to-run stability too, so
# no separate repeatability error is added in fits_copy.py.
# Tuesday was before the angle fix, so these peaks sit ~0.12 deg high; only the difference between the runs is used.
# The 1st-order Kβ peak is overlaid as a cross-check but not used.
# Importing this file (as fits_copy.py does) gives sigma_heat without printing or plotting anything.
unheated = load("temperature_full_scans.csv", "unheated", 1)
heated = load("temperature_full_scans.csv", "heated", 1)
results = {name: overlay_shift(unheated, heated, (low + tuesday_offset, high + tuesday_offset))
           for name, (low, high) in windows.items()}

sigma_heat = abs(results["Kα n=1"]["shift"])  # deg
sigma_heat_err = results["Kα n=1"]["err"]

if __name__ == "__main__":
    print("Heated against unheated crystal (Tuesday, identical settings):")
    for name, result in results.items():
        print(f"    {shift_text(name, result)}{'' if name == 'Kα n=1' else '  [cross-check only]'}")
    print(f"Heating error: sigma_heat = |Kα n=1 shift| = {sigma_heat:.4f} +/- {sigma_heat_err:.4f} deg")
    print(f"    Kβ n=1 cross-check gives {abs(results['Kβ n=1']['shift']):.4f} deg")

    # Both runs from 15.7 deg up, each with a Gaussian fit and its centre marked, unshifted and unscaled (same current
    # and dwell), and the Kα n=1 overlay shift between them. Angles are Tuesday's, before the angle fix, so the peaks
    # sit ~0.12 deg high.
    plt.rcParams.update(calib_style)
    fig, gaussian_fits = plot_pair(unheated, heated, ["Unheated", "Heated"],
                                   (figure_start, first_order_region[1] + tuesday_offset),
                                   shift_box_text("heated", "unheated", results["Kα n=1"]),
                                   title="Source Heating Error")
    (unheated_popt, unheated_perr), (heated_popt, heated_perr) = gaussian_fits
    print(f"Gaussian centres in the figure (one Gaussian each, display only): unheated {unheated_popt[1]:.4f} +/- "
          f"{unheated_perr[1]:.4f}, heated {heated_popt[1]:.4f} +/- {heated_perr[1]:.4f}, difference "
          f"{heated_popt[1] - unheated_popt[1]:+.4f} deg")
    save_figure("calib_4_temperature_error", fig)
    plt.show()
