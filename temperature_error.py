import matplotlib.pyplot as plt
from save_figure import save_figure
from peak_overlay import load, overlay_shift, shift_text, plot_overlay, windows, tuesday_offset

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

    fig = plot_overlay(unheated, heated, ("unheated", "heated"), results, tuesday_offset,
                       f"$\\sigma_{{heat}}$ = |shift| = {sigma_heat:.4f} $\\pm$ {sigma_heat_err:.4f}$^\\circ$",
                       "Crystal heating: heated scan overlaid on the unheated one (Tuesday, both 1 mA and 1 s per "
                       "step)\nAngles are Tuesday's, before the angle fix; only the shift between the runs is used")
    save_figure("calib_4_temperature_error", fig)
    plt.show()
