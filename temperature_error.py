import csv
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

# Temperature systematic on d. Two full scans on Tuesday, one with the NaCl crystal at room temperature and one
# heated, compared line by line. Tuesday was before the angle fix, so these centres sit ~0.12 deg high and are only
# used as heated - unheated differences, never in absolute terms.
# Centres are Gaussian fits made by build_error_csvs.py in "Error analysis CSVs"; each centre's error there is
# already scaled up by sqrt(chi2_red). Only the K-alpha/K-beta lines are used; the unidentified side peaks are not.
# Importing this file (as fits_copy.py does) gives thermal_d_over_d and sigma_stability without printing or plotting.
peak_file = Path(__file__).parent / "Error analysis CSVs" / "temperature_peak_positions.csv"
with open(peak_file, encoding="utf-8") as f:
    rows = [row for row in csv.DictReader(f) if row["line"] in ("Kα", "Kβ")]

lines = [("Kβ", 1), ("Kα", 1), ("Kβ", 2), ("Kα", 2), ("Kα", 3)]

# Linear expansion coefficient of NaCl near room temperature
alpha = 40e-6  # 1/K
# ASSUMED: uncertainty of the lab temperature during the d measurements. Replace with the real value if known.
sigma_T = 2.0  # K
# Measured heated - unheated temperature difference, if it was recorded (None if not)
delta_T_recorded = None  # K


# The row for one run and line as floats
def peak(run, line, n):
    for row in rows:
        if row["run"] == run and row["line"] == line and int(row["order_n"]) == n:
            return {key: float(value) if key not in ("run", "line") and value else value for key, value in row.items()}


# Weighted linear least squares y = sum_k p_k * columns_k, weighted by 1/err^2. The parameter errors are not
# scaled by chi2_red: the point errors are taken as absolute, and chi2_red is returned to judge the fit.
def weighted_fit(columns, y, err):
    design = np.column_stack(columns) / err[:, None]
    cov = np.linalg.inv(design.T @ design)
    params = cov @ design.T @ (y / err)
    chi2 = np.sum((design @ params - y / err) ** 2)
    return params, np.sqrt(np.diag(cov)), chi2, len(y) - len(params)


# Bragg angle of each line from the unheated run, and the heated - unheated shift with its error, all in rad
theta = np.radians([peak("unheated", *key)["mu_deg"] for key in lines])
shift = np.radians([peak("heated", *key)["shift_vs_unheated_deg"] for key in lines])
shift_err = np.radians([peak("heated", *key)["shift_err_deg"] for key in lines])
per_line = np.array([peak("heated", *key)["delta_d_over_d"] for key in lines])

# Differentiating Bragg's law at fixed lambda, a change in d moves each line by -tan(theta) * (delta d / d), so
# real expansion shifts the high-order lines most. A shift of the angle scale between the runs moves every line
# by the same delta beta_0. Model: shift = delta_beta_0 - tan(theta) * (delta d / d), and both one-term versions.
# Each model gives its fit columns at a given tan(theta), and the names of its parameters.
models = {"constant + expansion": (lambda tan: [np.ones_like(tan), -tan], ["delta_beta_0", "delta d/d"]),
          "constant shift only": (lambda tan: [np.ones_like(tan)], ["delta_beta_0"]),
          "expansion only": (lambda tan: [-tan], ["delta d/d"])}
fits, fit_quality = {}, {}
for name, (columns, param_names) in models.items():
    params, errs, chi2, dof = weighted_fit(columns(np.tan(theta)), shift, shift_err)
    fits[name] = dict(zip(param_names, zip(params, errs)))
    fit_quality[name] = chi2, dof
(delta_beta_0, delta_beta_0_err), (expansion, expansion_err) = fits["constant + expansion"].values()
delta_beta_0, delta_beta_0_err = np.degrees(delta_beta_0), np.degrees(delta_beta_0_err)

# No expansion is detected, so quote an upper limit. Heating can only increase d, so the limit is one-sided:
# delta d/d < (fitted value + 2 sigma). The per-line column above treats every shift as expansion and so
# overestimates delta d/d at 1st order, where a constant angle shift looks largest relative to tan(theta).
limit = expansion + 2 * expansion_err

# Two error terms for the budget:
#   angle stability: the constant shift between two runs taken the same day with nothing deliberately changed in
#   the goniometer. It is the same kind of error as the run-to-run repeatability in alignment_error.py (and about
#   the same size), so fits_copy.py uses that one and prints this only as a cross-check, so it is not counted twice.
#   thermal: expansion of the crystal over the uncertainty of the lab temperature, sigma_d/d = alpha * sigma_T.
sigma_stability = abs(delta_beta_0)
thermal_d_over_d = alpha * sigma_T

if __name__ == "__main__":
    print("Heated - unheated centre shift:")
    for key, t, s, e, dd in zip(lines, theta, shift, shift_err, per_line):
        print(f"    {key[0]} n={key[1]} at {np.degrees(t):5.2f} deg: shift = {np.degrees(s):+.4f} +/- "
              f"{np.degrees(e):.4f} deg  (as pure expansion, -cot(theta) * shift = {dd:+.1e})")

    print("\nFits of shift(rad) = delta_beta_0 - tan(theta) * (delta d / d), weighted by 1/err^2:")
    for name in models:
        chi2, dof = fit_quality[name]
        terms = [f"delta_beta_0 = {np.degrees(p):+.4f} +/- {np.degrees(e):.4f} deg" if param == "delta_beta_0"
                 else f"delta d/d = ({p * 1e4:+.1f} +/- {e * 1e4:.1f}) x 1e-4"
                 for param, (p, e) in fits[name].items()]
        print(f"    {name:21s}: {', '.join(terms)}, chi2/dof = {chi2:.2f}/{dof} = {chi2 / dof:.2f}")

    print(f"\nExpansion between runs: delta d/d = ({expansion * 1e4:+.1f} +/- {expansion_err * 1e4:.1f}) x 1e-4, "
          f"{abs(expansion) / expansion_err:.1f} sigma from 0")
    print(f"    Upper limit (2 sigma, one-sided): delta d/d < {limit:.1e}, "
          f"i.e. delta T < {limit / alpha:.0f} K for alpha = {alpha * 1e6:.0f}e-6 /K")
    print(f"    The per-line 'pure expansion' values at 1st order ({per_line[0]:.1e}, {per_line[1]:.1e}) overestimate it")
    if delta_T_recorded is not None:
        expected = alpha * delta_T_recorded
        print(f"    Recorded delta T = {delta_T_recorded:.1f} K predicts delta d/d = {expected:.1e}, "
              f"{(expansion - expected) / expansion_err:+.1f} sigma from the fit")

    print(f"\nFor the error budget:")
    print(f"    Run-to-run angle stability: sigma_stability = |delta_beta_0| = {sigma_stability:.4f} deg "
          f"(fitted +/- {delta_beta_0_err:.4f} deg); cross-check of sigma_repeat in alignment_error.py")
    for key, t in zip(lines, theta):
        print(f"        {key[0]} n={key[1]} at {np.degrees(t):5.2f} deg -> sigma_d/d = "
              f"{np.radians(sigma_stability) / np.tan(t):.2e}")
    print(f"    Thermal: sigma_d/d = alpha * sigma_T = {alpha * 1e6:.0f}e-6 /K x {sigma_T:.1f} K = "
          f"{thermal_d_over_d:.1e} (sigma_T = {sigma_T:.1f} K is ASSUMED), the same for every line")

    tan_fine = np.linspace(0, np.tan(np.radians(57)), 200)
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.errorbar(np.tan(theta), np.degrees(shift), yerr=np.degrees(shift_err), fmt="o", color="k", capsize=3,
                label="heated - unheated")
    for (name, (columns, _)), colour in zip(models.items(), ["C3", "C0", "C1"]):
        params = [p for p, _ in fits[name].values()]
        ax.plot(tan_fine, np.degrees(np.column_stack(columns(tan_fine)) @ params), "-", color=colour, linewidth=1.2,
                label=name)
    ax.axhline(0, color="k", linewidth=0.8)
    for key, t in zip(lines, theta):
        ax.annotate(f"{key[0]} n={key[1]}", (np.tan(t), 0.012), rotation=90, fontsize=8, ha="center")
    ax.set_xlabel("tan $\\theta$")
    ax.set_ylabel("Centre shift (deg)")
    ax.set_title("Heated crystal: shift = $\\Delta\\beta_0$ - tan $\\theta \\cdot \\Delta d/d$")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    plt.show()
