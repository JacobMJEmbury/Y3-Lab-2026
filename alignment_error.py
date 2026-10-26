import csv
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

# Crystal-alignment systematic on the Bragg angle. The NaCl crystal was flipped in its holder about the roll and
# yaw axes (Friday Task 3) and the peak centres compared with normal-orientation scans of the same lines.
# Centres are Gaussian fits made by build_error_csvs.py in "Error analysis CSVs"; each centre's error there is
# already scaled up by sqrt(chi2_red). Only the K-alpha/K-beta lines are used; the unidentified side peaks
# (~13.4 and ~15.4 deg) are left out.
# Importing this file (as fits_copy.py does) gives sigma_align_constant and sigma_repeat_pooled without printing
# or plotting anything.
peak_file = Path(__file__).parent / "Error analysis CSVs" / "alignment_peak_positions.csv"
with open(peak_file, encoding="utf-8") as f:
    rows = [row for row in csv.DictReader(f) if row["line"] in ("Kα", "Kβ")]

lines = [("Kβ", 1), ("Kα", 1), ("Kβ", 2), ("Kα", 2), ("Kα", 3)]
reference = "normal_thu_0.6mA"
normal_runs = ["normal_thu_0.6mA", "normal_thu_0.12mA", "normal_fri_0.12mA", "normal_fri_0.6mA"]
flipped_runs = ["flipped_roll", "flipped_yaw"]


# The row for one run and line as floats (None if that line was not in that scan)
def peak(run, line, n):
    for row in rows:
        if row["run"] == run and row["line"] == line and int(row["order_n"]) == n:
            return {key: float(value) if key not in ("run", "line") and value else value for key, value in row.items()}
    return None


# Weighted linear least squares y = sum_k p_k * columns_k, weighted by 1/err^2. The parameter errors are not
# scaled by chi2_red: the point errors are taken as absolute, and chi2_red is returned to judge the fit.
def weighted_fit(columns, y, err):
    design = np.column_stack(columns) / err[:, None]
    cov = np.linalg.inv(design.T @ design)
    params = cov @ design.T @ (y / err)
    chi2 = np.sum((design @ params - y / err) ** 2)
    return params, np.sqrt(np.diag(cov)), chi2, len(y) - len(params)


def chi2_text(chi2, dof):
    return f"chi2/dof = {chi2:.2f}/{dof} = {chi2 / dof:.2f}" if dof > 0 else "chi2/dof undefined (dof = 0)"


# Bragg angle of each line from the reference run, in rad. Any small zero offset left in it barely changes cot.
theta = {key: np.radians(peak(reference, *key)["mu_deg"]) for key in lines}

# --- A: baseline repeatability -------------------------------------------------------------------------------
# Scatter of the centre between normal-orientation runs (two currents on Thursday, two on Friday). It is the noise
# floor that an alignment shift has to beat. Each line has only 2-4 runs, so its own std dev is uncertain by up to
# ~70%; the value pooled over all lines (7 dof) is the one used in the error budget.
repeat_runs, sigma_repeat = {}, {}
pooled_sum, pooled_dof = 0, 0
for key in lines:
    repeat_runs[key] = [peak(run, *key) for run in normal_runs if peak(run, *key) is not None]
    mus = np.array([p["mu_deg"] for p in repeat_runs[key]])
    sigma_repeat[key] = np.std(mus, ddof=1)
    pooled_sum += (len(mus) - 1) * sigma_repeat[key] ** 2
    pooled_dof += len(mus) - 1
sigma_repeat_pooled = np.sqrt(pooled_sum / pooled_dof)

# --- B: alignment shift ----------------------------------------------------------------------------------------
# Shift of each line's centre in a flipped scan relative to the reference. If the shift is a constant angle
# (b = 0) it is a tilt or zero-type error; if it comes from the crystal it scales like a change in d, as tan(theta).
# Model: shift = a + b * tan(theta), in deg. Each entry of shifts holds the lines, shifts, their errors, then for
# flipped runs the two fits as (params, errors, chi2, dof).
shifts = {}
for run in flipped_runs + normal_runs[1:]:
    keys = [key for key in lines if peak(run, *key) is not None]
    shift = np.array([peak(run, *key)[f"shift_vs_{reference}_deg"] for key in keys])
    shift_err = np.array([peak(run, *key)["shift_err_deg"] for key in keys])
    shifts[run] = [keys, shift, shift_err]
for run in flipped_runs:
    keys, shift, shift_err = shifts[run]
    tan_theta = np.tan([theta[key] for key in keys])
    shifts[run] += [weighted_fit([np.ones_like(shift), tan_theta], shift, shift_err),
                    weighted_fit([np.ones_like(shift)], shift, shift_err)]

# The alignment error of each line is half the roll shift: if flipping reverses the misalignment, the true centre
# lies midway between the normal and flipped scans. Roll is the main estimate, since its shift is the same at
# 1st and 2nd order and its intensity only fell ~2.5x. Yaw lost ~9x in intensity (the crystal was badly
# misaligned after that flip) and is only a cross-check. The flipped scans stop at 30 deg, so the 2nd-order
# K-alpha and 3rd-order lines get the constant roll shift, which holds if b is consistent with 0.
roll_keys, roll_shift = shifts["flipped_roll"][:2]
roll_constant = shifts["flipped_roll"][4][0][0]
yaw_keys, yaw_shift = shifts["flipped_yaw"][:2]
sigma_align = {key: abs(roll_shift[roll_keys.index(key)]) / 2 if key in roll_keys else abs(roll_constant) / 2
               for key in lines}

# The value used by fits_copy.py: the roll shift is the same for every line (b consistent with 0), so the
# alignment error is one constant angle error, half the constant fitted to the roll shift. As a constant it moves
# every peak by the same amount, so it cancels in a doublet separation but not in an absolute angle.
sigma_align_constant = abs(roll_constant) / 2

if __name__ == "__main__":
    print("A. Repeatability of normal-orientation centres (sample std dev between runs):")
    for key in lines:
        found = repeat_runs[key]
        fit_errs = np.array([p["mu_err_deg"] for p in found])
        values = ", ".join(f"{p['mu_deg']:.4f}" for p in found)
        print(f"    {key[0]} n={key[1]}: {len(found)} runs ({values}) deg -> sigma_repeat = {sigma_repeat[key]:.4f} "
              f"deg, typical fit error {np.median(fit_errs):.4f} deg")
    print(f"    Pooled over all lines: sigma_repeat = {sigma_repeat_pooled:.4f} deg ({pooled_dof} dof). "
          f"With only 2 runs a line's own std dev is itself uncertain by ~70%.")

    print(f"\nB. Shift of the flipped scans relative to {reference}:")
    for run in flipped_runs:
        keys, shift, shift_err, ((a, b), (a_err, b_err), chi2, dof), ((c,), (c_err,), chi2_c, dof_c) = shifts[run]
        print(f"  {run}:")
        for key, s, e in zip(keys, shift, shift_err):
            ratio = peak(run, *key)["height_per_s_per_mA"] / peak(reference, *key)["height_per_s_per_mA"]
            print(f"    {key[0]} n={key[1]} at {np.degrees(theta[key]):5.2f} deg: shift = {s:+.4f} +/- {e:.4f} deg "
                  f"= {abs(s) / sigma_repeat[key]:4.1f} x sigma_repeat, intensity per mA x{ratio:.2f}")
        # With two lines (yaw) the two-term fit passes exactly through both points, so it says nothing about b
        print(f"    shift = a + b tan(theta): a = {a:+.4f} +/- {a_err:.4f} deg, b = {b:+.4f} +/- {b_err:.4f} deg "
              f"({abs(b) / b_err:.1f} sigma from 0), {chi2_text(chi2, dof)}")
        print(f"    constant shift only:     a = {c:+.4f} +/- {c_err:.4f} deg, {chi2_text(chi2_c, dof_c)}")

    # An angle error sigma_theta gives sigma_d / d = cot(theta) * sigma_theta (rad), and the same for lambda
    print("\nAlignment error per line, sigma_align = |roll shift| / 2, and its effect on d:")
    for key in lines:
        source = "roll shift" if key in roll_keys else "constant roll shift (not scanned)"
        line_text = (f"    {key[0]} n={key[1]} at {np.degrees(theta[key]):5.2f} deg: sigma_align = "
                     f"{sigma_align[key]:.4f} deg -> sigma_d/d = {np.radians(sigma_align[key]) / np.tan(theta[key]):.2e}"
                     f"  [{source}]")
        if key in yaw_keys:
            yaw_half = abs(yaw_shift[yaw_keys.index(key)]) / 2
            line_text += f"; yaw cross-check {yaw_half:.4f} deg -> {np.radians(yaw_half) / np.tan(theta[key]):.2e}"
        print(line_text)

    print("\nFor the error budget (deg):")
    print("    line     theta    sigma_repeat  sigma_align  sigma_d/d(align)")
    for key in lines:
        print(f"    {key[0]} n={key[1]}  {np.degrees(theta[key]):6.2f}   {sigma_repeat[key]:.4f}        "
              f"{sigma_align[key]:.4f}       {np.radians(sigma_align[key]) / np.tan(theta[key]):.2e}")
    print(f"    Used in fits_copy.py: sigma_align = {sigma_align_constant:.4f} deg (constant), "
          f"sigma_repeat = {sigma_repeat_pooled:.4f} deg (pooled)")

    tan_fine = np.linspace(0, np.tan(np.radians(57)), 200)
    fig, ax = plt.subplots(figsize=(8, 5))
    for run, colour, marker in zip(flipped_runs + normal_runs[1:], ["C3", "C1", "C0", "C2", "C4"], "os^vD"):
        keys, shift, shift_err = shifts[run][:3]
        ax.errorbar(np.tan([theta[key] for key in keys]), shift, yerr=shift_err, fmt=marker, color=colour, capsize=3,
                    label=run)
        if run in flipped_runs:
            (a, b), _, _, dof = shifts[run][3]
            if dof > 0:
                ax.plot(tan_fine, a + b * tan_fine, "-", color=colour, linewidth=1)
            ax.axhline(shifts[run][4][0][0], color=colour, linestyle=":", linewidth=1)
    ax.axhspan(-sigma_repeat[("Kα", 1)], sigma_repeat[("Kα", 1)], color="grey", alpha=0.2,
               label="$\\pm\\sigma_{repeat}$ (K$\\alpha$ n=1)")
    ax.axhline(0, color="k", linewidth=0.8)
    for key in lines:
        ax.annotate(f"{key[0]} n={key[1]}", (np.tan(theta[key]), 0.07), rotation=90, fontsize=8, ha="center")
    ax.set_xlabel("tan $\\theta$")
    ax.set_ylabel(f"Centre shift vs {reference} (deg)")
    ax.set_title("Crystal alignment: shift = a + b tan $\\theta$ (solid), constant only (dotted)")
    ax.legend(fontsize=8, loc="center right")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    plt.show()
