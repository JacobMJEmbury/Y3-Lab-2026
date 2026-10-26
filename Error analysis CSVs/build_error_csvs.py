"""Condense the crystal-alignment and temperature scans into CSVs for error analysis.

Reads Leybold X-ray apparatus .xry files (XR01 format):
  - header block "&b" gives: start angle, end angle, ..., angle step (deg)
  - "R" block is followed by "C N" then N rows of C columns (1/s); each column is
    one scan on the shared angle grid start + step * row, NAN where not measured
  - the C lines after the data are each scan's: U (kV), U, I (mA), dwell (s), step (deg)

Each rate gets a Poisson error sqrt(R·Δt)/Δt (at least 1 count). Peaks are fitted with a
Gaussian on a linear background (±0.4° window); the error on the centre is scaled up by
sqrt(χ²_red) when χ²_red > 1. See README.md for which runs were kept and why.
"""
import csv
from pathlib import Path

import numpy as np
from scipy.optimize import curve_fit

HERE = Path(__file__).resolve().parent
WEEK = HERE.parent

# (label, file, column, note) — the runs kept for each analysis
ALIGNMENT_FULL = [
    ("normal_thu_0.6mA", "Thursday/Angle_fixed/High Peak/Cu_NaCl_long_3.5Kpeak.xry", 1,
     "normal orientation, Thursday after angle fix"),
    ("normal_thu_0.12mA", "Thursday/Angle_fixed/Low Peak/Cu_NaCl_long_1.2KPeak.xry", 1,
     "normal orientation, Thursday after angle fix (repeat)"),
    ("flipped_roll", "Friday/Task3/Flipped_Roll.xry", 1, "crystal flipped about roll axis"),
    ("flipped_yaw", "Friday/Task3/Flipped_Yaw.xry", 1, "crystal flipped about yaw axis"),
]
ALIGNMENT_FRIDAY = [
    ("normal_fri_0.12mA", "Friday/Task2/1st_Order_high_and_low.xry", 1,
     "normal orientation, Friday, same day as flipped scans"),
    ("normal_fri_0.6mA", "Friday/Task2/1st_Order_high_and_low.xry", 2,
     "normal orientation, Friday, same day as flipped scans"),
]
TEMPERATURE = [
    ("unheated", "Tuesday/Cu_NaCl_mono_full.xry", 1, "room temperature"),
    ("heated", "Tuesday/Cu_NaCl_mono_full_heated.xry", 1, "heated"),
]
ALIGN_RANGE = (2.5, 30.0)  # deg — the range the flipped scans cover

# Search windows (deg) for each line; Cu Kα/Kβ reflections off NaCl
LINES = [
    ("unidentified ~13.4", 1, 13.1, 13.7),
    ("Kβ", 1, 14.2, 14.8),
    ("unidentified ~15.4", 1, 15.1, 15.7),
    ("Kα", 1, 15.8, 16.4),
    ("Kβ", 2, 29.4, 30.2),
    ("Kα", 2, 33.0, 33.8),
    ("Kα", 3, 54.8, 55.8),
]
WINDOW = 0.4


def read_xry(path):
    """Return one dict per column in the file."""
    lines = [l.strip() for l in path.read_text(encoding="latin-1").splitlines()]
    start, _, _, step = map(float, lines[4].split()[:4])
    r = lines.index("R")
    ncol, n = map(int, lines[r + 3].split())
    data = np.array([[float(v) for v in lines[r + 4 + i].split()[:ncol]] for i in range(n)])
    angle = np.round(start + step * np.arange(n), 2)
    scans = []
    for c in range(ncol):
        parts = lines[r + 4 + n + c].split()
        u_kv, i_ma, dwell = (float(parts[0]), float(parts[2]), float(parts[3])) if len(parts) == 5 \
            else (np.nan, np.nan, np.nan)
        ok = ~np.isnan(data[:, c])
        scans.append(dict(angle=angle[ok], rate=data[ok, c], u_kv=u_kv, i_ma=i_ma, dwell=dwell))
    return scans


def load(runs):
    out = []
    for label, rel, col, note in runs:
        s = read_xry(WEEK / rel)[col - 1]
        s["err"] = np.sqrt(np.maximum(s["rate"] * s["dwell"], 1)) / s["dwell"]
        out.append(dict(s, label=label, file=rel, col=col, note=note))
    return out


def write_scans(path, scans, lo=-np.inf, hi=np.inf):
    grid = sorted({a for s in scans for a in s["angle"] if lo <= a <= hi})
    with path.open("w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["beta_deg"] + [f"{s['label']}_{k}" for s in scans
                                   for k in ("rate_per_s", "err_per_s")])
        for a in grid:
            row = [f"{a:.1f}"]
            for s in scans:
                i = np.flatnonzero(s["angle"] == a)
                row += [f"{s['rate'][i[0]]:g}", f"{s['err'][i[0]]:.3f}"] if len(i) else ["", ""]
            w.writerow(row)
    print(f"Saved {path.name} ({len(grid)} angles, {len(scans)} runs)")


def line_model(x, a, mu, sigma, c, k):
    return a * np.exp(-(x - mu) ** 2 / (2 * sigma ** 2)) + c + k * (x - mu)


def fit_line(s, lo, hi):
    """Fit the strongest point in [lo, hi]; None if outside the scan or no clear peak."""
    m = (s["angle"] >= lo) & (s["angle"] <= hi)
    if m.sum() < 3:
        return None
    seed = s["angle"][m][np.argmax(s["rate"][m])]
    w = (s["angle"] > seed - WINDOW - 1e-9) & (s["angle"] < seed + WINDOW + 1e-9)
    x, y, e = s["angle"][w], s["rate"][w], s["err"][w]
    if len(x) < 6 or seed - x[0] < 0.2 or x[-1] - seed < 0.2:
        return None  # peak too close to the scan edge to fit
    try:
        p, cov = curve_fit(line_model, x, y, p0=[y.max() - y.min(), seed, 0.12, y.min(), 0],
                           sigma=e, absolute_sigma=True, maxfev=10000)
    except RuntimeError:
        return None
    dof = len(x) - len(p)
    chi2 = np.sum(((y - line_model(x, *p)) / e) ** 2) / dof
    err = np.sqrt(np.diag(cov)) * np.sqrt(max(chi2, 1))
    if not (lo - 0.2 <= p[1] <= hi + 0.2) or p[0] <= 0 or p[0] < 3 * err[0]:
        return None
    return dict(mu=p[1], mu_err=err[1], height=p[0], fwhm=2.3548 * abs(p[2]), chi2_red=chi2)


def fit_all(scans):
    return {s["label"]: {(name, n): fit_line(s, lo, hi) for name, n, lo, hi in LINES}
            for s in scans}


def write_peaks(path, scans, fits, ref, extra=None):
    """One row per (run, line); shift is relative to the `ref` run's fitted centre."""
    header = ["run", "line", "order_n", "I_mA", "dwell_s", "mu_deg", "mu_err_deg", "height_per_s",
              "height_per_s_per_mA", "fwhm_deg", "chi2_red", f"shift_vs_{ref}_deg",
              "shift_err_deg"]
    if extra:
        header += [h for h, _ in extra]
    with path.open("w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(header)
        for s in scans:
            for name, n, _, _ in LINES:
                f, r = fits[s["label"]][(name, n)], fits[ref][(name, n)]
                if f is None:
                    continue
                shift = [f"{f['mu'] - r['mu']:+.4f}", f"{np.hypot(f['mu_err'], r['mu_err']):.4f}"] \
                    if r is not None and s["label"] != ref else ["", ""]
                row = [s["label"], name, n, s["i_ma"], s["dwell"], f"{f['mu']:.4f}",
                       f"{f['mu_err']:.4f}", f"{f['height']:.1f}", f"{f['height'] / s['i_ma']:.1f}",
                       f"{f['fwhm']:.4f}", f"{f['chi2_red']:.2f}"] + shift
                if extra:
                    row += [fn(f, r) if s["label"] != ref and r is not None else ""
                            for _, fn in extra]
                w.writerow(row)
    print(f"Saved {path.name}")


def dd_over_d(f, r):
    """Fractional change in d implied by the shift: Δd/d = −cot θ · Δθ (rad)."""
    th = np.radians(r["mu"])
    return f"{-np.radians(f['mu'] - r['mu']) / np.tan(th):+.3e}"


def dd_over_d_err(f, r):
    th = np.radians(r["mu"])
    return f"{np.radians(np.hypot(f['mu_err'], r['mu_err'])) / np.tan(th):.1e}"


def write_runs(path, groups):
    with path.open("w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["analysis", "run", "source_file", "column", "U_kV", "I_mA", "dwell_s",
                    "beta_min_deg", "beta_max_deg", "n_points", "note"])
        for analysis, scans in groups:
            for s in scans:
                w.writerow([analysis, s["label"], s["file"], s["col"], s["u_kv"], s["i_ma"],
                            s["dwell"], f"{s['angle'][0]:.1f}", f"{s['angle'][-1]:.1f}",
                            len(s["angle"]), s["note"]])
    print(f"Saved {path.name}")


def main():
    align = load(ALIGNMENT_FULL)
    align_fri = load(ALIGNMENT_FRIDAY)
    temp = load(TEMPERATURE)

    write_scans(HERE / "alignment_full_scans.csv", align, *ALIGN_RANGE)
    write_scans(HERE / "alignment_friday_normal_1st_order.csv", align_fri)
    write_scans(HERE / "temperature_full_scans.csv", temp)

    a_scans = align + align_fri
    write_peaks(HERE / "alignment_peak_positions.csv", a_scans, fit_all(a_scans),
                ref="normal_thu_0.6mA")
    write_peaks(HERE / "temperature_peak_positions.csv", temp, fit_all(temp), ref="unheated",
                extra=[("delta_d_over_d", dd_over_d), ("delta_d_over_d_err", dd_over_d_err)])
    write_runs(HERE / "runs_used.csv",
               [("alignment", a_scans), ("temperature", temp)])


if __name__ == "__main__":
    main()
