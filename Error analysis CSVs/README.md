# Error analysis CSVs: crystal alignment and temperature

Run `python3 build_error_csvs.py` in this folder to regenerate every file here from the `.xry` files.

All scans are at 35 kV with a 0.1° step. Each rate has a Poisson error column, `err = √(R·Δt)/Δt`, counting at least 1 count. Peak centres come from a Gaussian on a linear background fitted over a ±0.4° window. Each centre's error is scaled up by √χ²_red whenever χ²_red > 1.

## Files

| File | Contents |
|---|---|
| `alignment_full_scans.csv` | 2.5–30° scans: two normal-orientation scans (Thursday) and the two flipped-crystal scans (Friday Task 3) |
| `alignment_friday_normal_1st_order.csv` | Friday Task 2 normal-orientation 1st-order scans (14.5–18°, 10 s). These were taken on the same day as the flipped scans |
| `alignment_peak_positions.csv` | Fitted centre, height, height per mA and FWHM for each line and run. Each shift is relative to `normal_thu_0.6mA` |
| `temperature_full_scans.csv` | Tuesday's unheated and heated full scans (2.5–78.2°) |
| `temperature_peak_positions.csv` | The same peak table for the temperature runs. Each shift is relative to `unheated`, with Δd/d = −cot θ · Δθ |
| `runs_used.csv` | Source file, column and settings for every run used |

## Runs left out and why

| Source | Column | Why it was left out |
|---|---|---|
| `Thursday/Angle_fixed/Na_Cl_Cu_distance_perf.xry` | 2 | Identical point-for-point to `Tuesday/Cu_NaCl_mono_full.xry` (a reference overlay the software kept) |
| `Thursday/Angle_fixed/Na_Cl_Cu_distance_adj_angle_wrong.xry` | 1 | The same Tuesday copy again |
| `Na_Cl_Cu_distance_perf.xry` | 1, 3 | Partial runs (2.5–8.0° and 8.2–19.2°) |
| `Na_Cl_Cu_distance_adj_angle_wrong.xry` | 2, 3, 4 | Partial runs (2.5–17.0°, 17.7–21.1° and 2.5–25.9°) |
| `Friday/Task2/1st_Order_low.xry` | 1 | Duplicate of column 1 of `1st_Order_high_and_low.xry` |

About the Thursday `Na_Cl_Cu_distance_*` runs:
- They have the same settings as the flipped scans (1 mA, 1 s).
- They put Kα₁ at 16.04–16.07°.
- Thursday's Low Peak and High Peak scans and Friday's normal scans all agree with each other at 16.11–16.12°.
- So the `Na_Cl_Cu_distance_*` runs look like they were taken before the angle was set correctly, as the `adj_angle_wrong` filename suggests.

The baselines used here are Thursday's Low and High Peak full scans, which cover the same 2.5–30° range as the flipped scans. Friday's 1st-order scans agree with them to within 0.003–0.007°, which confirms the setup didn't change overnight.

**Don't mix Tuesday peak positions with Thursday or Friday ones.** Tuesday's peaks sit about 0.12° higher because they were taken before the angle fix. Use Tuesday only for heated − unheated differences.

## What the numbers show

**Alignment**
- **Flipped roll:** every line shifts by about −0.12° to −0.13°, in both 1st and 2nd order. The intensity per mA falls by about 2.5×.
- **Flipped yaw:** Kα₁ shifts by −0.085 ± 0.021° and Kβ₁ by −0.047 ± 0.022°. The intensity falls by about 9×, so the yaw-flipped crystal was badly misaligned.
- **Baseline repeatability:** the four normal runs agree to about 0.01°. That's the noise floor any alignment shift has to beat.

**Temperature**
- Every heated peak moves to a lower angle by about 0.005–0.015°.
- The `delta_d_over_d` column treats each line's shift as pure thermal expansion. That gives about 9 × 10⁻⁴ from the 1st-order lines but only 0.5–3 × 10⁻⁴ from the 2nd and 3rd orders.
- Real expansion should shift peaks in proportion to tan θ, which means much larger shifts at high order. These shifts don't grow with order.
- I fitted Δθ = Δβ₀ − tan θ · (Δd/d) across the five Kα/Kβ lines:
  - Constant angle shift: Δβ₀ = −0.015 ± 0.006°
  - Expansion: Δd/d = (−1.0 ± 2.8) × 10⁻⁴, so no expansion is detected
  - χ²_red = 0.09
- A constant shift alone fits about as well (χ²_red = 0.10). Expansion alone fits worse (χ²_red = 1.4).
- **Conclusion:** most of the heated shift looks like a small zero or angle shift between the two runs, not a change in d.
  - Quote Δd/d as an upper limit: under about 5 × 10⁻⁴ at 2σ. That's ΔT under about 12 K if α(NaCl) ≈ 40 × 10⁻⁶ K⁻¹.
  - Use the actual temperature difference if you recorded it.
