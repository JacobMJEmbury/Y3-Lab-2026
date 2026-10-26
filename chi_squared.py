import warnings
import numpy as np
import matplotlib
from scipy import stats

# Goodness of fit of the single Gaussian and the two-Gaussian fit with I2/I1 = 0.518 (shared sigma) to each K-alpha
# peak, and how closely each fitted curve follows the makima spline through the data.
# fits_copy.py runs the whole analysis and draws its figures as it goes. Importing it reuses exactly the same
# baseline-free peaks, fits and splines (nominal variant); its figures are drawn off-screen.
matplotlib.use("Agg")
warnings.filterwarnings("ignore", message="FigureCanvasAgg is non-interactive")
import fits_copy as fits

# The straight baseline under each peak (slope and intercept) was fitted to the flanks of these same points in
# fits_copy.subtract_background, so it also uses up 2 degrees of freedom. The Kramers background comes from the full
# scan, a different data set, so it costs none here.
baseline_params = 2

# Each model: the function, its fitted parameters for order i, and its number of free parameters.
# The ratio fit is stored expanded to six numbers, so it is evaluated with two_gaussians; it has 4 free parameters.
models = {"Single Gaussian": (fits.gaussian, lambda i: fits.gaussian_fits["nominal"][i][0], 3),
          "2 Gaussians, I2/I1=0.518": (fits.two_gaussians, lambda i: fits.ratio_fits["nominal"][i][0], 4)}


# chi^2 of one fitted model against the measured points, with the raw-count Poisson errors.
#   chi2_nu_err  the spread chi^2_nu would have from counting noise alone if the model were right, sqrt(2/dof)
#   p_value      the chance of a chi^2 at least this large from counting noise alone if the model were right
#   pull         (chi2_nu - 1) / chi2_nu_err; consistent is True when it is within 2, i.e. counting noise alone
#                explains the scatter, and False when the shape misses something beyond counting noise
#   spline_rms  RMS difference between the fitted curve and the spline over the scan, in % of the peak height.
#                At the measured angles the spline IS the data, so a chi^2 against the spline there is the chi^2
#                above; between the points it only interpolates, adding no information, so this is descriptive only.
def goodness_of_fit(model, params, n_params, peak, spline):
    peak_angle, peak_rate, peak_sigma, _ = peak
    chi2 = np.sum(((peak_rate - model(peak_angle, *params)) / peak_sigma) ** 2)
    dof = len(peak_rate) - n_params - baseline_params
    angle_fine = np.linspace(peak_angle.min(), peak_angle.max(), 2000)
    spline_fine = spline(angle_fine)
    spline_rms = np.sqrt(np.mean((model(angle_fine, *params) - spline_fine) ** 2)) / spline_fine.max() * 100
    pull = (chi2 / dof - 1) / np.sqrt(2 / dof)
    return dict(points=len(peak_rate), n_params=n_params, dof=dof, chi2=chi2, chi2_nu=chi2 / dof,
                chi2_nu_err=np.sqrt(2 / dof), p_value=stats.chi2.sf(chi2, dof), pull=pull, consistent=abs(pull) <= 2,
                spline_rms=spline_rms)


results = {label: {name: goodness_of_fit(model, params(i), n_params, fits.peaks["nominal"][i], fits.splines[i])
                   for name, (model, params, n_params) in models.items()}
           for i, label in enumerate(fits.order_labels)}

# The single Gaussian is the two-Gaussian model with mean_2 = mean_1 (the sum is then one Gaussian of height
# 1.518 A1), so the two models are nested and an F-test says whether the one extra parameter (the second line's
# position) lowers chi^2 by more than chance. Using the two-Gaussian chi^2_nu as the noise scale keeps the test fair
# when chi^2_nu > 1, i.e. when neither model matches the counting noise.
#   better              the model with the lower chi2_nu
#   second_line_needed  the F-test p-value is below 5%
f_tests = {}
for label, by_model in results.items():
    single, double = by_model.values()
    f_stat = (single["chi2"] - double["chi2"]) / (double["chi2"] / double["dof"])
    p_f = stats.f.sf(f_stat, 1, double["dof"])
    f_tests[label] = dict(f_stat=f_stat, p_value=p_f,
                          better="two Gaussians" if double["chi2_nu"] < single["chi2_nu"] else "single Gaussian",
                          second_line_needed=p_f < 0.05)
