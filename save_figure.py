from pathlib import Path
import matplotlib.pyplot as plt

# Every script saves its figures here as PNGs, overwriting the previous run's, so the folder always matches the
# latest analysis. See figures/README.md for what each figure shows.
figure_dir = Path(__file__).parent / "figures"

# Report style: every piece of text on a figure (title, axis labels, tick labels, legend, text boxes) is the same
# size, 1.5x matplotlib's default 10 pt. Apply with plt.rcParams.update(report_style), or plt.rc_context(report_style)
# for one figure among several; don't pass fontsize= to individual calls.
report_text_size = 15  # pt
report_style = {key: report_text_size for key in ["font.size", "axes.titlesize", "axes.labelsize", "xtick.labelsize",
                                                  "ytick.labelsize", "legend.fontsize", "legend.title_fontsize",
                                                  "figure.titlesize"]}
# The calibration figures (calib_1 to calib_4) are the exception: their titles are double the report size.
calib_style = {**report_style, "axes.titlesize": 2 * report_text_size}


def save_figure(name, fig=None):
    figure_dir.mkdir(exist_ok=True)
    (fig or plt.gcf()).savefig(figure_dir / f"{name}.png", dpi=200, bbox_inches="tight")
