from pathlib import Path
import matplotlib.pyplot as plt

# Every script saves its figures here as PNGs, overwriting the previous run's, so the folder always matches the
# latest analysis. See figures/README.md for what each figure shows.
figure_dir = Path(__file__).parent / "figures"


def save_figure(name, fig=None):
    figure_dir.mkdir(exist_ok=True)
    (fig or plt.gcf()).savefig(figure_dir / f"{name}.png", dpi=200, bbox_inches="tight")
