"""Shared matplotlib style: validated categorical palette (fixed order), recessive axes, thin marks."""
import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402

BLUE, ORANGE, AQUA, YELLOW, MAGENTA, GREEN, VIOLET, RED = (
    '#2a78d6', '#eb6834', '#1baf7a', '#eda100', '#e87ba4', '#008300', '#4a3aa7', '#e34948')
GRAY = '#8a8984'
INK, INK2 = '#0b0b0b', '#52514e'


def setup():
    plt.rcParams.update({
        'figure.dpi': 110, 'savefig.dpi': 160, 'savefig.bbox': 'tight',
        'font.size': 9, 'axes.titlesize': 10, 'axes.labelsize': 9,
        'axes.edgecolor': '#c9c8c2', 'axes.labelcolor': INK2, 'xtick.color': INK2, 'ytick.color': INK2,
        'axes.spines.top': False, 'axes.spines.right': False,
        'axes.grid': True, 'grid.color': '#ecebe7', 'grid.linewidth': 0.6,
        'lines.linewidth': 2.0, 'lines.markersize': 5, 'legend.frameon': False,
        'figure.facecolor': '#fcfcfb', 'axes.facecolor': '#fcfcfb', 'savefig.facecolor': '#fcfcfb',
    })


def show_images(ax_grid, images, title=None):
    for ax, im in zip(ax_grid, images):
        ax.imshow(im, interpolation='nearest')
        ax.set_xticks([])
        ax.set_yticks([])
        ax.grid(False)
        for s in ax.spines.values():
            s.set_visible(False)
    if title is not None:
        ax_grid[0].set_ylabel(title, rotation=0, ha='right', va='center', color=INK2)
