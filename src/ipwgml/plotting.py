"""
ipwgml.plotting
===============

Provides plotting-related functionality.
"""
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns


def set_style():
    """
    Set the IPWGML matplotlib style.
    """
    plt.style.use(Path(__file__).parent / "files" / "ipwgml.mplstyle")


cmap_precip = sns.cubehelix_palette(start=1.50, rot=-0.9, as_cmap=True, hue=0.8, dark=0.2, light=0.9)
cmap_tbs = sns.cubehelix_palette(start=2.2, rot=0.9, as_cmap=True, hue=1.3, dark=0.2, light=0.8, reverse=True)
cmap_tbs = sns.color_palette("rocket_r", as_cmap=True)
