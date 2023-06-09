"""
A script that makes matplotlib figures look awesome.
"""

import logging
import warnings

from matplotlib import pyplot as plt
from matplotlib.mathtext import MathTextWarning


def fix(ax, minor_flag: bool = True, flag_3d: bool = False):
    """
    A method that fixes known issues with attempting to render
    plots that look nicer.

    Parameters
    ----------
    ax : axis instance
        The plotting axis you want to fix.

    minor_flag : bool
        A flag to specify whether the minor axis or major axis is
        to be formatted.

    flag_3d :
        A flag to specify whether it is a 3d plot or not.

    Returns
    -------
    None

    Example
    ------------------------
    from matplotlib import pyplot as plt
    import matplotlib
    from axs_fixer import fix

    MEDIUM_SIZE = 30
    BIGGER_SIZE = 40

    plt.rc("font", size=MEDIUM_SIZE)  # controls default text sizes
    plt.rc("axes", titlesize=MEDIUM_SIZE)  # fontsize of the axes title
    plt.rc("axes", labelsize=MEDIUM_SIZE)  # fontsize of the x and y labels
    plt.rc("xtick", labelsize=MEDIUM_SIZE)  # fontsize of the tick labels
    plt.rc("ytick", labelsize=MEDIUM_SIZE)  # fontsize of the tick labels
    plt.rc("legend", fontsize=MEDIUM_SIZE)  # legend fontsize
    plt.rc("figure", titlesize=BIGGER_SIZE)  # fontsize of the figure title
    plt.rc("figure", figsize=(16, 10), dpi=600)
    plt.rc("savefig", dpi=600, format="pdf")
    plt.rc("grid", linestyle="--")

    matplotlib.rcParams.update(
        {  # Use mathtext, not LaTeX
            "text.usetex": False,
            # Use the Computer modern font
            "font.family": "serif",
            "font.serif": "cmr10",
            "mathtext.fontset": "cm",
            # Use ASCII minus
            "axes.unicode_minus": False,
        }
    )

    plt.ioff()

    # Create a figure
    fig, ax = plt.subplots(figsize = (8, 8))

    # Plot something
    ax.scatter(range(10), range(10), marker = "x", color = "b")
    ax.set_xlabel(r"$x_{range}$")
    ax.set_ylabel(r"$y_{range}$")
    ax.grid()

    # Fix the figure
    fix(ax,
        minor_flag = True,
        flag_3d = False)

    fig.tight_layout()
    plt.show()

    #
    """
    if ax is None:
        ax = plt.gca()
    fig = ax.get_figure()

    # Force the figure to be drawn
    logger = logging.getLogger("matplotlib.mathtext")
    original_level = logger.getEffectiveLevel()
    logger.setLevel(logging.ERROR)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=MathTextWarning)
        fig.canvas.draw()
    logger.setLevel(original_level)

    # Remove '\mathdefault' from all minor tick labels
    # label.get_text()
    # .replace("\\times", u"\u00D7")#"}$.$\\mathdefault{")
    # .replace("-", "$\\mathdefault{-}$")
    # .replace("−", "$\\mathdefault{-}$")

    # To suppress any FixedLocator errors
    if not minor_flag:
        ax.set_xticks(ax.get_xticks().tolist())
        ax.set_yticks(ax.get_yticks().tolist())

    labels_x = [
        label.get_text().replace("\mathdefault", "")
        for label in (
            ax.get_xminorticklabels() if minor_flag else ax.get_xmajorticklabels()
        )
    ]
    ax.set_xticklabels(labels_x, minor=minor_flag)

    labels_y = [
        label.get_text().replace("\mathdefault", "")
        for label in (
            ax.get_yminorticklabels() if minor_flag else ax.get_ymajorticklabels()
        )
    ]
    ax.set_yticklabels(labels_y, minor=minor_flag)

    if flag_3d:
        if not minor_flag:
            ax.set_zticks(ax.get_zticks().tolist())

        labels_z = [
            label.get_text().replace("\mathdefault", "")
            for label in (
                ax.get_zminorticklabels() if minor_flag else ax.get_zmajorticklabels()
            )
        ]
        ax.set_zticklabels(labels_z, minor=minor_flag)
