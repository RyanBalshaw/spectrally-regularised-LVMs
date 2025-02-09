"""
MIT License
-----------

Copyright (c) 2023 Ryan Balshaw

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.

--------------------------------------------------------

This is a utility script for enhancing the appearance and consistency of matplotlib figures.

This script provides functionality to improve the visual quality and formatting of
matplotlib plots by addressing common rendering issues and inconsistencies. The main
function 'fix()' handles:

- Removal of '\mathdefault' from tick labels
- Proper handling of mathematical text and symbols
- Support for both 2D and 3D plots
- Configuration of minor and major tick labels
- Suppression of common matplotlib warnings during rendering

Dependencies:
    - matplotlib
    - logging
    - warnings

The script is particularly useful when creating publication-quality figures that
require consistent formatting and proper rendering of mathematical notation.
See the example in the fix() function docstring for typical usage.
"""

import logging

from matplotlib import pyplot as plt


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
