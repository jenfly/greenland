import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np
import pandas as pd

from .utils import pdfmerge


def savefig(filenm, save_on=True, savedir='../figures/', timestamp=True,
            **kwargs):
    """Save current figure if save_on flag is True.

    If input timestamp is True, the current date is prepended to file name.
    """
    if not save_on:
        return None
    if timestamp:
        filenm = pd.datetime.now().strftime('%Y-%m-%d') + '_' + filenm
    filenm = savedir + filenm
    print('Saving to ' + filenm)
    plt.savefig(filenm, **kwargs)
    return filenm


def save_figures(figs, filestr, save_figs=True, ext='pdf', merge=True,
                 delete_indiv=True, verbose=True):
    """Save multiple figures and optionally merge into a single PDF"""
    if not save_figs:
        return None
    filenms = []
    for i, fig in enumerate(figs):
        filenm = f'{filestr}{i:02d}.{ext}'
        print('Saving to ' + filenm)
        fig.savefig(filenm, bbox_inches='tight')
        filenms.append(filenm)
    if merge:
        outfile = f'{filestr}.{ext}'
        print('Merging to ' + outfile)
        pdfmerge(filenms, outfile, delete_indiv=delete_indiv)
    return filenms


def legend_2ax(ax1, ax2, **kwargs):
    """Create a combined legend for two y-axes."""
    h1, l1 = ax1.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax1.legend(h1 + h2, l1 + l2, **kwargs)
    return None


def weekly_gridlines(ax=None):
    """Set x-axis minor ticks and gridlines to weekly interval."""
    if ax is None:
        ax = plt.gca()
    ax.xaxis.set_minor_locator(mdates.WeekdayLocator(byweekday=(1),interval=1))
    ax.grid(which='minor')
