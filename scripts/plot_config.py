"""
plot_config.py
--------------
Centralised plotting conventions for the DwarfJeansAnalysis pipeline.

Importing this module applies all rcParams (LaTeX, fonts, ticks, colors,
figure size, legend style). Runner scripts only need::

    from plot_config import COLORS, DPI

Palette: Okabe-Ito colorblind-safe, with yellow replaced by dark goldenrod
(#B8860B) for better contrast on white backgrounds.
"""

import os
import matplotlib as mpl
from cycler import cycler

# texlive PATH fix for cluster (conditional for portability)
_texlive = "/global/software/sl-7.x86_64/modules/tools/texlive/2016/bin/x86_64-linux/"
if os.path.isdir(_texlive):
    os.environ["PATH"] += os.pathsep + _texlive

# Okabe-Ito palette (yellow replaced with dark goldenrod)
COLORS = [
    '#000000',  # black
    '#E69F00',  # orange
    '#56B4E9',  # sky blue
    '#009E73',  # bluish green
    '#B8860B',  # dark goldenrod (replaces yellow #F0E442)
    '#0072B2',  # blue
    '#D55E00',  # vermillion
    '#CC79A7',  # reddish purple
]

DPI = 150

# LaTeX
mpl.rcParams['text.usetex'] = True

# Fonts
mpl.rcParams['font.size']        = 15
mpl.rcParams['font.family']      = 'DejaVu Sans'
mpl.rcParams['font.serif']       = 'CMU Serif'
mpl.rcParams['font.sans-serif']  = (
    'CMU Sans Serif, DejaVu Sans, Bitstream Vera Sans, Lucida Grande, '
    'Verdana, Geneva, Lucid, Arial, Helvetica, Avant Garde, sans-serif'
)
mpl.rcParams['axes.labelsize']   = 15
mpl.rcParams['axes.titlesize']   = 14
mpl.rcParams['xtick.labelsize']  = 15
mpl.rcParams['ytick.labelsize']  = 15
mpl.rcParams['legend.fontsize']  = 15

mpl.rcParams['figure.figsize'] = (6, 6)

# Lines
mpl.rcParams['lines.linewidth']       = 1.5
mpl.rcParams['lines.antialiased']     = True
mpl.rcParams['lines.dashed_pattern']  = [2.8, 1.5]
mpl.rcParams['lines.dashdot_pattern'] = [4.8, 1.5, 0.8, 1.5]
mpl.rcParams['lines.dotted_pattern']  = [1.1, 1.1]
mpl.rcParams['lines.scale_dashes']    = True

mpl.rcParams['axes.prop_cycle'] = cycler('color', COLORS)

# Axes
mpl.rcParams['axes.linewidth'] = 1.0
mpl.rcParams['axes.labelpad']  = 9.0

# Tick marks — inward, minor visible, on all four sides
mpl.rcParams['xtick.top']           = True
mpl.rcParams['xtick.major.size']    = 5
mpl.rcParams['xtick.minor.size']    = 2.5
mpl.rcParams['xtick.major.width']   = 1.0
mpl.rcParams['xtick.minor.width']   = 0.75
mpl.rcParams['xtick.major.pad']     = 8
mpl.rcParams['xtick.direction']     = 'in'
mpl.rcParams['xtick.minor.visible'] = True
mpl.rcParams['ytick.right']         = True
mpl.rcParams['ytick.major.size']    = 5
mpl.rcParams['ytick.minor.size']    = 2.5
mpl.rcParams['ytick.major.width']   = 1.0
mpl.rcParams['ytick.minor.width']   = 0.75
mpl.rcParams['ytick.major.pad']     = 8
mpl.rcParams['ytick.direction']     = 'in'
mpl.rcParams['ytick.minor.visible'] = True

# Legend
mpl.rcParams['legend.frameon']        = False
mpl.rcParams['legend.framealpha']     = 0.8
mpl.rcParams['legend.fancybox']       = True
mpl.rcParams['legend.borderpad']      = 0.4
mpl.rcParams['legend.labelspacing']   = 0.5
mpl.rcParams['legend.handlelength']   = 1.5
mpl.rcParams['legend.handleheight']   = 0.7
mpl.rcParams['legend.handletextpad']  = 0.5
mpl.rcParams['legend.borderaxespad']  = 0.5
mpl.rcParams['legend.columnspacing']  = 2.0

# Save
mpl.rcParams['savefig.bbox']       = 'tight'
mpl.rcParams['savefig.pad_inches'] = 0.1
