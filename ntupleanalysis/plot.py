import os
stylepath = os.path.realpath(os.path.join(os.path.dirname(__file__),'atlas.mplstyle'))
# import matplotlib
import matplotlib.pyplot as plt
from matplotlib import rcParams
import matplotlib.mlab as mlab
import matplotlib.ticker as ticker
from matplotlib.ticker import FuncFormatter
from matplotlib.ticker import MaxNLocator
from matplotlib.colors import LogNorm
from matplotlib import rc
import matplotlib.gridspec as gridspec

def plot_setup():
    plt.style.use(stylepath)
    rcParams['pdf.fonttype'] = 42
    # dictionaries to hold the styles for re-use
    text_fd = dict(ha='left', va='center')
    atlas_fd = dict(weight='bold', style='italic', size=24, **text_fd)
    internal_fd = dict(size=24, **text_fd)
    text_fd = dict(size=18, **text_fd)

def draw_atlas_text(ax=None, simStatus='Internal', lines=[r'$\sqrt{s} = $13 TeV, MC16e', r'$Z\rightarrow\mu\mu$+jets, $p_T^{\mu\mu} > 200$ GeV'],
                    side='left'):
    if ax is None:
        ax = plt.gca()
    if side=='left':
        xCoord = 0.04
        alignment='left'
        shift1 = .185
        shift0 = 0.
    else:
        xCoord = .96
        alignment='right'
        shift1 = 0.
        shift0 = -.47
    shift  = .92-.84
    ytop = .92
    ax.text(xCoord+shift0,ytop,'ATLAS', transform=ax.transAxes, weight='bold', style='italic', size=24, ha=alignment, va='center',
            bbox=dict(facecolor='white',edgecolor='none', alpha=0.7))
    ax.text(xCoord+shift1,ytop, 'Simulation '+simStatus, transform=ax.transAxes, size=24, ha=alignment, va='center',
            bbox=dict(facecolor='white',edgecolor='none', alpha=0.7))
    for i, line in enumerate(lines):
        ax.text(xCoord, ytop-(i+1)*shift, line, ha=alignment, va='center', 
                transform=ax.transAxes, size=18, bbox=dict(facecolor='white',edgecolor='none', alpha=0.7))
