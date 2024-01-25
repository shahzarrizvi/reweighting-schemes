# Plotting imports
import numpy as np

from matplotlib import pyplot as plt
from matplotlib.patches import Patch
from matplotlib import rc
import matplotlib.font_manager
import matplotlib.ticker
rc('font', family='serif')
rc('text', usetex=True)
rc('font', size=10)        #22 #10
rc('xtick', labelsize=8)  #15 #8
rc('ytick', labelsize=8)  #15 #8
rc('legend', fontsize=8)  #15 #8
#rc('font', size=6)        #22
#rc('xtick', labelsize=5)  #15
#rc('ytick', labelsize=5)  #15
#rc('legend', fontsize=5)  #15
rc('text.latex', preamble=r'\usepackage{amsmath}')

cs = ['brown', 'green', 'red', 'blue']
lss = [':', '--', '-.', ':']
lw = 0.75
#lw = 2

# Plotting functions
def get_preds(model_lrs, xs):
    # Takes in model_lrs, a list of model likelihood ratios and xs, a list of 
    # values on which to compute the likelihood ratios. Returns a 2D array. The 
    # nth row is the likelihood ratio predictions from the nth model in 
    # model_lrs.
    return np.array([model_lr(xs) for model_lr in model_lrs])
    
def avg_lr(preds):
    # Takes in a 2D array of multiple models' likelihood ratio predictions. 
    # Returns the average likelihood ratio prediction.
    return preds.mean(axis=0)

def avg_lrr(lr, preds, xs):
    # Takes in a 2D array of multiple models' likelihood ratio predictions. 
    # Returns the average ratio of predicted likelihood to true likelihood
    lrr_preds = preds / lr(xs)
    return lrr_preds.mean(axis=0)
    
def lr_plot(ensembles,
            lr,
            bkgd = None, 
            sgnl = None,
            xs = np.linspace(-6, 6, 1000), 
            bins = np.linspace(-6, 6, 100),
            figsize = (10, 6),
            title = None,
            filename = None):
    # Takes in a list of pairs (lr_avg, lr_err). Plots them against the true 
    # likelihood.
    fig, ax_1 = plt.subplots(figsize = figsize)
    
    # Plot true likelihood
    plt.plot(xs, lr(xs), label = 'Exact', c='k', ls='-')
    
    # Plot model likelihoods
    for i in range(len(ensembles)):
        avg, err, lbl = ensembles[i]
        plt.plot(xs, avg, label=lbl, c=cs[i], ls=lss[i])
        #plt.fill_between(xs, avg - err, avg + err, color=cs[i], alpha=0.1)
    plt.legend()
    ax_1.minorticks_on()
    ax_1.tick_params(direction='in', which='both',length=5)
    plt.ylabel('Likelihood Ratio')
    #plt.ylim(0, 10)

    if bkgd and sgnl:
        # Plot background and signal
        ax_2 = ax_1.twinx()
        #bins = np.linspace(4, 16, 100)
        #bins = np.linspace(0, 20, 100)
        X_bkgd = bkgd.rvs(size = 10**6)
        X_sgnl = sgnl.rvs(size = 10**6)
        plt.hist(X_sgnl, alpha=0.1, bins=bins)
        plt.hist(X_bkgd, alpha=0.1, bins=bins)
        ax_2.minorticks_on()
        ax_2.tick_params(direction='in', which='both',length=5)
        plt.ylabel('Count')

    plt.xlim(xs[0], xs[-1])
    plt.xlabel(r'$x$')
    #plt.title(r"$\mu_{\rm{sgnl}}="+str(10.1)+r", \mu_{\rm{bkgd}}="+str(9.9)+r"$",loc="right",fontsize=20)
    #plt.title(r"$r_{\rm{sgnl}}="+str(r)+r", r_{\rm{bkgd}}="+str(r+1)+r"$",loc="right",fontsize=20)
    plt.title(params,loc="right",fontsize=20)
    if title != None:
        plt.title(title, loc="left", fontsize=20)
    if filename != None:
        plt.savefig(filename, dpi=1200, bbox_inches='tight')

def lrr_plot(ensembles,
             lr,
             bkgd, sgnl,
             xs = np.linspace(-6, 6, 1000),
             bins = np.linspace(-6, 6, 100),
             figsize = (10, 6),
             title = None,
             filename = None):
    # Takes in a list of pairs (lrr_avg, lrr_err). Plots them.
    fig, ax_1 = plt.subplots(figsize = figsize)
    
    # Plot ratios of likelihood ratios
    for i in range(len(ensembles)):
        avg, err, lbl = ensembles[i]
        plt.plot(xs, avg, label=lbl, c=cs[i], ls=lss[i])
        #plt.fill_between(xs, avg - err, avg + err, color=cs[i], alpha=0.1)
    plt.axhline(1,ls=":",color="grey", lw=0.5)
    plt.axvline(0,ls=":",color="grey", lw=0.5)
    plt.legend()
    ax_1.minorticks_on()
    ax_1.tick_params(direction='in', which='both',length=5)
    plt.ylim(0.94, 1.06)
    plt.ylabel('Ratio')

    # Plot background and signal
    ax_2 = ax_1.twinx()
    #bins = np.linspace(4, 16, 100)
    #bins = np.linspace(0, 20, 100)
    X_bkgd = bkgd.rvs(size = 10**6)
    X_sgnl = sgnl.rvs(size = 10**6)
    plt.hist(X_sgnl, alpha=0.1, bins=bins)
    plt.hist(X_bkgd, alpha=0.1, bins=bins)
    ax_2.minorticks_on()
    ax_2.tick_params(direction='in', which='both',length=5)
    plt.ylabel('Count')

    plt.xlim(xs[0], xs[-1])
    plt.xlabel(r'$x$')
    #plt.title(r"$\mu_{\rm{sgnl}}="+str(10.1)+r", \mu_{\rm{bkgd}}="+str(9.9)+r"$",loc="right",fontsize=20)
    #plt.title(r"$r_{\rm{sgnl}}="+str(r)+r", r_{\rm{bkgd}}="+str(r+1)+r"$",loc="right",fontsize=20)
    plt.title(params, loc="right", fontsize=20)
    if title != None:
        plt.title(title, loc="left", fontsize=20)
    if filename != None:
        plt.savefig(filename, dpi=1200, bbox_inches='tight')

def ratio_plot(ensembles,
               labels,
               lr,
               xs,
               bkgd = None, sgnl = None, 
               y_lim = None,
               ratio_y_lim = (0.95, 1.05),
               figsize = (8, 8),
               cs = None, 
               lss = None,
               title = None, 
               filename = None):
    
    fig, axs = plt.subplots(2, 1,
                            figsize = figsize,
                            sharex = True, 
                            gridspec_kw = {'height_ratios': [2, 1]})
    
    if not cs:
        cs = np.repeat(['brown', 'green', 'red', 'blue'], len(ensembles))
    if not lss:
        lss = np.repeat([':', '--', '-.', ':'], len(ensembles))
    
    # Plot likelihood ratios
    #axs[0].plot(xs, lr(xs), label = 'Exact', c = 'k', lw = 0.75)
    axs[0].plot(xs, lr(xs), label = 'Exact', c = 'k', lw = lw)
    
    n = len(ensembles)
    
    lrs = [None] * n
    lrrs = [None] * n
    for i in range(n):
        lrs[i] = avg_lr(ensembles[i])
        lrrs[i] = avg_lrr(lr, ensembles[i], xs)
    
    for i in range(n):
        axs[0].plot(xs, 
                    lrs[i], 
                    label = labels[i],
                    c = cs[i], 
                    ls = lss[i],
                    lw = lw)
                    #lw = 0.75)
        
    axs[0].set_xlim(xs[0], xs[-1])
    if y_lim:
        axs[0].set_ylim(y_lim[0], y_lim[1])
    axs[0].legend(frameon = False)
    axs[0].minorticks_on()
    axs[0].tick_params(which = 'minor', length = 3)
    axs[0].tick_params(which = 'major', length = 5)
    axs[0].tick_params(which = 'both', direction='in')
    axs[0].set_ylabel('$\mathcal{L}(x)$')

    # Plot exact histograms
    if bkgd and sgnl:
        hist_ax = axs[0].twinx()
        bins = np.linspace(xs[0] - 0.05, xs[-1] + 0.05, 122)
        weights = bkgd.cdf(bins)[1:] - bkgd.cdf(bins[:-1])
        plt.hist(bins[:-1], bins = bins, weights = weights, alpha = 0.1)
        weights = sgnl.cdf(bins)[1:] - sgnl.cdf(bins[:-1])
        plt.hist(bins[:-1], bins = bins, weights = weights, alpha = 0.1);
        hist_ax.set_yticks([]);

    # Plot likelihood ratio ratios
    for i in range(n):
        axs[1].plot(xs, 
                    lrrs[i],
                    c = cs[i],
                    ls = lss[i],
                    lw = lw)
                    #lw = 0.75)

    axs[1].axhline(1,ls=":",color="grey", lw=0.5)
    axs[1].axvline(0,ls=":",color="grey", lw=0.5)
    axs[1].set_ylim(ratio_y_lim[0], ratio_y_lim[1]);
    axs[1].minorticks_on()
    axs[1].tick_params(which = 'minor', length = 3)
    axs[1].tick_params(which = 'major', length = 5)
    axs[1].tick_params(which = 'both', direction='in')
    axs[1].set_ylabel(r'$\hat{\mathcal{L}}(x) / \mathcal{L}(x)$')

    plt.subplots_adjust(hspace = 0.1)
    axs[1].set_xlabel(r'$x$')
                    
    if title:
        axs[0].set_title(title, loc = 'right')
    if filename:
        plt.savefig(filename, 
                    dpi = 1200,
                    transparent = True,
                    bbox_inches = 'tight')

def mpe_plot(mpes,
             labels,
             Ns,
             stds = None,
             figsize = (8, 8),
             y_lim = None,
             cs = None,
             lss = None,
             title = None,
             filename = None):
    
    plt.figure(figsize = figsize)
    
    if not cs:
        cs = np.repeat(['brown', 'green', 'red', 'blue'], len(mpes))
    if not lss:
        lss = np.repeat([':', '--', '-.', ':'], len(mpes))
    
    for i in range(len(mpes)):
        plt.plot(Ns, 
                 mpes[i],
                 label = labels[i],
                 c = cs[i],
                 ls = lss[i],
                 lw = 0.75)
        if stds:
            plt.fill_between(Ns, 
                             mpes[i] - stds[i], 
                             mpes[i] + stds[i], 
                             color = cs[i], 
                             alpha=0.1)
            
    plt.legend(frameon = False)
    
    plt.xlim(Ns[0], Ns[-1])
    if y_lim:
        plt.ylim(y_lim[0], y_lim[1])

    plt.minorticks_on()
    plt.tick_params(axis = 'y', which = 'minor', length = 3)
    plt.tick_params(axis = 'y', which = 'major', length = 5)
    plt.tick_params(which = 'both', direction='in')
    plt.xscale("log", base=10)
    plt.ylabel('Mean Percent Error')
    plt.xlabel(r'$N$')
    
    if title:
        plt.title(title, loc="right");
    if filename:
        plt.savefig(filename,
                    transparent = True,
                    dpi=1200, 
                    bbox_inches='tight')
        
def mae_plot(maes,
             labels,
             Ns,
             stds = None,
             figsize = (8, 8),
             y_lim = None,
             cs = None,
             lss = None,
             title = None,
             filename = None):
    
    plt.figure(figsize = figsize)
    
    if not cs:
        cs = np.repeat(['brown', 'green', 'red', 'blue'], len(maes))
    if not lss:
        lss = np.repeat([':', '--', '-.', ':'], len(maes))
    
    for i in range(len(maes)):
        plt.plot(Ns, 
                 maes[i],
                 label = labels[i],
                 c = cs[i],
                 ls = lss[i],
                 lw = 0.75)
        if stds:
            plt.fill_between(Ns, 
                             maes[i] - stds[i], 
                             maes[i] + stds[i], 
                             color = cs[i], 
                             alpha=0.1)
            
    plt.legend(frameon = False)
    
    plt.xlim(Ns[0], Ns[-1])
    if y_lim:
        plt.ylim(y_lim[0], y_lim[1])

    plt.minorticks_on()
    plt.tick_params(axis = 'y', which = 'minor', length = 3)
    plt.tick_params(axis = 'y', which = 'major', length = 5)
    plt.tick_params(axis = 'x', which = 'minor', bottom = False)
    plt.tick_params(which = 'both', direction='in')
    plt.xscale("log", base=10)
    plt.xticks(Ns)
    plt.ylabel('MAE (100 trials)')
    plt.xlabel(r'$N$')
    
    if title:
        #plt.title(title, loc="right");
        plt.title(title, loc = 'right')
    if filename:
        plt.savefig(filename,
                    transparent = True,
                    dpi=1200, 
                    bbox_inches='tight')

def diff_plot(preds,
              lr,
              g,
              aa, bb,
              figsize = (10, 8),
              title = None, 
              filename = None):
    
    plt.figure(figsize = figsize)
    
    dd = (preds - lr(g)).reshape(aa.shape[0] - 1, aa.shape[1] - 1)
    
    plt.pcolormesh(aa, bb, dd, cmap = 'bwr', shading = 'auto', vmin = -0.2, vmax = 0.2)
    plt.colorbar()
    plt.gca().set_aspect('equal')
    
    if title:
        plt.title(title, loc="right");
    if filename:
        plt.savefig(filename,
                    transparent = True,
                    dpi=1200, 
                    bbox_inches='tight')