# Plotting imports
import numpy as np
from matplotlib import pyplot as plt
from matplotlib.patches import Patch
from matplotlib import rc
import matplotlib.font_manager
rc('font', family='serif')
rc('text', usetex=True)
rc('font', size=22)
rc('xtick', labelsize=15)
rc('ytick', labelsize=15)
rc('legend', fontsize=15)

# Plotting functions
def get_preds(model_lrs, xs=np.linspace(4, 16, 1000)):
    # Takes in model_lrs, a list of model likelihood ratios and xs, a list of 
    # values on which to compute the likelihood ratios. Returns a 2D array. The 
    # nth row is the likelihood ratio predictions from the nth model in 
    # model_lrs.
    return np.array([model_lr(xs) for model_lr in model_lrs])
    
def avg_lr(preds):
    # Takes in a 2D array of multiple models' likelihood ratio predictions. 
    # Returns the average likelihood ratio prediction and its error.
    return preds.mean(axis=0), preds.std(axis=0)

def avg_lrr(preds, xs=np.linspace(4, 16, 1000)):
    # Takes in a 2D array of multiple models' likelihood ratio predictions. 
    # Returns the average ratio of predicted likelihood to true likelihood and 
    # its error.
    lrr_preds = preds / lr(xs)
    return lrr_preds.mean(axis=0), lrr_preds.std(axis=0)
    
def lr_plot(ensembles,
            title=None,
            filename=None,
            cs = ['brown', 'green', 'red', 'blue'],
            lss = ['-', '--', '-.', ':'],
            xs=np.linspace(0, 20, 1000)):
            #xs=np.linspace(4, 16, 1000)):
            #xs=np.linspace(-6, 6, 1000)):
    # Takes in a list of pairs (lr_avg, lr_err). Plots them against the true 
    # likelihood.
    fig, ax_1 = plt.subplots(figsize = (10, 6))
    
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

    # Plot background and signal
    ax_2 = ax_1.twinx()
    #bins = np.linspace(4, 16, 100)
    bins = np.linspace(0, 20, 100)
    #bins = np.linspace(-6, 6, 100)
    plt.hist(sgnl, alpha=0.1, bins=bins)
    plt.hist(bkgd, alpha=0.1, bins=bins)
    ax_2.minorticks_on()
    ax_2.tick_params(direction='in', which='both',length=5)
    plt.ylabel('Count')

    plt.xlim(xs[0], xs[-1])
    plt.xlabel(r'$x$')
    #plt.title(r"$\mu_{\rm{sgnl}}="+str(10.1)+r", \mu_{\rm{bkgd}}="+str(9.9)+r"$",loc="right",fontsize=20)
    plt.title(r"$r_{\rm{sgnl}}="+str(r)+r", r_{\rm{bkgd}}="+str(r+1)+r"$",loc="right",fontsize=20)
    #plt.title(r"$\mu_{\rm{sgnl}}="+str(mu)+r", \mu_{\rm{bkgd}}="+str(-mu)+r"$",loc="right",fontsize=20)
    if title != None:
        plt.title(title, loc="left", fontsize=20)
    if filename != None:
        plt.savefig(filename, dpi=1200, bbox_inches='tight')

def lrr_plot(ensembles,
             title=None,
             filename=None,
             cs = ['brown', 'green', 'red', 'blue'],
             lss = [':', '--', '-.', ':'],
             xs=np.linspace(0, 20, 1000)):
             #xs=np.linspace(4, 16, 1000)):
             #xs=np.linspace(-6, 6, 1000)):
    # Takes in a list of pairs (lrr_avg, lrr_err). Plots them.
    fig, ax_1 = plt.subplots(figsize = (10, 6))
    
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
    bins = np.linspace(0, 20, 100)
    #bins = np.linspace(-6, 6, 100)
    plt.hist(sgnl, alpha=0.1, bins=bins)
    plt.hist(bkgd, alpha=0.1, bins=bins)
    ax_2.minorticks_on()
    ax_2.tick_params(direction='in', which='both',length=5)
    plt.ylabel('Count')

    plt.xlim(xs[0], xs[-1])
    plt.xlabel(r'$x$')
    #plt.title(r"$\mu_{\rm{sgnl}}="+str(10.1)+r", \mu_{\rm{bkgd}}="+str(9.9)+r"$",loc="right",fontsize=20)
    plt.title(r"$r_{\rm{sgnl}}="+str(r)+r", r_{\rm{bkgd}}="+str(r+1)+r"$",loc="right",fontsize=20)
    #plt.title(r"$\mu_{\rm{sgnl}}="+str(mu)+r", \mu_{\rm{bkgd}}="+str(-mu)+r"$",loc="right",fontsize=20)
    if title != None:
        plt.title(title, loc="left", fontsize=20)
    if filename != None:
        plt.savefig(filename, dpi=1200, bbox_inches='tight')