import matplotlib.pyplot as plt
import sklearn.datasets as datasets

import torch
from torch import nn
from torch import optim
from torch.utils.data import DataLoader

from nflows.flows.base import Flow
from nflows.distributions.normal import StandardNormal
from nflows.transforms.base import CompositeTransform
from nflows.transforms.autoregressive import MaskedAffineAutoregressiveTransform
from nflows.transforms.permutations import ReversePermutation

import tqdm as tqdm

import numpy as np
from scipy import stats
# Plots have dimension (w,h)
w = 3.5
h = 3.25 

import sklearn.metrics as metrics
from sklearn.ensemble import AdaBoostClassifier
from sklearn.model_selection import train_test_split

def ratio_hist(truth, fit,
               labels,
               color = "gray",
               figsize = (8, 8),
               x_lim = None,
               y_lim = None,
               title = None,
               filename = None):
    fig, axs = plt.subplots(2, 1,
                            figsize = figsize,
                            sharex = True, 
                            gridspec_kw = {'height_ratios': [2, 1]})
    
    truth = truth[(truth > x_lim[0]) & (truth < x_lim[1])]
    fit = fit[(fit > x_lim[0]) & (fit < x_lim[1])]
    
    bins = np.linspace(x_lim[0], x_lim[1], 81)
    
    t_hist = axs[0].hist(truth, bins = bins, density = True, color = color, alpha = 0.25, label = labels[0])
    f_hist = axs[0].hist(fit, bins = bins, density = True, histtype = 'step', color = 'red', label = labels[1])
    
    axs[0].minorticks_on()
    axs[0].tick_params(direction='in', which='both')
    axs[0].legend()
    
    if y_lim:
        axs[0].set_ylim(y_lim[0], y_lim[1])
    if x_lim:
        axs[0].set_xlim(x_lim[0], x_lim[1])
    
    bins = (f_hist[1] + np.diff(f_hist[1]).mean() / 2)[:-1]
    axs[1].scatter(bins, f_hist[0] / t_hist[0], marker = '_', c = 'red', lw = 0.75)
    axs[1].axhline(1,color="gray", lw=0.5)
    
    axs[1].minorticks_on()
    axs[1].tick_params(direction='in', which='both')
    axs[1].set_ylim(0, 2)
    
    plt.xlabel(labels[2])
    
    if title:
        axs[0].set_title(title, loc = "right")
    if filename:
        plt.savefig(filename, 
                    dpi = 300,
                    transparent = True,
                    bbox_inches = 'tight')

def calculate_auc(fake, real):
    np.random.seed(666)
    n, d = fake.shape
    y = stats.bernoulli.rvs(0.5, size = n).astype('float32')
    X = np.zeros_like(fake)
    X[y == 0] = real[y == 0]
    X[y == 1] = fake[y == 1]

    # Take 70% of data to be training data.
    X_trn, X_tst, y_trn, y_tst = train_test_split(X, y, train_size = 0.7)
    ab_clf = AdaBoostClassifier()
    ab_clf.fit(X_trn, y_trn)
    
    y_hat = ab_clf.predict_proba(X_tst)[:, 1]
    auc = metrics.roc_auc_score(y_tst, y_hat)
    return auc

# Load in data.
mc = np.load("../data/zenodo/Pythia21_Zjet_pTZ-200GeV_0.npz")

mc_pt =  mc['sim_jets'][:, 0] 
mc_eta = mc['sim_jets'][:, 1]
mc_phi = mc['sim_jets'][:, 2]
mc_m =   mc['sim_jets'][:, 3]
mc_w = mc['sim_widths']
mc_sdms = mc['sim_sdms']

mc = np.vstack([mc_pt, mc_eta, mc_m, mc_w, mc_sdms]).T


dt = np.load("../data/zenodo/Herwig_Zjet_pTZ-200GeV_0.npz")

dt_pt =  dt['sim_jets'][:, 0] 
dt_eta = dt['sim_jets'][:, 1]
dt_phi = dt['sim_jets'][:, 2]
dt_m =   dt['sim_jets'][:, 3]
dt_w = dt['sim_widths']
dt_sdms = dt['sim_sdms']

dt = np.vstack([dt_pt, dt_eta, dt_m, dt_w, dt_sdms]).T

n, d = mc.shape

num_layers = 5
base_dist = StandardNormal(shape=[d])

transforms = []
for _ in range(num_layers):
    transforms.append(ReversePermutation(features=d))
    transforms.append(MaskedAffineAutoregressiveTransform(features=d, 
                                                          hidden_features=8))
transform = CompositeTransform(transforms)

flow = Flow(transform, base_dist)

experiment = 'flows/sim/5'

ckpt = torch.load('flows/sim/5/ckpt_best')
flow.load_state_dict(ckpt['model_state_dict'])
flow.eval()

smp = flow.sample(n).detach().numpy()

print(-flow.log_prob(inputs=mc.astype('float32')).mean())
print(calculate_auc(smp, mc))

ratio_hist(mc[:, 0], smp[:, 0], 
           labels = ['MC', 'Flow', r'$p_T$'],
           color = 'blue',
           figsize = (w, h),
           title = r'$p_T$ (MC)',
           x_lim = (0, 750),
           filename = '../plots/zenodo/flows/{}/pT.png'.format(experiment)
          )

ratio_hist(mc[:, 1], smp[:, 1], 
           labels = ['MC', 'Flow', r'$\eta$'],
           color = 'blue',
           figsize = (w, h),
           title = r'$\eta$ (MC)',
           x_lim = (-5, 5),
           filename = '../plots/zenodo/flows/{}/eta.png'.format(experiment)
          )

ratio_hist(mc[:, 2], smp[:, 2], 
           labels = ['MC', 'Flow', r'$m$'],
           color = 'blue',
           figsize = (w, h),
           title = r'$m$ (MC)',
           x_lim = (0, 80),
           filename = '../plots/zenodo/flows/{}/m.png'.format(experiment)
          )

ratio_hist(mc[:, 3], smp[:, 3], 
           labels = ['MC', 'Flow', r'$w$'],
           color = 'blue',
           figsize = (w, h),
           title = r'$w$ (MC)',
           x_lim = (0, 0.6),
           filename = '../plots/zenodo/flows/{}/w.png'.format(experiment)
          )

ratio_hist(mc[:, 4], smp[:, 4], 
           labels = ['MC', 'Flow', r'sdms'],
           color = 'blue',
           figsize = (w, h),
           title = r'$m_{\rm SD}$ (MC)',
           x_lim = (0, 100),
           filename = '../plots/zenodo/flows/{}/sdms.png'.format(experiment)
          )

