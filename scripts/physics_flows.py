# General imports
import os
import pickle
from scipy import stats
import numpy as np
import tensorflow as tf

from utils.plotting import *
import seaborn as sns

rc('font', size=15)        #22
rc('xtick', labelsize=10)  #15
rc('ytick', labelsize=10)  #15
rc('legend', fontsize=10)  #15
rc('text.latex', preamble=r'\usepackage{amsmath}')

w = 5.3
h = 3.5
lw = 2

os.environ["CUDA_VISIBLE_DEVICES"] = "0" # pick a number < 4 on ML4HEP; < 3 on Voltan 
physical_devices = tf.config.list_physical_devices('GPU') 
tf.config.experimental.set_memory_growth(physical_devices[0], True)

mc = np.load("data/zenodo/Pythia21_Zjet_pTZ-200GeV_0.npz")

sim_pt =  mc['sim_jets'][:, 0]
sim_y  = mc['sim_jets'][:, 1]
sim_phi = mc['sim_jets'][:, 2]
sim_m =   mc['sim_jets'][:, 3]

data = np.load("data/zenodo/Herwig_Zjet_pTZ-200GeV_0.npz")
dat_pt =  data['sim_jets'][:, 0]
dat_y  = data['sim_jets'][:, 1]
dat_phi = data['sim_jets'][:, 2]
dat_m =   data['sim_jets'][:, 3]

# Specify contours at 1 sigma, 2 sigma, 3 sigma
# Combine flows and corner into one?
# Possibly add ratios under the histograms and have both.

fig, axs = plt.subplots(4, 4, figsize = (10, 7))
levels = np.append(1.0 - np.exp(-0.5 * np.arange(0, 3.1, 1) ** 2), 1)

X = np.load('data/zenodo/fold/4/X_trn.npy')
y = np.load('data/zenodo/fold/4/y_trn.npy')
sim_flow = X[y == 0]
dat_flow = X[y == 1]

# pT, pT
bins = np.linspace(0, 600, 50)
axs[0, 0].hist(sim_pt, bins = bins, alpha = 0.35, color = 'crimson', density = True)
axs[0, 0].hist(dat_pt, bins = bins, alpha = 0.35, color = 'dodgerblue', density = True)
axs[0, 0].hist(sim_flow[:, 0], bins = bins, color = 'crimson', 
               histtype = 'step', density = True)
axs[0, 0].hist(dat_flow[:, 0], bins = bins, color = 'dodgerblue', 
               histtype = 'step', density = True)
axs[0, 0].set_ylabel(r'$p_T$ [GeV]')
axs[0, 0].set_yticks([])
axs[0, 0].set_xticks([])
#axs[0, 0].set_ylim(0, 11500)

# pT, y
sns.kdeplot(x = sim_pt, y = sim_y, fill = False, cmap = 'Reds', 
            ax = axs[1, 0], levels = levels, alpha = 0.5)
sns.kdeplot(x = dat_pt, y = dat_y, fill = False, cmap = 'Blues', 
            ax = axs[1, 0], levels = levels, alpha = 0.5)
axs[1, 0].set_xlim(0, 600)
axs[1, 0].set_ylim(-5, 5)
axs[1, 0].set_ylabel(r'$y$')
axs[1, 0].set_xticks([])

# y, y
bins = np.linspace(-5, 5, 50)
axs[1, 1].hist(sim_y, bins = bins, alpha = 0.35, color = 'crimson', density = True)
axs[1, 1].hist(dat_y, bins = bins, alpha = 0.35, color = 'dodgerblue', density = True)
axs[1, 1].hist(sim_flow[:, 1], bins = bins, color = 'crimson', 
               histtype = 'step', density = True)
axs[1, 1].hist(dat_flow[:, 1], bins = bins, color = 'dodgerblue', 
               histtype = 'step', density = True)
axs[1, 1].set_yticks([])
axs[1, 1].set_xticks([])
#axs[1, 1].set_ylim(0, 8000)

# pT, phi
sns.kdeplot(x = sim_pt, y = sim_phi, fill = False, cmap = 'Reds', 
            ax = axs[2, 0], levels = levels, alpha = 0.5)
sns.kdeplot(x = dat_pt, y = dat_phi, fill = False, cmap = 'Blues', 
            ax = axs[2, 0], levels = levels, alpha = 0.5)
axs[2, 0].set_xlim(0, 600)
axs[2, 0].set_ylim(-0.1, 2*np.pi + 0.1)
axs[2, 0].set_ylabel(r'$\phi$')
axs[2, 0].set_xticks([])

# y, phi
sns.kdeplot(x = sim_y, y = sim_phi, fill = False, cmap = 'Reds', 
            ax = axs[2, 1], levels = levels, alpha = 0.5)
sns.kdeplot(x = dat_y, y = dat_phi, fill = False, cmap = 'Blues', 
            ax = axs[2, 1], levels = levels, alpha = 0.5)
axs[2, 1].set_xlim(-5, 5)
axs[2, 1].set_ylim(-0.1, 2*np.pi + 0.1)
axs[2, 1].set_yticks([])
axs[2, 1].set_xticks([])

# phi, phi
bins = np.linspace(0, 2*np.pi, 50)
axs[2, 2].hist(sim_phi, bins = bins, alpha = 0.35, color = 'crimson', density = True)
axs[2, 2].hist(dat_phi, bins = bins, alpha = 0.35, color = 'dodgerblue', density = True)
axs[2, 2].hist(sim_flow[:, 2], bins = bins, color = 'crimson', 
               histtype = 'step', density = True)
axs[2, 2].hist(dat_flow[:, 2], bins = bins, color = 'dodgerblue', 
               histtype = 'step', density = True)
#axs[2, 2].set_ylim(0, 3200)
axs[2, 2].set_yticks([])
axs[2, 2].set_xticks([])

# pT, m
sns.kdeplot(x = sim_pt, y = sim_m, fill = False, cmap = 'Reds', 
            ax = axs[3, 0], levels = levels, alpha = 0.5)
sns.kdeplot(x = dat_pt, y = dat_m, fill = False, cmap = 'Blues', 
            ax = axs[3, 0], levels = levels, alpha = 0.5)
axs[3, 0].set_xlim(0, 600)
axs[3, 0].set_ylim(0, 80)
axs[3, 0].set_ylabel(r'$m$ [GeV]')
axs[3, 0].set_xlabel(r'$p_T$ [GeV]')

# y, m
sns.kdeplot(x = sim_y, y = sim_m, fill = False, cmap = 'Reds', 
            ax = axs[3, 1], levels = levels, alpha = 0.5)
sns.kdeplot(x = dat_y, y = dat_m, fill = False, cmap = 'Blues', 
            ax = axs[3, 1], levels = levels, alpha = 0.5)
axs[3, 1].set_xlim(-5, 5)
axs[3, 1].set_ylim(0, 80)
axs[3, 1].set_xlabel(r'$y$')
axs[3, 1].set_yticks([])

# phi, m
sns.kdeplot(x = sim_phi, y = sim_m, fill = False, cmap = 'Reds', 
            ax = axs[3, 2], levels = levels, alpha = 0.5)
sns.kdeplot(x = dat_phi, y = dat_m, fill = False, cmap = 'Blues', 
            ax = axs[3, 2], levels = levels, alpha = 0.5)
axs[3, 2].set_xlim(0 - 0.1, 2*np.pi + 0.1)
axs[3, 2].set_ylim(0, 80)
axs[3, 2].set_xlabel(r'$\phi$')
axs[3, 2].set_yticks([])

# m, m
bins = np.linspace(0, 80, 50)
axs[3, 3].hist(sim_m, bins = bins, alpha = 0.35, color = 'crimson', density = True)
axs[3, 3].hist(dat_m, bins = bins, alpha = 0.35, color = 'dodgerblue', density = True)
axs[3, 3].hist(sim_flow[:, 3], bins = bins, color = 'crimson', 
               histtype = 'step', density = True)
axs[3, 3].hist(dat_flow[:, 3], bins = bins, color = 'dodgerblue', 
               histtype = 'step', density = True)
axs[3, 3].set_xlabel(r'$m$ [GeV]')
axs[3, 3].set_yticks([])
#axs[3, 3].set_ylim(0, 15000)

axs[0, 1].axis('off')
axs[0, 2].axis('off')
axs[0, 3].axis('off')
axs[1, 2].axis('off')
axs[1, 3].axis('off')
axs[2, 3].axis('off')

legend = [Patch(facecolor='crimson', edgecolor=None, alpha = 0.5, label='Monte Carlo'),
          Patch(facecolor='dodgerblue', edgecolor=None, alpha = 0.5, label = 'Data'),
          Patch(facecolor='w', edgecolor='crimson', label = 'Monte Carlo Flow'),
          Patch(facecolor='w', edgecolor='dodgerblue', label = 'Data Flow')]
axs[0, 2].legend(handles = legend, loc='center', frameon = False)

plt.savefig('poster/physics_flows.png', transparent = True, dpi = 1200,
            bbox_inches = 'tight')