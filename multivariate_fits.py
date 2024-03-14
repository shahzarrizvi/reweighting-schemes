# General imports
import os
import pickle
from scipy import stats
import tensorflow as tf
import numpy as np

from utils.plotting import *
from utils.training import *

rc('font', size=15)        #22
rc('xtick', labelsize=10)  #15
rc('ytick', labelsize=10)  #15
rc('legend', fontsize=10)  #15
rc('text.latex', preamble=r'\usepackage{amsmath}')

w = 7
h = 3.5

np.random.seed(666) # Need to do more to ensure data is the same across runs.

os.environ["CUDA_VISIBLE_DEVICES"] = "0" # pick a number < 4 on ML4HEP; < 3 on Voltan 
physical_devices = tf.config.list_physical_devices('GPU') 
tf.config.experimental.set_memory_growth(physical_devices[0], True)

num = 0

x = np.linspace(-2, 2, 201)
y = np.linspace(-2, 2, 201)
xx, yy = np.meshgrid(x, y)

pos = np.empty(xx.shape + (2,))
pos[:, :, 0] = xx; pos[:, :, 1] = yy

N = 400
g = np.meshgrid(np.linspace(-2, 2, N + 1), np.linspace(-2, 2, N + 1))
g = np.append(g[0].reshape(-1, 1), g[1].reshape(-1, 1), axis = 1)

aa, bb = np.meshgrid(np.linspace(-2.005, 2.005, N + 2), np.linspace(-2.005, 2.005, N + 2))

mu_bkgd = np.array([-0.1, 0])
mu_sgnl = np.array([+0.1, 0])
sg_bkgd = np.eye(2)
sg_sgnl = np.eye(2)

bkgd = stats.multivariate_normal(mu_bkgd, sg_bkgd)
sgnl = stats.multivariate_normal(mu_sgnl, sg_sgnl)

lr = make_lr(bkgd, sgnl)

zz = np.zeros((N + 1)**2)
zz[sgnl.pdf(g) / bkgd.pdf(g) <= 1] = 1
zz[sgnl.pdf(g) / bkgd.pdf(g) > 1] = -1

vmin = -0.1
vmax = 0.1
ticks = np.arange(-0.1, 0.11, 0.05)



fig, axs = plt.subplots(1, 5, figsize = (10.5, 2), sharex = True, sharey = True)

axs[0].pcolormesh(aa, bb, zz.reshape(N + 1, N + 1), cmap = 'bwr', 
                     zorder = -2, rasterized = True)
axs[0].contour(xx, yy, 0.5 * (bkgd.pdf(pos) + sgnl.pdf(pos)), colors='white', 
                  linewidths = 0.75, zorder = -1)
axs[0].set_aspect('equal')
axs[0].set_yticks([-2, -1, 0, 1, 2])
axs[0].set_xticks([-2, -1, 0, 1, 2])
axs[0].tick_params(axis = 'y', which = 'minor', bottom = False)
axs[0].tick_params(axis = 'x', which = 'minor', bottom = False)
axs[0].tick_params(direction='in', which='both',length=5)
axs[0].set_title(r'Checker')
axs[0].set_xlabel(r'$x_1$', labelpad = -0.075)
axs[0].set_ylabel(r'$x_2$', labelpad = -3)
axs[0].set_xlim(-2, 2)
axs[0].set_ylim(-2, 2)

# BCE
filestr = 'models/multivariate/c_bce/set_{}/'.format(num)
preds_1 = np.load(filestr + 'preds_1.npy')
dd = (preds_1 - lr(g)).reshape(aa.shape[0] - 1, aa.shape[1] - 1)
plot_1 = axs[1].pcolormesh(aa, bb, dd, cmap = 'bwr', shading = 'auto', 
                              vmin = vmin, vmax  = vmax, rasterized = True)
axs[1].set_aspect('equal')
axs[1].set_yticks([-2, -1, 0, 1, 2])
axs[1].set_xticks([-2, -1, 0, 1, 2])
#axs[1].set_xticklabels([])
axs[1].set_yticklabels([])
axs[1].tick_params(axis = 'y', which = 'minor', bottom = False)
axs[1].tick_params(axis = 'x', which = 'minor', bottom = False)
axs[1].tick_params(direction='in', which='both',length=5)
axs[1].set_title(r'BCE')
#axs[1].set_ylabel(r'$x_2$', labelpad = -3)
axs[1].set_xlabel(r'$x_1$', labelpad = -0.75)
axs[1].set_xlim(-2, 2)
axs[1].set_ylim(-2, 2)

# MSE
filestr = 'models/multivariate/c_mse/set_{}/'.format(num)
preds_1 = np.load(filestr + 'preds_1.npy')
dd = (preds_1 - lr(g)).reshape(aa.shape[0] - 1, aa.shape[1] - 1)
plot_1 = axs[2].pcolormesh(aa, bb, dd, cmap = 'bwr', shading = 'auto', 
                              vmin = vmin, vmax  = vmax, rasterized = True)
axs[2].set_aspect('equal')
axs[2].set_yticks([-2, -1, 0, 1, 2])
axs[2].set_xticks([-2, -1, 0, 1, 2])
axs[2].set_xticklabels([])
axs[2].set_yticklabels([])
axs[2].tick_params(axis = 'y', which = 'minor', bottom = False)
axs[2].tick_params(axis = 'x', which = 'minor', bottom = False)
axs[2].tick_params(direction='in', which='both',length=5)
axs[2].set_title(r'MSE')
#axs[2].set_ylabel(r'$x_2$', labelpad = -0.75)
axs[2].set_xlabel(r'$x_1$', labelpad = -0.75)
axs[2].set_xlim(-2, 2)
axs[2].set_ylim(-2, 2)

# MLC
filestr = 'models/multivariate/c_mlc/set_{}/'.format(num)
preds_3 = np.load(filestr + 'preds_3.npy')
dd = (preds_3 - lr(g)).reshape(aa.shape[0] - 1, aa.shape[1] - 1)
plot_3 = axs[3].pcolormesh(aa, bb, dd, cmap = 'bwr', shading = 'auto', 
                              vmin = vmin, vmax  = vmax, rasterized = True)
axs[3].set_aspect('equal')
axs[3].set_yticks([-2, -1, 0, 1, 2])
axs[3].set_xticks([-2, -1, 0, 1, 2])
#axs[3].set_xticklabels([])
axs[3].set_yticklabels([])
axs[3].tick_params(axis = 'y', which = 'minor', bottom = False)
axs[3].tick_params(axis = 'x', which = 'minor', bottom = False)
axs[3].tick_params(direction='in', which='both',length=5)
axs[3].set_title(r'MLC')
#axs[3].set_ylabel(r'$x_2$', labelpad = -0.75)
axs[3].set_xlabel(r'$x_1$', labelpad = -0.75)
axs[3].set_xlim(-2, 2)
axs[3].set_ylim(-2, 2)

# SQR
filestr = 'models/multivariate/c_sqr/set_{}/'.format(num)
preds_3 = np.load(filestr + 'preds_3.npy')
dd = (preds_3 - lr(g)).reshape(aa.shape[0] - 1, aa.shape[1] - 1)
plot_3 = axs[4].pcolormesh(aa, bb, dd, cmap = 'bwr', shading = 'auto', 
                              vmin = vmin, vmax  = vmax, rasterized = True)
axs[4].set_aspect('equal')
axs[4].set_yticks([-2, -1, 0, 1, 2])
axs[4].set_xticks([-2, -1, 0, 1, 2])
#axs[4].set_xticklabels([])
axs[4].set_yticklabels([])
axs[4].tick_params(axis = 'y', which = 'minor', bottom = False)
axs[4].tick_params(axis = 'x', which = 'minor', bottom = False)
axs[4].tick_params(direction='in', which='both',length=5)
axs[4].set_title(r'SQR')
#axs[4].set_ylabel(r'$x_2$', labelpad = -0.75)
axs[4].set_xlabel(r'$x_1$', labelpad = -0.75)
axs[4].set_xlim(-2, 2)
axs[4].set_ylim(-2, 2)

fig.colorbar(plot_3, ax = axs.ravel(), fraction=0.0084, pad=0.02, 
             ticks = ticks)
#fig.tight_layout()
plt.savefig('poster/multivariate_fits.png', 
            dpi = 1200, 
            transparent = True, 
            bbox_inches = 'tight')