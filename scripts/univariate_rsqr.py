# General imports
import os
import pickle
from scipy import stats
import numpy as np
import tensorflow as tf

from utils.plotting import *

rc('font', size=15)        #22
rc('xtick', labelsize=10)  #15
rc('ytick', labelsize=10)  #15
rc('legend', fontsize=10)  #15
rc('text.latex', preamble=r'\usepackage{amsmath}')

w = 5.3
h = 3.5
lw = 2

np.random.seed(666) # Need to do more to ensure data is the same across runs.

os.environ["CUDA_VISIBLE_DEVICES"] = "0" # pick a number < 4 on ML4HEP; < 3 on Voltan 
physical_devices = tf.config.list_physical_devices('GPU') 
tf.config.experimental.set_memory_growth(physical_devices[0], True)

# Experiment parameters
num = 0
reps = 20

# File parameters
filestr = 'models/univariate/ab_sqr/set_{}/'.format(num)
lin_filestr = filestr + 'relu/model_{}_{}.h5'
exp_filestr = filestr + 'exponential/model_{}_{}.h5'

# Data parameters
#N = 10**6
#X = np.load('data/normal/0.1/X_trn.npy')[:N]
#y = np.load('data/normal/0.1/y_trn.npy')[:N]
#data, m, s = split_data(X, y)

rs = np.sort(np.append(np.round(np.linspace(-2, 2, 81), 2),
                       np.round(np.linspace(-0.05, 0.05, 26), 3)[1:-1]))

# True distribution information
#bkgd = stats.norm(-0.1, 1)
#sgnl = stats.norm(+0.1, 1)

#lr = make_lr(bkgd, sgnl)
#mae = make_mae(bkgd, sgnl, 'data/normal/0.1/')

plt.figure(figsize = (w, h))

exp_avgs = np.load(filestr + 'exp_avgs.npy')
plt.plot(rs, exp_avgs, c='orangered', lw = lw, label = r'$\exp(z)$')

plt.minorticks_on()
plt.tick_params(which = 'minor', length = 3)
plt.tick_params(which = 'major', length = 5)
plt.tick_params(which = 'both', direction='in')
plt.ylabel(r'MAE (20 trials)')
plt.xlabel(r'$r$')
plt.ylim(0, 0.02)
plt.xlim(-2, 2)
plt.yticks(np.arange(0, 0.021, 0.002))
#plt.legend(frameon = False)
plt.gca().annotate(' ', (1.0, -0.0015), 
                   arrowprops = dict(color = 'k', headlength = 2.5, headwidth = 2.5),
                   annotation_clip = False)

#plt.title(r"$r$-SQR Losses", loc="right");
plt.savefig('poster/univariate_rsqr.png',
            transparent = True,
            dpi=1200, 
            bbox_inches='tight')