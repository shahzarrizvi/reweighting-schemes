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

w = 3.4
h = 4
lw = 2

np.random.seed(666) # Need to do more to ensure data is the same across runs.

os.environ["CUDA_VISIBLE_DEVICES"] = "0" # pick a number < 4 on ML4HEP; < 3 on Voltan 
physical_devices = tf.config.list_physical_devices('GPU') 
tf.config.experimental.set_memory_growth(physical_devices[0], True)

rs = np.sort(np.append(np.round(np.linspace(-2, 2, 81), 2),
                       np.round(np.linspace(-0.05, 0.05, 26), 3)[1:-1]))


plt.figure(figsize = (w, h))

num = 4
filestr = 'models/multivariate/ab_sqr/set_{}/'.format(num)
exp_avgs = np.load(filestr + 'exp_avgs.npy')
plt.plot(rs, exp_avgs, label=r'$\exp{z}$', c='orangered', lw = lw)
#plt.legend(frameon = False)
plt.minorticks_on()
plt.tick_params(which = 'minor', length = 3)
plt.tick_params(which = 'major', length = 5)
plt.tick_params(which = 'both', direction='in')
#plt.set_yticklabels([])
plt.xlabel(r'$r$')
plt.ylabel('MAE (20 trials)')
plt.ylim(0, 0.06)
plt.xlim(-2, 2)
plt.gca().annotate(' ', (1.0, -0.004), 
                   arrowprops = dict(color = 'k', headlength = 2.5, headwidth = 2.5),
                   annotation_clip = False)

#plt.title(r"$r$-SQR Losses", loc="right");
plt.savefig('poster/multivariate_rsqr.png',
            transparent = True,
            dpi=1200, 
            bbox_inches='tight')