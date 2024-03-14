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

ps = np.round(np.linspace(-2, 2, 101), 2)
num = 4
filestr = 'models/multivariate/ab_mse/set_{}/'.format(num)
avgs = np.load(filestr + 'avgs.npy')

plt.figure(figsize = (w, h))
plt.plot(ps, avgs, c='lightseagreen', lw = lw, label = r'$\sigma(z)$')
plt.minorticks_on()
plt.tick_params(which = 'minor', length = 3)
plt.tick_params(which = 'major', length = 5)
plt.tick_params(which = 'both', direction='in')
#plt.xticklabels([])
plt.xlabel(r'$p$')
plt.ylabel('MAE (20 trials)')
plt.ylim(0, 0.06)
plt.xlim(-2, 2)
plt.gca().annotate(' ', (2.0, -0.004), 
                   arrowprops = dict(color = 'k', headlength = 2.5, headwidth = 2.5),
                   annotation_clip = False)

plt.savefig('poster/multivariate_pmse.png',
            transparent = True,
            dpi=1200, 
            bbox_inches='tight')