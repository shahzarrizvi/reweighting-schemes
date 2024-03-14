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
filestr = 'models/univariate/ab_mse/set_{}/'.format(num)
mse_filestr = filestr + 'model_{}_{}.h5'

ps = np.round(np.linspace(-2, 2, 101), 2)
avgs = np.load(filestr + 'avgs.npy')

plt.figure(figsize = (w, h))
plt.plot(ps, avgs, c='lightseagreen', lw = lw, label = r'$\sigma(z)$')

plt.minorticks_on()
plt.tick_params(which = 'minor', length = 3)
plt.tick_params(which = 'major', length = 5)
plt.tick_params(which = 'both', direction='in')
plt.ylabel(r'MAE (20 trials)')
plt.xlabel(r'$p$')
plt.ylim(0, 0.02)
plt.xlim(-2, 2)
plt.yticks(np.arange(0, 0.021, 0.002))
#plt.legend(frameon = False)
plt.gca().annotate(' ', (2.0, -0.0015), 
                   arrowprops = dict(color = 'k', headlength = 2.5, headwidth = 2.5),
                   annotation_clip = False)

plt.savefig('poster/univariate_pmse.png',
            transparent = True,
            dpi=1200, 
            bbox_inches='tight')