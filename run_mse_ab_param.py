# General imports
import os
import tensorflow as tf
from scipy import stats

# Utility imports
from utils.losses import *
from utils.plotting import *
from utils.training import *

np.random.seed(666) # Need to do more to ensure data is the same across runs.

os.environ["CUDA_VISIBLE_DEVICES"] = "0" # pick a number < 4 on ML4HEP; < 3 on Voltan 
physical_devices = tf.config.list_physical_devices('GPU') 
tf.config.experimental.set_memory_growth(physical_devices[0], True)

# Experiment parameters
#num = 0    # bkgd: normal(-0.1, 1)     sgnl: normal(0.1, 1)
#num = 1    # bkgd: normal(-0.2, 1)     sgnl: normal(0.2, 1)
#num = 2    # bkgd: normal(-0.3, 1)     sgnl: normal(0.3, 1)
#num = 3    # bkgd: normal(-0.4, 1)     sgnl: normal(0.4, 1)
#num = 4    # bkgd: normal(-0.5, 1)     sgnl: normal(0.5, 1)
#num = 5    # bkgd: beta(2, 3)          sgnl: beta(3, 2)
num = 6    # bkgd: gamma(5, 1)         sgnl: gamma(6, 1)
reps = 20

# Data generation
N = 10**6
#bkgd = stats.norm(-0.1, 1)
#sgnl = stats.norm(0.1, 1)
#bkgd = stats.norm(-0.2, 1)
#sgnl = stats.norm(0.2, 1)
#bkgd = stats.norm(-0.3, 1)
#sgnl = stats.norm(0.3, 1)
#bkgd = stats.norm(-0.4, 1)
#sgnl = stats.norm(0.4, 1)
#bkgd = stats.norm(-0.5, 1)
#sgnl = stats.norm(0.5, 1)
#bkgd = stats.beta(2, 3)
#sgnl = stats.beta(3, 2)
bkgd = stats.gamma(5, 1)
sgnl = stats.gamma(6, 1)

filestr = 'models/univariate/mse_ab_param/set_{}/'.format(num)
mse_filestr = filestr + 'model_{}_{}.h5'

lr = make_lr(bkgd, sgnl)
mae = make_mae(bkgd, sgnl)

data, m, s = make_data(bkgd, sgnl, N)
np.save(filestr + 'm.npy', m)
np.save(filestr + 's.npy', s)

ps = np.round(np.linspace(-2, 2, 101), 2)

for p in ps:
    print('===================================================\n{}'.format(p))
    params = {'loss': get_mse(p)}
    for i in range(reps):
        print(i, end = '\t')
        model, trace = train(data, **params)
        print()
        model.save_weights(mse_filestr.format(p, i))