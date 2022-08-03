# General imports
import os
import tensorflow as tf
from scipy import stats

# Utility imports
from utils.losses import *
from utils.plotting import *
from utils.training import *

np.random.seed(666) # Need to do more to ensure data is the same across runs.

os.environ["CUDA_VISIBLE_DEVICES"] = "1" # pick a number < 4 on ML4HEP; < 3 on Voltan 
physical_devices = tf.config.list_physical_devices('GPU') 
tf.config.experimental.set_memory_growth(physical_devices[0], True)

# Experiment parameters
num = 2    # bkgd: normal(-0.1, 1)     sgnl: normal(0.1, 1)
#num = 3    # bkgd: beta(2, 3)          sgnl: beta(3, 2)
#num = 4    # bkgd: gamma(5, 1)         sgnl: gamma(6, 1)
#num = 5    # bkgd: normal(-0.2, 1)     sgnl: normal(0.2, 1)
#num = 6    # bkgd: normal(-0.3, 1)     sgnl: normal(0.3, 1)
#num = 7    # bkgd: normal(-0.4, 1)     sgnl: normal(0.4, 1)
#num = 8    # bkgd: normal(-0.5, 1)     sgnl: normal(0.5, 1)
reps = 20

# Data generation
N = 10**6
bkgd = stats.norm(-0.1, 1)
sgnl = stats.norm(-0.1, 1)
#bkgd = stats.beta(2, 3)
#sgnl = stats.beta(3, 2)
#bkgd = stats.gamma(5, 1)
#sgnl = stats.gamma(6, 1)
#bkgd = stats.norm(-0.2, 1)
#sgnl = stats.norm(0.2, 1)
#bkgd = stats.norm(-0.3, 1)
#sgnl = stats.norm(0.3, 1)
#bkgd = stats.norm(-0.4, 1)
#sgnl = stats.norm(0.4, 1)
#bkgd = stats.norm(-0.5, 1)
#sgnl = stats.norm(0.5, 1)

filestr = 'models/univariate/sqr_ab_param/set_{}/'.format(num)
sqr_filestr = filestr + 'linear/model_{}_{}.h5'
exp_filestr = filestr + 'exp/model_{}_{}.h5'

lr = make_lr(bkgd, sgnl)
mae = make_mae(bkgd, sgnl)

data, m, s = make_data(bkgd, sgnl, N)
np.save(filestr + 'm.npy', m)
np.save(filestr + 's.npy', s)

rs = np.sort(np.append(np.round(np.linspace(-2, 2, 81), 2),
                       np.round(np.linspace(-0.05, 0.05, 26), 3)[1:-1]))
#rs = rs[rs < 0]
rs = rs[rs >= 0]

for r in rs:
    print('===================================================\n{}'.format(r))
    sqr_params = {'loss': get_sqr(r), 'output':'relu'}
    exp_params = {'loss': get_exp_sqr(r), 'output':'linear'}
    for i in range(reps):
        print(i, end = '\t')
        sqr_model, trace = train(data, **sqr_params)
        exp_model, trace = train(data, **exp_params)
        print()
        sqr_model.save_weights(sqr_filestr.format(r, i))
        exp_model.save_weights(exp_filestr.format(r, i))