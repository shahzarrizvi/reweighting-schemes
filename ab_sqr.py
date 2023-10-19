# General imports
import os
import tensorflow as tf
from scipy import stats

# Utility imports
from utils.losses import *
from utils.plotting import *
from utils.training import *

np.random.seed(666) # Need to do more to ensure data is the same across runs.

os.environ["CUDA_VISIBLE_DEVICES"] = "3" # pick a number < 4 on ML4HEP; < 3 on Voltan 
physical_devices = tf.config.list_physical_devices('GPU') 
tf.config.experimental.set_memory_growth(physical_devices[0], True)

# Experiment parameters
# Physics
#num = 0 
num = 2

# Multivariate
#num = 0    # vertical
#num = 1    # slant
#num = 2    # circle
#num = 3    # hyperbola
#num = 4    # checker

# Univariate
#num = 0    # bkgd: normal(-0.1, 1)     sgnl: normal(0.1, 1)
#num = 1    # bkgd: normal(-0.2, 1)     sgnl: normal(0.2, 1)
#num = 2    # bkgd: normal(-0.3, 1)     sgnl: normal(0.3, 1)
#num = 3    # bkgd: normal(-0.4, 1)     sgnl: normal(0.4, 1)
#num = 4    # bkgd: normal(-0.5, 1)     sgnl: normal(0.5, 1)
#num = 5    # bkgd: beta(2, 3)          sgnl: beta(3, 2)
#num = 6    # bkgd: gamma(5, 1)         sgnl: gamma(6, 1)
reps = 20

# Model parameters
#dirstr = 'models/univariate/simple/ab_sqr/set_{}/'.format(num)
dirstr = 'models/zenodo/ab_sqr/set_{}/'.format(num)
filestr_1 = dirstr + 'relu/model_{}_{}.h5'
filestr_2 = dirstr + 'exponential/model_{}_{}.h5'

if not os.path.isdir(dirstr):
    os.mkdir(dirstr)

if not os.path.isdir(dirstr + 'relu/'):
    os.mkdir(dirstr + 'relu/')

if not os.path.isdir(dirstr + 'exponential/'):
    os.mkdir(dirstr + 'exponential/')

# Data parameters
N = 10**6
X = np.load('data/zenodo/fold/X_trn.npy')[:N]
y = np.load('data/zenodo/fold/y_trn.npy')[:N].astype('float32')
data, m, s = split_data(X, y)

rs = np.sort(np.append(np.round(np.linspace(-2, 2, 81), 2),
                       np.round(np.linspace(-0.05, 0.05, 26), 3)[1:-1]))
#rs = rs[rs < 0]
rs = rs[rs >= 0]

for r in rs:
    print('===================================================\n{}'.format(r))
    #params_1 = {'loss': get_sqr(r), 'activation':'relu'}
    #params_2 = {'loss': get_exp_sqr(r), 'activation':'linear'}
    params_1 = {'loss': get_sqr(r), 'd': 4, 'output':'relu'}
    params_2 = {'loss': get_exp_sqr(r), 'd': 4, 'output':'linear'}
    for i in range(reps):
        print(i, end = '\t')
        #sqr_model, trace = train_simple(data, **params_1)
        #exp_model, trace = train_simple(data, **params_2)
        sqr_model, trace = train(data, **params_1)
        exp_model, trace = train(data, **params_2)
        print()
        sqr_model.save_weights(filestr_1.format(r, i))
        exp_model.save_weights(filestr_2.format(r, i))