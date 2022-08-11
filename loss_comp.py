# General imports
import os
import tensorflow as tf
from scipy import stats

# Utility imports
from utils.losses import *
from utils.plotting import *
from utils.training import *

np.random.seed(666) # Need to do more to ensure data is the same across runs.

os.environ["CUDA_VISIBLE_DEVICES"] = "2" # pick a number < 4 on ML4HEP; < 3 on Voltan 
physical_devices = tf.config.list_physical_devices('GPU') 
tf.config.experimental.set_memory_growth(physical_devices[0], True)

# Experiment parameters
num = 3
reps = 100

# Model parameters
p = 1.24
r = 0.018
bce_params = {'loss':bce}
mse_params = {'loss':get_mse(p)}
mlc_params = {'loss':exp_mlc, 'output':'linear'}
sqr_params = {'loss':get_exp_sqr(r), 'output':'linear'}

filestr = 'models/univariate/loss_comp/set_{}/'.format(num)
bce_filestr = filestr + 'bce/model_{}_{}.h5'
mse_filestr = filestr + 'mse/model_{}_{}.h5'
mlc_filestr = filestr + 'mlc/model_{}_{}.h5'
sqr_filestr = filestr + 'sqr/model_{}_{}.h5'

if not os.path.isdir(filestr):
    os.mkdir(filestr)

if not os.path.isdir(filestr + 'bce/'):
    os.mkdir(filestr + 'bce/')
    
if not os.path.isdir(filestr + 'mse/'):
    os.mkdir(filestr + 'mse/')

if not os.path.isdir(filestr + 'mlc/'):
    os.mkdir(filestr + 'mlc/')
    
if not os.path.isdir(filestr + 'sqr/'):
    os.mkdir(filestr + 'sqr/')

# Data parameters
Ns = 10**np.arange(2, 8)
X = np.load('data/normal/0.1/X_trn.npy')
y = np.load('data/normal/0.1/y_trn.npy')

for N in Ns:
    print('===================================================\n{}'.format(N))
    # Take the first N samples.
    data, m, s = split_data(X[:N], y[:N])
    
    for i in range(reps):
        print(i, end = ' ')
        bce_model, trace = train(data, **bce_params)
        mse_model, trace = train(data, **mse_params)
        mlc_model, trace = train(data, **mlc_params)
        sqr_model, trace = train(data, **sqr_params)
        print()
        bce_model.save_weights(bce_filestr.format(N, i))
        mse_model.save_weights(mse_filestr.format(N, i))
        mlc_model.save_weights(mlc_filestr.format(N, i))
        sqr_model.save_weights(sqr_filestr.format(N, i))