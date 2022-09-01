# General imports
import tensorflow as tf
import os

# Utility imports
from utils.losses import *
from utils.plotting import *
from utils.training import *

np.random.seed(666) # Need to do more to ensure data is the same across runs.

os.environ["CUDA_VISIBLE_DEVICES"] = "0" # pick a number < 4 on ML4HEP; < 3 on Voltan 
physical_devices = tf.config.list_physical_devices('GPU') 
tf.config.experimental.set_memory_growth(physical_devices[0], True)

# Experiment parameters
# Multivariate
#num = 0    # vertical
#num = 1    # slant
#num = 2    # circle
#num = 3    # hyperbola
num = 4    # checker

#Univariate
#num = 0    # bkgd: normal(-0.1, 1)     sgnl: normal(0.1, 1)
#num = 1    # bkgd: normal(-0.2, 1)     sgnl: normal(0.2, 1)
#num = 2    # bkgd: normal(-0.3, 1)     sgnl: normal(0.3, 1)
#num = 3    # bkgd: normal(-0.4, 1)     sgnl: normal(0.4, 1) 
#num = 4    # bkgd: normal(-0.5, 1)     sgnl: normal(0.5, 1)
#num = 5    # bkgd: beta(2, 3)          sgnl: beta(3, 2)
#num = 6    # bkgd: gamma(5, 1)         sgnl: gamma(6, 1)
reps = 20

# File parameters
filestr = 'models/multivariate/ab_mse/set_{}/'.format(num)
mse_filestr = filestr + 'model_{}_{}.h5'

if not os.path.isdir(filestr):
    os.mkdir(filestr)

# Data parameters
N = 10**6
X = np.load('data/mvn/checker/X_trn.npy')[:N]
y = np.load('data/mvn/checker/y_trn.npy')[:N]
data, m, s = split_data(X, y)

ps = np.round(np.linspace(-2, 2, 101), 2)

for p in ps:
    print('===================================================\n{}'.format(p))
    params = {'loss': get_mse(p), 'd': 2}
    for i in range(reps):
        print(i, end = '\t')
        model, trace = train(data, **params)
        print()
        model.save_weights(mse_filestr.format(p, i))