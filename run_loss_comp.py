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

num = 3
reps = 100
Ns = [10**7]

# Model parameters
bce_params = {'loss':bce}
mse_params = {'loss':mse}
mlc_params = {'loss':mlc, 'output':'relu'}
sqr_params = {'loss':sqr, 'output':'relu'}

filestr = 'models/univariate/loss_comp/set_{}/'.format(num)
bce_filestr = filestr + 'bce/model_{}_{}.h5'
mse_filestr = filestr + 'mse/model_{}_{}.h5'
mlc_filestr = filestr + 'mlc/model_{}_{}.h5'
sqr_filestr = filestr + 'sqr/model_{}_{}.h5'

for N in Ns:
    print('===================================================\n{}'.format(N))
    # Generate data
    bkgd = stats.norm(-0.1, 1)
    sgnl = stats.norm(+0.1, 1)
    data, m, s = make_data(bkgd, sgnl, N)
    np.save(filestr + 'm_{}.npy'.format(N), m)
    np.save(filestr + 's_{}.npy'.format(N), s)
    
    for i in range(reps):
        print(i, end = ' ')
        bce_model, trace = train(data, **bce_params)
        mse_model, trace = train(data, **mse_params)
        
        mlc_model, trace = train(data, **mlc_params)
        while trace.history['val_loss'][-1] > 0:
            mlc_model, trace = train(data, **mlc_params)
        
        sqr_model, trace = train(data, **sqr_params)
        while trace.history['val_loss'][-1] > 1:
            sqr_model, trace = train(data, **sqr_params)
        print()
            
        bce_model.save_weights(bce_filestr.format(N, i))
        mse_model.save_weights(mse_filestr.format(N, i))
        mlc_model.save_weights(mlc_filestr.format(N, i))
        sqr_model.save_weights(sqr_filestr.format(N, i))