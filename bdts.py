# General imports
import os
import tensorflow as tf
from scipy import stats
from sklearn.ensemble import GradientBoostingClassifier
from xgboost import XGBClassifier
from joblib import dump, load 

# Utility imports
from utils.losses import *
from utils.plotting import *
from utils.training import *

np.random.seed(666) # Need to do more to ensure data is the same across runs.

os.environ["CUDA_VISIBLE_DEVICES"] = "0" # pick a number < 4 on ML4HEP; < 3 on Voltan 
physical_devices = tf.config.list_physical_devices('GPU') 
tf.config.experimental.set_memory_growth(physical_devices[0], True)

# Experiment parameters
num = 0
reps = 100
d = 1
Ns = 10**np.arange(2, 8)

# Model parameters
bce_params = {'loss':bce, 'd': d}

filestr = 'models/bdts/{}/set_{}/'.format(d, num)
bce_filestr = filestr + 'bce/model_{}_{}.h5'
bdt_filestr = filestr + 'bdt/model_{}.h5'
gbc_filestr = filestr + 'gbc/model_{}_{}.h5'

if not os.path.isdir(filestr):
    os.mkdir(filestr)

if not os.path.isdir(filestr + 'bce/'):
    os.mkdir(filestr + 'bce/')
    
if not os.path.isdir(filestr + 'bdt/'):
    os.mkdir(filestr + 'bdt/')
    
if not os.path.isdir(filestr + 'gbc/'):
    os.mkdir(filestr + 'gbc/')

# Data parameters
X = np.load('data/bdts/{}/X_trn.npy'.format(d)).reshape(-1, 1)
y = np.load('data/bdts/{}/y_trn.npy'.format(d)).astype('float32')


for N in Ns:
    print('===================================================\n{}'.format(N))
    # Take the first N samples.
    data, m, s = split_data(X[:N], y[:N])
    
    # Train BDT model (only need to train 1)
    bdt_model = XGBClassifier(early_stopping_rounds = 10)
    X_trn, X_vld, y_trn, y_vld = data
    bdt_model.fit(X_trn, y_trn, eval_set = [(X_vld, y_vld)], verbose = 0)
    trace = bdt_model.evals_result()['validation_0']
    print(trace['logloss'][-1], '\t', len(trace['logloss']), end = '\n')
    bdt_model.save_model(bdt_filestr.format(N))

    for i in range(reps):
        print(i, end = ' ')
        # Train BCE model
        bce_model, trace = train(data, **bce_params)
        bce_model.save_weights(bce_filestr.format(N, i))
        
        # Train GBC model
        gbc_model = GradientBoostingClassifier(validation_fraction = 0.25,
                                               n_iter_no_change = 10)
        gbc_model.fit(X[:N], y[:N])
        dump(gbc_model, gbc_filestr.format(N, i))
    print()

 # Experiment parameters
num = 0
reps = 100
d = 2
Ns = 10**np.arange(2, 8)

# Model parameters
bce_params = {'loss':bce, 'd': d}

filestr = 'models/bdts/{}/set_{}/'.format(d, num)
bce_filestr = filestr + 'bce/model_{}_{}.h5'
bdt_filestr = filestr + 'bdt/model_{}.h5'
gbc_filestr = filestr + 'gbc/model_{}_{}.h5'

if not os.path.isdir(filestr):
    os.mkdir(filestr)

if not os.path.isdir(filestr + 'bce/'):
    os.mkdir(filestr + 'bce/')
    
if not os.path.isdir(filestr + 'bdt/'):
    os.mkdir(filestr + 'bdt/')
    
if not os.path.isdir(filestr + 'gbc/'):
    os.mkdir(filestr + 'gbc/')

# Data parameters
X = np.load('data/bdts/{}/X_trn.npy'.format(d))
y = np.load('data/bdts/{}/y_trn.npy'.format(d)).astype('float32')   

for N in Ns:
    print('===================================================\n{}'.format(N))
    # Take the first N samples.
    data, m, s = split_data(X[:N], y[:N])
    
    # Train BDT model (only need to train 1)
    bdt_model = XGBClassifier(early_stopping_rounds = 10)
    X_trn, X_vld, y_trn, y_vld = data
    bdt_model.fit(X_trn, y_trn, eval_set = [(X_vld, y_vld)], verbose = 0)
    trace = bdt_model.evals_result()['validation_0']
    print(trace['logloss'][-1], '\t', len(trace['logloss']), end = '\n')
    bdt_model.save_model(bdt_filestr.format(N))

    for i in range(reps):
        print(i, end = ' ')
        # Train BCE model
        bce_model, trace = train(data, **bce_params)
        bce_model.save_weights(bce_filestr.format(N, i))
        
        # Train GBC model
        gbc_model = GradientBoostingClassifier(validation_fraction = 0.25,
                                               n_iter_no_change = 10)
        gbc_model.fit(X[:N], y[:N])
        dump(gbc_model, gbc_filestr.format(N, i))
    print()

# Experiment parameters
num = 0
reps = 100
d = 4
Ns = 10**np.arange(2, 8)

# Model parameters
bce_params = {'loss':bce, 'd': d}

filestr = 'models/bdts/{}/set_{}/'.format(d, num)
bce_filestr = filestr + 'bce/model_{}_{}.h5'
bdt_filestr = filestr + 'bdt/model_{}.h5'
gbc_filestr = filestr + 'gbc/model_{}_{}.h5'

if not os.path.isdir(filestr):
    os.mkdir(filestr)

if not os.path.isdir(filestr + 'bce/'):
    os.mkdir(filestr + 'bce/')
    
if not os.path.isdir(filestr + 'bdt/'):
    os.mkdir(filestr + 'bdt/')
    
if not os.path.isdir(filestr + 'gbc/'):
    os.mkdir(filestr + 'gbc/')

# Data parameters
X = np.load('data/bdts/{}/X_trn.npy'.format(d))
y = np.load('data/bdts/{}/y_trn.npy'.format(d)).astype('float32')
    
for N in Ns:
    print('===================================================\n{}'.format(N))
    # Take the first N samples.
    data, m, s = split_data(X[:N], y[:N])
    
    # Train BDT model (only need to train 1)
    bdt_model = XGBClassifier(early_stopping_rounds = 10)
    X_trn, X_vld, y_trn, y_vld = data
    bdt_model.fit(X_trn, y_trn, eval_set = [(X_vld, y_vld)], verbose = 0)
    trace = bdt_model.evals_result()['validation_0']
    print(trace['logloss'][-1], '\t', len(trace['logloss']), end = '\n')
    bdt_model.save_model(bdt_filestr.format(N))

    for i in range(reps):
        print(i, end = '\t')
        # Train BCE model
        bce_model, trace = train(data, **bce_params)
        bce_model.save_weights(bce_filestr.format(N, i))
        
        # Train GBC model
        gbc_model = GradientBoostingClassifier(validation_fraction = 0.25,
                                               n_iter_no_change = 10)
        gbc_model.fit(X[:N], y[:N])
        dump(gbc_model, gbc_filestr.format(N, i))
        print()
    print()    


# Experiment parameters
num = 0
reps = 100
d = 8
Ns = 10**np.arange(2, 8)

# Model parameters
bce_params = {'loss':bce, 'd': d}

filestr = 'models/bdts/{}/set_{}/'.format(d, num)
bce_filestr = filestr + 'bce/model_{}_{}.h5'
bdt_filestr = filestr + 'bdt/model_{}.h5'
gbc_filestr = filestr + 'gbc/model_{}_{}.h5'

if not os.path.isdir(filestr):
    os.mkdir(filestr)

if not os.path.isdir(filestr + 'bce/'):
    os.mkdir(filestr + 'bce/')
    
if not os.path.isdir(filestr + 'bdt/'):
    os.mkdir(filestr + 'bdt/')
    
if not os.path.isdir(filestr + 'gbc/'):
    os.mkdir(filestr + 'gbc/')

# Data parameters
X = np.load('data/bdts/{}/X_trn.npy'.format(d))
y = np.load('data/bdts/{}/y_trn.npy'.format(d)).astype('float32')

for N in Ns:
    print('===================================================\n{}'.format(N))
    # Take the first N samples.
    data, m, s = split_data(X[:N], y[:N])
    
    # Train BDT model (only need to train 1)
    bdt_model = XGBClassifier(early_stopping_rounds = 10)
    X_trn, X_vld, y_trn, y_vld = data
    bdt_model.fit(X_trn, y_trn, eval_set = [(X_vld, y_vld)], verbose = 0)
    trace = bdt_model.evals_result()['validation_0']
    print(trace['logloss'][-1], '\t', len(trace['logloss']), end = '\n')
    bdt_model.save_model(bdt_filestr.format(N))

    for i in range(reps):
        print(i, end = ' ')
        # Train BCE model
        bce_model, trace = train(data, **bce_params)
        bce_model.save_weights(bce_filestr.format(N, i))
        
        # Train GBC model
        gbc_model = GradientBoostingClassifier(validation_fraction = 0.25,
                                               n_iter_no_change = 10)
        gbc_model.fit(X[:N], y[:N])
        dump(gbc_model, gbc_filestr.format(N, i))
    print()
    
# Experiment parameters
num = 0
reps = 100
d = 16
Ns = 10**np.arange(2, 8)

# Model parameters
bce_params = {'loss':bce, 'd': d}

filestr = 'models/bdts/{}/set_{}/'.format(d, num)
bce_filestr = filestr + 'bce/model_{}_{}.h5'
bdt_filestr = filestr + 'bdt/model_{}.h5'
gbc_filestr = filestr + 'gbc/model_{}_{}.h5'

if not os.path.isdir(filestr):
    os.mkdir(filestr)

if not os.path.isdir(filestr + 'bce/'):
    os.mkdir(filestr + 'bce/')
    
if not os.path.isdir(filestr + 'bdt/'):
    os.mkdir(filestr + 'bdt/')
    
if not os.path.isdir(filestr + 'gbc/'):
    os.mkdir(filestr + 'gbc/')

# Data parameters
X = np.load('data/bdts/{}/X_trn.npy'.format(d))
y = np.load('data/bdts/{}/y_trn.npy'.format(d)).astype('float32')

for N in Ns:
    print('===================================================\n{}'.format(N))
    # Take the first N samples.
    data, m, s = split_data(X[:N], y[:N])
    
    # Train BDT model (only need to train 1)
    bdt_model = XGBClassifier(early_stopping_rounds = 10)
    X_trn, X_vld, y_trn, y_vld = data
    bdt_model.fit(X_trn, y_trn, eval_set = [(X_vld, y_vld)], verbose = 0)
    trace = bdt_model.evals_result()['validation_0']
    print(trace['logloss'][-1], '\t', len(trace['logloss']), end = '\n')
    bdt_model.save_model(bdt_filestr.format(N))

    for i in range(reps):
        print(i, end = ' ')
        # Train BCE model
        bce_model, trace = train(data, **bce_params)
        bce_model.save_weights(bce_filestr.format(N, i))
        
        # Train GBC model
        gbc_model = GradientBoostingClassifier(validation_fraction = 0.25,
                                               n_iter_no_change = 10)
        gbc_model.fit(X[:N], y[:N])
        dump(gbc_model, gbc_filestr.format(N, i))
    print()
    
# Experiment parameters
num = 0
reps = 100
d = 32
Ns = 10**np.arange(2, 8)

# Model parameters
bce_params = {'loss':bce, 'd': d}

filestr = 'models/bdts/{}/set_{}/'.format(d, num)
bce_filestr = filestr + 'bce/model_{}_{}.h5'
bdt_filestr = filestr + 'bdt/model_{}.h5'
gbc_filestr = filestr + 'gbc/model_{}_{}.h5'

if not os.path.isdir(filestr):
    os.mkdir(filestr)

if not os.path.isdir(filestr + 'bce/'):
    os.mkdir(filestr + 'bce/')
    
if not os.path.isdir(filestr + 'bdt/'):
    os.mkdir(filestr + 'bdt/')
    
if not os.path.isdir(filestr + 'gbc/'):
    os.mkdir(filestr + 'gbc/')

# Data parameters
X = np.load('data/bdts/{}/X_trn.npy'.format(d))
y = np.load('data/bdts/{}/y_trn.npy'.format(d)).astype('float32')

for N in Ns:
    print('===================================================\n{}'.format(N))
    # Take the first N samples.
    data, m, s = split_data(X[:N], y[:N])
    
    # Train BDT model (only need to train 1)
    bdt_model = XGBClassifier(early_stopping_rounds = 10)
    X_trn, X_vld, y_trn, y_vld = data
    bdt_model.fit(X_trn, y_trn, eval_set = [(X_vld, y_vld)], verbose = 0)
    trace = bdt_model.evals_result()['validation_0']
    print(trace['logloss'][-1], '\t', len(trace['logloss']), end = '\n')
    bdt_model.save_model(bdt_filestr.format(N))

    for i in range(reps):
        print(i, end = ' ')
        # Train BCE model
        bce_model, trace = train(data, **bce_params)
        bce_model.save_weights(bce_filestr.format(N, i))
        
        # Train GBC model
        gbc_model = GradientBoostingClassifier(validation_fraction = 0.25,
                                               n_iter_no_change = 10)
        gbc_model.fit(X[:N], y[:N])
        dump(gbc_model, gbc_filestr.format(N, i))
    print()