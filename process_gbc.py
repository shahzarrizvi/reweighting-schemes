# General imports
import os
import pickle
from scipy import stats
import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, Input, Dropout
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
from joblib import dump, load

# Utility imports
from utils.losses import *
from utils.plotting import *
from utils.training import *

np.random.seed(666) # Need to do more to ensure data is the same across runs.
w = 3.5
h = 3.25              # Plots have dimension (w,h)

os.environ["CUDA_VISIBLE_DEVICES"] = "2" # pick a number < 4 on ML4HEP; < 3 on Voltan 
physical_devices = tf.config.list_physical_devices('GPU') 
tf.config.experimental.set_memory_growth(physical_devices[0], True)

# Experiment parameters
num = 0
reps = 100
d = 1
Ns = 10**np.arange(2, 8)

# Model parameters
bce_params = {'loss':bce, 'd': d}

filestr = 'models/trees/{}/set_{}/'.format(d, num)
bce_filestr = filestr + 'bce/model_{}_{}.h5'
gbc_filestr = filestr + 'gbc/model_{}_{}.h5'
bdt_filestr = filestr + 'bdt/model_{}.h5'

# True distribution information
bkgd = stats.norm(-0.1, 1)
sgnl = stats.norm(+0.1, 1)

lr = make_lr(bkgd, sgnl)
mae = make_mae(bkgd, sgnl, 'data/trees/{}/'.format(d))

X = np.load('data/trees/{}/X_trn.npy'.format(d)).reshape(-1, 1)
y = np.load('data/trees/{}/y_trn.npy'.format(d))

# Calculate mean absolute errors
gbc_avg = np.array([])
for N in Ns:
    print(N, end = '\t')
    data, m, s = split_data(X[:N], y[:N])
    gbc_lrs = [None] * reps
    for i in range(reps):
        gbc_model = load(gbc_filestr.format(N, i))
        gbc_lrs[i] = tree_lr(gbc_model)
    gbc_maes = [mae(lr) for lr in gbc_lrs]
    gbc_avg = np.append(gbc_avg, np.mean(gbc_maes))
    np.save(filestr + 'gbc_avg', gbc_avg)
    print(gbc_maes[-1], end = '\t')

bce_avg = np.load(filestr + 'bce_avg')
bdt_avg = np.load(filestr + 'bdt_avg')

mae_plot([bce_avg, gbc_avg, bdt_avg],
         ['BCE', 'GBC', 'BDT'],
         Ns,
         figsize = (w, h),
         title = r'\it $d = {}$'.format(d),
         filename = 'plots/paper/gbc_d{}.png'.format(d))

# Experiment parameters
num = 0
reps = 100
d = 2
Ns = 10**np.arange(2, 8)

# Model parameters
bce_params = {'loss':bce, 'd': d}

filestr = 'models/trees/{}/set_{}/'.format(d, num)
bce_filestr = filestr + 'bce/model_{}_{}.h5'
gbc_filestr = filestr + 'gbc/model_{}_{}.h5'
bdt_filestr = filestr + 'bdt/model_{}.h5'

# True distribution information
mu_bkgd = np.array([-0.1] + [0]*(d - 1))
mu_sgnl = np.array([+0.1] + [0]*(d - 1))
sg_bkgd = np.eye(d)
sg_sgnl = np.eye(d)

bkgd = stats.multivariate_normal(mu_bkgd, sg_bkgd)
sgnl = stats.multivariate_normal(mu_sgnl, sg_sgnl)

lr = make_lr(bkgd, sgnl)
mae = make_mae(bkgd, sgnl, 'data/trees/{}/'.format(d))

X = np.load('data/trees/{}/X_trn.npy'.format(d))
y = np.load('data/trees/{}/y_trn.npy'.format(d))

# Calculate mean absolute errors
gbc_avg = np.array([])
for N in Ns:
    print(N, end = '\t')
    data, m, s = split_data(X[:N], y[:N])
    gbc_lrs = [None] * reps
    for i in range(reps):
        gbc_model = load(gbc_filestr.format(N, i))
        gbc_lrs[i] = tree_lr(gbc_model)
    gbc_maes = [mae(lr) for lr in gbc_lrs]
    gbc_avg = np.append(gbc_avg, np.mean(gbc_maes))
    np.save(filestr + 'gbc_avg', gbc_avg)
    print(gbc_maes[-1], end = '\t')

bce_avg = np.load(filestr + 'bce_avg')
bdt_avg = np.load(filestr + 'bdt_avg')

mae_plot([bce_avg, gbc_avg, bdt_avg],
         ['BCE', 'GBC', 'BDT'],
         Ns,
         figsize = (w, h),
         title = r'\it $d = {}$'.format(d),
         filename = 'plots/paper/gbc_d{}.png'.format(d))

# Experiment parameters
num = 0
reps = 100
d = 4
Ns = 10**np.arange(2, 8)

# Model parameters
bce_params = {'loss':bce, 'd': d}

filestr = 'models/trees/{}/set_{}/'.format(d, num)
bce_filestr = filestr + 'bce/model_{}_{}.h5'
gbc_filestr = filestr + 'gbc/model_{}_{}.h5'
bdt_filestr = filestr + 'bdt/model_{}.h5'

# True distribution information
mu_bkgd = np.array([-0.1] + [0]*(d - 1))
mu_sgnl = np.array([+0.1] + [0]*(d - 1))
sg_bkgd = np.eye(d)
sg_sgnl = np.eye(d)

bkgd = stats.multivariate_normal(mu_bkgd, sg_bkgd)
sgnl = stats.multivariate_normal(mu_sgnl, sg_sgnl)

lr = make_lr(bkgd, sgnl)
mae = make_mae(bkgd, sgnl, 'data/trees/{}/'.format(d))

X = np.load('data/trees/{}/X_trn.npy'.format(d))
y = np.load('data/trees/{}/y_trn.npy'.format(d))

# Calculate mean absolute errors
gbc_avg = np.array([])
for N in Ns:
    print(N, end = '\t')
    data, m, s = split_data(X[:N], y[:N])
    gbc_lrs = [None] * reps
    for i in range(reps):
        gbc_model = load(gbc_filestr.format(N, i))
        gbc_lrs[i] = tree_lr(gbc_model)
    gbc_maes = [mae(lr) for lr in gbc_lrs]
    gbc_avg = np.append(gbc_avg, np.mean(gbc_maes))
    np.save(filestr + 'gbc_avg', gbc_avg)
    print(gbc_maes[-1], end = '\t')

bce_avg = np.load(filestr + 'bce_avg')
bdt_avg = np.load(filestr + 'bdt_avg')

mae_plot([bce_avg, gbc_avg, bdt_avg],
         ['BCE', 'GBC', 'BDT'],
         Ns,
         figsize = (w, h),
         title = r'\it $d = {}$'.format(d),
         filename = 'plots/paper/gbc_d{}.png'.format(d))

# Experiment parameters
num = 0
reps = 100
d = 8
Ns = 10**np.arange(2, 8)

# Model parameters
bce_params = {'loss':bce, 'd': d}

filestr = 'models/trees/{}/set_{}/'.format(d, num)
bce_filestr = filestr + 'bce/model_{}_{}.h5'
gbc_filestr = filestr + 'gbc/model_{}_{}.h5'
bdt_filestr = filestr + 'bdt/model_{}.h5'

# True distribution information
mu_bkgd = np.array([-0.1] + [0]*(d - 1))
mu_sgnl = np.array([+0.1] + [0]*(d - 1))
sg_bkgd = np.eye(d)
sg_sgnl = np.eye(d)

bkgd = stats.multivariate_normal(mu_bkgd, sg_bkgd)
sgnl = stats.multivariate_normal(mu_sgnl, sg_sgnl)

lr = make_lr(bkgd, sgnl)
mae = make_mae(bkgd, sgnl, 'data/trees/{}/'.format(d))

X = np.load('data/trees/{}/X_trn.npy'.format(d))
y = np.load('data/trees/{}/y_trn.npy'.format(d))

# Calculate mean absolute errors
gbc_avg = np.array([])
for N in Ns:
    print(N, end = '\t')
    data, m, s = split_data(X[:N], y[:N])
    gbc_lrs = [None] * reps
    for i in range(reps):
        gbc_model = load(gbc_filestr.format(N, i))
        gbc_lrs[i] = tree_lr(gbc_model)
    gbc_maes = [mae(lr) for lr in gbc_lrs]
    gbc_avg = np.append(gbc_avg, np.mean(gbc_maes))
    np.save(filestr + 'gbc_avg', gbc_avg)
    print(gbc_maes[-1], end = '\t')

bce_avg = np.load(filestr + 'bce_avg')
bdt_avg = np.load(filestr + 'bdt_avg')

mae_plot([bce_avg, gbc_avg, bdt_avg],
         ['BCE', 'GBC', 'BDT'],
         Ns,
         figsize = (w, h),
         title = r'\it $d = {}$'.format(d),
         filename = 'plots/paper/gbc_d{}.png'.format(d))

# Experiment parameters
num = 0
reps = 100
d = 16
Ns = 10**np.arange(2, 8)

# Model parameters
bce_params = {'loss':bce, 'd': d}

filestr = 'models/trees/{}/set_{}/'.format(d, num)
bce_filestr = filestr + 'bce/model_{}_{}.h5'
gbc_filestr = filestr + 'gbc/model_{}_{}.h5'
bdt_filestr = filestr + 'bdt/model_{}.h5'

# True distribution information
mu_bkgd = np.array([-0.1] + [0]*(d - 1))
mu_sgnl = np.array([+0.1] + [0]*(d - 1))
sg_bkgd = np.eye(d)
sg_sgnl = np.eye(d)

bkgd = stats.multivariate_normal(mu_bkgd, sg_bkgd)
sgnl = stats.multivariate_normal(mu_sgnl, sg_sgnl)

lr = make_lr(bkgd, sgnl)
mae = make_mae(bkgd, sgnl, 'data/trees/{}/'.format(d))

X = np.load('data/trees/{}/X_trn.npy'.format(d))
y = np.load('data/trees/{}/y_trn.npy'.format(d))

# Calculate mean absolute errors
gbc_avg = np.array([])
for N in Ns:
    print(N, end = '\t')
    data, m, s = split_data(X[:N], y[:N])
    gbc_lrs = [None] * reps
    for i in range(reps):
        gbc_model = load(gbc_filestr.format(N, i))
        gbc_lrs[i] = tree_lr(gbc_model)
    gbc_maes = [mae(lr) for lr in gbc_lrs]
    gbc_avg = np.append(gbc_avg, np.mean(gbc_maes))
    np.save(filestr + 'gbc_avg', gbc_avg)
    print(gbc_maes[-1], end = '\t')

bce_avg = np.load(filestr + 'bce_avg')
bdt_avg = np.load(filestr + 'bdt_avg')

mae_plot([bce_avg, gbc_avg, bdt_avg],
         ['BCE', 'GBC', 'BDT'],
         Ns,
         figsize = (w, h),
         title = r'\it $d = {}$'.format(d),
         filename = 'plots/paper/gbc_d{}.png'.format(d))

# Experiment parameters
num = 0
reps = 100
d = 32
Ns = 10**np.arange(2, 8)

# Model parameters
bce_params = {'loss':bce, 'd': d}

filestr = 'models/trees/{}/set_{}/'.format(d, num)
bce_filestr = filestr + 'bce/model_{}_{}.h5'
gbc_filestr = filestr + 'gbc/model_{}_{}.h5'
bdt_filestr = filestr + 'bdt/model_{}.h5'

# True distribution information
mu_bkgd = np.array([-0.1] + [0]*(d - 1))
mu_sgnl = np.array([+0.1] + [0]*(d - 1))
sg_bkgd = np.eye(d)
sg_sgnl = np.eye(d)

bkgd = stats.multivariate_normal(mu_bkgd, sg_bkgd)
sgnl = stats.multivariate_normal(mu_sgnl, sg_sgnl)

lr = make_lr(bkgd, sgnl)
mae = make_mae(bkgd, sgnl, 'data/trees/{}/'.format(d))

X = np.load('data/trees/{}/X_trn.npy'.format(d))
y = np.load('data/trees/{}/y_trn.npy'.format(d))

# Calculate mean absolute errors
gbc_avg = np.array([])
for N in Ns:
    print(N, end = '\t')
    data, m, s = split_data(X[:N], y[:N])
    gbc_lrs = [None] * reps
    for i in range(reps):
        gbc_model = load(gbc_filestr.format(N, i))
        gbc_lrs[i] = tree_lr(gbc_model)
    gbc_maes = [mae(lr) for lr in gbc_lrs]
    gbc_avg = np.append(gbc_avg, np.mean(gbc_maes))
    np.save(filestr + 'gbc_avg', gbc_avg)
    print(gbc_maes[-1], end = '\t')

bce_avg = np.load(filestr + 'bce_avg')
bdt_avg = np.load(filestr + 'bdt_avg')

mae_plot([bce_avg, gbc_avg, bdt_avg],
         ['BCE', 'GBC', 'BDT'],
         Ns,
         figsize = (w, h),
         title = r'\it $d = {}$'.format(d),
         filename = 'plots/paper/gbc_d{}.png'.format(d))


