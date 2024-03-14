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
import pandas as pd
import seaborn as sns

# Utility imports
#import sys
#sys.path.append('../')
from utils.losses import *
from utils.plotting import *
from utils.training import *
from flows.flows import *

rc('font', size=15)        #22
rc('xtick', labelsize=10)  #15
rc('ytick', labelsize=10)  #15
rc('legend', fontsize=10)  #15
rc('text.latex', preamble=r'\usepackage{amsmath}')

w = 7
h = 3.5

np.random.seed(666) # Need to do more to ensure data is the same across runs.

os.environ["CUDA_VISIBLE_DEVICES"] = "0" # pick a number < 4 on ML4HEP; < 3 on Voltan 
physical_devices = tf.config.list_physical_devices('GPU') 
tf.config.experimental.set_memory_growth(physical_devices[0], True)

num = 3
reps = 100
Ns = 10**np.arange(2, 8)

mu = 0.1
bkgd = stats.norm(-mu, 1)
sgnl = stats.norm(mu, 1)

mae = make_mae(bkgd, sgnl, 'data/normal/0.1/')
lr = make_lr(bkgd, sgnl)

# Model parameters
p = 1.24
r = 0.018
bce_params = {'loss':bce}
mse_params = {'loss':mse}
pmse_params = {'loss':get_mse(p)}
mlc_params = {'loss':exp_mlc, 'output':'linear'}
sqr_params = {'loss':exp_sqr, 'output':'linear'}
rsqr_params = {'loss':get_exp_sqr(r), 'output':'linear'}

p_null = 1.08   #null
r_null = 0.042  #null
pmse_params = {'loss':get_mse(p_null)}
rsqr_params = {'loss':get_exp_sqr(r_null), 'output': 'linear'}

p_mr = 1.15   #null
r_mr = 0.1  #null
pmse_mr_params = {'loss':get_mse(p_mr)}
rsqr_mr_params = {'loss':get_exp_sqr(r_mr), 'output': 'linear'}

filestr = 'models/univariate/loss_comp/set_{}/'.format(num)
bce_filestr = filestr + 'bce/model_{}_{}.h5'
mse_filestr = filestr + 'mse/model_{}_{}.h5'
mlc_filestr = filestr + 'mlc/model_{}_{}.h5'
sqr_filestr = filestr + 'sqr/model_{}_{}.h5'

#pmse_filestr = filestr + 'pmse/model_{}_{}.h5'
#rsqr_filestr = filestr + 'rsqr/model_{}_{}.h5'
pmse_filestr = filestr + 'pmse_null/model_{}_{}.h5'
rsqr_filestr = filestr + 'rsqr_null/model_{}_{}.h5'

pmse_mr_filestr = filestr + 'pmse_mr/model_{}_{}.h5'
rsqr_mr_filestr = filestr + 'rsqr_mr/model_{}_{}.h5'

xs = np.load(filestr + 'xs.npy')
bce_preds = np.load(filestr + 'bce_preds.npy')
mse_preds = np.load(filestr + 'mse_preds.npy')
mlc_preds = np.load(filestr + 'mlc_preds.npy')
sqr_preds = np.load(filestr + 'sqr_preds.npy')
pmse_preds = np.load(filestr + 'pmse_preds.npy')
rsqr_preds = np.load(filestr + 'rsqr_preds.npy')

cs = ['dodgerblue', 'seagreen', 'crimson', 'darkorange', 'lightseagreen', 'orangered']
lss = [':', '--', ':', '--', '-.', '-.']
ratio_plot([bce_preds, mse_preds, mlc_preds, sqr_preds, pmse_preds, rsqr_preds],
           ['BCE', 'MSE', 'MLC', 'SQR', r'$p^*$-MSE', r'$r^*$-SQR'],
           lr,
           xs,
           bkgd, sgnl,
           figsize = (w, h),
           y_lim = (0, 4),
           cs = cs,
           lss = lss,
           title = None,
           filename = 'poster/univariate_fits.png')