import tensorflow as tf
from scipy import stats
import numpy as np
import sklearn.metrics as metrics
from sklearn.ensemble import AdaBoostClassifier
from sklearn.model_selection import train_test_split

from matplotlib import pyplot as plt
from matplotlib.patches import Patch
from matplotlib import rc
import matplotlib.font_manager
rc('font', family='serif')
rc('text', usetex=True)
rc('font', size=10)        #22
rc('xtick', labelsize=8)  #15
rc('ytick', labelsize=8)  #15
rc('legend', fontsize=8)  #15
rc('text.latex', preamble=r'\usepackage{amsmath}')

# Plots have dimension (w,h)
w = 3.5
h = 3.25       

from flows import *

import sys
import os

sys.path.append('../')
from utils.training import *
from utils.losses import *

np.random.seed(666)

os.environ["CUDA_VISIBLE_DEVICES"] = "3" # pick a number < 4 on ML4HEP; < 3 on Voltan 
physical_devices = tf.config.list_physical_devices('GPU') 
tf.config.experimental.set_memory_growth(physical_devices[0], True)

dat = np.load("../data/zenodo/Herwig_Zjet_pTZ-200GeV_0.npz")

dat_pt =  dat['sim_jets'][:, 0] # dat[:, 0]
dat_eta = dat['sim_jets'][:, 1]
dat_phi = dat['sim_jets'][:, 2]
dat_m =   dat['sim_jets'][:, 3]
dat_w = dat['sim_widths']
dat_sdms = dat['sim_sdms']

dat = np.vstack([dat_pt, dat_eta, dat_phi, dat_m, dat_w, dat_sdms]).T

n, d = dat.shape

def calculate_auc(fake, real):
    np.random.seed(666)
    n, d = fake.shape
    y = stats.bernoulli.rvs(0.5, size = n).astype('float32')
    X = np.zeros_like(fake)
    X[y == 0] = real[y == 0]
    X[y == 1] = fake[y == 1]

    # Take 70% of data to be training data.
    X_trn, X_tst, y_trn, y_tst = train_test_split(X, y, train_size = 0.7)
    ab_clf = AdaBoostClassifier()
    ab_clf.fit(X_trn, y_trn)
    
    y_hat = ab_clf.predict_proba(X_tst)[:, 1]
    auc = metrics.roc_auc_score(y_tst, y_hat)
    return auc

start = 70001
end = 83000
ns = np.arange(start, end + 1)

aucs = np.zeros(end)
aucs[:start - 1] = np.load('dat_aucs.npy')[:start - 1]

for i in ns:
    dat_target = make_target(d)
    dat_ckpt = tf.train.Checkpoint(dat_target)
    dat_ckpt.restore('dat/full/ckpt-{}'.format(i))
    
    dat_smp = dat_target.sample(n)
    aucs[i - 1] = calculate_auc(dat_smp, dat)
    if i % 100 == 0:
        print(i, '\t', round(aucs[i - 1], 4))
        np.save('dat_aucs.npy', aucs)