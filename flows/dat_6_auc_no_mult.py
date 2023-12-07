# Pick GPU.
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

import torch
from torch import nn
from torch import optim
from torch.utils.data import DataLoader

print(torch.cuda.is_available())
print(torch.cuda.device_count())
print(torch.cuda.current_device())
print(torch.cuda.device(0))
print(torch.cuda.get_device_name(0))
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)

from nflows.flows.base import Flow
from nflows.distributions.normal import StandardNormal
from nflows.transforms.base import CompositeTransform
from nflows.transforms.autoregressive import MaskedAffineAutoregressiveTransform
from nflows.transforms.permutations import ReversePermutation

from scipy import stats
import numpy as np
import sklearn.metrics as metrics
from sklearn.ensemble import AdaBoostClassifier
from sklearn.model_selection import train_test_split     

np.random.seed(666)

# Load in data
dt = np.load("../data/zenodo/Herwig_Zjet_pTZ-200GeV_0.npz")

dt_pt =  dt['sim_jets'][:, 0] 
dt_eta = dt['sim_jets'][:, 1]
dt_phi = dt['sim_jets'][:, 2]
dt_m =   dt['sim_jets'][:, 3]
dt_w = dt['sim_widths']
dt_sdms = dt['sim_sdms']
dt_mults = dt['sim_mults']
dt_tau21 = dt['sim_tau2s'] / (dt_w + 10**-50)
dt_jm = np.log(dt_sdms**2 / dt_pt**2 + 10**-100)

jm_mask = dt_jm > -20
dt_pt = dt_pt[jm_mask]
dt_eta = dt_eta[jm_mask]
dt_phi = dt_phi[jm_mask]
dt_m = dt_m[jm_mask]
dt_w = dt_w[jm_mask]
dt_sdms = dt_sdms[jm_mask]
dt_mults = dt_mults[jm_mask]
dt_tau21 = dt_tau21[jm_mask]
dt_jm = dt_jm[jm_mask]

dt = np.vstack([dt_m, dt_mults, dt_w, dt_jm, dt_tau21, dt_pt]).T
n, d = dt.shape

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


# Initialize flow
num_layers = 5
base_dist = StandardNormal(shape=[d])

transforms = []
for _ in range(num_layers):
    transforms.append(ReversePermutation(features=d))
    transforms.append(MaskedAffineAutoregressiveTransform(features=d, 
                                                          hidden_features=8))
transform = CompositeTransform(transforms)
flow = Flow(transform, base_dist)
flow.to(device)


# Calculate AUCs
aucs = np.zeros(100)
idx = [0, 2, 3, 4, 5]
for i in range(100):
    ckpt = torch.load('nflows/dat/6/best/ckpt_{}'.format(i))
    flow.load_state_dict(ckpt['model_state_dict'])
    flow.eval()
    
    smp = flow.sample(n).cpu().detach().numpy()
    aucs[i] = calculate_auc(smp[:, idx], dt[:, idx])
    print(aucs[i])
np.save('dat_6_aucs_no_mult.npy', aucs)