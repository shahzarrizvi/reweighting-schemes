# Pick GPU.
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "5"

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

import tqdm as tqdm
import numpy as np

# Load in data.
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

data = torch.tensor(dt, dtype = torch.float32).cuda()
data.to(device)
#dataset = DataLoader(data, batch_size = 2**6, shuffle = True)

# Checkpointing methods
def make_checkpoint(flow, optimizer, loss, filename):
    torch.save({'model_state_dict': flow.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': loss,
               }, 
               filename)


# Initialize flow.
num_layers = 5
base_dist = StandardNormal(shape=[d])

transforms = []
for _ in range(num_layers):
    transforms.append(ReversePermutation(features=d))
    transforms.append(MaskedAffineAutoregressiveTransform(features=d, 
                                                          hidden_features=16))
transform = CompositeTransform(transforms)

flow = Flow(transform, base_dist)
flow.to(device)
optimizer = optim.Adam(flow.parameters())


# Reset old checkpoint
#ckpt = torch.load('nflows/dat/6/ckpt_3000000')
#flow.load_state_dict(ckpt['model_state_dict'])
#optimizer.load_state_dict(ckpt['optimizer_state_dict'])
#loss = ckpt['loss']
#flow.train()


# Train flow.
trn_dir = 'nflows/dat/8/'
num_iter = 5000000 # Use dataset 100,000 times.
losses = np.zeros(num_iter)
#losses[:3000000] = np.load(trn_dir + 'losses.npy')[:3000000]

best = np.inf * np.ones(100)
print(-flow.log_prob(inputs=data).mean())
for i in tqdm.trange(num_iter):
    optimizer.zero_grad()
    loss = -flow.log_prob(inputs=data).mean()
    losses[i] = loss
    
    loss.backward()
    optimizer.step()
    if i % 1000 == 0:
        np.save(trn_dir + 'losses.npy', losses)
        if i % 10000 == 0:
            make_checkpoint(flow, optimizer, loss, trn_dir + 'ckpt_{}'.format(i))
        
    if losses[i] < max(best):
        idx = np.argmax(best) # Get the index of the maximum best loss
        make_checkpoint(flow, optimizer, loss, trn_dir + 'best/ckpt_{}'.format(idx))
        best[idx] = losses[i]
        
make_checkpoint(flow, optimizer, loss, trn_dir + 'ckpt_{}'.format(num_iter))
np.save(trn_dir + 'losses.npy', losses)