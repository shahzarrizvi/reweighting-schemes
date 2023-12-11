# Pick GPU.
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

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
mc = np.load("../data/zenodo/Pythia21_Zjet_pTZ-200GeV_0.npz")

mc_pt =  mc['sim_jets'][:, 0] 
mc_eta = mc['sim_jets'][:, 1]
mc_phi = mc['sim_jets'][:, 2]
mc_m =   mc['sim_jets'][:, 3]
mc_w = mc['sim_widths']
mc_sdms = mc['sim_sdms']
mc_mults = mc['sim_mults']
mc_tau21 = mc['sim_tau2s'] / (mc_w + 10**-50)
mc_jm = np.log(mc_sdms**2 / mc_pt**2 + 10**-100)

jm_mask = mc_jm > -20
mc_pt = mc_pt[jm_mask]
mc_eta = mc_eta[jm_mask]
mc_phi = mc_phi[jm_mask]
mc_m = mc_m[jm_mask]
mc_w = mc_w[jm_mask]
mc_sdms = mc_sdms[jm_mask]
mc_mults = mc_mults[jm_mask]
mc_tau21 = mc_tau21[jm_mask]
mc_jm = mc_jm[jm_mask]

mc = np.vstack([mc_m, mc_mults, mc_w, mc_jm, mc_tau21, mc_pt]).T
n, d = mc.shape

data = torch.tensor(mc, dtype = torch.float32).cuda()
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
                                                          hidden_features=8))
transform = CompositeTransform(transforms)

flow = Flow(transform, base_dist)
flow.to(device)
optimizer = optim.Adam(flow.parameters())

# Reset old checkpoint
ckpt = torch.load('nflows/sim/8/ckpt_3460000')
flow.load_state_dict(ckpt['model_state_dict'])
optimizer.load_state_dict(ckpt['optimizer_state_dict'])
loss = ckpt['loss']
flow.train()


# Train flow.
trn_dir = 'nflows/sim/8/'
num_iter = 10000000 # Use dataset 100,000 times.
losses = np.zeros(num_iter)
losses[:3460000] = np.load(trn_dir + 'losses.npy')[:3460000]

best = np.inf * np.ones(100)
print(-flow.log_prob(inputs=data).mean())
for i in tqdm.trange(3460001, num_iter):
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
        idx = np.argmax(best)
        make_checkpoint(flow, optimizer, loss, trn_dir + 'best/ckpt_{}'.format(idx))
        best[idx] = losses[i]
        
make_checkpoint(flow, optimizer, loss, trn_dir + 'ckpt_{}'.format(num_iter))
np.save(trn_dir + 'losses.npy', losses)