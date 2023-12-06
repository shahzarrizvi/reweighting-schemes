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

mc = np.vstack([mc_pt, mc_eta, mc_m, mc_w, mc_sdms, mc_tau21]).T
n, d = mc.shape

data = torch.tensor(mc, dtype = torch.float32)
dataset = DataLoader(data, batch_size = 2**6, shuffle = True)

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
optimizer = optim.Adam(flow.parameters())

# Reset old checkpoint
ckpt = torch.load('flows/sim/7/ckpt_100000')
flow.load_state_dict(ckpt['model_state_dict'])
optimizer.load_state_dict(ckpt['optimizer_state_dict'])
loss = ckpt['loss']
flow.train()

# Train flow.
trn_dir = 'flows/sim/7/'
num_iter = 300000 # Use dataset 100,000 times.
losses = np.zeros(num_iter)
losses[:200000] = np.load(trn_dir + 'losses.npy')[:200000]

best_loss = min(losses[losses > 0])
print(-flow.log_prob(inputs=mc.astype('float32')).mean())
for i in tqdm.trange(200001, num_iter):
    optimizer.zero_grad()
    loss = -flow.log_prob(inputs=data).mean()
    losses[i] = loss
    
    loss.backward()
    optimizer.step()
    if i % 100 == 0:
        make_checkpoint(flow, optimizer, loss, trn_dir + 'ckpt_{}'.format(i))
        np.save(trn_dir + 'losses.npy', losses)
        
    if losses[i] < best_loss:
        make_checkpoint(flow, optimizer, loss, trn_dir + 'ckpt_best')
        best_loss = losses[i]
        
make_checkpoint(flow, optimizer, loss, trn_dir + 'ckpt_{}'.format(num_iter))
np.save(trn_dir + 'losses.npy', losses)