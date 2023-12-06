# Pick GPU.
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "7"

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
dat = np.load("../data/zenodo/Herwig_Zjet_pTZ-200GeV_0.npz")

dat_pt =  dat['sim_jets'][:, 0] # dat[:, 0]
dat_eta = dat['sim_jets'][:, 1]
dat_phi = dat['sim_jets'][:, 2]
dat_m =   dat['sim_jets'][:, 3]
dat_w = dat['sim_widths']
dat_sdms = dat['sim_sdms']

dat = np.vstack([dat_pt, dat_eta, dat_phi, dat_m, dat_w, dat_sdms]).T
data = torch.tensor(dat, dtype = torch.float32)
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
base_dist = StandardNormal(shape=[6])

transforms = []
for _ in range(num_layers):
    transforms.append(ReversePermutation(features=6))
    transforms.append(MaskedAffineAutoregressiveTransform(features=6, 
                                                          hidden_features=8))
transform = CompositeTransform(transforms)

flow = Flow(transform, base_dist)
optimizer = optim.Adam(flow.parameters())

# Train flow.
num_iter = 100000 # Use dataset 100,000 times.
losses = np.zeros(num_iter)
best_loss = np.inf
for i in tqdm.trange(num_iter):
    optimizer.zero_grad()
    loss = -flow.log_prob(inputs=data).mean()
    losses[i] = loss
    
    loss.backward()
    optimizer.step()
    if i % 100 == 0:
        make_checkpoint(flow, optimizer, loss, 'nflows/total_2/ckpt_{}'.format(i))
        np.save('nflows/total_2/losses.npy', losses)
        
    if losses[i] < best_loss:
        make_checkpoint(flow, optimizer, loss, 'nflows/total_2/ckpt_best')
        best_loss = losses[i]
        
make_checkpoint(flow, optimizer, loss, 'nflows/total_2/ckpt_{}'.format(num_iter))
np.save('nflows/total_2/losses.npy', losses)