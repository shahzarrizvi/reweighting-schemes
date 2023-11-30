# Pick GPU.
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "3, 4"

import torch
from torch import nn
from torch import optim
from torch.utils.data import DataLoader

print(torch.cuda.is_available())
print(torch.cuda.device_count())
print(torch.cuda.current_device())
print(torch.cuda.device(0))
print(torch.cuda.get_device_name(0))