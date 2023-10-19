import tensorflow as tf
from scipy import stats
import numpy as np
import matplotlib.pyplot as plt

from flows import *

import os

os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3" # pick a number < 4 on ML4HEP; < 3 on Voltan 
physical_devices = tf.config.list_physical_devices('GPU') 
for gpu in physical_devices:
    tf.config.experimental.set_memory_growth(gpu, True)

data = np.load("../data/zenodo/Herwig_Zjet_pTZ-200GeV_0.npz")

dat_pt =  data['sim_jets'][:, 0]
dat_eta = data['sim_jets'][:, 1]
dat_phi = data['sim_jets'][:, 2]
dat_m =   data['sim_jets'][:, 3]
dat = np.vstack([dat_pt, dat_eta, dat_phi, dat_m]).T

dat_target = distributed_flow(dat, 
                              ckpt_path = 'dat/2/ckpt',
                              batch_size = 2**10, 
                              num_epochs = 2000, 
                              lr = 1e-3)