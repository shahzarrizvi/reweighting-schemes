import tensorflow as tf
from scipy import stats
import numpy as np
import matplotlib.pyplot as plt

from flows import *

import os

os.environ["CUDA_VISIBLE_DEVICES"] = "3" # pick a number < 4 on ML4HEP; < 3 on Voltan 
physical_devices = tf.config.list_physical_devices('GPU') 
tf.config.experimental.set_memory_growth(physical_devices[0], True)

data = np.load("../data/zenodo/Herwig_Zjet_pTZ-200GeV_0.npz")

dat_pt =  data['sim_jets'][:, 0]
dat_eta = data['sim_jets'][:, 1]
dat_phi = data['sim_jets'][:, 2]
dat_m =   data['sim_jets'][:, 3]
dat = np.vstack([dat_pt, dat_eta, dat_phi, dat_m]).T

dat_target = flow(dat, 
                  ckpt_path = 'dat7/ckpt', 
                  num_hidden = 64,
                  batch_size = 2**7, 
                  num_epochs = 600, 
                  lr = 1e-3)