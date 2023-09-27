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

mc = np.load("../data/zenodo/Pythia21_Zjet_pTZ-200GeV_0.npz")

sim_pt =  mc['sim_jets'][:, 0]
sim_eta = mc['sim_jets'][:, 1]
sim_phi = mc['sim_jets'][:, 2]
sim_m =   mc['sim_jets'][:, 3]
sim = np.vstack([sim_pt, sim_eta, sim_phi, sim_m]).T

sim_target = distributed_flow(sim, 
                  ckpt_path = 'sim7/ckpt', 
                  batch_size = 2**10, 
                  num_epochs = 4000, 
                  lr = 1e-3)