import tensorflow as tf
from scipy import stats
import numpy as np
import matplotlib.pyplot as plt

from flows import *

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1" # pick a number < 4 on ML4HEP; < 3 on Voltan 
physical_devices = tf.config.list_physical_devices('GPU') 
tf.config.experimental.set_memory_growth(physical_devices[0], True)

mc = np.load("../data/zenodo/Pythia21_Zjet_pTZ-200GeV_0.npz")

sim_pt =   mc['sim_jets'][:, 0]
sim_eta =  mc['sim_jets'][:, 1]
sim_phi =  mc['sim_jets'][:, 2]
sim_m =    mc['sim_jets'][:, 3]
sim_w =    mc['sim_widths']
sim_sdms = mc['sim_sdms']
sim = np.vstack([sim_pt, sim_eta, sim_phi, sim_m, sim_w, sim_sdms]).T

sim_target = flow(sim, 
                  exact = False,
                  ckpt_path = 'sim/full_2/ckpt', 
                  batch_size = 2**7, 
                  num_epochs = 4000, 
                  lr = 1e-3)