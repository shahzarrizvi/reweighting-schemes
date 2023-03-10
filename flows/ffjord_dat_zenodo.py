import tensorflow as tf
from scipy import stats
import numpy as np
import matplotlib.pyplot as plt

from flows import *

import os

os.environ["CUDA_VISIBLE_DEVICES"] = "0" # pick a number < 4 on ML4HEP; < 3 on Voltan 
physical_devices = tf.config.list_physical_devices('GPU') 
tf.config.experimental.set_memory_growth(physical_devices[0], True)

data = np.load("../data/zenodo/Herwig_Zjet_pTZ-200GeV_0.npz")

dat_pt =  data['sim_jets'][:, 0]
dat_eta = data['sim_jets'][:, 1]
dat_phi = data['sim_jets'][:, 2]
dat_m =   data['sim_jets'][:, 3]
dat_mults = data['sim_mults']
dat_lhas = data['sim_lhas']
dat_widths = data['sim_widths']
dat_ang2s = data['sim_ang2s']
dat_tau2s = data['sim_tau2s']
dat_sdms = data['sim_sdms']
dat_zgs = data['sim_zgs']
dat = np.vstack([dat_pt, dat_eta, dat_phi, dat_m, dat_mults, dat_lhas, dat_widths, dat_ang2s]).T

dat_target = flow(dat, 
                  ckpt_path = 'zenodo/dat8/ckpt', 
                  batch_size = 2**7, 
                  num_epochs = 400, 
                  lr = 1e-3)