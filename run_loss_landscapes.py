import numpy as np
from scipy import stats
import scipy.integrate as integrate
import os
import tensorflow as tf

np.random.seed(666)
eps = 1e-7

os.environ["CUDA_VISIBLE_DEVICES"] = "1" # pick a number < 4 on ML4HEP; < 3 on Voltan 
physical_devices = tf.config.list_physical_devices('GPU') 
tf.config.experimental.set_memory_growth(physical_devices[0], True)

def sig(x):
    return 1 / (1 + np.exp(-x))

def relu(x):
    return np.maximum(0, x)

# Numerical estimation of losses
bkgd = stats.norm(-0.1, 1)
sgnl = stats.norm(+0.1, 1)

def nbce(a, b):
    g = lambda x: -(sgnl.pdf(x) * np.log(sig(a*x + b) + eps) + \
                    bkgd.pdf(x) * np.log(1 - sig(a*x + b) + eps) )
    return integrate.quad_vec(g, -np.inf, np.inf)[0]

def nmse(a, b):
    g = lambda x: -(sgnl.pdf(x) * -(1 - sig(a*x + b))**2 + \
                    bkgd.pdf(x) * -(sig(a*x + b)**2) )
    return integrate.quad_vec(g, -np.inf, np.inf)[0]

def nmlc(a, b):
    g = lambda x: -(sgnl.pdf(x) * np.log(relu(a*x + b) + eps) + \
                    bkgd.pdf(x) * (1 - relu(a*x + b)) )
    return integrate.quad_vec(g, -np.inf, np.inf)[0]

def nsqr(a, b):
    g = lambda x: -(sgnl.pdf(x) * -1 / (relu(a*x + b) + eps)**0.5 + \
                    bkgd.pdf(x) * -relu(a*x + b)**0.5 )
    return integrate.quad_vec(g, -np.inf, np.inf)[0]

aa, bb = np.meshgrid(np.linspace(-1.5, 1.5, 301), np.linspace(-1.5, 1.5, 301))

nbce_zs = nbce(aa, bb)
nmse_zs = nmse(aa, bb)
nmlc_zs = nmlc(aa, bb)
nsqr_zs = nsqr(aa, bb)

np.save('anims/nbce_zs_2', nbce_zs)
np.save('anims/nmse_zs_2', nmse_zs)
np.save('anims/nmlc_zs_2', nmlc_zs)
np.save('anims/nsqr_zs_2', nsqr_zs)

print('Numerical Estimation')
print(g[nbce_zs == min(nbce_zs)])
print(g[nmse_zs == min(nmse_zs)])
print(g[nmlc_zs == min(nmlc_zs)])
print(g[nsqr_zs == min(nsqr_zs)])