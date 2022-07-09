import numpy as np
from scipy import stats
import scipy.integrate as integrate
import os
import tensorflow as tf

np.random.seed(666)
eps = 1e-7

def sig(x):
    return 1 / (1 + np.exp(-x))

def relu(x):
    return np.maximum(0, x)

bkgd = stats.norm(-0.1, 1)
sgnl = stats.norm(+0.1, 1)

# Empiric estimation of losses
N = 5 * 10**3
X = np.concatenate((bkgd.rvs(size = N), sgnl.rvs(size = N)))
y = np.concatenate((np.zeros(N), np.ones(N)))

def bce(a, b):
    XX = np.resize(X, X.shape + np.shape(a))
    yy = np.resize(y, y.shape + np.shape(a))
    y_hat = sig(a*XX + b)
    return 2 * -np.mean((yy) * np.log(y_hat + eps) + (1 - yy) * np.log(1 - y_hat + eps), axis = 0)

def mse(a, b):
    XX = np.resize(X, X.shape + np.shape(a))
    yy = np.resize(y, y.shape + np.shape(a))
    y_hat = sig(a*XX + b)
    return 2 * -np.mean((yy) * -(1 - y_hat)**2 + (1 - yy) * -(y_hat)**2, axis = 0)

def mse_p(a, b, p):
    XX = np.resize(X, X.shape + np.shape(a))
    yy = np.resize(y, y.shape + np.shape(a))
    y_hat = sig(a*XX + b)
    return 2 * -np.mean((yy) * -(1 - y_hat)**p + (1 - yy) * -(y_hat)**p, axis = 0)

def mlc(a, b):
    XX = np.resize(X, X.shape + np.shape(a))
    yy = np.resize(y, y.shape + np.shape(a))
    y_hat = relu(a*XX + b)
    return 2 * -np.mean((yy) * np.log(y_hat + eps) + (1 - yy) * (1 - y_hat), axis = 0)

def exp_mlc(a, b):
    XX = np.resize(X, X.shape + np.shape(a))
    yy = np.resize(y, y.shape + np.shape(a))
    y_hat = relu(a*XX + b)
    return 2 * -np.mean((yy) * y_hat + (1 - yy) * (1 - np.exp(y_hat)), axis = 0)

def sqr(a, b):
    XX = np.resize(X, X.shape + np.shape(a))
    yy = np.resize(y, y.shape + np.shape(a))
    y_hat = relu(a*XX + b)
    return 2 * -np.mean((yy) * -(1 / (y_hat + eps)**0.5) + (1 - yy) * -(y_hat)**0.5, axis = 0)

def exp_sqr(a, b):
    XX = np.resize(X, X.shape + np.shape(a))
    yy = np.resize(y, y.shape + np.shape(a))
    y_hat = relu(a*XX + b)
    return 2 * -np.mean((yy) * -(1 / np.exp(y_hat)**0.5) + (1 - yy) * -(np.exp(y_hat))**0.5, axis = 0)

def sqr_r(a, b, r):
    XX = np.resize(X, X.shape + np.shape(a))
    yy = np.resize(y, y.shape + np.shape(a))
    y_hat = relu(a*XX + b)
    return 2 * -np.mean((yy) * -(1 / (y_hat + eps)**(r/2)) + (1 - yy) * -(y_hat)**(r/2), axis = 0) 

# Numerical estimation of losses
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

aa, bb = np.meshgrid(np.linspace(-3, 3, 601), np.linspace(-3, 3, 601))

#print('Empirical')
#bce_zs = bce(aa, bb)
#print('BCE', end = ' ')
#mse_zs = mse(aa, bb)
#print('MSE', end = ' ')
#mlc_zs = mlc(aa, bb)
#print('MLC', end = ' ')
#sqr_zs = sqr(aa, bb)
#print('SQR', end = '\n\n')

print('Numerical')
nbces = nbce(aa, bb)
print('NBCE', end = ' ')
nmses = nmse(aa, bb)
print('NMSE', end = ' ')
nmlcs = nmlc(aa, bb)
print('NMLC', end = ' ')
nsqrs = nsqr(aa, bb)
print('NSQR', end = '\n\n')

#np.save('anims/bce_zs_3.npy', bce_zs)
#np.save('anims/mse_zs_3.npy', mse_zs)
#np.save('anims/mlc_zs_3.npy', mlc_zs)
#np.save('anims/sqr_zs_3.npy', sqr_zs)

np.save('anims/nbces.npy', nbces)
np.save('anims/nmses.npy', nmses)
np.save('anims/nmlcs.npy', nmlcs)
np.save('anims/nsqrs.npy', nsqrs)

#print('Empirical Estimation')
#print(aa[bce_zs == np.min(bce_zs)], bb[bce_zs == np.min(bce_zs)])
#print(aa[mse_zs == np.min(mse_zs)], bb[mse_zs == np.min(mse_zs)])
#print(aa[mlc_zs == np.min(mlc_zs)], bb[mlc_zs == np.min(mlc_zs)])
#print(aa[sqr_zs == np.min(sqr_zs)], bb[sqr_zs == np.min(sqr_zs)])
#print()
print('Numerical Estimation')
print(aa[nbces == np.min(nbces)], bb[nbces == np.min(nbces)])
print(aa[nmses == np.min(nmses)], bb[nmses == np.min(nmses)])
print(aa[nmlcs == np.min(nmlcs)], bb[nmlcs == np.min(nmlcs)])
print(aa[nsqrs == np.min(nsqrs)], bb[nsqrs == np.min(nsqrs)])