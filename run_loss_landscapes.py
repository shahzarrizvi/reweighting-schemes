import numpy as np
import os
from scipy import stats
import scipy.integrate as integrate
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

bkgd = stats.norm(-0.1, 1)
sgnl = stats.norm(+0.1, 1)

# Empiric estimation of losses

N = 5 * 10**5
X = np.concatenate((bkgd.rvs(size = N), sgnl.rvs(size = N)))
y = np.concatenate((np.zeros(N), np.ones(N)))

def bce(a, b):
    XX = np.repeat(X, np.prod(np.shape(a))).reshape(X.shape + np.shape(a))
    yy = np.repeat(y, np.prod(np.shape(a))).reshape(y.shape + np.shape(a))
    y_hat = sig(a*XX + b)
    return 2 * -np.mean((yy) * np.log(y_hat + eps) + (1 - yy) * np.log(1 - y_hat + eps), axis = 0)

def mse(a, b):
    XX = np.repeat(X, np.prod(np.shape(a))).reshape(X.shape + np.shape(a))
    yy = np.repeat(y, np.prod(np.shape(a))).reshape(y.shape + np.shape(a))
    y_hat = sig(a*XX + b)
    return 2 * -np.mean((yy) * -(1 - y_hat)**2 + (1 - yy) * -(y_hat)**2, axis = 0)

def mse_p(a, b, p):
    XX = np.repeat(X, np.prod(np.shape(a))).reshape(X.shape + np.shape(a))
    yy = np.repeat(y, np.prod(np.shape(a))).reshape(y.shape + np.shape(a))
    y_hat = sig(a*XX + b) 
    return 2 * -np.mean((yy) * -(1 - y_hat)**p + (1 - yy) * -(y_hat)**p, axis = 0)

def mlc(a, b):
    XX = np.repeat(X, np.prod(np.shape(a))).reshape(X.shape + np.shape(a))
    yy = np.repeat(y, np.prod(np.shape(a))).reshape(y.shape + np.shape(a))
    y_hat = relu(a*XX + b)
    return 2 * -np.mean((yy) * np.log(y_hat + eps) + (1 - yy) * (1 - y_hat), axis = 0)

def exp_mlc(a, b):
    XX = np.repeat(X, np.prod(np.shape(a))).reshape(X.shape + np.shape(a))
    yy = np.repeat(y, np.prod(np.shape(a))).reshape(y.shape + np.shape(a))
    y_hat = relu(a*XX + b)
    return 2 * -np.mean((yy) * y_hat + (1 - yy) * (1 - np.exp(y_hat)), axis = 0)

def sqr(a, b):
    XX = np.repeat(X, np.prod(np.shape(a))).reshape(X.shape + np.shape(a))
    yy = np.repeat(y, np.prod(np.shape(a))).reshape(y.shape + np.shape(a))
    y_hat = relu(a*XX + b)
    return 2 * -np.mean((yy) * -(1 / (y_hat + eps)**0.5) + (1 - yy) * -(y_hat)**0.5, axis = 0)

def exp_sqr(a, b):
    XX = np.repeat(X, np.prod(np.shape(a))).reshape(X.shape + np.shape(a))
    yy = np.repeat(y, np.prod(np.shape(a))).reshape(y.shape + np.shape(a))
    y_hat = relu(a*XX + b)
    return 2 * -np.mean((yy) * -(1 / np.exp(y_hat)**0.5) + (1 - yy) * -(np.exp(y_hat))**0.5, axis = 0)

def sqr_r(a, b, r):
    XX = np.repeat(X, np.prod(np.shape(a))).reshape(X.shape + np.shape(a))
    yy = np.repeat(y, np.prod(np.shape(a))).reshape(y.shape + np.shape(a))
    y_hat = relu(a*XX + b)
    return 2 * -np.mean((yy) * -(1 / (y_hat + eps)**(r/2)) + (1 - yy) * -(y_hat)**(r/2), axis = 0) 

def gridded(loss, g_min, g_max, f, n_splits):
    chunks = [None] * n_splits
    span = (g_max - g_min) / n_splits
    n_points = int( span / f)
    n_total = int( (g_max - g_min) / f )

    for i in range(n_splits - 1):
        chunks[i] = np.linspace(g_min + i * span, g_min + (i + 1) * span - f, n_points)
    chunks[n_splits - 1] = np.linspace(g_min + (n_splits - 1) * span, g_min + n_splits * span, n_points + 1)
    
    aa, bb = np.meshgrid(np.linspace(g_min, g_max, n_total + 1), np.linspace(g_min, g_max, n_total + 1))
    
    grids = [None] * n_splits**2
    for i in range(n_splits):
        for j in range(n_splits):
            grids[n_splits*i + j] = np.meshgrid(chunks[i], chunks[j])

    losses = [None] * n_splits**2
    for i in range(n_splits**2):
        if i % n_splits == 0 and i > 0: print()
        print(i, end = '\t')
        losses[i] = loss(grids[i][0], grids[i][1])
    losses = np.hstack([np.vstack([losses[j] for j in range(i, i + n_splits)]) for i in range(0, n_splits**2, n_splits)])
    return losses, aa, bb

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

print('Empirical')
if not os.path.exists('anims/bces.npy'):
    print('BCE')
    bces, _, _ = gridded(bce,-3, 3, 0.01, 15)
    np.save('anims/bces', bces)
    print()
if not os.path.exists('anims/mses.npy')
    print('MSE')
    mses, _, _ = gridded(mse, -3, 3, 0.01, 15)
    np.save('anims/mses', mses)
    print()
if not os.path.exists('anims/mlcs.npy')
    print('MLC')
    mlcs, _, _ = gridded(mlc, -3, 3, 0.01, 15)
    np.save('anims/mlcs', mlcs)
    print()
if not os.path.exists('anims/sqrs.npy')
    print('SQR')
    sqrs, _, _ = gridded(sqr, -3, 3, 0.01, 15)
    np.save('anims/sqrs', sqrs)
    print()
print()

print('Numerical')
if not os.path.exists('anims/nbces.npy'):
    print('NBCE')
    nbces = nbce(aa, bb)
    np.save('anims/nbces.npy', nbces)
    print()
if not os.path.exists('anims/nmses.npy')
    print('NMSE')
    nmses = nmse(aa, bb)
    np.save('anims/nmses.npy', nmses)
    print()
if not os.path.exists('anims/nmlcs.npy')
    print('NMLC')
    nmlcs = nmlc(aa, bb)
    np.save('anims/nmlcs.npy', nmlcs)
    print()
if not os.path.exists('anims/nsqrs.npy')
    print('NBCE')
    nsqrs = nsqr(aa, bb)
    np.save('anims/nsqrs.npy', nsqrs)
    print()
print()

bces = np.load('anims/bces.npy')
mses = np.load('anims/mses.npy')
mlcs = np.load('anims/mlcs.npy')
sqrs = np.load('anims/sqrs.npy')

nbces = np.load('anims/nbces.npy')
nmses = np.load('anims/nmses.npy')
nmlcs = np.load('anims/nmlcs.npy')
nsqrs = np.load('anims/nsqrs.npy')

print('Empirical Estimation')
print(aa[bces == np.min(bces)], bb[bces == np.min(bces)])
print(aa[mses == np.min(mses)], bb[mses == np.min(mses)])
print(aa[mlcs == np.min(mlcs)], bb[mlcs == np.min(mlcs)])
print(aa[sqrs == np.min(sqrs)], bb[sqrs == np.min(sqrs)])
print()
print('Numerical Estimation')
print(aa[nbces == np.min(nbces)], bb[nbces == np.min(nbces)])
print(aa[nmses == np.min(nmses)], bb[nmses == np.min(nmses)])
print(aa[nmlcs == np.min(nmlcs)], bb[nmlcs == np.min(nmlcs)])
print(aa[nsqrs == np.min(nsqrs)], bb[nsqrs == np.min(nsqrs)])