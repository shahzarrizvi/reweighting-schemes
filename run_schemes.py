#!/usr/bin/env python
# coding: utf-8

# General imports
import numpy as np
import os

# Plotting
from matplotlib import pyplot as plt
from matplotlib.patches import Patch
from matplotlib import rc
import matplotlib.font_manager
rc('font', family='serif')
rc('text', usetex=True)
rc('font', size=22)
rc('xtick', labelsize=15)
rc('ytick', labelsize=15)
rc('legend', fontsize=15)

# ML imports
from sklearn.model_selection import train_test_split
import tensorflow as tf
import tensorflow.keras
import tensorflow.keras.backend as K
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, Input, Dropout
from tensorflow.keras.callbacks import EarlyStopping

# Configure GPUs
os.environ["CUDA_VISIBLE_DEVICES"] = "0" # pick a number < 4 on ML4HEP; < 3 on Voltan 
physical_devices = tf.config.list_physical_devices('GPU') 
tf.config.experimental.set_memory_growth(physical_devices[0], True)

# Data parameters
N = 10**6
mu = 0.1
sigma = 1

### Generate the data.
# Background is Normal(-μ, 1)
# Signal is Normal(μ, 1))
bkgd = np.random.normal(-mu, sigma, N)
sgnl = np.random.normal(mu, sigma, N)
X = np.concatenate([bkgd, sgnl])
y = np.concatenate([np.zeros(N), np.ones(N)])

X_train, X_test, y_train, y_test = train_test_split(X, y)

# Loss functions
def bce(y_true, y_pred):
    return -((y_true) * K.log(y_pred + K.epsilon()) + (1. - y_true) * K.log(1. - y_pred))

def mse(y_true, y_pred):
    return -((y_true) * -K.square(1. - y_pred) + (1. - y_true) * -K.square(y_pred + K.epsilon()))

def sqr(y_true, y_pred):
    return -((y_true) * -1. / K.sqrt(y_pred + K.epsilon()) + (1. - y_true) * -K.sqrt(y_pred + K.epsilon()))

def mlc(y_true, y_pred):
    return -((y_true) * K.log(y_pred + K.epsilon()) + (1. - y_true) * (1. - y_pred))

def square_mlc(y_true, y_pred):
    return -((y_true) * K.log( (y_pred + K.epsilon())**2 ) + (1. - y_true) * (1. - y_pred**2))

def exp_mlc(y_true, y_pred):
    return -((y_true) * y_pred + (1. - y_true) * (1. - K.exp(y_pred)))

def square_sqr(y_true, y_pred):
    return -((y_true) * -1. / K.sqrt(K.square(y_pred)) + (1. - y_true) * -K.sqrt(K.square(y_pred)))

def exp_sqr(y_true, y_pred):
    return -((y_true) * -1. / K.sqrt(K.exp(y_pred)) + (1. - y_true) * -K.sqrt(K.exp(y_pred)))

def sqr(y_true, y_pred):
    return -((y_true) * -1. / K.sqrt(y_pred + K.epsilon()) + (1. - y_true) * -K.sqrt(y_pred + K.epsilon()))
        
# Likelihood ratios
def lr(x):
    return np.exp(-(1/(2 * sigma**2)) * ( (x - mu)**2 - (x + mu)**2))

def get_bce_lr(model):
    def model_bce_lr(x):
        f = model.predict(x)
        return np.squeeze(f / (1. - f))
    return model_bce_lr

def get_mse_lr(model):
    def model_mse_lr(x):
        f = model.predict(x)
        return np.squeeze(f / (1. - f))
    return model_mse_lr

def get_mlc_lr(model):
    def model_mlc_lr(x):
        f = model.predict(x)
        return np.squeeze(f)
    return model_mlc_lr

def get_square_mlc_lr(model):
    def model_square_mlc_lr(x):
        f = model.predict(x)
        return np.squeeze(f**2)
    return model_square_mlc_lr

def get_exp_mlc_lr(model):
    def model_exp_mlc_lr(x):
        f = model.predict(x)
        return np.squeeze(np.exp(f))
    return model_exp_mlc_lr


def get_sqr_lr(model):
    def model_sqr_lr(x):
        f = model.predict(x)
        return np.squeeze(f)
    return model_sqr_lr

def get_square_sqr_lr(model):
    def model_square_sqr_lr(x):
        f = model.predict(x)
        return np.squeeze(f**2)
    return model_square_sqr_lr

def get_exp_sqr_lr(model):
    def model_exp_sqr_lr(x):
        f = model.predict(x)
        return np.squeeze(np.exp(f))
    return model_exp_sqr_lr

def mae(model_lr):
    # Takes in model_lr, a model likelihood ratio. Returns the expected absolute
    # error for that model.
    return np.abs(model_lr(X) - lr(X)).mean()

earlystopping = EarlyStopping(patience=10,
                              verbose=0,
                              restore_best_weights=True)

# Models
def train(loss, 
          hidden='relu', 
          output='sigmoid', 
          dropout=True, 
          optimizer='adam', 
          metrics=['accuracy'], 
          verbose=0):
    model = Sequential()
    if dropout:
        model.add(Dense(64, activation=hidden, input_shape=(1, )))
        model.add(Dropout(0.1))
        model.add(Dense(128, activation=hidden))
        model.add(Dropout(0.1))
        model.add(Dense(64, activation=hidden))
        model.add(Dropout(0.1))
        model.add(Dense(1, activation=output))
    else: 
        model.add(Dense(64, activation=hidden, input_shape=(1, )))
        model.add(Dense(128, activation=hidden))
        model.add(Dense(64, activation=hidden))
        model.add(Dense(1, activation=output))        
    
    model.compile(loss=loss,
                  optimizer=optimizer, 
                  metrics=metrics)
    
    trace = model.fit(X_train, 
                      y_train,
                      epochs = 100, 
                      batch_size=int(0.1*N), 
                      validation_data=(X_test, y_test),
                      callbacks=[earlystopping], 
                      verbose=verbose)
    print(trace.history['val_loss'][-1], end = ' ')
    
    lr_match = {bce: get_bce_lr, 
                mse: get_mse_lr, 
                mlc: get_mlc_lr,
                square_mlc: get_square_mlc_lr,
                exp_mlc: get_exp_mlc_lr,
                sqr: get_sqr_lr, 
                square_sqr: get_square_sqr_lr, 
                exp_sqr: get_exp_sqr_lr}
    model_lr = lr_match[loss](model)
    
    return model, model_lr

# Plotting functions
def get_preds(model_lrs, xs=np.linspace(-6, 6, 1000)):
    # Takes in model_lrs, a list of model likelihood ratios and xs, a list of 
    # values on which to compute the likelihood ratios. Returns a 2D array. The 
    # nth row is the likelihood ratio predictions from the nth model in 
    # model_lrs.
    return np.array([model_lr(xs) for model_lr in model_lrs])
    
def avg_lr(preds):
    # Takes in a 2D array of multiple models' likelihood ratio predictions. 
    # Returns the average likelihood ratio prediction and its error.
    return preds.mean(axis=0), preds.std(axis=0)

def avg_lrr(preds, xs=np.linspace(-6, 6, 1000)):
    # Takes in a 2D array of multiple models' likelihood ratio predictions. 
    # Returns the average ratio of predicted likelihood to true likelihood and 
    # its error.
    lrr_preds = preds / lr(xs)
    return lrr_preds.mean(axis=0), lrr_preds.std(axis=0)

# Error Calculations
reps = 100

bce_params = {'loss':bce}
mse_params = {'loss':mse}
mlc_params = {'loss':exp_mlc, 'output':'linear'}
sqr_params = {'loss':exp_sqr, 'output':'linear'}

Ns = 10**np.arange(2, 8)

# Train models
#bce_lrs = {}
#mse_lrs = {}
#mlc_lrs = {}
sqr_lrs = {}

for N in Ns:
    print('===================================================\n{}'.format(N))
    # Set up lists
    #bce_lrs[N] = [None] * reps
    #mse_lrs[N] = [None] * reps
    #mlc_lrs[N] = [None] * reps
    sqr_lrs[N] = [None] * reps
    
    # Generate data
    bkgd = np.random.normal(-mu, 1, N)
    sgnl = np.random.normal(mu, 1, N)
    X = np.concatenate([bkgd, sgnl])
    y = np.concatenate([np.zeros(N), np.ones(N)])

    X_train, X_test, y_train, y_test = train_test_split(X, y)

    for i in range(reps):
        print(i, end = ' ')
        #bce_model, bce_lrs[N][i] = train(**bce_params)
        #mse_model, mse_lrs[N][i] = train(**mse_params)
        #mlc_model, mlc_lrs[N][i] = train(**mlc_params)
        sqr_model, sqr_lrs[N][i] = train(**sqr_params)
        print()
        # Save models
        #bce_model.save_weights('models/maes/bce/bce_model_{}_{}.h5'.format(N, i))
        #mse_model.save_weights('models/maes/mse/mse_model_{}_{}.h5'.format(N, i))
        #mlc_model.save_weights('models/maes/mlc/mlc_model_{}_{}.h5'.format(N, i))
        sqr_model.save_weights('models/maes/sqr/sqr_model_{}_{}.h5'.format(N, i))
'''            
def mae(model_lr):
    # Takes in model_lr, a model likelihood ratio. Returns the expected absolute
    # error for that model.
    return np.abs(model_lr(X) - lr(X)).mean()
        
# Calculate mean absolute errors.
N = 10**4

bkgd = np.random.normal(-mu, 1, N)
sgnl = np.random.normal(mu, 1, N)
X = np.concatenate([bkgd, sgnl])

bce_mae_avg = []
mse_mae_avg = []
mlc_mae_avg = []
sqr_mae_avg = []

bce_mae_err = []
mse_mae_err = []
mlc_mae_err = []
sqr_mae_err = []

for N in Ns:
    bce_maes = [mae(lr) for lr in bce_lrs[N]]
    mse_maes = [mae(lr) for lr in mse_lrs[N]]
    mlc_maes = [mae(lr) for lr in mlc_lrs[N]]
    sqr_maes = [mae(lr) for lr in sqr_lrs[N]]
    
    bce_mae_avg += [np.mean(bce_maes)]
    bce_mae_err += [np.std(bce_maes)]
    
    mse_mae_avg += [np.mean(mse_maes)]
    mse_mae_err += [np.std(mse_maes)]
    
    mlc_mae_avg += [np.mean(mlc_maes)]
    mlc_mae_err += [np.std(mlc_maes)]
    
    sqr_mae_avg += [np.mean(sqr_maes)]
    sqr_mae_err += [np.std(sqr_maes)]

bce_mae_avg = np.array(bce_mae_avg)
mse_mae_avg = np.array(mse_mae_avg)
mlc_mae_avg = np.array(mlc_mae_avg)
sqr_mae_avg = np.array(sqr_mae_avg)

bce_mae_err = np.array(bce_mae_err)
mse_mae_err = np.array(mse_mae_err)
mlc_mae_err = np.array(mlc_mae_err)
sqr_mae_err = np.array(sqr_mae_err)

np.save('lr_data/bce_mae_avg.npy', bce_mae_avg)
np.save('lr_data/mse_mae_avg.npy', mse_mae_avg)
np.save('lr_data/mlc_mae_avg.npy', mlc_mae_avg)
np.save('lr_data/sqr_mae_avg.npy', sqr_mae_avg)

np.save('lr_data/bce_mae_err.npy', bce_mae_err)
np.save('lr_data/mse_mae_err.npy', mse_mae_err)
np.save('lr_data/mlc_mae_err.npy', mlc_mae_err)
np.save('lr_data/sqr_mae_err.npy', sqr_mae_err)

# Plot with errors
fig, ax = plt.subplots(figsize = (10, 6))

plt.plot(Ns, bce_mae_avg, c='brown', ls=':', label='BCE')
plt.plot(Ns, mse_mae_avg, c='green', ls='--', label='MSE')
plt.plot(Ns, mlc_mae_avg, c='red', ls='--', label='MLC')
plt.plot(Ns, sqr_mae_avg, c='blue', ls='-.', label='SQR')
plt.fill_between(Ns, bce_mae_avg - bce_mae_err, bce_mae_avg + bce_mae_err, color='brown', alpha=0.1)
plt.fill_between(Ns, mse_mae_avg - mse_mae_err, mse_mae_avg + mse_mae_err, color='green', alpha=0.1)
plt.fill_between(Ns, mlc_mae_avg - mlc_mae_err, mlc_mae_avg + mlc_mae_err, color='red', alpha=0.1)
plt.fill_between(Ns, sqr_mae_avg - sqr_mae_err, sqr_mae_avg + sqr_mae_err, color='blue', alpha=0.1)
plt.legend()

plt.xscale("log", base=10)
plt.minorticks_on()
plt.tick_params(direction='in', which='both',length=5)
plt.ylabel('Mean Absolute Error')
plt.xlabel(r'$N$')

plt.title(r"$\mu_{\rm{sgnl}}="+str(mu)+r", \mu_{\rm{bkgd}}="+str(-mu)+r"$",loc="right",fontsize=20);
plt.savefig('plots/maes.png', dpi=1200, bbox_inches='tight')

# Plot without errors
fig, ax = plt.subplots(figsize = (10, 6))

plt.plot(Ns, bce_mae_avg, c='brown', ls=':', label='BCE')
plt.plot(Ns, mse_mae_avg, c='green', ls='--', label='MSE')
plt.plot(Ns, mlc_mae_avg, c='red', ls='--', label='MLC')
plt.plot(Ns, sqr_mae_avg, c='blue', ls='-.', label='SQR')
plt.legend()

plt.xscale("log", base=10)
plt.minorticks_on()
plt.tick_params(direction='in', which='both',length=5)
plt.ylabel('Mean Absolute Error')
plt.xlabel(r'$N$')

plt.title(r"$\mu_{\rm{sgnl}}="+str(mu)+r", \mu_{\rm{bkgd}}="+str(-mu)+r"$",loc="right",fontsize=20);
plt.savefig('plots/maes_no_err.png', dpi=1200, bbox_inches='tight')
'''