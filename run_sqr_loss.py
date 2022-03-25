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

os.environ["CUDA_VISIBLE_DEVICES"] = "0" # pick a number < 4 on ML4HEP; < 3 on Voltan 
physical_devices = tf.config.list_physical_devices('GPU') 
tf.config.experimental.set_memory_growth(physical_devices[0], True)

### Generate the data.
# Data parameters
N = 10**6
mu = 0.1
sigma = 1

# Background is Normal(-μ, 1)
# Signal is Normal(μ, 1))
bkgd = np.random.normal(-mu, sigma, N)
sgnl = np.random.normal(mu, sigma, N)
X = np.concatenate([bkgd, sgnl])
y = np.concatenate([np.zeros(N), np.ones(N)])

X_train, X_test, y_train, y_test = train_test_split(X, y)

# Loss functions
def exp_sqr(y_true, y_pred):
    return -((y_true) * -1. / K.sqrt(K.exp(y_pred)) + (1. - y_true) * -K.sqrt(K.exp(y_pred)))

def sqr(y_true, y_pred):
    return -((y_true) * -1. / K.sqrt(y_pred + K.epsilon()) + (1. - y_true) * -K.sqrt(y_pred + K.epsilon()))

def get_sqr(p):
    def sqr_p(y_true, y_pred):
        return -((y_true) * -K.pow(y_pred + K.epsilon(), -p/2) + (1. - y_true) * -K.pow(y_pred + K.epsilon(), p/2))
    return sqr_p

def get_exp_sqr(p):
    def exp_sqr_p(y_true, y_pred):
        return -((y_true) * -K.pow(K.exp(y_pred), -p/2) + (1. - y_true) * -K.pow(K.exp(y_pred), p/2))
    return exp_sqr_p
        
# Likelihood ratios
def lr(x):
    return np.exp(-(1/(2 * sigma**2)) * ( (x - mu)**2 - (x + mu)**2))

def pure_lr(model):
    def model_lr(x):
        f = model.predict(x)
        return np.squeeze(f)
    return model_lr

def exp_lr(model):
    def model_lr(x):
        f = model.predict(x)
        return np.squeeze(np.exp(f))
    return model_lr

def make_mae(N_mae=10**4):
    mu = 0.1
    sigma = 1

    bkgd_mae = np.random.normal(-mu, sigma, N_mae)
    sgnl_mae = np.random.normal(mu, sigma, N_mae)
    X_mae = np.concatenate([bkgd_mae, sgnl_mae])
    
    def mae(model_lr):
        nonlocal X_mae
        return np.abs(model_lr(X_mae) - lr(X_mae)).mean()
    return mae
    
mae = make_mae()

earlystopping = EarlyStopping(patience=10,
                              verbose=0,
                              restore_best_weights=True)

# Models
def create_model(loss,
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
    
    return model

def train(loss,
          hidden='relu', 
          output='sigmoid', 
          dropout=True, 
          optimizer='adam', 
          metrics=['accuracy'], 
          verbose=0):
    model = create_model(loss, hidden, output, dropout, optimizer, verbose)
    
    trace = model.fit(X_train, 
                      y_train,
                      epochs = 100, 
                      batch_size=int(0.1*N), 
                      validation_data=(X_test, y_test),
                      callbacks=[earlystopping], 
                      verbose=verbose)
    print(trace.history['val_loss'][-1], end = '\t')
    
    return model

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
    
def lr_plot(ensembles,
            title=None,
            filename=None,
            cs = ['brown', 'green', 'red', 'blue'],
            lss = [':', '--', '-.', ':'],
            xs=np.linspace(-6, 6, 1000)):
    # Takes in a list of pairs (lr_avg, lr_err). Plots them against the true 
    # likelihood.
    fig, ax_1 = plt.subplots(figsize = (10, 6))
    
    # Plot true likelihood
    plt.plot(xs, lr(xs), label = 'Exact', c='k', ls='-')
    
    # Plot model likelihoods
    for i in range(len(ensembles)):
        avg, err, lbl = ensembles[i]
        plt.plot(xs, avg, label=lbl)
        plt.fill_between(xs, avg - err, avg + err, alpha=0.1)
    plt.legend()
    ax_1.minorticks_on()
    ax_1.tick_params(direction='in', which='both',length=5)
    plt.ylabel('Likelihood Ratio')

    # Plot background and signal
    ax_2 = ax_1.twinx()
    bins = np.linspace(-6, 6, 100)
    plt.hist(sgnl, alpha=0.1, bins=bins)
    plt.hist(bkgd, alpha=0.1, bins=bins)
    ax_2.minorticks_on()
    ax_2.tick_params(direction='in', which='both',length=5)
    plt.ylabel('Count')

    plt.xlim(-6, 6)
    plt.xlabel(r'$x$')
    plt.title(r"$\mu_{\rm{sgnl}}="+str(mu)+r", \mu_{\rm{bkgd}}="+str(-mu)+r"$",loc="right",fontsize=20);
    if title != None:
        plt.title(title, loc="left", fontsize=20)
    if filename != None:
        plt.savefig(filename, dpi=1200, bbox_inches='tight')

def lrr_plot(ensembles,
             title=None,
             filename=None,
             cs = ['brown', 'green', 'red', 'blue'],
             lss = [':', '--', '-.', ':'],
             xs=np.linspace(-6, 6, 1000)):
    # Takes in a list of pairs (lrr_avg, lrr_err). Plots them.
    fig, ax_1 = plt.subplots(figsize = (10, 6))
    
    # Plot ratios of likelihood ratios
    for i in range(len(ensembles)):
        avg, err, lbl = ensembles[i]
        plt.plot(xs, avg, label=lbl)
        plt.fill_between(xs, avg - err, avg + err, alpha=0.1)
    plt.axhline(1,ls=":",color="grey", lw=0.5)
    plt.axvline(0,ls=":",color="grey", lw=0.5)
    plt.legend()
    ax_1.minorticks_on()
    ax_1.tick_params(direction='in', which='both',length=5)
    plt.ylim(0.94, 1.06)
    plt.ylabel('Ratio')

    # Plot background and signal
    ax_2 = ax_1.twinx()
    bins = np.linspace(-6, 6, 100)
    plt.hist(sgnl, alpha=0.1, bins=bins)
    plt.hist(bkgd, alpha=0.1, bins=bins)
    ax_2.minorticks_on()
    ax_2.tick_params(direction='in', which='both',length=5)
    plt.ylabel('Count')

    plt.xlim(-6, 6)
    plt.xlabel(r'$x$')
    plt.title(r"$\mu_{\rm{sgnl}}="+str(mu)+r", \mu_{\rm{bkgd}}="+str(-mu)+r"$",loc="right",fontsize=20)
    if title != None:
        plt.title(title, loc="left", fontsize=20)
    if filename != None:
        plt.savefig(filename, dpi=1200, bbox_inches='tight')

        
# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++ #
# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++ #


# Train
reps = 50

ps = np.linspace(-6, 6, 121)
sqr_filestr = 'models/sqr/model_{}_{}.h5'
exp_filestr = 'models/exp/model_{}_{}.h5'

sqr_lrs = {}
exp_lrs = {}
for p in ps:
    print('===================================================\n{}'.format(p))
    sqr_lrs[p] = [None] * reps
    exp_lrs[p] = [None] * reps
    sqr_params = {'loss': get_sqr(p), 'output':'relu'}
    exp_params = {'loss': get_exp_sqr(p), 'output':'linear'}
    for i in range(10):
        print(i, end = '\t')
        sqr_model = train(**sqr_params)
        exp_model = train(**exp_params)
        sqr_lrs[p][i] = pure_lr(sqr_model)
        exp_lrs[p][i] = exp_lr(exp_model)
        print()
        sqr_model.save_weights(sqr_filestr.format(p, i))
        exp_model.save_weights(exp_filestr.format(p, i))

    
# Mean absolute errors
sqr_mae_avg = []
sqr_mae_err = []
exp_mae_avg = []
exp_mae_err = []

for p in ps:
    print(p, end = '\t')
    sqr_maes = [mae(lr) for lr in sqr_lrs[p]]
    sqr_mae_avg += [np.mean(sqr_maes)]
    sqr_mae_err += [np.std(sqr_maes)]
    print(sqr_mae_avg[-1], end = '\t')
    exp_maes = [mae(lr) for lr in exp_lrs[p]]
    exp_mae_avg += [np.mean(exp_maes)]
    exp_mae_err += [np.std(exp_maes)]
    print(exp_mae_avg[-1])

sqr_mae_avg = np.array(sqr_mae_avg)
sqr_mae_err = np.array(sqr_mae_err)
exp_mae_avg = np.array(exp_mae_avg)
exp_mae_err = np.array(exp_mae_err)

fig, ax = plt.subplots(figsize = (10, 6))

plt.plot(ps, sqr_mae_avg, c='blue', label='SQR (linear)')
plt.plot(ps, exp_mae_avg, c='red', label='SQR (exponential)')
plt.legend()
plt.minorticks_on()
plt.tick_params(direction='in', which='both',length=5)
plt.ylim(0, 1)
plt.ylabel('Mean Absolute Error')
plt.xlabel(r'$p$')

plt.savefig('plots/new_sqr_loss_params_maes.png', dpi=1200, bbox_inches='tight')


fig, ax = plt.subplots(figsize = (10, 6))

plt.plot(ps, sqr_mae_avg, c='blue', label='SQR (linear)')
plt.plot(ps, exp_mae_avg, c='red', label='SQR (exponential)')
plt.fill_between(ps, sqr_mae_avg - sqr_mae_err, sqr_mae_avg + sqr_mae_err, c='blue')
plt.fill_between(ps, exp_mae_avg - exp_mae_err, exp_mae_avg + exp_mae_err, c='red')

plt.legend()
plt.minorticks_on()
plt.tick_params(direction='in', which='both',length=5)
plt.ylim(0, 1)
plt.ylabel('Mean Absolute Error')
plt.xlabel(r'$p$')

plt.savefig('plots/new_sqr_loss_params_maes_2.png', dpi=1200, bbox_inches='tight')