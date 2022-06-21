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

np.random.seed(666) # Need to do more to ensure data is the same across runs.

os.environ["CUDA_VISIBLE_DEVICES"] = "0" # pick a number < 4 on ML4HEP; < 3 on Voltan 
physical_devices = tf.config.list_physical_devices('GPU') 
tf.config.experimental.set_memory_growth(physical_devices[0], True)

### Generate the data.
N = 10**6
mu = 0.1
sigma = 1

# Background is Normal(-μ, σ). Signal is Normal(μ, σ))
bkgd = np.random.normal(-mu, sigma, N)
sgnl = np.random.normal(mu, sigma, N)
X = np.concatenate([bkgd, sgnl])
y = np.concatenate([np.zeros(N), np.ones(N)])

X_train, X_test, y_train, y_test = train_test_split(X, y)

# Loss functions
def bce(y_true, y_pred):
    # Clipping to (ɛ, 1 - ɛ) is fine since the final activation is sigmoid.
    y_pred = K.clip(y_pred, K.epsilon(), 1. - K.epsilon())
    
    return -((y_true) * K.log(y_pred + K.epsilon()) + 
             (1. - y_true) * K.log(1. - y_pred))

def square_bce(y_true, y_pred):
    return -((y_true) * K.log(y_pred**2) + 
             (1. - y_true) * K.log(1. - y_pred**2))

def exp_bce(y_true, y_pred):
    return -((y_true) * (y_pred) + 
             (1. - y_true) * (1. - y_pred))

def mse(y_true, y_pred):
    # Clipping to (ɛ, 1 - ɛ) is fine since the final activation is sigmoid.
    y_pred = K.clip(y_pred, K.epsilon(), 1. - K.epsilon())
    
    return -((y_true) * -K.square(1. - y_pred) + 
             (1. - y_true) * -K.square(y_pred))

def square_mse(y_true, y_pred):
    return -((y_true) * -K.square(1. - y_pred) 
             + (1. - y_true) * -K.square(y_pred + K.epsilon()))

def exp_mse(y_true, y_pred):
    return -((y_true) * -K.square(1. - K.exp(y_pred)) + 
             (1. - y_true) * -K.square(K.exp(y_pred))) 

def get_mse(p):
    def mse_p(y_true, y_pred):
        y_pred = K.clip(y_pred, K.epsilon(), 1. - K.epsilon())
    
        return -((y_true) * -K.pow(1. - y_pred, p) + 
                 (1. - y_true) * -K.pow(y_pred, p))
    return mse_p
             
def mlc(y_true, y_pred):
    return -((y_true) * K.log(y_pred + K.epsilon()) + 
             (1. - y_true) * (1. - y_pred))

def square_mlc(y_true, y_pred):
    return -((y_true) * K.log(y_pred**2) + 
             (1. - y_true) * (1. - y_pred**2))

def exp_mlc(y_true, y_pred):
    return -((y_true) * (y_pred) + 
             (1. - y_true) * (1. - K.exp(y_pred)))

def sqr(y_true, y_pred):
    return -((y_true) * -1. / K.sqrt(y_pred + K.epsilon()) + 
             (1. - y_true) * -K.sqrt(y_pred + K.epsilon()))

def square_sqr(y_true, y_pred):
    return -((y_true) * -1. / K.sqrt(K.square(y_pred)) + 
             (1. - y_true) * -K.sqrt(K.square(y_pred)))

def exp_sqr(y_true, y_pred):
    return -((y_true) * -1. / K.sqrt(K.exp(y_pred)) + 
             (1. - y_true) * -K.sqrt(K.exp(y_pred)))

def get_sqr(p):
    def sqr_p(y_true, y_pred):
        return -((y_true) * -K.pow(y_pred + K.epsilon(), -p/2) + 
                 (1. - y_true) * -K.pow(y_pred + K.epsilon(), p/2))
    return sqr_p

def get_exp_sqr(p):
    def exp_sqr_p(y_true, y_pred):
        return -((y_true) * -K.pow(K.exp(y_pred), -p/2) + 
                 (1. - y_true) * -K.pow(K.exp(y_pred), p/2))
    return exp_sqr_p
        
# Likelihood ratios
def lr(x):
    return np.exp(-(1/(2 * sigma**2)) * ( (x - mu)**2 - (x + mu)**2))

def odds_lr(model):
    def model_lr(x):
        f = model.predict(x)
        return np.squeeze(f / (1. - f))
    return model_lr

def square_odds_lr(model):
    def model_lr(x):
        f = model.predict(x)
        return np.squeeze(f**2 / (1. - f**2))
    return model_lr

def exp_odds_lr(model):
    def model_lr(x):
        f = model.predict(x)
        return np.squeeze(np.exp(f) / (1. - np.exp(f)))
    return model_lr

def pure_lr(model):
    def model_lr(x):
        f = model.predict(x)
        return np.squeeze(f)
    return model_lr

def square_lr(model):
    def model_lr(x):
        f = model.predict(x)
        return np.squeeze(f**2)
    return model_lr

def exp_lr(model):
    def model_lr(x):
        f = model.predict(x)
        return np.squeeze(np.exp(f))
    return model_lr

def pow_lr(model, p):
    def model_lr(x):
        f = model.predict(x)
        return np.squeeze(f**p)
    return model_lr

def exp_pow_lr(model, p):
    def model_lr(x):
        f = model.predict(x)
        return np.squeeze(np.exp(f)**p)
    return model_lr

def pow_odds_lr(model, p):
    def model_lr(x):
        f = model.predict(x)
        return np.squeeze( (f / (1. - f))**p)
    return model_lr

def make_mae(mu_mae, sigma_mae, N_mae=10**4):
    bkgd_mae = np.random.normal(-mu_mae, sigma_mae, N_mae)
    sgnl_mae = np.random.normal(mu_mae, sigma_mae, N_mae)
    X_mae = np.concatenate([bkgd_mae, sgnl_mae])
    
    def mae(model_lr):
        nonlocal X_mae
        return np.abs(model_lr(X_mae) - lr(X_mae)).mean()
    return mae
    
mae = make_mae(mu, sigma)

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
    
    return model

def train(loss,
          hidden='relu', 
          output='sigmoid', 
          dropout=True, 
          optimizer='adam', 
          metrics=['accuracy'], 
          verbose=0):
    model = create_model(loss, hidden, output, dropout, optimizer, metrics, verbose)      
    
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
            lss = ['-', '--', '-.', ':'],
            xs=np.linspace(-6, 6, 1000)):
    # Takes in a list of pairs (lr_avg, lr_err). Plots them against the true 
    # likelihood.
    fig, ax_1 = plt.subplots(figsize = (10, 6))
    
    # Plot true likelihood
    plt.plot(xs, lr(xs), label = 'Exact', c='k', ls='-')
    
    # Plot model likelihoods
    for i in range(len(ensembles)):
        avg, err, lbl = ensembles[i]
        plt.plot(xs, avg, label=lbl, c=cs[i], ls=lss[i])
        #plt.fill_between(xs, avg - err, avg + err, color=cs[i], alpha=0.1)
    plt.legend()
    ax_1.minorticks_on()
    ax_1.tick_params(direction='in', which='both',length=5)
    plt.ylabel('Likelihood Ratio')
    #plt.ylim(0, 6)

    # Plot background and signal
    ax_2 = ax_1.twinx()
    bins = np.linspace(-6, 6, 100)
    plt.hist(sgnl, alpha=0.1, bins=bins)
    plt.hist(bkgd, alpha=0.1, bins=bins)
    ax_2.minorticks_on()
    ax_2.tick_params(direction='in', which='both',length=5)
    plt.ylabel('Count')

    plt.xlim(xs[0], xs[-1])
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
        plt.plot(xs, avg, label=lbl, c=cs[i], ls=lss[i])
        #plt.fill_between(xs, avg - err, avg + err, color=cs[i], alpha=0.1)
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

    plt.xlim(xs[0], xs[-1])
    plt.xlabel(r'$x$')
    plt.title(r"$\mu_{\rm{sgnl}}="+str(mu)+r", \mu_{\rm{bkgd}}="+str(-mu)+r"$",loc="right",fontsize=20)
    if title != None:
        plt.title(title, loc="left", fontsize=20)
    if filename != None:
        plt.savefig(filename, dpi=1200, bbox_inches='tight')

#########################################################################
#                                                                       #
#########################################################################
num = 0
reps = 20

ps = np.round(np.append(np.linspace(-4, -2, 41),
                        np.linspace(2, 4, 41)),
              2)
filestr = 'models/mse_ab_param/set_' + str(num) + '/model_{}_{}.h5'

# Train models
for p in ps:
    print('===================================================\n{}'.format(p))
    params = {'loss': get_mse(p)}
    for i in range(reps):
        print(i, end = ' ')
        model = train(**params)
        print()
        model.save_weights(filestr.format(p, i))
        
# Get model likelihood ratios.
lrs = {}
for p in ps:
    print(p)
    lrs[p] = [None] * reps
    params = {'loss': get_mse(p)}
    for i in range(reps):
        model = create_model(**params)
        model.load_weights(filestr.format(p, i))
        lrs[p][i] = pow_odds_lr(model, p)
        print(i, end = ' ')
    print()

mae_avg = []
mae_err = []

for p in ps:
    maes = [mae(lr) for lr in lrs[p]]
    mae_avg += [np.mean(maes)]
    mae_err += [np.std(maes)]
    print(p, '\t', mae_avg[-1])

mae_avg = np.array(mae_avg)
mae_err = np.array(mae_err)

fig, ax = plt.subplots(figsize = (10, 6))

plt.plot(ps, mae_avg, c='blue')

plt.minorticks_on()
plt.tick_params(direction='in', which='both',length=5)
plt.ylabel('Mean Absolute Error')
plt.xlabel(r'$p$')

plt.title(r"$\mu_{\rm{sgnl}}="+str(mu)+r", \mu_{\rm{bkgd}}="+str(-mu)+r"$",
          loc="right",
          fontsize=20);
plt.title(r"MSE $A/B$ Parametrization",loc="left",fontsize=20);
plt.savefig('plots/mse_ab_param/set_{}/maes.png'.format(num), 
            dpi=1200, 
            bbox_inches='tight')