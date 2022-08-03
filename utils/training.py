# ML imports
#import tensorflow as tf
#import tensorflow.keras
#import tensorflow.keras.backend as K
import numpy as np
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, Input, Dropout
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.model_selection import train_test_split

# Training functions
earlystopping = EarlyStopping(patience=10,
                              verbose=0,
                              restore_best_weights=True)

def create_model(loss,
                 d = 1,
                 hidden = 'relu', 
                 output = 'sigmoid', 
                 dropout = True, 
                 optimizer = 'adam', 
                 metrics = ['accuracy'], 
                 verbose = 0):
    model = Sequential()
    if dropout:
        model.add(Dense(64, activation=hidden, input_shape=(d, )))
        model.add(Dropout(0.1))
        model.add(Dense(128, activation=hidden))
        model.add(Dropout(0.1))
        model.add(Dense(64, activation=hidden))
        model.add(Dropout(0.1))
        model.add(Dense(1, activation=output))
    else: 
        model.add(Dense(64, activation=hidden, input_shape=(d, )))
        model.add(Dense(128, activation=hidden))
        model.add(Dense(64, activation=hidden))
        model.add(Dense(1, activation=output))        
    
    return model

def train(data, 
          loss,
          d = 1,
          hidden = 'relu', 
          output = 'sigmoid', 
          dropout = True, 
          optimizer = 'adam', 
          metrics = ['accuracy'], 
          verbose = 0):
    X_train, X_test, y_train, y_test = data
    
    N = (len(X_train) + len(X_test)) / 2
    
    model = create_model(loss, d, hidden, output, dropout, optimizer, metrics, verbose)      
    
    model.compile(loss = loss,
                  optimizer = optimizer, 
                  metrics = metrics)
    
    trace = model.fit(X_train, 
                      y_train,
                      epochs = 100, 
                      batch_size = int(0.1*N), 
                      validation_data = (X_test, y_test),
                      callbacks = [earlystopping], 
                      verbose = verbose)
    print(trace.history['val_loss'][-1], '\t', len(trace.history['val_loss']), end = '\t')
    
    return model, trace

def make_data(bkgd, sgnl, N):
    X_bkgd = bkgd.rvs(size = N)
    X_sgnl = sgnl.rvs(size = N)
    y_bkgd = np.zeros(N)
    y_sgnl = np.ones(N)
    
    X = np.concatenate([X_bkgd, X_sgnl])
    y = np.concatenate([y_bkgd, y_sgnl])
    
    # Split in train and test sets.
    X_train, X_test, y_train, y_test = train_test_split(X, y)
    
    # Normalize both the train and the test set.
    m = np.mean(X_train, axis = 0)
    s = np.std(X_train, axis = 0)
    X_train = (X_train - m) / s
    X_test = (X_test - m) / s
    
    return (X_train, X_test, y_train, y_test), m, s

def make_lr(bkgd, sgnl):
    return lambda x: sgnl.pdf(x) / bkgd.pdf(x)

def make_mae(bkgd, sgnl, N_mae = 10**4):
    bkgd_mae = bkgd.rvs(size = N_mae)
    sgnl_mae = sgnl.rvs(size = N_mae)
    X_mae = np.concatenate([bkgd_mae, sgnl_mae])
    
    lr = make_lr(bkgd, sgnl)
    
    def mae(model_lr):
        nonlocal X_mae
        return np.abs(model_lr(X_mae) - lr(X_mae)).mean()
    return mae

def make_mpe(bkgd, sgnl, N_mpe = 10**4):
    bkgd_mpe = bkgd.rvs(size = N_mpe)
    sgnl_mpe = sgnl.rvs(size = N_mpe)
    X_mpe = np.concatenate([bkgd_mpe, sgnl_mpe])
    
    lr = make_lr(bkgd, sgnl)
    
    def mpe(model_lr):
        nonlocal X_mpe
        return np.abs((model_lr(X_mpe) - lr(X_mpe)) / lr(X_mpe)).mean() * 100
    return mae
    
def odds_lr(model, m = 0, s = 1):
    def model_lr(x):
        f = model.predict((x - m) / s)
        return np.squeeze(f / (1. - f))
    return model_lr

def square_odds_lr(model, m = 0, s = 1):
    def model_lr(x):
        f = model.predict((x - m) / s)
        return np.squeeze(f**2 / (1. - f**2))
    return model_lr

def exp_odds_lr(model, m = 0, s = 1):
    def model_lr(x):
        f = model.predict((x - m) / s)
        return np.squeeze(np.exp(f) / (1. - np.exp(f)))
    return model_lr

def pure_lr(model, m = 0, s = 1):
    def model_lr(x):
        f = model.predict((x - m) / s)
        return np.squeeze(f)
    return model_lr

def square_lr(model, m = 0, s = 1):
    def model_lr(x):
        f = model.predict((x - m) / s)
        return np.squeeze(f**2)
    return model_lr

def exp_lr(model, m = 0, s = 1):
    def model_lr(x):
        f = model.predict((x - m) / s)
        return np.squeeze(np.exp(f))
    return model_lr

def pow_lr(model, p, m = 0, s = 1):
    def model_lr(x):
        f = model.predict((x - m) / s)
        return np.squeeze(f**p)
    return model_lr

def exp_pow_lr(model, p, m = 0, s = 1):
    def model_lr(x):
        f = model.predict((x - m) / s)
        return np.squeeze(np.exp(f)**p)
    return model_lr

def pow_odds_lr(model, p, m = 0, s = 1):
    def model_lr(x):
        f = model.predict((x - m) / s)
        return np.squeeze( (f / (1. - f))**(p - 1))
    return model_lr

def t_tanh(x):
    return 0.5 * (np.tanh(x) + 1)

def tanh_lr(model, m = 0, s = 1):
    def model_lr(x):
        f = model.predict((x - m) / s)
        return np.squeeze(t_tanh(f) / (1. - t_tanh(f)))
    return model_lr

def t_atan(x):
    return 0.5 + (np.arctan(x) / np.pi)

def atan_lr(model, m = 0, s = 1):
    def model_lr(x):
        f = model.predict((x - m) / s)
        return np.squeeze(t_atan(f) / (1. - t_atan(f)))
    return model_lr

'''
#bkgd sgnl maybe an array
# distribution primarily only needed for plotting
def compare(params, bkgd, sgnl, reps=100, N = 10**6, num, filestrs):
    #assert len(params) == len(filestrs)
    data = make_data(bkgd, sgnl, N) + [N]
    for i in range(reps):
        for j in len(params):
            model, trace = train(data, **params[i])
            model.save_weights(filestrs[i])

match = {bce: [bce, square_bce, exp_bce],
         mse: [mse, square_mse, exp_mse],
         mlc: [mlc, square_mlc, exp_mlc],
         sqr: [sqr, square_sqr, exp_sqr]
        }
def c_param(loss, bkgd, sgnl):
    losses = match[loss]
    linear_param = {losses[0], ...}   # the remaining information will have to be put into match as well
    square_param = {losses[1], ...}   # such as whether the activation is relu, sigmoid, or linear
    exponl_param = {losses[2], ...}
    
    params = [linear_param, square_param, exponl_param]
    
    linear_filestr = 'models/demo/...' # need some way to figure out num and other values, like the loss
    square_filestr = 'models/demo/...' # possibly also included in match
    exponl_filestr = 'models/demo/...'
    
    filestrs = [linear_filestr, square_filestr, exponl_filestr]
    
    compare(params, filestrs, bkgd, sgnl, num = ..., filestrs)
    #print statement like "you can find your models in [path to dir]
'''
        