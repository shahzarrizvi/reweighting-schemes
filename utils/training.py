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

def create_model(hidden='relu', 
                 output='sigmoid', 
                 dropout=True):
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

def train(data_args, 
          loss,
          hidden='relu', 
          output='sigmoid', 
          dropout=True, 
          optimizer='adam', 
          metrics=['accuracy'], 
          verbose=0):
    X_train, X_test, y_train, y_test, N = data_args
    
    model = create_model(hidden, output, dropout)      
    
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

def make_data(bkgd, sgnl, N):
    X_bkgd = bkgd.rvs(size = N)
    X_sgnl = bkgd.rvs(size = N)
    y_bkgd = np.zeros(N)
    y_sgnl = np.ones(N)
    
    X = np.concatenate([X_bkgd, X_sgnl])
    y = np.concatenate([y_bkgd, y_sgnl])
    
    return train_test_split(X, y)

def make_lr(bkgd, sgnl):
    return lambda x: sgnl.pdf(x) / bkgd.pdf(x)

def make_mae(bkgd, sgnl, N_mae=10**4):
    bkgd_mae = bkgd.rvs(size = N_mae)
    sgnl_mae = sgnl.rvs(size = N_mae)
    X_mae = np.concatenate([bkgd_mae, sgnl_mae])
    
    lr = make_lr(bkgd, sgnl)
    
    def mae(model_lr):
        nonlocal X_mae
        return np.abs(model_lr(X_mae) - lr(X_mae)).mean()
    return mae

def make_mpe(bkgd, sgnl, N_mpe=10**4):
    bkgd_mpe = bkgd.rvs(size = N_mpe)
    sgnl_mpe = sgnl.rvs(size = N_mpe)
    X_mpe = np.concatenate([bkgd_mpe, sgnl_mpe])
    
    lr = make_lr(bkgd, sgnl)
    
    def mpe(model_lr):
        nonlocal X_mpe
        return np.abs((model_lr(X_mpe) - lr(X_mpe)) / lr(X_mpe)).mean() * 100
    return mae
    
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
        return np.squeeze( (f / (1. - f))**(p - 1))
    return model_lr