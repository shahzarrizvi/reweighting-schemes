import numpy as np
import tensorflow.keras.backend as K
#from scipy import stats

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
    return -((y_true) * K.log(y_pred**2 + K.epsilon()) + 
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