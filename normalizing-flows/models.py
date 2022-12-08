import tensorflow as tf
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Lambda, Dense, Flatten
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras import backend as K
from tensorflow.keras.layers import Input
import numpy as np
from sklearn.metrics import roc_curve, auc
import flow

tf.random.set_seed(1234)

def _AE(NFEAT,LAYERSIZE,ENCODESIZE):
    inputs = Input((NFEAT, ))
    layer = Dense(LAYERSIZE[0], activation='relu', use_bias=False)(inputs)
    #Encoder
    for il in range(1,len(LAYERSIZE)):
        layer = Dense(LAYERSIZE[il], activation='relu', use_bias=False)(layer)


    layer = Dense(ENCODESIZE, activation='linear')(layer)
    #Decoder
    for il in range(len(LAYERSIZE)):
        layer = Dense(LAYERSIZE[len(LAYERSIZE)-il-1], activation='relu')(layer)
    #layer = Dropout(0.25)(layer)
    outputs = Dense(NFEAT, activation='linear')(layer)

    return inputs,outputs

def AE(reference,
       checkpoint_name,
       val_data=[],
       ndim=2,
       LAYERSIZE=[50,20,10],ENCODESIZE=1,
       load=False,
       lr=1e-3,max_epoch=500):
    '''
    Inputs:
    reference: Background events used for training
    val_data: validation data to estimate the training performance
    checkpoint_name: Name of the folder to store trained models
    Outputs:
    fpr: false positive rate calculated using the reconstruction loss of the autoencoder
    tpr: true positive rate calculated using the reconstruction loss of the autoencoder
    '''
    K.clear_session()
    opt = tf.keras.optimizers.Adam(learning_rate=lr)
    callbacks = [tf.keras.callbacks.EarlyStopping(monitor='loss', patience=10)]

    inputs,outputs = _AE(ndim,LAYERSIZE,ENCODESIZE)
    model = Model(inputs, outputs)
    if load:
        model.load_weights('../checkpoint/{}.hdf5'.format(checkpoint_name))
        val_s,val_b=val_data
        AE_b = model.predict(val_b,batch_size=1000)
        mse_AE_b = np.mean(np.square(AE_b - val_b),-1)
        AE_s = model.predict(val_s,batch_size=1000)
        mse_AE_s = np.mean(np.square(AE_s - val_s),-1)
        mse = np.concatenate([mse_AE_b,mse_AE_s],0)
        label = np.concatenate([np.zeros(mse_AE_b.shape[0]),np.ones(mse_AE_s.shape[0])],0)
        
        fpr, tpr, _ = roc_curve(label,mse, pos_label=1)    
        print("AE AUC: {}".format(auc(fpr, tpr)))

        return fpr,tpr

    else:
        
        model.compile(loss="mse", optimizer=opt, metrics=['accuracy'])
        checkpoint = ModelCheckpoint('../checkpoint/{}.hdf5'.format(checkpoint_name),
                                 save_best_only=True,mode='auto',period=1,save_weights_only=True)

        hist = model.fit(reference,reference,
                         epochs=max_epoch, 
                         callbacks=callbacks+[checkpoint],
                         validation_split=0.2,
                         batch_size=10000)



def FFJORD(reference,
           checkpoint_name,
           val_data=[],
           LAYERSIZE=[50,20,10],STACKSIZE=2,
           load=False,
           ndim=2,
           lr=1e-3,max_epoch=500):
    '''
    Inputs:
    reference: Background events used for training
    val_data: validation data to estimate the training performance
    checkpoint_name: Name of the folder to store trained models
    Outputs:
    fpr: false positive rate calculated using the reconstruction loss of the autoencoder
    tpr: true positive rate calculated using the reconstruction loss of the autoencoder
    '''
    K.clear_session()
    opt = tf.keras.optimizers.Adam(learning_rate=lr)
    callbacks = [tf.keras.callbacks.EarlyStopping(monitor='loss', patience=10)]
    checkpoint = ModelCheckpoint('../checkpoint/{}'.format(checkpoint_name),
                                 save_best_only=True,mode='auto',period=1,save_weights_only=True)
    
    stacked_mlps = []
    for _ in range(STACKSIZE):
        mlp_model = flow.MLP_ODE(LAYERSIZE, num_output=ndim)
        stacked_mlps.append(mlp_model)

    #Create the model

    batch_size = 10000
    model = flow.FFJORD(stacked_mlps,batch_size=batch_size,num_output=ndim)
    model.compile(optimizer=opt)
    hist = model.fit(reference,
                     epochs=max_epoch, 
                     callbacks=callbacks+[checkpoint],
                     validation_split=0.2,
                     batch_size=batch_size)
    if load:
        model.load_weights('../checkpoint/{}'.format(checkpoint_name,save_format='tf')).expect_partial()
        val_s,val_b=val_data
        prob_b = np.clip(-model.prob(val_b),-1e4,1e4)
        prob_s = np.clip(-model.prob(val_s),-1e4,1e4)
        prob = np.concatenate([prob_b,prob_s],0)
        label = np.concatenate([np.zeros(prob_b.shape[0]),np.ones(prob_s.shape[0])],0)
        
        fpr, tpr, _ = roc_curve(label,prob, pos_label=1)    
        print("AE AUC: {}".format(auc(fpr, tpr)))
        return fpr,tpr
    else:
        model.compile(optimizer=opt)
        hist = model.fit(reference,
                         epochs=max_epoch, 
                         callbacks=callbacks+[checkpoint],
                         validation_split=0.2,
                         batch_size=batch_size)

    



def _classifier(NFEAT,LAYERSIZE):
    inputs = Input((NFEAT, ))
    layer = Dense(LAYERSIZE[0], activation='relu')(inputs)
    layer = Dense(LAYERSIZE[0], activation='relu')(layer)
    for il in range(1,len(LAYERSIZE)):
        layer = Dense(LAYERSIZE[il], activation='relu')(layer)
        layer = Dense(LAYERSIZE[il], activation='relu')(layer)
    outputs = Dense(1, activation='sigmoid')(layer)
    return inputs,outputs



def CWoLa(data,reference,
          checkpoint_name,
          val_data=[],
          LAYERSIZE=[50,20,10],
          ndim=2,
          load=False,
          lr=1e-3,max_epoch=500):
    '''
    Inputs: 
    data: full dataset containing both signal and background events
    reference: Background only events
    val_data: validation data to estimate the training performance
    checkpoint_name: Name of the folder to store trained models

    Outputs:
    fpr: false positive rate calculated using the ratio ps/pb
    tpr: true positive rate calculated using the ratio ps/pb
    '''
    K.clear_session()
    opt = tf.keras.optimizers.Adam(learning_rate=lr)
    callbacks = [tf.keras.callbacks.EarlyStopping(monitor='loss', patience=20)]

    inputs,outputs = _classifier(ndim,LAYERSIZE)
    model = Model(inputs, outputs)
    if load:
        model.load_weights('../checkpoint/{}.hdf5'.format(checkpoint_name))
        val_s,val_b=val_data
        eps = 1e-5
        pred_s = model.predict(val_s,batch_size=1000)
        pred_s = pred_s/(1-pred_s + eps)
        pred_b = model.predict(val_b,batch_size=1000)
        pred_b = pred_b/(1-pred_b + eps)
        
        pred = np.concatenate([pred_b,pred_s],0)
        label = np.concatenate([np.zeros(pred_b.shape[0]),np.ones(pred_s.shape[0])],0)
    
        fpr, tpr, _ = roc_curve(label,pred, pos_label=1)    
        print("CWoLa AUC: {}".format(auc(fpr, tpr)))
        return fpr,tpr

    else:
        model.compile(loss="binary_crossentropy", optimizer=opt, metrics=['accuracy'])
        checkpoint = ModelCheckpoint('../checkpoint/{}.hdf5'.format(checkpoint_name),
                                     save_best_only=True,mode='auto',period=1,save_weights_only=True)

        train_data = np.concatenate([data,reference],0)
        label = np.concatenate([np.ones(data.shape[0]),np.zeros(reference.shape[0])],0)
        X_train, X_test, y_train, y_test = train_test_split(train_data, label, test_size=0.2)

        hist = model.fit(X_train,y_train,
                         epochs=max_epoch, 
                         callbacks=callbacks+[checkpoint],
                         validation_data=(X_test,y_test),
                         batch_size=50000)
    
