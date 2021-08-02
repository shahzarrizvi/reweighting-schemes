import numpy as np
from sklearn.model_selection import train_test_split

import tensorflow as tf
import tensorflow.keras
import tensorflow.keras.backend as K
from tensorflow.keras.layers import Dense, Input
from tensorflow.keras.models import Model
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import EarlyStopping

# from NN (DCTR)
def reweight(model, events):
    f = model.predict(events, batch_size=1000)
    weights = f / (1. - f)
    return np.squeeze(weights)

# Binary crossentropy for classifying two samples with weights
# Weights are "hidden" by zipping in y_true (the labels)

def weighted_binary_crossentropy(y_true, y_pred):
    weights = tf.gather(y_true, [1], axis=1) # event weights
    y_true = tf.gather(y_true, [0], axis=1) # actual y_true for loss
    
    # Clip the prediction value to prevent NaN's and Inf's
    epsilon = K.epsilon()
    y_pred = K.clip(y_pred, epsilon, 1. - epsilon)

    t_loss = -weights * ((y_true) * K.log(y_pred) +
                         (1 - y_true) * K.log(1 - y_pred))
    
    return K.mean(t_loss)

def unifold(iterations,
            sim_truth,
            sim_reco,
            data_reco,
            sim_truth_weights_MC=None,
            sim_reco_weights_MC=None,
            data_reco_weights_MC=None,
            dummyval=None,
            average_weight=True,
            model_init_filepath=None,
            verbose=0):
    """    
    Arguments:

    iterations: number of iterations (integer)

    sim_truth: MC Truth observable as Numpy array

    sim_reco: MC Reco observable as Numpy array

    data_reco: Data Reco observable 
                        to be unfolded as Numpy arrays

    sim_truth_weights_MC: (optional) initial MC weights for truth simulation sample

    sim_reco_weights_MC: (optional) initial MC weights for reco simulation sample

    data_reco_weights_MC: (optional) initial MC weights for reco "data" sample
    
    dummyval: (optional) flag value for fake or efficiency factors

    verbose: (optional) (integer) 0 supresses all output; 1 is normal output

    Returns:

    weights: A Numpy array of weights to reweight distribution of 
                sim_truth to the unfolded distribution of data_reco
                (MC weights still need to be applied)

    model: The model used to calculate those weights
    """

    if sim_truth_weights_MC is None:
        sim_truth_weights_MC = np.ones(len(sim_truth))

    if sim_reco_weights_MC is None:
        sim_reco_weights_MC = np.ones(len(sim_reco))

    if data_reco_weights_MC is None:
        data_reco_weights_MC = np.ones(len(data_reco))

    # initialize training data and weights
    labels_sim = np.zeros(len(sim_reco))
    labels_data = np.ones(len(data_reco))

    sim_reco_mask = sim_reco!=-99
    sim_truth_mask = sim_truth!=-99
    data_reco_mask = data_reco!=-99

    xvals_1 = np.concatenate((sim_reco[sim_reco_mask], data_reco[data_reco_mask]))
    yvals_1 = np.concatenate((labels_sim[sim_reco_mask], labels_data[data_reco_mask]))

    if dummyval is not None:
        sim_mask = sim_reco_mask * sim_truth_mask
        xvals_1b = np.concatenate(
            (sim_truth[sim_mask],
             sim_truth[sim_mask]))
        yvals_1b = np.concatenate(
            (labels_sim[sim_mask],
             labels_sim[sim_mask]+1))
        xvals_2b = np.concatenate((sim_reco[sim_mask],
                                   sim_reco[sim_mask]))
        yvals_2b = np.concatenate(
            (labels_sim[sim_mask],
             labels_sim[sim_mask]+1))

    xvals_2 = np.concatenate(
        (sim_truth[sim_truth_mask], sim_truth[sim_truth_mask]))
    yvals_2 = np.concatenate((labels_sim[sim_truth_mask],
                              (labels_sim[sim_truth_mask] + 1.)))



    weights = np.empty(shape=(iterations, 2, len(sim_truth)))
    # shape = (iteration, step, event)

    weights_pull = np.ones_like(sim_truth_weights_MC)
    weights_push = np.ones_like(sim_reco_weights_MC)

    # initialize model
    inputs = Input((1, ))
    hidden_layer_1 = Dense(50, activation='relu')(inputs)
    hidden_layer_2 = Dense(50, activation='relu')(hidden_layer_1)
    hidden_layer_3 = Dense(50, activation='relu')(hidden_layer_2)
    outputs = Dense(1, activation='sigmoid')(hidden_layer_3)

    model = Model(inputs=inputs, outputs=outputs)
    
    if model_init_filepath is not None:
        model.load_weights(model_init_filepath)

    earlystopping = EarlyStopping(patience=10,
                                  verbose=verbose,
                                  restore_best_weights=True)

    for i in range(iterations):
        if verbose == 1:
            print("\nITERATION: {}\n".format(i + 1))

        # STEP 1: classify MC Reco (which is reweighted by weights_push) to Data Reco
        # this reweights reweighted MC Reco --> Data Reco
        if verbose == 1:
            print("STEP 1\n")

        # iterative weights for MC Reco, initial weights for Data Reco
        weights_1 = np.concatenate(
            ((weights_push * sim_reco_weights_MC)[sim_reco_mask],
             data_reco_weights_MC[data_reco_mask]))

        X_train_1, X_test_1, Y_train_1, Y_test_1, w_train_1, w_test_1 = train_test_split(
            xvals_1, yvals_1, weights_1)

        # zip ("hide") the weights with the labels
        Y_train_1 = np.stack((Y_train_1, w_train_1), axis=1)
        Y_test_1 = np.stack((Y_test_1, w_test_1), axis=1)

        # compile and train model
        model.compile(loss=weighted_binary_crossentropy,
                      optimizer='Adam',
                      metrics=['accuracy'])
        model.fit(X_train_1,
                  Y_train_1,
                  epochs=1000,
                  batch_size=10000,
                  validation_data=(X_test_1, Y_test_1),
                  callbacks=[earlystopping],
                  verbose=verbose)

        # calculate, normalize, and clip weights
        weights_pull = weights_push * reweight(model, sim_reco)

        if dummyval is not None:
            if average_weight:
                # STEP 1b: Efficiency factors (estimate <w|x_true>)
                if verbose == 1:
                    print("\nSTEP 1b\n")
                weights_pull = np.nan_to_num(weights_pull)
                weights_pull = np.clip(weights_pull, -100, 100)
                weights_1b = np.concatenate(
                    (sim_truth_weights_MC[sim_mask],
                     (weights_pull *
                      sim_truth_weights_MC)[sim_mask]))

                X_train_1b, X_test_1b, Y_train_1b, Y_test_1b, w_train_1b, w_test_1b = train_test_split(
                    xvals_1b, yvals_1b, weights_1b)

                # zip ("hide") the weights with the labels
                Y_train_1b = np.stack((Y_train_1b, w_train_1b), axis=1)
                Y_test_1b = np.stack((Y_test_1b, w_test_1b), axis=1)

                # compile and train model
                model.compile(loss=weighted_binary_crossentropy,
                              optimizer='Adam',
                              metrics=['accuracy'])
                model.fit(X_train_1b,
                          Y_train_1b,
                          epochs=1000,
                          batch_size=10000,
                          validation_data=(X_test_1b, Y_test_1b),
                          callbacks=[earlystopping],
                          verbose=verbose)

                weights_pull[np.invert(sim_reco_mask)] = reweight(
                    model, sim_truth[np.invert(sim_reco_mask)])
            else:
                weights_pull[np.invert(sim_reco_mask)] = 1
        weights_pull = np.nan_to_num(weights_pull)
        weights_pull = np.clip(weights_pull, -100, 100)
        weights_pull /= np.mean(weights_pull)
        weights[i, :1, :] = weights_pull

        # STEP 2: classify nominal MC Truth to reweighted (by weights_pull) MC Truth
        # this reweights nominal MC Truth --> reweighted MC Truth
        if verbose == 1:
            print("\nSTEP 2\n")

        # MC weights for MC Truth, pulled weights for (reweighted) MC Truth
        weights_2 = np.concatenate(
            (sim_truth_weights_MC[sim_truth_mask],
             (weights_pull * sim_truth_weights_MC)[sim_truth_mask]))

        X_train_2, X_test_2, Y_train_2, Y_test_2, w_train_2, w_test_2 = train_test_split(
            xvals_2, yvals_2,
            weights_2)

        # zip ("hide") the weights with the labels
        Y_train_2 = np.stack((Y_train_2, w_train_2), axis=1)
        Y_test_2 = np.stack((Y_test_2, w_test_2), axis=1)

        # compile and train model
        model.compile(loss=weighted_binary_crossentropy,
                      optimizer='Adam',
                      metrics=['accuracy'])
        model.fit(X_train_2,
                  Y_train_2,
                  epochs=1000,
                  batch_size=10000,
                  validation_data=(X_test_2, Y_test_2),
                  callbacks=[earlystopping],
                  verbose=verbose)

        # calculate, normalize, and clip weights
        weights_push = reweight(model, sim_truth)

        if dummyval is not None:
            if average_weight:
                # STEP 2b: Fake factors (estimate <w|x_reco>)
                if verbose == 1:
                    print("\nSTEP 2b\n")
                weights_push = np.nan_to_num(weights_push)
                weights_push = np.clip(weights_push, -100, 100)
                weights_2b = np.concatenate(
                    (sim_reco_weights_MC[sim_mask],
                     (weights_push *
                      sim_reco_weights_MC)[sim_mask]))

                X_train_2b, X_test_2b, Y_train_2b, Y_test_2b, w_train_2b, w_test_2b = train_test_split(
                    xvals_2b, yvals_2b, weights_2b)
                # zip ("hide") the weights with the labels
                Y_train_2b = np.stack((Y_train_2b, w_train_2b), axis=1)
                Y_test_2b = np.stack((Y_test_2b, w_test_2b), axis=1)

                # compile and train model
                model.compile(loss=weighted_binary_crossentropy,
                              optimizer='Adam',
                              metrics=['accuracy'])
                model.fit(X_train_2b,
                          Y_train_2b,
                          epochs=1000,
                          batch_size=10000,
                          validation_data=(X_test_2b, Y_test_2b),
                          callbacks=[earlystopping],
                          verbose=verbose)

                weights_push[np.invert(sim_truth_mask)] = reweight(
                    model, sim_reco[np.invert(sim_truth_mask)])
            else:
                # STEP 2b: Fake factors (take the prior weight w=1)
                weights_push[np.invert(sim_truth_mask)] = 1
        weights_push = np.nan_to_num(weights_push)
        weights_push = np.clip(weights_push, -100, 100)
        weights_push /= np.mean(weights_push)
        weights[i, 1:2, :] = weights_push
    return weights, model