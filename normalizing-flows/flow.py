import numpy as np
import os,re
import sklearn.datasets as skd
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, Input
import time
import matplotlib.pyplot as plt

import tensorflow_probability as tfp
tfb = tfp.bijectors
tfd = tfp.distributions


class MLP_ODE(keras.Model):
    """Multi-layer NN ode_fn."""
    def __init__(self, layer_sizes, num_output,name='mlp_ode'):
        super(MLP_ODE, self).__init__()
        self._num_output = num_output
        self._layer_sizes = layer_sizes
        self._modules = []
        
        #Fully connected layers with tanh activation and linear output
        self._modules.append(Input(shape=(1+self._num_output))) #time is part of the inputs
        for i in range(len(self._layer_sizes) - 1):
            self._modules.append(layers.Dense(self._layer_sizes[i],activation='tanh'))
            
        self._modules.append(layers.Dense(self._num_output,activation=None))
        self._model = keras.Sequential(self._modules)
        
    @tf.function
    def call(self, t, data):
        #We add time to each transformation so they are parameterized for time
        t = t*tf.ones([data.shape[0],1])
        inputs = tf.concat([t, data], -1)
        return self._model(inputs)
        
class FFJORD(keras.Model):
    def __init__(self, stacked_mlps, batch_size,num_output,name='FFJORD'):
        super(FFJORD, self).__init__()
        self._num_output=num_output
        self._batch_size = batch_size 
        ode_solve_fn = tfp.math.ode.DormandPrince(atol=1e-5).solve
        #Gaussian noise to trace solver
        trace_augmentation_fn = tfb.ffjord.trace_jacobian_hutchinson
        
        bijectors = []
        for imlp,mlp in enumerate(stacked_mlps):
            ffjord = tfb.FFJORD(
                state_time_derivative_fn=mlp,
                ode_solve_fn=ode_solve_fn,
                trace_augmentation_fn=trace_augmentation_fn,              
            )
            bijectors.append(ffjord)

        #Reverse the bijector order
        self.chain = tfb.Chain(list(reversed(bijectors)))

        self.loss_tracker = keras.metrics.Mean(name="loss")
        #Determien the base distribution
        self.base_distribution = tfp.distributions.MultivariateNormalDiag(
            loc=self._num_output*[0.0], scale_diag=self._num_output*[1.0]
        )
        
        self.flow=self.Transform()
        self._variables = self.flow.variables
        
    @property
    def metrics(self):
        """List of the model's metrics.
        We make sure the loss tracker is listed as part of `model.metrics`
        so that `fit()` and `evaluate()` are able to `reset()` the loss tracker
        at the start of each epoch and at the start of an `evaluate()` call.
        """
        return [self.loss_tracker]
    
    @tf.function
    def call(self, inputs):
        return self.flow.bijector.forward(inputs)
            
    def Transform(self):        
        return tfd.TransformedDistribution(distribution=self.base_distribution, bijector=self.chain)

    
    @tf.function
    def log_loss(self,_x):
        loss = -tf.reduce_mean(self.flow.log_prob(_x))
        return loss

    def prob(self,_x):
        return self.flow.prob(_x)


    
    @tf.function()
    def train_step(self, data):
        #Full shape needs to be given when using tf.dataset
        data.set_shape((self._batch_size,self._num_output))
        with tf.GradientTape() as tape:
            loss = self.log_loss(data)
            
        g = tape.gradient(loss, self.trainable_variables)
        self.optimizer.apply_gradients(zip(g, self.trainable_variables))
        self.loss_tracker.update_state(loss)

        return {"loss": self.loss_tracker.result()}
    
    @tf.function
    def test_step(self, data):
        #Full shape needs to be given when using tf.dataset
        data.set_shape((self._batch_size,self._num_output))
        
        loss = self.log_loss(data)
        self.loss_tracker.update_state(loss)
        return {"loss": self.loss_tracker.result()}


if __name__ == '__main__':

    LR = 1e-2
    NUM_EPOCHS = 20
    STACKED_FFJORDS = 4 #Number of stacked transformations
    NUM_LAYERS = 8 #Hiddden layers per bijector
    NUM_OUTPUT = 2 #Output dimension
    NUM_HIDDEN = 4*NUM_OUTPUT #Hidden layer node size

    
    #Target dataset: half moon
    DATASET_SIZE = 1024 * 8
    BATCH_SIZE = 256 

    moons = skd.make_moons(n_samples=DATASET_SIZE, noise=.06)[0].astype("float32")
    print(moons.shape)
    
    #Stack of bijectors 
    stacked_mlps = []
    for _ in range(STACKED_FFJORDS):
        mlp_model = MLP_ODE(NUM_HIDDEN, NUM_LAYERS, NUM_OUTPUT)
        stacked_mlps.append(mlp_model)

    #Create the model
    model = FFJORD(stacked_mlps,BATCH_SIZE,NUM_OUTPUT)
    model.compile(optimizer=keras.optimizers.Adam(learning_rate=LR))
    
    history = model.fit(
        moons,
        batch_size=BATCH_SIZE,
        epochs=NUM_EPOCHS,
        verbose=1,
    )

    NSAMPLES = 10000
    #Sample the learned distribution
    base_distribution = model.base_distribution.sample(NSAMPLES)
    transformed = model.flow.sample(NSAMPLES)
    
    #Plotting    
    fig = plt.figure(figsize=(8, 6))
    plt.subplot(211)
    plt.scatter(base_distribution[:, 0], base_distribution[:, 1], color="r")
    plt.subplot(212)
    plt.scatter(transformed[:, 0], transformed[:, 1], color="r")
    plot_folder = '../plots'
    if not os.path.exists(plot_folder):
        os.makedirs(plot_folder)
    
    fig.savefig('{}/double_moon.pdf'.format(plot_folder))


