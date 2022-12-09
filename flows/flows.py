import numpy as np
from scipy import stats

import tensorflow.compat.v2 as tf
import tensorflow_probability as tfp
import sonnet as snt
tf.enable_v2_behavior()

tfb = tfp.bijectors
tfd = tfp.distributions

import tqdm as tqdm

class MLP_ODE(snt.Module):
    """Multi-layer NN ode_fn."""
    def __init__(self, num_hidden, num_layers, num_output, name='mlp_ode'):
        super(MLP_ODE, self).__init__(name=name)
        self._num_hidden = num_hidden
        self._num_output = num_output
        self._num_layers = num_layers
        self._modules = []
        for _ in range(self._num_layers - 1):
            self._modules.append(snt.Linear(self._num_hidden))
            self._modules.append(tf.math.tanh)
        self._modules.append(snt.Linear(self._num_output))
        self._model = snt.Sequential(self._modules)

    def __call__(self, t, inputs):
        inputs = tf.concat([tf.broadcast_to(t, inputs.shape), inputs], -1)
        return self._model(inputs)

def create_ffjords(num_ffjords = 4,
                   num_hidden = 8, 
                   num_layers = 3, 
                   d = 2):
    solver = tfp.math.ode.DormandPrince(atol=1e-5)
    ode_solve_fn = solver.solve
    trace_augmentation_fn = tfb.ffjord.trace_jacobian_exact

    bijectors = []
    for _ in range(num_ffjords):
        mlp_model = MLP_ODE(num_hidden, num_layers, d)
        next_ffjord = tfb.FFJORD(state_time_derivative_fn = mlp_model,
                                 ode_solve_fn = ode_solve_fn,
                                 trace_augmentation_fn = trace_augmentation_fn)
        bijectors.append(next_ffjord)
    
    ffjords = tfb.Chain(bijectors[::-1])
    return ffjords

@tf.function
def train_step(tsfm_dist, optimizer, target_sample):
    with tf.GradientTape() as tape:
        loss = -tf.reduce_mean(tsfm_dist.log_prob(target_sample))
    variables = tape.watched_variables()
    gradients = tape.gradient(loss, variables)
    optimizer.apply(gradients, variables)
    return loss

def flow(data,
         ckpt_path = 'ckpt',
         num_ffjords = 4, 
         num_hidden = 8, 
         num_layers = 3, 
         batch_size = 2**10, 
         lr = 1e-2, 
         num_epochs = 80):
    n, d = data.shape
    
    ffjords = create_ffjords(num_ffjords, num_hidden, num_layers, d)
    base_dist = tfd.MultivariateNormalDiag(d*[0], d*[1])
    tsfm_dist = tfd.TransformedDistribution(distribution = base_dist, bijector = ffjords)
    
    #ckpt = tf.train.Checkpoint(tsfm_dist)
    
    dataset = tf.data.Dataset.from_tensor_slices(data.astype(np.float32)) \
                .prefetch(tf.data.experimental.AUTOTUNE) \
                .cache() \
                .shuffle(n) \
                .batch(batch_size)
    
    learning_rate = tf.Variable(lr, trainable=False)
    optimizer = snt.optimizers.Adam(learning_rate)

    for epoch in tqdm.trange(num_epochs // 2):
        for batch in dataset:
            _ = train_step(tsfm_dist, optimizer, batch)
            #ckpt.save(ckpt_path)
    
    return tsfm_dist