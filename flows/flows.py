import numpy as np
from scipy import stats

import tensorflow.compat.v2 as tf
import tensorflow_probability as tfp
import sonnet as snt
import wandb
from sklearn.ensemble import AdaBoostClassifier
from sklearn import metrics
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
    trace_augmentation_fn = tfb.ffjord.trace_jacobian_hutchinson

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
         batch_size = 2**6, 
         lr = 1e-2, 
         num_epochs = 100):
    
    n, d = data.shape
    
    ffjords = create_ffjords(num_ffjords, num_hidden, num_layers, d)
    base_dist = tfd.MultivariateNormalDiag(d*[0], d*[1])
    tsfm_dist = tfd.TransformedDistribution(distribution = base_dist, bijector = ffjords)
    
    ckpt = tf.train.Checkpoint(tsfm_dist)
    manager = tf.train.CheckpointManager(ckpt, ckpt_path, max_to_keep=10)
    
    dataset = tf.data.Dataset.from_tensor_slices(data.astype(np.float32)) \
                .prefetch(tf.data.experimental.AUTOTUNE) \
                .cache() \
                .shuffle(n) \
                .batch(batch_size)
    
    learning_rate = tf.Variable(lr, trainable=False)
    optimizer = snt.optimizers.Adam(learning_rate)

    i = 1
    for epoch in tqdm.trange(num_epochs // 2):
        for batch in dataset:
            loss = train_step(tsfm_dist, optimizer, batch)
            
            #if i % 1000 == 0:
            #    ckpt.save(ckpt_path)
            manager.save()
            i += 1
    #ckpt.save(ckpt_path)
    
    return tsfm_dist


def make_target(d = 2, 
                num_ffjords = 4,
                num_hidden = 8, 
                num_layers = 3):
    ffjords = create_ffjords(num_ffjords, num_hidden, num_layers, d)
    base_dist = tfd.MultivariateNormalDiag(d*[0], d*[1])
    tsfm_dist = tfd.TransformedDistribution(distribution = base_dist, bijector = ffjords)
    return tsfm_dist

def calculate_auc(data, flow):
    n, d = data.shape 
    
    X_bkgd = flow
    X_sgnl = data
    
    np.random.seed(666)
    
    # Create full dataset; randomly sample points from Data or Flow with chance 1/2.
    y = stats.bernoulli.rvs(0.5, size = n).astype('float32')
    X = np.zeros_like(X_bkgd)

    X[y == 0] = X_bkgd[y == 0]
    X[y == 1] = X_sgnl[y == 1]

    # Take 70% of data to be training data.
    N_trn = int(0.7*n)
    trn_idx = np.random.choice(range(n), N_trn, replace = False)
    tst_idx = [n for n in range(n) if n not in trn_idx]

    X_trn = X[trn_idx]
    y_trn = y[trn_idx]

    X_tst = X[tst_idx]
    y_tst = y[tst_idx]
    
    ab_clf = AdaBoostClassifier()
    ab_clf.fit(X_trn, y_trn)
    
    y_hat = ab_clf.predict_proba(X_tst)[:, 1]
    return metrics.roc_auc_score(y_tst, y_hat)

def distributed_flow(data,
         ckpt_path = 'ckpt',
         num_ffjords = 4, 
         num_hidden = 8, 
         num_layers = 3, 
         batch_size = 2**6, 
         lr = 1e-2, 
         num_epochs = 100):
    
    run = wandb.init(project="four-vector-flows",
                     config={"learning_rate": lr,
                             "epochs": num_epochs,
                             "batch_size": batch_size
                            }
                    )
    
    n, d = data.shape
    
    strategy = tf.distribute.MirroredStrategy(devices=["/gpu:0", "/gpu:1", "/gpu:2", "/gpu:3"])
    
    with strategy.scope():
        ffjords = create_ffjords(num_ffjords, num_hidden, num_layers, d)
        base_dist = tfd.MultivariateNormalDiag(d*[0], d*[1])
        tsfm_dist = tfd.TransformedDistribution(distribution = base_dist, bijector = ffjords)

    ckpt = tf.train.Checkpoint(tsfm_dist)

    dataset = tf.data.Dataset.from_tensor_slices(data.astype(np.float32)) \
                .prefetch(tf.data.experimental.AUTOTUNE) \
                .cache() \
                .shuffle(n) \
                .batch(batch_size)

    learning_rate = tf.Variable(lr, trainable=False)
    optimizer = snt.optimizers.Adam(learning_rate)

    i = 1
    for epoch in tqdm.trange(num_epochs // 2):
        for batch in dataset:
            loss = train_step(tsfm_dist, optimizer, batch)
            wandb.log({'loss': loss})

            if i % 1000 == 0:
                ckpt.save(ckpt_path)
            i += 1
    ckpt.save(ckpt_path)
    
    return tsfm_dist