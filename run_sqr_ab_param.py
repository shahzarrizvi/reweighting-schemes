# General imports
import os
import tensorflow as tf
from scipy import stats

# Utility imports
from utils.losses import *
from utils.plotting import *
from utils.training import *

np.random.seed(666) # Need to do more to ensure data is the same across runs.

os.environ["CUDA_VISIBLE_DEVICES"] = "2" # pick a number < 4 on ML4HEP; < 3 on Voltan 
physical_devices = tf.config.list_physical_devices('GPU') 
tf.config.experimental.set_memory_growth(physical_devices[0], True)

# Experiment parameters
#num = 3    # bkgd: beta(2, 3)          sgnl: beta(3, 2)
#num = 4    # bkgd: gamma()             sgnl: gamma()
#num = 5    # bkgd: normal(-0.2, 1)     sgnl: normal(0.2, 1)
#num = 6    # bkgd: normal(-0.3, 1)     sgnl: normal(0.3, 1)
#num = 7    # bkgd: normal(-0.4, 1)     sgnl: normal(0.4, 1)
num = 8    # bkgd: normal(-0.5, 1)     sgnl: normal(0.5, 1)
reps = 20

# Data generation
N = 10**6
#bkgd = stats.beta(2, 3)
#sgnl = stats.beta(3, 2)
#bkgd = stats.gamma(5, 1)
#sgnl = stats.gamma(6, 1)
#bkgd = stats.norm(-0.2, 1)
#sgnl = stats.norm(0.2, 1)
#bkgd = stats.norm(-0.3, 1)
#sgnl = stats.norm(0.3, 1)
#bkgd = stats.norm(-0.4, 1)
#sgnl = stats.norm(0.4, 1)
bkgd = stats.norm(-0.5, 1)
sgnl = stats.norm(0.5, 1)

lr = make_lr(bkgd, sgnl)
mae = make_mae(bkgd, sgnl)
data = make_data(bkgd, sgnl, N) + [N]

rs = np.sort(np.append(np.round(np.linspace(-2, 2, 81), 2),
                       np.round(np.linspace(-0.05, 0.05, 26), 3)[1:-1]))
rs = rs[rs >= 0.8]
sqr_filestr = 'models/sqr_ab_param/set_' + str(num) + '/linear/model_{}_{}.h5'
exp_filestr = 'models/sqr_ab_param/set_' + str(num) + '/exp/model_{}_{}.h5'

sqr_lrs = {}
exp_lrs = {}
for r in rs:
    print('===================================================\n{}'.format(r))
    sqr_lrs[r] = [None] * reps
    exp_lrs[r] = [None] * reps
    sqr_params = {'loss': get_sqr(r), 'output':'relu'}
    exp_params = {'loss': get_exp_sqr(r), 'output':'linear'}
    for i in range(reps):
        print(i, end = '\t')
        sqr_model = train(data, **sqr_params)
        exp_model = train(data, **exp_params)
        sqr_lrs[r][i] = pow_lr(sqr_model, r)
        exp_lrs[r][i] = exp_pow_lr(exp_model, r)
        print()
        sqr_model.save_weights(sqr_filestr.format(r, i))
        exp_model.save_weights(exp_filestr.format(r, i))