import os
import tensorflow as tf
from scipy import stats

# Utility imports
from utils.losses import *
from utils.plotting import *
from utils.training import *

np.random.seed(666) # Need to do more to ensure data is the same across runs.

os.environ["CUDA_VISIBLE_DEVICES"] = "0" # pick a number < 4 on ML4HEP; < 3 on Voltan 
physical_devices = tf.config.list_physical_devices('GPU') 
tf.config.experimental.set_memory_growth(physical_devices[0], True)

num = 2          # bkgd: normal(-0.1, 1)     sgnl: normal(0.1, 1)
#num = 3          # bkgd: beta(2, 3)          sgnl: beta(3, 2)
#num = 4          # bkgd: gamma(5, 1)         sgnl: gamma(6, 1)
#num = 5          # bkgd: normal(-0.2, 1)     sgnl: normal(0.2, 1)
#num = 6          # bkgd: normal(-0.3, 1)     sgnl: normal(0.3, 1)
#num = 7          # bkgd: normal(-0.4, 1)     sgnl: normal(0.4, 1)
#num = 8          # bkgd: normal(-0.5, 1)     sgnl: normal(0.5, 1)
reps = 20

# Data generation
N = 10**6
#bkgd = stats.beta(2, 3)
#sgnl = stats.beta(3, 2)
#bkgd = stats.gamma(5, 1)
#sgnl = stats.gamma(6, 1)
bkgd = stats.norm(-0.1, 1)
sgnl = stats.norm(0.1, 1)
#bkgd = stats.norm(-0.2, 1)
#sgnl = stats.norm(0.2, 1)
#bkgd = stats.norm(-0.3, 1)
#sgnl = stats.norm(0.3, 1)
#bkgd = stats.norm(-0.4, 1)
#sgnl = stats.norm(0.4, 1)
#bkgd = stats.norm(-0.5, 1)
#sgnl = stats.norm(0.5, 1)

filestr = 'models/univariate/sqr_ab_param/set_{}/'.format(num)
sqr_filestr = filestr + '/linear/model_{}_{}.h5'
exp_filestr = filestr + '/exp/model_{}_{}.h5'

lr = make_lr(bkgd, sgnl)
mae = make_mae(bkgd, sgnl)

m = np.load(filestr + 'm.npy')
s = np.load(filestr + 's.npy')
#m = (bkgd.mean() + sgnl.mean()) / 2
#s = ((bkgd.var() + sgnl.var()) / 2 + np.var([bkgd.mean(), sgnl.mean()]))**0.5

rs = np.sort(np.append(np.round(np.linspace(-2, 2, 81), 2),
                       np.round(np.linspace(-0.05, 0.05, 26), 3)[1:-1]))

print(num)
# Get model likelihood ratios.
sqr_avgs = []
exp_avgs = []
for r in rs:
    print(r)
    sqr_lrs = [None] * reps
    exp_lrs = [None] * reps
    sqr_params = {'loss': get_sqr(r), 'output':'relu'}
    exp_params = {'loss': get_exp_sqr(r), 'output':'linear'}
    for i in range(reps):
        sqr_model = create_model(**sqr_params)
        exp_model = create_model(**exp_params)
        sqr_model.load_weights(sqr_filestr.format(r, i))
        exp_model.load_weights(exp_filestr.format(r, i))
        sqr_lrs[i] = pow_lr(sqr_model, r, m, s)
        exp_lrs[i] = exp_pow_lr(exp_model, r, m, s)
        print(i, end = ' ')
    print()
    
    sqr_maes = [mae(lr) for lr in sqr_lrs]
    exp_maes = [mae(lr) for lr in exp_lrs]
    
    sqr_avgs += [np.mean(sqr_maes)]
    exp_avgs += [np.mean(exp_maes)]
    print(sqr_avgs[-1], '\t', exp_avgs[-1])
    print()

sqr_avgs = np.array(sqr_avgs)
exp_avgs = np.array(exp_avgs)

np.save(filestr + 'lin_shift', sqr_avgs)
np.save(filestr + 'exp_shift', exp_avgs)

fig, ax = plt.subplots(figsize = (8, 8))

plt.plot(rs, sqr_avgs, c='blue', label='linear')
plt.plot(rs, exp_avgs, c='red', label='exponential')
plt.legend()

plt.minorticks_on()
plt.tick_params(direction='in', which='both',length=5)
plt.ylabel('Mean Absolute Error')
plt.xlabel(r'$r$')
#plt.ylim(0, 0.2)

#plt.title(r"$r_{\rm{sgnl}}="+str(3)+r", s_{\rm{sgnl}}="+str(2)+ r"$" + 
#          "\n" + r"$r_{\rm{bkgd}}="+str(2)+r", s_{\rm{bkgd}}=" + str(3) + r"$",
#          loc="right",
#          fontsize=20);
#plt.title(r"$r_{\rm{sgnl}}="+str(6)+r", r_{\rm{bkgd}}="+str(6)+ r"$",
#          loc="right",
#          fontsize=20);
#plt.title(r"$\mu_{\rm{sgnl}}="+str(0.2)+r", \mu_{\rm{bkgd}}="+str(-0.2)+r"$",
#          loc="right",
#          fontsize=20);
plt.title(r"SQR $A/B$ Parametrization",loc="left",fontsize=20);
plt.savefig('plots/sqr_ab_norm_0.1.png'.format(num), 
            dpi=1200, 
            bbox_inches='tight')