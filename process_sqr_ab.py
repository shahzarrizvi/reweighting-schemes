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
bkgd = stats.norm(-0.1, 1)    # num = 2
sgnl = stats.norm(0.1, 1)
#bkgd = stats.beta(2, 3)    # num = 3
#sgnl = stats.beta(3, 2)
#bkgd = stats.gamma(5, 1)    # num = 4
#sgnl = stats.gamma(6, 1)
#bkgd = stats.norm(-0.2, 1)    # num = 5
#sgnl = stats.norm(0.2, 1)
#bkgd = stats.norm(-0.3, 1)    # num = 6
#sgnl = stats.norm(0.3, 1)
#bkgd = stats.norm(-0.4, 1)    # num = 6
#sgnl = stats.norm(0.4, 1)
#bkgd = stats.norm(-0.5, 1)    # num = 8
#sgnl = stats.norm(0.5, 1)

filestr = 'models/univariate/sqr_ab_param/set_{}/'.format(num)
lin_filestr = filestr + 'linear/model_{}_{}.h5'
exp_filestr = filestr + 'exp/model_{}_{}.h5'

lr = make_lr(bkgd, sgnl)
mae = make_mae(bkgd, sgnl)

m = np.load(filestr + 'm.npy')    # (bkgd.mean() + sgnl.mean()) / 2
s = np.load(filestr + 's.npy')    # ((bkgd.var() + sgnl.var()) / 2 + np.var([bkgd.mean(), sgnl.mean()]))**0.5

rs = np.sort(np.append(np.round(np.linspace(-2, 2, 81), 2),
                       np.round(np.linspace(-0.05, 0.05, 26), 3)[1:-1]))

print(num)
# Get model likelihood ratios.
lin_avgs = []
exp_avgs = []
for r in rs:
    print(r)
    print(p)
    lin_lrs = [None] * reps
    exp_lrs = [None] * reps
    lin_params = {'loss': get_sqr(r), 'output':'relu'}
    exp_params = {'loss': get_exp_sqr(r), 'output':'linear'}
    for i in range(reps):
        lin_model = create_model(**lin_params)
        exp_model = create_model(**exp_params)
        lin_model.load_weights(lin_filestr.format(r, i))
        exp_model.load_weights(exp_filestr.format(r, i))
        lin_lrs[i] = pow_lr(lin_model, r, m, s)
        exp_lrs[i] = pow_exp_lr(exp_model, r, m, s)
        print(i, end = ' ')
    print()
    
    lin_maes = [mae(lr) for lr in lin_lrs]
    exp_maes = [mae(lr) for lr in exp_lrs]
    
    lin_avgs += [np.mean(lin_maes)]
    exp_avgs += [np.mean(exp_maes)]
    print(lin_avgs[-1], '\t', exp_avgs[-1])
    print()

lin_avgs = np.array(lin_avgs)
exp_avgs = np.array(exp_avgs)

np.save(filestr + 'lin', lin_avgs)
np.save(filestr + 'exp', exp_avgs)

fig, ax = plt.subplots(figsize = (8, 8))

plt.plot(rs, lin_avgs, c='blue', label=r'$f$')
plt.plot(rs, exp_avgs, c='red', label=r'\exp{f}')
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
plt.savefig('plots/sqr_ab_norm_0.1.png', 
            dpi=300, 
            bbox_inches='tight')