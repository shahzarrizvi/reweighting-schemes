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

#num = 0          # bkgd: normal(-0.1, 1)     sgnl: normal(0.1, 1)
#num = 1          # bkgd: normal(-0.2, 1)     sgnl: normal(0.2, 1)
#num = 2          # bkgd: normal(-0.3, 1)     sgnl: normal(0.3, 1)
#num = 3          # bkgd: normal(-0.4, 1)     sgnl: normal(0.4, 1)
#num = 4          # bkgd: normal(-0.5, 1)     sgnl: normal(0.5, 1)
#num = 5          # bkgd: beta(2, 3)          sgnl: beta(3, 2)
#num = 6          # bkgd: gamma(5, 1)         sgnl: gamma(6, 1)
reps = 20

# Data generation
N = 10**6
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
#bkgd = stats.beta(2, 3)
#sgnl = stats.beta(3, 2)
#bkgd = stats.gamma(5, 1)
#sgnl = stats.gamma(6, 1)

filestr = 'models/univariate/mse_ab_param/set_{}/'.format(num)
mse_filestr = filestr + 'model_{}_{}.h5'

lr = make_lr(bkgd, sgnl)
mae = make_mae(bkgd, sgnl)

m = np.load(filestr + 'm.npy')
s = np.load(filestr + 's.npy')

ps = np.round(np.linspace(-2, 2, 101), 2)

print(num)
# Get model likelihood ratios.
avgs = []
for p in ps:
    print(p)
    lrs = [None] * reps
    params = {'loss':get_mse(p)}
    for i in range(reps):
        model = create_model(**params)
        model.load_weights(mse_filestr.format(p, i))
        lrs[i] = pow_odds_lr(model, p, m, s)
        print(i, end = ' ')
    print()
    
    maes = [mae(lr) for lr in lrs]
    avgs += [np.mean(maes)]
    print(avgs[-1])
    print()

avgs = np.array(avgs)

np.save(filestr + 'norm_0.1', avgs)

fig, ax = plt.subplots(figsize = (8, 8))

plt.plot(ps, avgs, c = 'blue')
plt.legend()

plt.minorticks_on()
plt.tick_params(direction='in', which='both',length=5)
plt.ylabel('Mean Absolute Error')
plt.xlabel(r'$p$')
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
plt.title(r"MSE $A/B$ Parametrization",loc="left",fontsize=20);
plt.savefig('plots/mse_ab_norm_0.1.png', 
            dpi=300, 
            bbox_inches='tight')