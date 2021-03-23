"""
plot_emd.py

This one makes a couple of plots from the pre-computed EMD values.

Matt LeBlanc (CERN) 2021
<matt.leblanc@cern.ch>
"""

import numpy as np
import h5py
import argparse
import matplotlib.pyplot as plt

#import energyflow as ef

# get EMD values
infile = "emd_vals.h5"
h5f = h5py.File(infile,'r')
emd_vals = []
emd_vals.extend(h5f['emd_vals'][:])

##############################
# EMD distribution
##############################

#print(emd_vals)
plt.hist(emd_vals, bins=100, range=(0.0,1.0), histtype='step', label='R=1.0 Jets')
plt.legend(loc='center right', frameon=False, fontsize="x-large")
plt.xlabel('EMD (normalized)', fontsize="x-large");
plt.show()
plt.savefig("out_origami/emds.png")

################################
# Fractal correlation dimension
################################

# prepare for histograms
bins = 10**np.linspace(-2, 0, 60)
reg = 10**-30
midbins = (bins[:-1] + bins[1:])/2
dmidbins = np.log(midbins[1:]) - np.log(midbins[:-1]) + reg
midbins2 = (midbins[:-1] + midbins[1:])/2

# compute the correlation dimensions
dims = []
uemds = np.triu(emd_vals)
counts = np.cumsum(np.histogram(uemds[uemds > 0], bins=bins)[0])
dims.append((np.log(counts[1:] + reg) - np.log(counts[:-1] + reg))/dmidbins)

# plot the correlation dimensions
plt.plot(midbins2, dims[0], '-', color='red', label='R=1.0 Jets')

# labels
plt.legend(loc='center right', frameon=False)

# plot style
plt.xscale('log')
plt.xlabel('Energy Scale Q/pT'); plt.ylabel('Correlation Dimension')
plt.xlim(0.02, 1); plt.ylim(0, 10)

plt.show()
plt.savefig("out_origami/corr_dim.png")
