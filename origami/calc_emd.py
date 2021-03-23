"""
calc_emd.py

This one calculates EMDs using the pre-processed .h5's
Matt LeBlanc (CERN) 2021
<matt.leblanc@cern.ch>
"""

import numpy as np
import h5py
import argparse
import matplotlib.pyplot as plt

import energyflow as ef

def getTracks(infile):
    h5f = h5py.File(infile,'r')
    tracks = []
    tracks.extend(h5f['track_array'][:])

    tracks_array = np.asarray(tracks)
    tracks_array = np.squeeze(tracks_array)
    tracks_array = tracks_array[:,:,0:3] #if reco else jet_array[:,:,3:6]
    tracksList = tracks_array.tolist()
    #tracksList = [[j for j in i if j != [-99.0, -99.0, -99.0]] for i in tracksList]    
    tracksList = [[j for j in i if j != [0.0, 0.0, 0.0]] for i in tracksList]
    tracksList = [x for x in tracksList if x != []]
    return tracksList

parser = argparse.ArgumentParser("EMDchunk")
parser.add_argument('--infile', type=str)
args = parser.parse_args()

tracks = getTracks(args.infile)

print('calculating EMD for '+str(len(tracks))+' jets.')

print(tracks[0])

tracks = tracks[0:20000]

emd_vals = ef.emd.emds_wasserstein(np.array(tracks, dtype=object),
                                   R=1.0,
                                   beta=1.0,
                                   norm=True,
                                   verbose=1,
                                   n_jobs=-1,
                                   print_every=100000)
emd_vals = np.array(emd_vals, dtype=np.float16)

print(emd_vals)

h5f_out = h5py.File('emd_vals.h5', 'w')
h5f_out.create_dataset('emd_vals', data=emd_vals)
h5f_out.close()
