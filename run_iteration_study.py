### Generic imports
import os
import numpy as np
import pandas as pd
import scipy.stats as stats
from tqdm import tqdm
import argparse
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.ticker import MaxNLocator

### ML imports
from sklearn.model_selection import train_test_split
import sklearn.preprocessing as preprocessing
import tensorflow as tf

### Custom functions
from omnifold import *
from omnifold.utilities import *

# ### GPU Setup
# os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
# os.environ["CUDA_VISIBLE_DEVICES"] = "1" # pick a number < 4 on ML4HEP; < 3 on Voltan 
# physical_devices = tf.config.list_physical_devices('GPU') 
# tf.config.experimental.set_memory_growth(physical_devices[0], True)

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("variable", type=int, help="Index of variable of interest.")
    parser.add_argument("n_trials", type=int, help="Number of trials.")
    parser.add_argument("n_iterations", type=int, help="Number of iterations per trial.")
    return parser.parse_args()

if __name__ == "__main__":
    args = get_args()

    ### Plotting setup
    plot_setup()
    plt.rcParams.update({"font.family": "serif",})
    plot_dir = './plots/'

    ### Load data
    folder = '/data0/mpettee/omnifold_data/'
    f_mc = uproot.open(folder+"ZjetOmnifold_Jun25_PowhegPythia_mc16e_slim.root")
    f_data = uproot.open(folder+"ZjetOmnifold_Jun25_Sherpa221_mc16e_slim.root")
    columns = f_mc['OmniTree'].keys()

    mc = f_mc['OmniTree'].arrays(
        [col for col in columns if not col.endswith("_tracks")],
        library="pd")

    data = f_data['OmniTree'].arrays(
        [col for col in columns if not col.endswith("_tracks")],
        library="pd")

    print("MC has {:,} events with {} columns.".format(mc.shape[0],mc.shape[1]))
    print("Data has {:,} events with {} columns.".format(data.shape[0],data.shape[1]))

    ### Apply event selection. The sample we're using already has these cuts applied, but we'll keep them in for future datasets. 
    mc['pass190'] = mc['pass190'].astype('bool')
    mc['truth_pass190'] = mc['truth_pass190'].astype('bool')
    data['pass190'] = data['pass190'].astype('bool')
    data['truth_pass190'] = data['truth_pass190'].astype('bool')
    mc_truth_weights = mc[(mc.pass190 | mc.truth_pass190)].weight_mc
    mc_reco_weights = mc[(mc.pass190 | mc.truth_pass190)].weight
    data_truth_weights = data[data.truth_pass190].weight_mc
    data_reco_weights = data[data.pass190].weight

    ### Normalize the weights
    for weights in [mc_truth_weights, mc_reco_weights, data_truth_weights, data_reco_weights]:
        weights /= np.mean(weights)

    ### Load IBU histograms to get binning & other metadata
    file_labels = [
        'Ntracks_trackj1', 'Ntracks_trackj2', 'm_trackj1', 'm_trackj2',
        'pT_trackj1', 'pT_trackj2', 'y_trackj1', 'y_trackj2', 'phi_trackj1',
        'phi_trackj2', 'tau1_trackj1', 'tau1_trackj2', 'tau2_trackj1',
        'tau2_trackj2', 'tau3_trackj1', 'tau3_trackj2', 'pT_ll', 'y_ll', 'pT_l1',
        'pT_l2', 'eta_l1', 'eta_l2', 'phi_l1', 'phi_l2'
    ]

    plot_labels = [
        r'Leading track jet $n_{\textrm{ch}}$ ',
        r'Subleading track jet $n_{\textrm{ch}}$', 'Leading track jet $m$ [GeV]',
        r'Subleading track jet $m$ [GeV]', r'Leading track jet $p_T$ [GeV]',
        r'Subleading track jet $p_T$ [GeV]', r'Leading track jet $y$',
        r'Subleading track jet $y$', r'Leading track jet $\phi$',
        r'Subleading track jet $\phi$', r'Leading track jet $\tau_1$',
        r'Subleading track jet $\tau_1$', r'Leading track jet $\tau_2$',
        r'Subleading track jet $\tau_2$', r'Leading track jet $\tau_3$',
        r'Subleading track jet $\tau_3$', r'$p^{\mu \mu}_T$ [GeV]',
        r'$y_{\mu \mu}$', r'$p^{\mu 1}_{T}$ [GeV]', r'$p^{\mu 2}_{T}$ [GeV]',
        '$\eta_{\mu 1}$', '$\eta_{\mu 2}$', '$\phi_{\mu 1}$', '$\phi_{\mu 2}$'
    ]

    IBU_hists = uproot.open(folder+'unfoldingPlotsJune14_UnfoldedHists.root')

    bins = []
    for label in file_labels:
        bins += [IBU_hists['SherpaUnfoldWPythia_2018_'+label].to_numpy()[1]]

    labels_and_bins = zip(file_labels, plot_labels, bins)

    ibu_hists = []
    for file_label, plot_label, plot_bins in labels_and_bins:
        ibu_hists += [{
            'file_label': file_label,
            'plot_label': plot_label,
            'bins': plot_bins
        }]

    ### Replace bins with Laura's new binning! 
    ibu_hists[0]['bins'] = np.array([1, 7, 11, 15, 20, 30, 40]) # leading jet n_charged_tracks
    ibu_hists[4]['bins'] = np.array([0,  50,  100,  150,  200,  300,  1000]) # leading jet pT
    ibu_hists[10]['bins'] = np.array([0,  0.05,  0.1,  0.17,  0.25,  0.35,  0.5,  0.9]) # leading jet tau_1

    ### RUN ITERATION STUDY   
    var_i = args.variable # index of variable of interest (num_tracks for leading jet) 
    n_trials = args.n_trials        # how many full runs of unfolding     #30
    n_iterations = args.n_iterations    # how many iterations per unfolding process    #10
    save_label0 = 'iteration_study_'+str(n_trials)+"x"+str(n_iterations)
    bins = ibu_hists[var_i]['bins']
    # Unifold result
    reco_unfolded_hists = np.zeros((n_trials, n_iterations+1, len(bins)-1)) 
    truth_unfolded_hists = np.zeros((n_trials, n_iterations+1, len(bins)-1))

    # Chi-squared distances
    reco_distances = np.zeros((n_trials, n_iterations+1, len(bins)-1)) 
    truth_distances = np.zeros((n_trials, n_iterations+1, len(bins)-1))

    for i in tqdm(range(n_trials), desc="Trial"):

        K.clear_session()
        unifold_weights = np.zeros(shape=(len(ibu_hists), len(mc_truth_weights)))
        x_label = ibu_hists[var_i]['plot_label']
        file_label = ibu_hists[var_i]['file_label']
        os.makedirs(plot_dir+'UniFold/'+file_label, exist_ok=True)
        save_label = plot_dir+'UniFold/'+file_label+'/'+save_label0
        print("Saving as {}.".format(save_label))

        dummyval = -99
        mc_truth = mc[(mc.pass190 | mc.truth_pass190)]['truth_' + file_label]
        mc_truth[mc.truth_pass190 == False] = dummyval
        mc_reco = mc[(mc.pass190 | mc.truth_pass190)][file_label]
        mc_reco[mc.pass190 == False] = dummyval

        data_truth = data[data['truth_pass190']]['truth_' + file_label]
        data_reco = data[data['pass190']][file_label]

        # z-score standardization of data
        mc_truth_z, mc_reco_z, data_reco_z = standardize(np.array(mc_truth), 
                                                       np.array(mc_reco),
                                                       np.array(data_reco))

        ### Fill these arrays for Iteration 0, i.e. before unfolding, for context
        hR0, _ = np.histogram(mc_reco[mc_reco!=dummyval],
                           weights=mc_reco_weights[mc_reco!=dummyval],
                           bins=bins, density=True)

        hR2, _ = np.histogram(data_reco[data_reco!=dummyval],
                           weights=data_reco_weights[data_reco!=dummyval],
                           bins=bins, density=True)

        hT0, _ = np.histogram(mc_truth[mc_truth!=dummyval],
                           weights=mc_truth_weights[mc_truth!=dummyval],
                           bins=bins, density=True)

        hT2, _ = np.histogram(data_truth[data_truth!=dummyval],
                           weights=data_truth_weights[data_truth!=dummyval],
                           bins=bins, density=True)


        reco_distances[i][0] = 0.5*np.sum((hR0-hR2)**2/(hR0+hR2))
        truth_distances[i][0] = 0.5*np.sum((hT0-hT2)**2/(hT0+hT2))
            
        reco_unfolded_hists[i][0] = hR0
        truth_unfolded_hists[i][0] = hT0

        ### Reweight using Poisson distribution with λ=1 (new distribution for each trial)
        this_trial_reweight = np.random.poisson(lam=1.0, size=(len(mc_reco_weights)))
        poisson_mc_reco_weights = mc_reco_weights*this_trial_reweight
        poisson_mc_truth_weights = mc_truth_weights*this_trial_reweight
        # data_reco_weights = data_reco_weights*this_trial_reweight

        ### Perform all iterations of unfolding, and fill results in "weights" array
        weights, _ = unifold(iterations=n_iterations,
                     sim_truth=mc_truth_z,
                     sim_reco=mc_reco_z,
                     sim_truth_weights_MC=poisson_mc_truth_weights,
                     sim_reco_weights_MC=poisson_mc_reco_weights,
                     data_reco=data_reco_z,
                     data_reco_weights_MC=data_reco_weights,
                     dummyval=dummyval,
                     verbose=0,
                    )

        ### Make plots
        plot_results(sim_truth=mc_truth,
                 sim_reco=mc_reco,
                 sim_truth_weights_MC=poisson_mc_truth_weights,
                 sim_reco_weights_MC=poisson_mc_reco_weights,
                 data_truth=data_truth,
                 data_reco=data_reco,
                 data_truth_weights_MC=data_truth_weights,
                 data_reco_weights_MC=data_reco_weights,
                 weights=weights,
                 bins=bins,
                 x_label=x_label,
                 flavor_label='UniFold',
                 save_label=save_label+'_trial_'+str(i)
                )

        ### Fill arrays evaluating each iteration
        for j in np.arange(n_iterations):
            hR1, _ = np.histogram(mc_reco[mc_reco!=dummyval],
                                   weights=(poisson_mc_reco_weights * weights[j, 0, :])[mc_reco!=dummyval],
                                   bins=bins,density=True)
            hR2, _ = np.histogram(data_reco[data_reco!=dummyval],
                                   weights=data_reco_weights[data_reco!=dummyval],
                                   bins=bins,density=True)
            reco_distances[i][j+1] = 0.5*np.sum((hR1-hR2)**2/(hR1+hR2))
            reco_unfolded_hists[i][j+1] = hR1
            
            hT1, _ = np.histogram(mc_truth[mc_truth!=dummyval],
                               weights=(poisson_mc_truth_weights * weights[j, 1, :])[mc_truth!=dummyval],
                               bins=bins,density=True)
            if data_truth is not None:
                hT2, _ = np.histogram(data_truth[data_truth!=dummyval],
                                   weights=data_truth_weights[data_truth!=dummyval],
                                   bins=bins,density=True)
            truth_distances[i][j+1] = 0.5*np.sum((hT1-hT2)**2/(hT1+hT2))
            truth_unfolded_hists[i][j+1] = hT1

    ### Calculate summary statistics
    mean_reco_distances = np.mean(reco_distances, axis=(0,2))
    mean_truth_distances = np.mean(truth_distances, axis=(0,2))
    reco_stat_uncert = np.mean(np.var(reco_unfolded_hists, axis=0), axis=1)
    truth_stat_uncert = np.mean(np.var(truth_unfolded_hists, axis=0), axis=1)

    ### Make summary plots
    fig, axs = plt.subplots(nrows = 1, ncols = 2, figsize=(15,6), constrained_layout=True)
    ax = axs[0]
    ax.plot(np.arange(len(weights)+1), mean_reco_distances, label=r"Reco-level $\chi^2$ Distance", linewidth=2, color="saddlebrown")
    ax.plot(np.arange(len(weights)+1), mean_truth_distances, label=r"Truth-level $\chi^2$ Distance", linewidth=2, color="mediumseagreen")
    ax.set_xlabel("Iteration")
    ax.set_ylabel("Average Distance")
    ax.legend(fontsize=22)
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))

    ax = axs[1]
    ax.plot(np.arange(len(weights)+1), reco_stat_uncert, label=r"Reco-level $\sigma$", linewidth=2, color="saddlebrown")
    ax.plot(np.arange(len(weights)+1), truth_stat_uncert, label=r"Truth-level $\sigma$", linewidth=2, color="mediumseagreen")
    ax.set_xlabel("Iteration")
    ax.set_ylabel("Average Stat. Uncertainty")
    ax.legend(fontsize=22)
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))

    if save_label is not None:
        fig.savefig(save_label + '-distances-and-uncert.png',
                    bbox_inches='tight',
                    dpi=100)