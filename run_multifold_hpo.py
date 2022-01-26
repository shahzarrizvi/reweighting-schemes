### Generic imports
import os
import json
import numpy as np
import pandas as pd
import scipy.stats as stats
import argparse
from tqdm import tqdm
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.ticker import MaxNLocator
import wandb
from wandb.keras import WandbCallback

### ML imports
from sklearn.model_selection import train_test_split
import sklearn.preprocessing as preprocessing
import tensorflow as tf
from keras.models import load_model

### Custom functions
from omnifold import *
from omnifold.utilities import *

### GPU Setup
os.environ["CUDA_VISIBLE_DEVICES"] = "0" # pick a number < 4 on ML4HEP; < 3 on Voltan 
physical_devices = tf.config.list_physical_devices('GPU') 
tf.config.experimental.set_memory_growth(physical_devices[0], True)

### Plot setup
plot_setup()
plot_dir = './plots/'

### GPU Setup
os.environ["CUDA_VISIBLE_DEVICES"] = "1" # pick a number < 4 on ML4HEP; < 3 on Voltan 
physical_devices = tf.config.list_physical_devices('GPU') 
tf.config.experimental.set_memory_growth(physical_devices[0], True)

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--save_label", default='test', type=str, help="Folder name for saving training outputs & plots.")
    parser.add_argument("--n_layers", default=3, type=int, help="Number of hidden layers.")
    parser.add_argument("--layer_size", default=50, type=int, help="Number of nodes per layer.")
    parser.add_argument("--patience", default=10, type=int, help="How many epochs of no val_loss improvement before the training is stopped.")
    parser.add_argument("--epochs", default=1000, type=int, help="Number of training epochs.")
    parser.add_argument("--batch_size", default=10000, type=int, help="Batch size during training.")
    parser.add_argument("--livelossplot", action="store_true", help="Use this flag if you want to turn on the live loss plot visualization.") 
    return parser.parse_args()

if __name__ == "__main__":
    args = get_args()
    
    wandb.init(project="multifold", entity="mpettee")
    wandb.config = {
      "epochs": args.epochs,
      "patience": args.patience,
      "n_layers": args.n_layers,
      "batch_size": args.batch_size,
      "layer_size": args.layer_size, 
    }
    
    ### Plotting setup
    plot_setup()
    plt.rcParams.update({"font.family": "serif",})
    plot_dir = './plots/'

    ### Load data
    mc = pd.read_hdf('./omnifold_data/zjets_powhegpythia_mc16e.h5')
    data = pd.read_hdf('./omnifold_data/zjets_sherpa_mc16e.h5')
    data_truth = pd.read_hdf('./omnifold_data/zjets_sherpa_mc16e_truth.h5')

    with open(r'./omnifold_data/inputs.json', "r") as read_file:
        inputs = json.load(read_file)

    for dict in inputs:
        dict['bins'] = np.asarray(dict['bins']) # cast back to numpy arrays for plotting

    ### Add in 200 GeV cuts for plotting only 
    mc_pt200 = mc[(mc.truth_pT_ll > 200) | (mc.pT_ll > 200)]
    data_truth_pt200 = data_truth[data_truth.truth_pT_ll > 200]
    data_pt200 = data[data.pT_ll > 200]
    mc_filter = (mc.truth_pT_ll > 200) | (mc.pT_ll > 200)

    dummyval = -99
    save_label0 = args.save_label
    
    # Prepare to unfold 
    K.clear_session()
    print("Unfolding {} variables.".format(len(inputs)))

    mc_truth_plots = [None] * len(inputs)
    mc_reco_plots = [None] * len(inputs)
    data_truth_plots = [None] * len(inputs)
    data_reco_plots = [None] * len(inputs)

    mc_truth_hists = [None] * len(inputs)
    mc_reco_hists = [None] * len(inputs)
    data_truth_hists = [None] * len(inputs)
    data_reco_hists = [None] * len(inputs)

    # z-score standardization of data
    mc_truth_z = [None] * len(inputs)
    mc_reco_z = [None] * len(inputs)
    data_reco_z = [None] * len(inputs)

    for i in tqdm(range(len(inputs))):
        file_label = inputs[i]['file_label']

        mc_truth_plots[i] = np.where(mc_pt200.truth_pass190, mc_pt200['truth_'+file_label], dummyval)
        mc_reco_plots[i] = np.where(mc_pt200.pass190, mc_pt200[file_label], dummyval)
        data_truth_plots[i] = data_truth_pt200['truth_'+file_label]
        data_reco_plots[i] = data_pt200[file_label]

        bins = inputs[i]['bins']
        x_label = inputs[i]['plot_label']
        file_label = inputs[i]['file_label']
        os.makedirs(plot_dir+'MultiFold/'+file_label, exist_ok=True)
        save_label = plot_dir+'MultiFold/'+file_label+'/'+save_label0

        mc_truth_hists[i] = np.where(mc.truth_pass190, mc['truth_'+file_label], dummyval)
        mc_reco_hists[i] = np.where(mc.pass190, mc[file_label], dummyval)
        data_truth_hists[i] = data_truth['truth_'+file_label]
        data_reco_hists[i] = data[file_label]

        mc_truth_z[i], mc_reco_z[i], data_reco_z[i] = standardize(
            np.array(mc_truth_hists[i]), 
            np.array(mc_reco_hists[i]), 
            np.array(data_reco_hists[i]))

    # Unfold! 
    weights, model = multifold(
                     sim_truth=mc_truth_z,
                     sim_reco=mc_reco_z,
                     sim_truth_weights_MC=mc.weight_mc,
                     sim_reco_weights_MC=mc.weight,
                     data_reco=data_reco_z,
                     data_reco_weights_MC=data.weight,
                     dummyval=dummyval,
                     verbose=1,
                     iterations=1,
                     layer_size=args.layer_size,
                     n_layers=args.n_layers,
                     epochs=args.epochs,
                     patience=args.patience,
                     batch_size=args.batch_size,
                     livelossplot=args.livelossplot,
                    )
    
    # Plot DeltaR 
    import vector

    def ll(pt_l1, eta_l1, phi_l1, pt_l2, eta_l2, phi_l2): 
            l1 = vector.array({"pt": pt_l1, "eta": eta_l1, "phi": phi_l1, "m": np.zeros(len(pt_l1))})
            l2 = vector.array({"pt": pt_l2, "eta": eta_l2, "phi": phi_l2, "m": np.zeros(len(pt_l2))})
            return l1.add(l2)

    def delta_r(y_ll, y_trackj1, phi_ll, phi_trackj1):
        delta_y = y_ll - y_trackj1
        delta_phi = phi_ll - phi_trackj1
        return np.sqrt(delta_y**2 + delta_phi**2)
    
    # DeltaR

    ### Define the combined dilepton phi first, before calculating DeltaR
    mc_pt200['truth_phi_ll'] = ll(mc_pt200['truth_pT_l1'], mc_pt200['truth_eta_l1'], mc_pt200['truth_phi_l1'], 
                                  mc_pt200['truth_pT_l2'], mc_pt200['truth_eta_l2'], mc_pt200['truth_phi_l2']).phi
    mc_pt200['phi_ll'] = ll(mc_pt200['pT_l1'], mc_pt200['eta_l1'], mc_pt200['phi_l1'], 
                            mc_pt200['pT_l2'], mc_pt200['eta_l2'], mc_pt200['phi_l2']).phi
    data_truth_pt200['truth_phi_ll'] = ll(data_truth_pt200['truth_pT_l1'], data_truth_pt200['truth_eta_l1'], data_truth_pt200['truth_phi_l1'], 
                                          data_truth_pt200['truth_pT_l2'], data_truth_pt200['truth_eta_l2'], data_truth_pt200['truth_phi_l2']).phi
    data_pt200['phi_ll'] = ll(data_pt200['pT_l1'], data_pt200['eta_l1'], data_pt200['phi_l1'], 
                              data_pt200['pT_l2'], data_pt200['eta_l2'], data_pt200['phi_l2']).phi

    mc_truth_plot = np.where(mc_pt200.truth_pass190, delta_r(mc_pt200['truth_y_ll'], mc_pt200['truth_y_trackj1'], mc_pt200['truth_phi_ll'], mc_pt200['truth_phi_trackj1']), dummyval)
    mc_reco_plot = np.where(mc_pt200.pass190, delta_r(mc_pt200['y_ll'], mc_pt200['y_trackj1'], mc_pt200['phi_ll'], mc_pt200['phi_trackj1']), dummyval)
    data_truth_plot = delta_r(data_truth_pt200['truth_y_ll'], data_truth_pt200['truth_y_trackj1'], data_truth_pt200['truth_phi_ll'], data_truth_pt200['truth_phi_trackj1'])
    data_reco_plot = delta_r(data_pt200['y_ll'], data_pt200['y_trackj1'], data_pt200['phi_ll'], data_pt200['phi_trackj1'])

    bins = np.linspace(0,6,30)
    os.makedirs(plot_dir+'MultiFold/'+ 'DeltaR', exist_ok=True)

    # Truth vs Reco
    plot_distributions(
        sim_truth=mc_truth_plot,
        sim_reco=mc_reco_plot,
        sim_truth_weights_MC=mc_pt200.weight_mc,
        sim_reco_weights_MC=mc_pt200.weight,
        data_truth=data_truth_plot,
        data_reco=data_reco_plot,
        data_truth_weights_MC=data_truth_pt200.weight_mc,
        data_reco_weights_MC=data_pt200.weight,
        bins=bins,
        x_label=r'$\Delta R(ll,j_1)$',
        save_label=plot_dir + '/MultiFold/' + 'DeltaR' + '/' + save_label0 + '-MultiFold-' + 'DeltaR'
    )

    # Unfolding 
    plot_results(sim_truth=mc_truth_plot,
                 sim_reco=mc_reco_plot,
                 sim_truth_weights_MC=mc_pt200.weight_mc,
                 sim_reco_weights_MC=mc_pt200.weight,
                 data_truth=data_truth_plot,
                 data_reco=data_reco_plot,
                 data_truth_weights_MC=data_truth_pt200.weight_mc,
                 data_reco_weights_MC=data_pt200.weight,
                 weights=weights[:,:,mc_filter],
                 flavor_label = "MultiFold",
                 bins=bins,
                 x_label=r'$\Delta R(ll,j_1)$',
                 save_label=plot_dir + '/MultiFold/' + 'DeltaR' + '/' + save_label0 + '-MultiFold-' + 'DeltaR'
                )