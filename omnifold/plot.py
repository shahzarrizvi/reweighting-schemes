from ntupleanalysis import *
from tqdm import tqdm
from matplotlib import pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

# Global plot settings
import matplotlib.font_manager

plt.rcParams.update({
    "text.usetex": True,
    "pgf.rcfonts": False,
    "font.family": "serif",
    "font.size": 22,
    "xtick.labelsize": 15,
    "ytick.labelsize": 15,
    "legend.fontsize": 15
})

# Define default plot styles
plot_style_0 = {
    'histtype': 'step',
    'color': 'black',
    'linewidth': 2,
    'linestyle': '--',
    'density': True
}

plot_style_1 = {
    'histtype': 'step',
    'color': 'black',
    'linewidth': 2,
    'density': True
}

plot_style_2 = {'histtype': 'stepfilled', 'alpha': 0.5, 'density': True}

pad = 20 # padding for the titles

def plot_distributions(sim_truth,
                       sim_reco,
                       data_reco,
                       bins,
                       x_label,
                       sim_truth_weights_MC=None,
                       sim_reco_weights_MC=None,
                       data_truth_weights_MC=None,
                       data_reco_weights_MC=None,
                       data_truth=None,
                       save_label=None):

    fig, ax = plt.subplots(nrows=2,
                           ncols=2,
                           figsize=(20, 6),
                           constrained_layout=True,
                           sharex=True,
                           sharey=False,
                           gridspec_kw = {'height_ratios':[3, 1]}
                          )

    hT0, _, _ = ax[0,0].hist(sim_truth,
                           weights=sim_truth_weights_MC,
                           bins=bins,
                           label='MC Truth',
                           **plot_style_2,
                           color='C0')
    hR0, _, _ = ax[0,0].hist(sim_reco,
                           weights=sim_reco_weights_MC,
                           bins=bins,
                           label='MC Reco',
                           **plot_style_2,
                           color='C1')
    ax[0,0].set_title(x_label,pad=pad)
    ax[0,0].set_ylabel('Events per bin (normalized)', fontsize=22)
    ax[0,0].set_ylim([0, 1.5 * np.max(np.concatenate((hR0, hT0)))])
    legend = ax[0,0].legend(title='Simulation', loc='upper right', frameon=False)
    plt.setp(legend.get_title(), multialignment='center')
    draw_atlas_text(ax=ax[0,0])
#     print("MC distance (normalized): {:.5f}".format(0.5*np.sum((hT0-hR0)**2/(hT0+hR0))))
    
    ### Ratio plot, with a rectangle marking the +/- 10% band around a ratio of 1
    ratio = np.divide(hR0, hT0, where=(hT0 != 0))
    bin_centers = 0.5 * (bins[1:] + bins[:-1])
    
    ### Calculate the error by taking the square root of the ratio of the two histograms in question (bin-by-bin), but with squared weights
    error_hT0 = np.sqrt(np.histogram(
                        sim_truth,
                        bins=bins, 
                        weights=sim_truth_weights_MC**2,
                        density=False)[0])
    error_hR0 = np.sqrt(np.histogram(
                        sim_reco,
                        bins=bins, 
                        weights=sim_reco_weights_MC**2,
                        density=False)[0])
    not_normalized_hT0 = np.histogram(
                        sim_truth,
                        bins=bins, 
                        weights=sim_truth_weights_MC,
                        density=False)[0]
    not_normalized_hR0 = np.histogram(
                    sim_reco,
                    bins=bins, 
                    weights=sim_reco_weights_MC,
                    density=False)[0]
#     print("MC distance: {:.2f}".format(0.5*np.sum((not_normalized_hT0-not_normalized_hR0)**2/(not_normalized_hT0+not_normalized_hR0))))

    rel_errors_hT0 = np.divide(error_hT0, not_normalized_hT0, where=(not_normalized_hT0 != 0))
    rel_errors_hR0 = np.divide(error_hR0, not_normalized_hR0, where=(not_normalized_hR0 != 0))
    errors = np.sqrt(rel_errors_hR0**2+rel_errors_hT0**2)
    ax[1,0].add_patch(matplotlib.patches.Rectangle((0., 0.9), width=bins[-1]-bins[0], height=0.2,
                 facecolor = 'mistyrose',
                 fill=True,
                ))
    ax[1,0].errorbar(bin_centers,ratio, yerr=errors, color='black',marker='.',capsize=3)
    ax[1,0].set_ylabel('Ratio', fontsize=22)
    ax[1,0].set_ylim([0.75,1.25])
    ax[1,0].set_title('MC Reco / MC Truth')

    hT1, _, _ = ax[0,1].hist(data_truth,
                           weights=data_truth_weights_MC,
                           bins=bins,
                           label='``Data" Truth',
                           **plot_style_2,
                           color='C2')
    hR1, _, _ = ax[0,1].hist(data_reco,
                           weights=data_reco_weights_MC,
                           bins=bins,
                           label='``Data" Reco',
                           **plot_style_2,
                           color='k')
    ax[0,1].set_title(x_label, pad=pad)
    ax[0,1].get_shared_y_axes().join(ax[0,0], ax[0,1])
    ax[0,1].set_ylim([0, 1.5 * np.max(np.concatenate((hR1, hT1)))])
    legend = ax[0,1].legend(title='``Data"', loc='upper right', frameon=False)
    plt.setp(legend.get_title(), multialignment='center')
    draw_atlas_text(ax=ax[0,1])
#     print("Data distance (normalized): {:.5f}".format(0.5*np.sum((hT1-hR1)**2/(hT1+hR1))))

    
    ### Ratio plot, with a rectangle marking the +/- 10% band around a ratio of 1
    ratio = np.divide(hR1, hT1, where=(hT1 != 0))
    bin_centers = 0.5 * (bins[1:] + bins[:-1])
    
    ### Calculate the error by taking the square root of the ratio of the two histograms in question (bin-by-bin), but with squared weights
    error_hT1 = np.sqrt(np.histogram(
                        data_truth,
                        bins=bins, 
                        weights=data_truth_weights_MC**2,
                        density=False)[0])
    error_hR1 = np.sqrt(np.histogram(
                        data_reco,
                        bins=bins, 
                        weights=data_reco_weights_MC**2,
                        density=False)[0])
    not_normalized_hT1 = np.histogram(
                        data_truth,
                        bins=bins, 
                        weights=data_truth_weights_MC,
                        density=False)[0]
    not_normalized_hR1 = np.histogram(
                        data_reco,
                        bins=bins, 
                        weights=data_reco_weights_MC,
                        density=False)[0]
#     print("Data distance: {:.2f}".format(0.5*np.sum((not_normalized_hT1-not_normalized_hR1)**2/(not_normalized_hT1+not_normalized_hR1))))
    rel_errors_hT1 = np.divide(error_hT1, not_normalized_hT1, where=(not_normalized_hT1 != 0))
    rel_errors_hR1 = np.divide(error_hR1, not_normalized_hR1, where=(not_normalized_hR1 != 0))
    errors = np.sqrt(rel_errors_hR1**2+rel_errors_hT1**2)
    ax[1,1].add_patch(matplotlib.patches.Rectangle((0., 0.9), width=bins[-1]-bins[0], height=0.2,
                 facecolor = 'mistyrose',
                 fill=True,
                ))
    ax[1,1].errorbar(bin_centers,ratio, yerr=errors, color='black',marker='.',capsize=3)
    ax[1,1].set_ylim([0.75,1.25])
    ax[1,1].set_title('``Data" Reco / ``Data" Truth')
    
    if save_label is not None:
#         fig.savefig(save_label + '-Distributions.pdf',
#                     bbox_inches='tight',
#                     backend='pgf')
        fig.savefig(save_label + '-Distributions.png',
                    bbox_inches='tight',
#                     backend='pgf',
                    dpi=100)

def plot_results(sim_truth,
                 sim_reco,
                 data_reco,
                 weights,
                 bins,
                 x_label,
                 flavor_label='OmniFold',
                 sim_truth_weights_MC=None,
                 sim_reco_weights_MC=None,
                 data_truth_weights_MC=None,
                 data_reco_weights_MC=None,
                 data_truth=None,
                 dummyval=-99,
                 save_label=None):

    if sim_truth_weights_MC is None:
        sim_truth_weights_MC = np.ones(len(sim_truth))
    if sim_reco_weights_MC is None:
        sim_reco_weights_MC = np.ones(len(sim_reco))

    if data_truth_weights_MC is None:
        data_truth_weights_MC = np.ones(len(data_truth))
    if data_reco_weights_MC is None:
        data_reco_weights_MC = np.ones(len(data_reco))

    reco_distances = []
    truth_distances = []
    
    ### First, plot the ratios without OmniFold
    fig, ax = plt.subplots(nrows=2, ncols=2, sharex=True, sharey=False, gridspec_kw = {'height_ratios':[3, 1]}, figsize=(20, 6), constrained_layout=True)

    ### Reco
    hR0, _, _ = ax[0,0].hist(sim_reco[sim_reco!=dummyval],
                           weights=sim_reco_weights_MC[sim_reco!=dummyval],
                           bins=bins,
                           label='MC Reco',
                           **plot_style_2,
                           color='C1')

    hR2, _, _ = ax[0,0].hist(data_reco[data_reco!=dummyval],
                           weights=data_reco_weights_MC[data_reco!=dummyval],
                           bins=bins,
                           label='``Data" Reco (Target)',
                           **plot_style_2,
                           color='k')
    ax[0,0].set_title(x_label, pad=pad)
    ax[0,0].set_ylabel("Events per bin (normalized)", fontsize=22)
    ax[0,0].set_ylim([0, 1.5 * np.max(np.concatenate((hR0, hR2)))])
    draw_atlas_text(ax=ax[0,0])
    ax[0,0].legend()

    ### Ratio plot, with a rectangle marking the +/- 5% band around a ratio of 1
    ratio_reco = np.divide(hR0, hR2, where=(hR2 != 0))
    bin_centers = 0.5 * (bins[1:] + bins[:-1])
    
    ### Calculate the error by taking the square root of the ratio of the two histograms in question (bin-by-bin), but with squared weights
    error_hR0 = np.sqrt(np.histogram(
                        sim_reco[sim_reco!=dummyval],
                        bins=bins, 
                        weights=sim_reco_weights_MC[sim_reco!=dummyval]**2,
                        density=False)[0])
    error_hR2 = np.sqrt(np.histogram(
                        data_reco[data_reco!=dummyval],
                        bins=bins, 
                        weights=data_reco_weights_MC[data_reco!=dummyval]**2,
                        density=False)[0])
    not_normalized_hR0 = np.histogram(
                        sim_reco[sim_reco!=dummyval],
                        bins=bins, 
                        weights=sim_reco_weights_MC[sim_reco!=dummyval],
                        density=False)[0]
    not_normalized_hR2 = np.histogram(
                        data_reco[data_reco!=dummyval],
                        bins=bins, 
                        weights=data_reco_weights_MC[data_reco!=dummyval],
                        density=False)[0]
    rel_errors_hR0 = np.divide(error_hR0, not_normalized_hR0, where=(not_normalized_hR0 != 0))
    rel_errors_hR2 = np.divide(error_hR2, not_normalized_hR2, where=(not_normalized_hR2 != 0))
    errors = np.sqrt(rel_errors_hR0**2+rel_errors_hR2**2)
    ax[1,0].add_patch(matplotlib.patches.Rectangle((0., 0.95), width=bins[-1]-bins[0], height=0.1,
                 facecolor = 'mistyrose',
                 fill=True,
                ))
    ax[1,0].errorbar(bin_centers,ratio_reco, yerr=errors, color='black',marker='.',capsize=3)
    ax[1,0].set_ylim([0.75,1.25])
    ax[1,0].set_ylabel('Ratio', fontsize=22)
    ax[1,0].set_title('MC Reco/Target, before '+flavor_label)

    reco_distance = 0.5*np.sum((hR0-hR2)**2/(hR0+hR2))
    reco_distances.append(reco_distance)

    ### Truth
    hT0, _, _ = ax[0,1].hist(sim_truth[sim_truth!=dummyval],
                           weights=sim_truth_weights_MC[sim_truth!=dummyval],
                           bins=bins,
                           label=r'MC Truth',
                           **plot_style_2)

    if data_truth is not None:
        hT2, _, _ = ax[0,1].hist(data_truth[data_truth!=dummyval],
                               weights=data_truth_weights_MC[data_truth!=dummyval],
                               bins=bins,
                               label='``Data" Truth (Target)',
                               **plot_style_2,
                               color='C2')
    ax[0,1].legend()
    ax[0,1].set_title(x_label, pad=pad)
    ax[0,1].set_ylim([0, 1.5 * np.max(np.concatenate((hT0, hT2)))])
    draw_atlas_text(ax=ax[0,1])

    ### Ratio plot, with a rectangle marking the +/- 5% band around a ratio of 1
    ratio_truth = np.divide(hT0, hT2, where=(hT2 != 0))
    bin_centers = 0.5 * (bins[1:] + bins[:-1])

    ### Calculate the error by taking the square root of the ratio of the two histograms in question (bin-by-bin), but with squared weights
    error_hT0 = np.sqrt(np.histogram(
                        sim_truth[sim_truth!=dummyval], 
                        bins=bins, 
                        weights=sim_truth_weights_MC[sim_truth!=dummyval]**2,
                        density=False)[0])
    error_hT2 = np.sqrt(np.histogram(
                        data_truth[data_truth!=dummyval],
                        bins=bins, 
                        weights=data_truth_weights_MC[data_truth!=dummyval]**2,
                        density=False)[0])
    not_normalized_hT0 = np.histogram(
                        sim_truth[sim_truth!=dummyval], 
                        bins=bins, 
                        weights=sim_truth_weights_MC[sim_truth!=dummyval],
                        density=False)[0]
    not_normalized_hT2 = np.histogram(
                        data_truth[data_truth!=dummyval],
                        bins=bins, 
                        weights=data_truth_weights_MC[data_truth!=dummyval],
                        density=False)[0]
    rel_errors_hT0 = np.divide(error_hT0, not_normalized_hT0, where=(not_normalized_hT0 != 0))
    rel_errors_hT2 = np.divide(error_hT2, not_normalized_hT2, where=(not_normalized_hT2 != 0))
    errors = np.sqrt(rel_errors_hT0**2+rel_errors_hT2**2)
    ax[1,1].add_patch(matplotlib.patches.Rectangle((0., 0.95), width=bins[-1]-bins[0], height=0.1,
                 facecolor = 'mistyrose',
                 fill=True,
                ))
    ax[1,1].errorbar(bin_centers,ratio_truth, yerr=errors, color='black',marker='.',capsize=3)
    ax[1,1].set_ylim([0.75,1.25])
    ax[1,1].set_title('MC Truth/Target, before '+flavor_label)
    truth_distance = 0.5*np.sum((hT0-hT2)**2/(hT0+hT2))
    truth_distances.append(truth_distance)
    if save_label is not None:
#             fig.savefig(save_label + '-Iteration{:02}.pdf'.format(i+1),
#                         bbox_inches='tight',
#                         backend='pgf')
        fig.savefig(save_label + '-Iteration{:02}.png'.format(0),
                bbox_inches='tight',
                dpi=100)
    
    ### Now plot OmniFold results
    for i in tqdm(range(len(weights))):
        fig, ax = plt.subplots(nrows=2, ncols=2, sharex=True, sharey=False, gridspec_kw = {'height_ratios':[3, 1]}, figsize=(20, 6), constrained_layout=True)

        if i == 0:
            weights_init = sim_reco_weights_MC
            label0 = ''
            label1 = ', iter-{}'.format(i + 1)
        else:
            weights_init = sim_reco_weights_MC * weights[i - 1, 1, :]
            label0 = ', iter-{}'.format(i)
            label1 = ', iter-{}'.format(i + 1)

        ### Reco
        hR0, _, _ = ax[0,0].hist(sim_reco[sim_reco!=dummyval],
                               weights=weights_init[sim_reco!=dummyval],
                               bins=bins,
                               label='MC Reco' + label0 + '\n' +
                               r'(wgt.$=\nu_{{{}}}$)'.format(i),
                               **plot_style_2,
                               color='C1')

        hR1, _, _ = ax[0,0].hist(sim_reco[sim_reco!=dummyval],
                               weights=(sim_reco_weights_MC * weights[i, 0, :])[sim_reco!=dummyval],
                               bins=bins,
                               label='MC Reco' + label1 + '\n' +
                               r'(wgt.$=\omega_{{{}}}$)'.format(i + 1),
                               **plot_style_1)
        hR2, _, _ = ax[0,0].hist(data_reco[data_reco!=dummyval],
                               weights=data_reco_weights_MC[data_reco!=dummyval],
                               bins=bins,
                               label='``Data" Reco (Target)',
                               **plot_style_2,
                               color='k')
        ax[0,0].set_title(x_label, pad=pad)
        ax[0,0].set_ylabel("Events per bin (normalized)", fontsize=22)
        ax[0,0].set_ylim([0, 1.5 * np.max(np.concatenate((hR0, hR1, hR2)))])
        draw_atlas_text(ax=ax[0,0])
        ax[0,0].legend()

        ### Ratio plot, with a rectangle marking the +/- 5% band around a ratio of 1
        ratio_reco = np.divide(hR1, hR2, where=(hR2 != 0))
        bin_centers = 0.5 * (bins[1:] + bins[:-1])
        ### Calculate the error by taking the square root of the ratio of the two histograms in question (bin-by-bin), but with squared weights
        error_hR1 = np.sqrt(np.histogram(
                            sim_reco[sim_reco!=dummyval],
                            bins=bins, 
                            weights=(sim_reco_weights_MC * weights[i, 0, :])[sim_reco!=dummyval]**2,
                            density=False)[0])
        error_hR2 = np.sqrt(np.histogram(
                            data_reco[data_reco!=dummyval],
                            bins=bins, 
                            weights=data_reco_weights_MC[data_reco!=dummyval]**2,
                            density=False)[0])
        not_normalized_hR1 = np.histogram(
                            sim_reco[sim_reco!=dummyval],
                            bins=bins, 
                            weights=(sim_reco_weights_MC * weights[i, 0, :])[sim_reco!=dummyval],
                            density=False)[0]
        not_normalized_hR2 = np.histogram(
                            data_reco[data_reco!=dummyval],
                            bins=bins, 
                            weights=data_reco_weights_MC[data_reco!=dummyval],
                            density=False)[0]
        rel_errors_hR1 = np.divide(error_hR1, not_normalized_hR1, where=(not_normalized_hR1 != 0))
        rel_errors_hR2 = np.divide(error_hR2, not_normalized_hR2, where=(not_normalized_hR2 != 0))
        errors = np.sqrt(rel_errors_hR1**2+rel_errors_hR2**2)
        ax[1,0].add_patch(matplotlib.patches.Rectangle((0., 0.95), width=bins[-1]-bins[0], height=0.1,
                     facecolor = 'mistyrose',
                     fill=True,
                    ))
        ax[1,0].errorbar(bin_centers,ratio_reco, yerr=errors, color='black',marker='.',capsize=3)
        ax[1,0].set_ylim([0.75,1.25])
        ax[1,0].set_ylabel('Ratio', fontsize=22)
        ax[1,0].set_title('MC Reco' + label1 +
                               r'(wgt.$=\omega_{{{}}}$)'.format(i + 1)+'/Target')
        
        reco_distance = 0.5*np.sum((hR1-hR2)**2/(hR1+hR2))
#         print("MC distance (normalized): {:.10f}".format(reco_distance))
        reco_distances.append(reco_distance)

        ### Truth
        hT0, _, _ = ax[0,1].hist(sim_truth[sim_truth!=dummyval],
                               weights=sim_truth_weights_MC[sim_truth!=dummyval],
                               bins=bins,
                               label=r'MC Truth',
                               **plot_style_2)

        hT1, _, _ = ax[0,1].hist(sim_truth[sim_truth!=dummyval],
                               weights=(sim_truth_weights_MC * weights[i, 1, :])[sim_truth!=dummyval],
                               bins=bins,
                               label=flavor_label + 'ed ``Data"' + label1 +
                               '\n' + r'(wgt.$=\nu_{{{}}}$)'.format(i + 1),
                               **plot_style_1)
        if data_truth is not None:
            hT2, _, _ = ax[0,1].hist(data_truth[data_truth!=dummyval],
                                   weights=data_truth_weights_MC[data_truth!=dummyval],
                                   bins=bins,
                                   label='``Data" Truth (Target)',
                                   **plot_style_2,
                                   color='C2')
        ax[0,1].legend()
        ax[0,1].set_title(x_label, pad=pad)
#         ax[0,1].get_shared_y_axes().join(ax[0,0], ax[0,1])
        ax[0,1].set_ylim([0, 1.5 * np.max(np.concatenate((hT0, hT1, hT2)))])
        draw_atlas_text(ax=ax[0,1])

        ### Ratio plot, with a rectangle marking the +/- 5% band around a ratio of 1
        ratio_truth = np.divide(hT1, hT2, where=(hT2 != 0))
        bin_centers = 0.5 * (bins[1:] + bins[:-1])

        ### Calculate the error by taking the square root of the ratio of the two histograms in question (bin-by-bin), but with squared weights
        error_hT1 = np.sqrt(np.histogram(
                            sim_truth[sim_truth!=dummyval], 
                            bins=bins, 
                            weights=(sim_truth_weights_MC * weights[i, 1, :])[sim_truth!=dummyval]**2,
                            density=False)[0])
        error_hT2 = np.sqrt(np.histogram(
                            data_truth[data_truth!=dummyval],
                            bins=bins, 
                            weights=data_truth_weights_MC[data_truth!=dummyval]**2,
                            density=False)[0])
        not_normalized_hT1 = np.histogram(
                            sim_truth[sim_truth!=dummyval], 
                            bins=bins, 
                            weights=(sim_truth_weights_MC * weights[i, 1, :])[sim_truth!=dummyval],
                            density=False)[0]
        not_normalized_hT2 = np.histogram(
                            data_truth[data_truth!=dummyval],
                            bins=bins, 
                            weights=data_truth_weights_MC[data_truth!=dummyval],
                            density=False)[0]
        rel_errors_hT1 = np.divide(error_hT1, not_normalized_hT1, where=(not_normalized_hT1 != 0))
        rel_errors_hT2 = np.divide(error_hT2, not_normalized_hT2, where=(not_normalized_hT2 != 0))
        errors = np.sqrt(rel_errors_hT1**2+rel_errors_hT2**2)
        ax[1,1].add_patch(matplotlib.patches.Rectangle((0., 0.95), width=bins[-1]-bins[0], height=0.1,
                     facecolor = 'mistyrose',
                     fill=True,
                    ))
        ax[1,1].errorbar(bin_centers,ratio_truth, yerr=errors, color='black',marker='.',capsize=3)
        ax[1,1].set_ylim([0.75,1.25])
        ax[1,1].set_title(flavor_label+'ed ``Data"/Target')
        truth_distance = 0.5*np.sum((hT1-hT2)**2/(hT1+hT2))
#         print("Data distance (normalized): {:.10f}".format(truth_distance))
        truth_distances.append(truth_distance)
        if save_label is not None:
#             fig.savefig(save_label + '-Iteration{:02}.pdf'.format(i+1),
#                         bbox_inches='tight',
#                         backend='pgf')
            fig.savefig(save_label + '-Iteration{:02}.png'.format(i+1),
                    bbox_inches='tight',
                    dpi=100)
                        
        
    from matplotlib.ticker import MaxNLocator

    fig = plt.figure()
    plt.plot(np.arange(len(reco_distances)), reco_distances, label=r"Reco-level $\chi^2$ Distance", linewidth=2, color="saddlebrown")
    plt.plot(np.arange(len(truth_distances)), truth_distances, label=r"Truth-level $\chi^2$ Distance", linewidth=2, color="mediumseagreen")
    plt.axes().xaxis.set_major_locator(MaxNLocator(integer=True))
    plt.xlabel("Iteration")
    plt.ylabel("Distance")
    plt.legend(fontsize=22)
    if save_label is not None:
        fig.savefig(save_label + '-distances.png',
                    bbox_inches='tight',
                    dpi=100)
        