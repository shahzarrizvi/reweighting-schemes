# z-score standardization of data

def standardize(sim_truth, sim_reco, data_reco, dummyval=-99):
    scaler_truth = preprocessing.StandardScaler()
    scaler_reco = preprocessing.StandardScaler()

    scaler_truth.fit(
        sim_truth[sim_truth != dummyval].astype(float).reshape(-1, 1))
    scaler_reco.fit(
        np.concatenate(
            (sim_reco[sim_reco != dummyval],
             data_reco[data_reco != dummyval])).astype(float).reshape(-1, 1))

    sim_truth_z = np.copy(sim_truth.astype(float))
    sim_reco_z = np.copy(sim_reco.astype(float))
    data_reco_z = np.copy(data_reco.astype(float))

    sim_truth_z[sim_truth != dummyval] = np.squeeze(
        scaler_truth.transform(
            sim_truth[sim_truth != dummyval].astype(float).reshape(-1, 1)))
    sim_reco_z[sim_reco != dummyval] = np.squeeze(
        scaler_reco.transform(
            sim_reco[sim_reco != dummyval].astype(float).reshape(-1, 1)))
    data_reco_z[data_reco != dummyval] = np.squeeze(
        scaler_reco.transform(
            data_reco[data_reco != dummyval].astype(float).reshape(-1, 1)))

    return sim_truth_z, sim_reco_z, data_reco_z

def get_uncertainty(data, weights, bins):
    sigma = np.empty(shape=((bins.size - 1), ))

    sum_weights = np.sum(weights)

    which_bin = np.digitize(data, bins)

    for i in range(sigma.size):
        sigma[i] = np.sqrt(np.sum(
            (weights[which_bin == (i + 1)])**2)) / sum_weights

    return sigma

def chi_square_dist(data_expected,
                    data_observed,
                    bins,
                    weights_expected=None,
                    weights_observed=None,
                    sigma_observed=None,
                    sigma_expected=None):

    if weights_expected is None:
        weights_expected = np.ones_like(data_expected)
    if weights_observed is None:
        weights_observed = np.ones_like(data_observed)

    H_expected, _ = np.histogram(data_expected,
                                 bins=bins,
                                 weights=weights_expected,
                                 density=True)

    H_observed, _ = np.histogram(data_observed,
                                 weights=weights_observed,
                                 bins=bins,
                                 density=True)

    if sigma_expected is None:
        sigma_expected = get_uncertainty(data_expected, weights_expected, bins)
    if sigma_observed is None:
        sigma_observed = get_uncertainty(data_observed, weights_observed, bins)

    uncertainty_squared = sigma_expected**2 + sigma_observed**2


    dist = (H_observed - H_expected)**2 / uncertainty_squared
    dist[np.isnan(dist)] = 0
    dist = np.sum(dist)

    return dist

def best_1D_reweighting(test,
                        target,
                        bins,
                        test_weights=None,
                        target_weights=None):

    dists = []
    for i in range(len(test_weights)):
        dists += [
            chi_square_dist(data_expected=target,
                            weights_expected=target_weights,
                            data_observed=test,
                            weights_observed=test_weights[i],
                            bins=bins)
        ]
    
    return test_weights[np.argmin(dists)]

def best_nD_reweighting(test,
                        target,
                        bins,
                        test_weights=None,
                        nominal_weights=None,
                        target_weights=None):

    dists_nominal = np.empty(shape=(len(test), len(test_weights)))
    dists_rewgt = np.empty(shape=(len(test), len(test_weights)))

    for i in range(dists_nominal.shape[0]):
        dists_nominal[i, :] = chi_square_dist(data_expected=target[i],
                                              weights_expected=target_weights,
                                              data_observed=test[i],
                                              weights_observed=nominal_weights,
                                              bins=bins[i])

        for j in range(dists_nominal.shape[1]):
            dists_rewgt[i, j] = chi_square_dist(data_expected=target[i],
                                                weights_expected=target_weights,
                                                data_observed=test[i],
                                                weights_observed=test_weights[j],
                                                bins=bins[i])

    rewgt_score = np.mean(dists_rewgt / dists_nominal, axis=0)

    return test_weights[np.argmin(rewgt_score)]
