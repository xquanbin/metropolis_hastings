# -*- coding: utf-8 -*-
# Author : xiequanbin
# Date:  18-06-01
# Email:  xquanbin072095@gmail.com

import time
import numpy as np
import metropolis as mt
import hiw_simulation as hiw
from scipy.stats import multivariate_normal, dirichlet


def state_sampling():

    pass


def beta_sampling(y, x, t_k, prior_mu, prior_sigma, sigma):
    post_mu = np.dot(np.linalg.inv(prior_sigma), prior_mu)
    post_sigma = np.linalg.inv(prior_sigma)
    for t in t_k:
        post_mu += np.dot(np.dot(x[:, :, t].T, np.linalg.inv(sigma)), y[t, :].T)
        post_sigma += np.dot(np.dot(x[:, :, t].T, np.linalg.inv(sigma)), x[:, :, t])

    post_sigma = np.linalg.inv(post_sigma)
    post_mu = np.dot(post_sigma, post_mu)

    # For some reason there are very small numerical issues such as round-off that make post_sigma not quite symmetric,
    # so here we use a usual trick to force it to be exactly symmetric.
    post_sigma = (post_sigma + post_sigma.T) / 2.
    beta_sample = multivariate_normal.rvs(post_mu, post_sigma)

    return beta_sample


def tran_matrix_sampling(prior_dir_delta, state_samples):
    T = len(state_samples)
    K = len(prior_dir_delta)

    N = np.zeros([K, K])
    tran_matrix = np.zeros([K, K])

    for i in range(0, K):
        for j in range(0, K):
            for t in range(0, T - 1):
                if (state_samples[t] == j) and (state_samples[t + 1] == i):
                    N[i, j] += 1

    post_dir_delta = prior_dir_delta + N
    for k in range(0, K):
        tran_matrix[k, :] = dirichlet.rvs(post_dir_delta[k, :])

    return tran_matrix


if __name__ == "__main__":

    # load test data
    data_list = []
    with open('./input/data_15.txt', 'r') as f:
        lines = f.readlines()
        for line in lines:
            sub_data = [float(i) for i in line.strip().split()]
            data_list.append(sub_data)
    data = np.array(data_list)
    (T, p) = data.shape

