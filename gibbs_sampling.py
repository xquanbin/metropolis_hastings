# -*- coding: utf-8 -*-
# Author : xiequanbin
# Date:  18-06-01
# Email:  xquanbin072095@gmail.com

import time
import numpy as np
import networkx as nx
import metropolis as mt
import hiw_simulation as hiw
from scipy.stats import multivariate_normal, dirichlet


def state_sampling(K, y, x, beta, sigma, tran_matrix, initial_state_prob):
    (T, p) = y.shape
    state_sample = np.zeros(T, dtype=int)

    y_prob = np.zeros([T, K])
    for t in range(0, T):
        for k in range(0, K):
            y_prob[t, k] = multivariate_normal.pdf(y[t], mean=np.dot(x[t].T, beta[k]), cov=sigma[k])

    predict_step = np.zeros([T, K])
    update_step = np.zeros([T, K])

    predict_step[0] = np.dot(initial_state_prob, tran_matrix)
    update_step[0] = predict_step[0] * y_prob[0] / np.dot(predict_step[0], y_prob[0])

    for t in range(1, T):
        predict_step[t] = np.dot(update_step[t - 1], tran_matrix)
        update_step[t] = predict_step[t] * y_prob[t] / np.dot(predict_step[t], y_prob[t])

    # generate sT from p(sT|y1:T)
    sT_cond_prob = update_step[-1]
    sT_cdf = np.cumsum(sT_cond_prob)
    rT = np.random.rand()
    if rT <=sT_cdf[0]:
        state_sample[-1] = 0
    else:
        for k in range(1, K):
            if (rT > sT_cdf[k - 1]) and (rT <= sT_cdf[k]):
                state_sample[-1] = k
                break

    # then generate st from p(st|st+1,y1:t)
    for t in range(-2, -T-1, -1):
        st_cond_prob = update_step[t] * tran_matrix[:, state_sample[t + 1]] / predict_step[t + 1, state_sample[t + 1]]
        st_cdf = np.cumsum(st_cond_prob)

        rt = np.random.rand()
        if rt <= st_cdf[0]:
            state_sample[t] = 0
        else:
            for k in range(1, K):
                if (rt > st_cdf[k - 1]) and (rT <= st_cdf[k]):
                    state_sample[t] = k
                    break

    return state_sample


def beta_sampling(y, x, t_k, prior_mu, prior_sigma, sigma):
    post_mu = np.dot(np.linalg.inv(prior_sigma), prior_mu)
    post_sigma = np.linalg.inv(prior_sigma)
    for t in t_k:
        post_mu += np.dot(np.dot(x[t], np.linalg.inv(sigma)), y[t])
        post_sigma += np.dot(np.dot(x[t], np.linalg.inv(sigma)), x[t].T)

    post_sigma = np.linalg.inv(post_sigma)
    post_mu = np.dot(post_sigma, post_mu)

    # For some reason there are very small numerical issues such as round-off that make post_sigma not quite symmetric,
    # so here we use a usual trick to force it to be exactly symmetric.
    post_sigma = (post_sigma + post_sigma.T) / 2.
    beta_sample = multivariate_normal.rvs(post_mu, post_sigma)

    return beta_sample


def tran_matrix_sampling(dir_prior_delta, state_samples):
    T = len(state_samples)
    K = len(dir_prior_delta)

    N = np.zeros([K, K])
    tran_matrix = np.zeros([K, K])

    for i in range(0, K):
        for j in range(0, K):
            for t in range(0, T - 1):
                if (state_samples[t] == j) and (state_samples[t + 1] == i):
                    N[i, j] += 1

    dir_post_delta = dir_prior_delta + N
    for k in range(0, K):
        tran_matrix[k] = dirichlet.rvs(dir_post_delta[k])

    return tran_matrix


if __name__ == "__main__":

    # load data
    RETURN_PATH = './intermediate/daily_return'
    FACTORS_PATH = './intermediate/factors'
    y = np.loadtxt(RETURN_PATH + '/bank.txt')
    thr_factors = np.loadtxt(FACTORS_PATH + '/thr_factors.txt')
    (T, p) = y.shape
    factor_num = thr_factors.shape[1]

    n = p * factor_num
    x = np.zeros([T, n, p])
    for i in range(0, T):
        x[i] = np.kron(np.eye(p), thr_factors[i]).T

    # ===============
    #   set params
    # ===============
    # random seed
    np.random.seed(12345)
    # params for metropolis-hastings algorithm
    delta = 3
    tau = 0.0004
    rho = 0
    # initial state probability and state types
    initial_state_prob = np.array([0.9, 0.1])
    K = len(initial_state_prob)
    # premium factors beta
    beta_prior_mu = np.zeros(n)
    beta_prior_sigma = 1000 * np.eye(n)
    # transition matrix
    dir_prior_delta = np.ones([K, K]) / 2.
    # iterations
    mh_steps = 1000
    hiw_sample_num = 1
    burn_in_sample_num = 2000
    gibbs_sample_num = 10000
    itr = burn_in_sample_num + gibbs_sample_num

    # initial output array
    state_samples = np.zeros([itr, T])
    sigma_samples = np.zeros([itr, K, p, p])
    G_samples = np.zeros([itr, K])
    beta_samples = np.zeros([itr, K, n])
    tran_matrix_samples = np.zeros([itr, K, K])

    # initial values in iterations
    tran_matrix_samples[0] = np.array([[0.95, 0.05], [0.05, 0.95]])
    for k in range(0, K):
        beta_samples[0][k] = multivariate_normal.rvs(beta_prior_mu, beta_prior_sigma)
        sigma_samples[0][k] = np.eye(p)
        G_samples[0][k] = nx.Graph().add_nodes_from(range(0, p))

    print np.sum(state_sampling(2, y, x, beta_samples[0], sigma_samples[0], tran_matrix_samples[0], initial_state_prob))
    # start gibbs sampling
    t1 = time.time()
    for i in range(1, itr):
        t2 = time.time()
        print "==> {}th gibbs sampling starts:".format(i)

        state_samples[i] = state_sampling(K, y, x, beta_samples[i - 1], sigma_samples[i - 1], tran_matrix_samples[i - 1], initial_state_prob)
        # print np.sum(state_samples[i])
        print "    state sampling finished!"

        for k in range(0, K):
            t_k = np.where(state_samples[i] == k)[0]
            if not len(t_k):
                beta_samples[i][k] = beta_samples[i - 1][k]
                sigma_samples[i][k] = sigma_samples[i - 1][k]
                continue

            residual = np.zeros([len(t_k), p])
            for t in range(0, len(t_k)):
                residual[t] = y[t_k[t]] - np.dot(x[t_k[t]].T, beta_samples[i - 1][k])

            G_k = mt.metropolis_hastings(residual, delta, tau, rho, mh_steps)
            print "    graph sampling under state {} finished!".format(k)

            cliques_k = list(nx.find_cliques(G_k))
            sorted_cliques_k = mt.get_perfect_ordering_of_cliques(G_k, cliques_k)
            post_delta = delta + len(t_k)
            post_phi = tau * rho * (np.ones([p, p]) - np.eye(p)) + tau * np.eye(p) + np.dot(residual.T, residual)
            sigma_k, omega_k = hiw.hiw_sim(sorted_cliques_k, post_delta, post_phi, hiw_sample_num)
            sigma_samples[i][k] = sigma_k
            print "    sigma sampling under state {} finished!".format(k)

            beta_samples[i][k] = beta_sampling(y, x, t_k, beta_prior_mu, beta_prior_sigma, sigma_samples[i][k])
            print "    beta sampling under state {} finished!".format(k)

        tran_matrix_samples[i] = tran_matrix_sampling(dir_prior_delta, state_samples[i])
        print "    transition matrix sampling finished!"
        print "{}th gibbs sampling finished, time cost: {}s".format(i, round(time.time() - t2, 2))