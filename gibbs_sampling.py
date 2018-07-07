# -*- coding: utf-8 -*-
# Author : xiequanbin
# Date:  18-06-01
# Email:  xquanbin072095@gmail.com


import os
import time
import pickle
import numpy as np
import networkx as nx
import metropolis as mt
import hiw_simulation as hiw
import matplotlib.pyplot as plt
from seaborn import color_palette
from matplotlib.colors import LinearSegmentedColormap
from scipy.stats import multivariate_normal, dirichlet
from mvnrnd import mvnrnd


# some useful paths
RETURN_PATH = './intermediate/daily_return'
INFO_PATH = './intermediate/stock_info'
FACTORS_PATH = './intermediate/factors'
OUTPUT_PATH = './output'
OUTPUT_FIGURE_PATH = OUTPUT_PATH + '/figure'
if not os.path.exists(OUTPUT_PATH):
    os.mkdir(OUTPUT_PATH)
if not os.path.exists(OUTPUT_FIGURE_PATH):
    os.mkdir(OUTPUT_FIGURE_PATH)


def Gibbs_sampling(y, x, mh_params, beta_params, tran_params, itr_params, save_path=OUTPUT_PATH, save_batch=True, batch_size=1000):

    if save_batch:
        temp_save_path = "./temp_output"
        if not os.path.exists(temp_save_path):
            os.mkdir(temp_save_path)

    # params
    # params for metropolis-hastings algorithm
    delta = mh_params['delta']
    tau = mh_params['tau']
    rho = mh_params['rho']
    # premium factors beta
    beta_prior_mu = beta_params['prior_mu']
    beta_prior_sigma = beta_params['prior_sigma']
    # initial state probability, state types and transition matrix
    initial_state_prob = tran_params['initial_state']
    K = len(initial_state_prob)
    dir_prior_delta = tran_params['prior_delta']
    # iterations
    mh_steps = itr_params['mh']
    hiw_sample_num = itr_params['hiw']
    burn_in_sample_num = itr_params['burn_in']
    gibbs_sample_num = itr_params['gibbs']
    itr = burn_in_sample_num + 2 * gibbs_sample_num     # storing every other of (2 * gibbs_sample_num) simulations

    # initial output array
    state_samples = np.zeros([itr, T])
    sigma_samples = np.zeros([itr, K, p, p])
    G_samples = np.zeros([itr, K]).tolist()
    beta_samples = np.zeros([itr, K, n])
    tran_matrix_samples = np.zeros([itr, K, K])

    # initial values in iterations
    tran_matrix_samples[0] = np.array([[0.95, 0.05], [0.05, 0.95]])
    for k in range(0, K):
        # beta_samples[0][k] = multivariate_normal.rvs(beta_prior_mu, beta_prior_sigma)
        beta_samples[0][k] = mvnrnd(beta_prior_mu, beta_prior_sigma)
        sigma_samples[0][k] = 100 * np.eye(p)
        graph_k = nx.Graph()
        graph_k.add_nodes_from(range(0, p))
        G_samples[0][k] = graph_k

    # start gibbs sampling
    t1 = time.time()
    for i in range(1, itr):
        t2 = time.time()
        print "==> {}th gibbs sampling starts:".format(i)

        state_samples[i] = state_sampling(K, y, x, beta_samples[i - 1], sigma_samples[i - 1], tran_matrix_samples[i - 1], initial_state_prob)
        print np.sum(state_samples[i])
        print "    state sampling is finished!"

        state_k_len = [len(np.where(state_samples[i] == k)[0]) for k in range(0, K)]
        if not np.prod(state_k_len):
            beta_samples[i] = beta_samples[i-1]
            sigma_samples[i] = sigma_samples[i-1]
            G_samples[i] = G_samples[i-1]
            tran_matrix_samples[i] = tran_matrix_samples[i-1]
            continue

        for k in range(0, K):
            t_k = np.where(state_samples[i] == k)[0]
            residual = np.zeros([len(t_k), p])
            for t in range(0, len(t_k)):
                residual[t] = y[t_k[t]] - np.dot(x[t_k[t]].T, beta_samples[i - 1][k])

            G_k = mt.metropolis_hastings(residual, delta, tau, rho, mh_steps)
            G_samples[i][k] = G_k
            print "    graph sampling under state {} is finished!".format(k)
            
            cliques_k = list(nx.find_cliques(G_k))
            sorted_cliques_k = mt.get_perfect_ordering_of_cliques(G_k, cliques_k)
            post_delta = delta + len(t_k)
            post_phi = tau * rho * (np.ones([p, p]) - np.eye(p)) + tau * np.eye(p) + np.dot(residual.T, residual)
            sigma_k, omega_k = hiw.hiw_sim(sorted_cliques_k, post_delta, post_phi, hiw_sample_num)
            sigma_samples[i][k] = sigma_k[-1]
            print "    sigma sampling under state {} is finished!".format(k)
            
            beta_samples[i][k] = beta_sampling(y, x, t_k, beta_prior_mu, beta_prior_sigma, omega_k[-1])
            print "    beta sampling under state {} is finished!".format(k)

        tran_matrix_samples[i] = tran_matrix_sampling(dir_prior_delta, state_samples[i])
        print "    transition matrix sampling is finished!"
        print "{}th gibbs sampling is finished, time cost: {}s".format(i, round(time.time() - t2, 2))

        # save the samples array to txt by batch
        if not (i+1) % batch_size:
            if save_batch:
                temp_output_dict = {'state_samples': state_samples[0: i+1], 'sigma_samples': sigma_samples[0: i+1],
                                    'graph_samples': G_samples[0: i+1], 'beta_samples': beta_samples[0: i+1],
                                    'transition_matrix_samples': tran_matrix_samples[0: i+1]}
                with open(temp_save_path + '/samples{}.txt'.format(i+1), 'wb') as temp_f:
                    pickle.dump(temp_output_dict, temp_f, -1)

    # save the samples array to txt
    output_dict = {'state_samples': state_samples[burn_in_sample_num::2], 'sigma_samples': sigma_samples[burn_in_sample_num::2],
                   'graph_samples': G_samples[burn_in_sample_num::2], 'beta_samples': beta_samples[burn_in_sample_num::2],
                   'transition_matrix_samples': tran_matrix_samples[burn_in_sample_num::2]}
    with open(save_path + '/samples.txt', 'wb') as f:
        pickle.dump(output_dict, f, -1)

    print "Congratulations, gibbs sampling is finished! time cost: {} min".format(round((time.time() - t1)/60., 2))

    return output_dict


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


def beta_sampling(y, x, t_k, prior_mu, prior_sigma, omega):
    post_mu = np.dot(np.linalg.inv(prior_sigma), prior_mu)
    post_sigma = np.linalg.inv(prior_sigma)
    for t in t_k:
        post_mu += np.dot(np.dot(x[t], omega), y[t])
        post_sigma += np.dot(np.dot(x[t], omega), x[t].T)

    post_sigma = np.linalg.inv(post_sigma)
    post_mu = np.dot(post_sigma, post_mu)

    # For some reason there are very small numerical issues such as round-off that make post_sigma not quite symmetric,
    # so here we use a usual trick to force it to be exactly symmetric.
    post_sigma = (post_sigma + post_sigma.T) / 2.
    # beta_sample = multivariate_normal.rvs(post_mu, post_sigma)
    beta_sample = mvnrnd(post_mu, post_sigma)

    return beta_sample


def tran_matrix_sampling(dir_prior_delta, state_samples):
    T = len(state_samples)
    K = len(dir_prior_delta)

    N = np.zeros([K, K])
    tran_matrix = np.zeros([K, K])

    for i in range(0, K):
        for j in range(0, K):
            for t in range(0, T - 1):
                # if (state_samples[t] == j) and (state_samples[t + 1] == i):
                if (state_samples[t] == i) and (state_samples[t + 1] == j):
                    N[i, j] += 1

    dir_post_delta = dir_prior_delta + N
    for k in range(0, K):
        tran_matrix[k] = dirichlet.rvs(dir_post_delta[k])

    return tran_matrix


if __name__ == "__main__":

    # load data
    y = np.loadtxt(RETURN_PATH + '/all.txt')
    stk_info = np.loadtxt(INFO_PATH + '/all_info.txt', dtype="str", encoding='utf-8')
    risk_factors = np.loadtxt(FACTORS_PATH + '/capm.txt')
    (T, p) = y.shape
    factor_num = risk_factors.shape[1]

    n = p * factor_num
    x = np.zeros([T, n, p])
    for i in range(0, T):
        x[i] = np.kron(np.eye(p), risk_factors[i]).T

    # ==============================================
    #                   set params
    # ==============================================
    # random seed
    np.random.seed(12345)
    # params for metropolis-hastings algorithm
    delta = 3
    tau = 0.0001
    rho = 0
    # premium factors beta
    beta_prior_mu = np.zeros(n)
    beta_prior_sigma = 100 * np.eye(n)
    for i in range(0, p):
        beta_prior_sigma[factor_num * i, factor_num * i] = 1   # special for the intercepts   
    # initial state probability, state types and transition matrix
    initial_state_prob = np.array([0.5, 0.5])
    K = len(initial_state_prob)
    dir_prior_delta = np.ones([K, K]) / 2.
    # iterations
    mh_steps = 1000
    hiw_sample_num = 1
    burn_in_sample_num = 2000
    gibbs_sample_num = 5000
    itr = burn_in_sample_num + 2 * gibbs_sample_num     # storing every other of (2 * gibbs_sample_num) simulations

    # if samples have not been generated, run the Gibbs_sampling, else read samples from samples.txt .
    if not os.path.exists(OUTPUT_PATH + '/samples.txt'):
        mh_params = {'delta': delta, 'tau': tau, 'rho': rho}
        beta_params = {'prior_mu': beta_prior_mu, 'prior_sigma': beta_prior_sigma}
        tran_params = {'initial_state': initial_state_prob, 'prior_delta': dir_prior_delta}
        itr_params = {'mh': mh_steps, 'hiw': hiw_sample_num, 'burn_in': burn_in_sample_num, 'gibbs': gibbs_sample_num}
        samples_dict = Gibbs_sampling(y, x, mh_params, beta_params, tran_params, itr_params, save_batch=False)
    else:
        with open(OUTPUT_PATH + '/samples.txt', 'rb') as f:
            samples_dict = pickle.load(f)

    state_samples = samples_dict['state_samples']
    sigma_samples = samples_dict['sigma_samples']
    G_samples = samples_dict['graph_samples']
    beta_samples = samples_dict['beta_samples']
    tran_matrix_samples = samples_dict['transition_matrix_samples']

    # ==================================================================
    #               statistical analysis of gibbs samples
    # ==================================================================
    # some information about industry
    stk_info[stk_info == "银行"] = "金融"
    industry_type = np.unique(stk_info[:, 3])
    industry_num = len(industry_type)
    industry_type_dict = {"信息": "Tech", "公用": "Utils", "医药": "Health Care", "可选": "Cons.Disc.",
                          "工业": "Industrials", "材料": "Materials", "消费": "Cons.", "电信": "Tel",
                          "能源": "Energy", "金融": "Financials"}
    pos_dict = {}
    pos_list = []
    for i in range(0, industry_num):
        pos_dict[industry_type[i]] = list(np.where(stk_info == industry_type[i])[0])
        pos_list += pos_dict[industry_type[i]]

    # identify systemic risk state
    idtf = np.mean(state_samples) > 0.5
    if idtf:
        labels = {0: "High", 1: "Low"}
        high_risk_state_samples = 1 - state_samples
    else:
        labels = {0: "Low", 1: "High"}
        high_risk_state_samples = state_samples
    
    # high systemic risk probability
    high_risk_prob = np.mean(high_risk_state_samples, axis=0)
    trading_date = np.loadtxt('./intermediate/trading_date.txt',dtype='str')
    fig1 = plt.figure(figsize=(12, 6))
    ax1 = fig1.add_subplot(111)
    # xticks = range(0, len(high_risk_prob))[::300]
    xticks = [0, 302, 607, 915, 1218, 1514, 1820, 2129, 2433]
    ax1.stem(high_risk_prob, linefmt='C0-', markerfmt='C0')
    plt.xlim([0, len(high_risk_prob)])
    plt.ylim([0, 1])
    ax1.set_xticks(xticks)
    ax1.set_xticklabels(trading_date[xticks])
    plt.xlabel('trading date')
    plt.ylabel('high risk probability')
    fig1.savefig(OUTPUT_FIGURE_PATH + "/high risk probability.png", dpi=fig1.dpi)

    # transition probabilities
    k2k_tran_prob = np.zeros([gibbs_sample_num, K])
    for k in range(0, K):
        k2k_tran_prob[:, k] = tran_matrix_samples[:, k, k]
    fig2 = plt.figure(figsize=(8, 6))
    ax2 = fig2.add_subplot(111)
    ax2.boxplot(k2k_tran_prob, 0, '')
    ax2.set_xticklabels(['From {} to {}'.format(labels[0], labels[0]), 
                         'From {} to {}'.format(labels[1], labels[1])])
    plt.ylabel('Transition Probability')
    fig2.savefig(OUTPUT_FIGURE_PATH + '/transition probabilities.png', dpi=fig2.dpi)

    # beta coefficients distribution
    fig3, axes = plt.subplots(nrows=factor_num // 2, ncols=2, figsize=(36, 12))
    fig3.tight_layout(pad=3, h_pad=3, w_pad=3)
    beta_title = {0: "Alphas", 1: "Market Betas", 2: "SMB Betas",
                  3: "HML Betas", 4: 'RMW Betas', 5: 'CMA Betas'}
    beta_diff = np.zeros([factor_num, gibbs_sample_num, p])
    for fn in range(0, factor_num):
        nn = range(fn, fn + n, factor_num)
        beta_diff[fn, :, :] = (beta_samples[:, 0, nn] - beta_samples[:, -1, nn]) * (-1)**idtf
        beta_diff[fn, :, :] = beta_diff[0][:, pos_list]     # sorted by Industry

        xticks_list = []
        for i in range(0, industry_num):
            xticks_list.append(pos_dict[industry_type[i]][len(pos_dict[industry_type[i]]) // 2])
        for i in range(0, len(xticks_list)):
            xticks_list[i] = pos_list.index(xticks_list[i])

        if factor_num // 2 > 1:
            axes[fn // 2][fn % 2].boxplot(beta_diff[fn, :, :], 0, '', whiskerprops={'linestyle': ':'})
            axes[fn // 2][fn % 2].set_xticks(xticks_list)
            axes[fn // 2][fn % 2].set_xticklabels([industry_type_dict[i] for i in industry_type], fontsize=15)
            axes[fn // 2][fn % 2].set_yticklabels(axes[fn // 2][fn % 2].get_yticks(), fontsize=14)
            axes[fn // 2][fn % 2].set_title(beta_title[fn], fontsize=20)
        else:
            axes[fn % 2].boxplot(beta_diff[fn, :, :], 0, '', whiskerprops={'linestyle': ':'})
            axes[fn % 2].set_xticks(xticks_list)
            axes[fn % 2].set_xticklabels([industry_type_dict[i] for i in industry_type], fontsize=15)
            axes[fn % 2].set_yticklabels(axes[fn % 2].get_yticks(), fontsize=14)
            axes[fn % 2].set_title(beta_title[fn], fontsize=20)
    fig3.savefig(OUTPUT_FIGURE_PATH + '/betas difference.png', dpi=fig3.dpi)

    # Draw weighted networks example
    fig4, axes4 = plt.subplots(nrows=2, ncols=1, figsize=(12, 30))
    nodes_name = {u: v for u, v in enumerate(stk_info[:, 2])}
    for k in range(0, K):
        G_example = G_samples[-1][k]
        pos = nx.spring_layout(G_example)
        importance_dict = nx.eigenvector_centrality_numpy(G_example)
        nx.draw_networkx_edges(G_example, pos, alpha=0.7, ax=axes4[k])
        nx.draw_networkx_nodes(G_example, pos, nodelist=list(importance_dict.keys()),
                               node_size=360,
                               node_color=list(importance_dict.values()),
                               cmap=LinearSegmentedColormap.from_list('a', color_palette("Reds", n_colors=12)[:8]),
                               ax=axes4[k])
        # nx.draw_networkx_labels(G_example, pos, labels=nodes_name, ax=axes4[k], font_color='black', font_size=10)
        nx.draw_networkx_labels(G_example, pos, ax=axes4[k], font_color='black', font_size=10)
        # nx.draw_networkx(G_example,
        #                  node_size=600,
        #                  node_color=importance_dict.values(),
        #                  cmap=plt.get_cmap('Reds'),
        #                  font_color = 'white',
        #                  font_size = 14,
        #                  ax=axes4[k])
        axes4[k].set_title(labels[k] + " Systemic Risk")
        axes4[k].axis('off')
    fig4.savefig(OUTPUT_FIGURE_PATH + "/networks.png")

    # Degree Centrality, Standard Eigenvector Centrality and Weighted Eigenvector Centrality for individual stocks
    dgr_cen = np.zeros([gibbs_sample_num, K, p])
    std_eig_cen = np.zeros([gibbs_sample_num, K, p])
    weighted_eig_cen = np.zeros([gibbs_sample_num, K, p])

    for i in range(0, gibbs_sample_num):
        for k in range(0, K):

            # in order to get weighted eigenvector centrality, weights are defined by the covariance terms.
            G_ik = G_samples[i][k].copy()
            for node1, node2 in G_ik.edges:
                G_ik[node1][node2]['weight'] = sigma_samples[i, k, node1, node2]

            dgr_dict = nx.degree_centrality(G_ik)
            std_eig_dict = nx.eigenvector_centrality_numpy(G_ik)
            weighted_eig_dict = nx.eigenvector_centrality_numpy(G_ik, weight='weight')
            dgr_cen[i, k, :] = [dgr_dict[e] for e in sorted(dgr_dict.keys())]
            std_eig_cen[i, k, :] = [std_eig_dict[e] for e in sorted(std_eig_dict.keys())]
            weighted_eig_cen[i, k, :] = [weighted_eig_dict[e] for e in sorted(weighted_eig_dict.keys())]

    median_dgr_cen = np.median(dgr_cen, axis=0)
    median_std_eig_cen = np.median(std_eig_cen, axis=0)
    median_weighted_eig_cen = np.median(weighted_eig_cen, axis=0)
    cen_dict = {# 'Median Degree Centrality': median_dgr_cen,
                'Median Standard Eigenvector Centrality': median_std_eig_cen,
                'Median Weighted Eigenvector Centrality': median_weighted_eig_cen}
    keys = cen_dict.keys()

    fig5, axes5 = plt.subplots(nrows=len(keys), ncols=1, figsize=(12, 10))
    fig5.tight_layout(pad=3, h_pad=4, w_pad=3)
    rank_num = 20
    rank = range(1, p + 1)[: rank_num]
    for m in range(0, len(keys)):
        for k in range(0, K):
            cen = cen_dict[keys[m]][k]
            ranked_node = np.argsort(-cen)[: rank_num]
            axes5[m].plot(cen[ranked_node], marker='^', label=labels[k] + " Systemic Risk")
            for i in rank:
                axes5[m].text(i - 1 + 0.1, cen[ranked_node[i - 1]], str(ranked_node[i - 1]), va='bottom', ha='left',
                              wrap=True)
        axes5[m].set_xticks(range(0, p)[: rank_num])
        axes5[m].set_xticklabels(rank)
        axes5[m].set_xlim([-1, np.min([p, rank_num]) - 1])
        axes5[m].legend()
        axes5[m].set_xlabel('Ranking')
        axes5[m].set_title(keys[m])
    fig5.savefig(OUTPUT_FIGURE_PATH + "/Individual Stocks' Centrality Rank.png")
    
    # Degree Centrality, Standard Eigenvector Centrality and Weighted Eigenvector Centrality for industries
    if industry_num > 1:
        ind_dgr_cen = np.zeros([gibbs_sample_num, K, industry_num])
        ind_std_eig_cen = np.zeros([gibbs_sample_num, K, industry_num])
        ind_weighted_eig_cen = np.zeros([gibbs_sample_num, K, industry_num])
        
        for i in range(0, industry_num):
            pos = pos_dict[industry_type[i]]
            ind_dgr_cen[:, :, i] = np.mean(dgr_cen[:, :, pos], axis=2)
            ind_std_eig_cen[:, :, i] = np.mean(std_eig_cen[:, :, pos], axis=2)
            ind_weighted_eig_cen[:, :, i] = np.mean(weighted_eig_cen[:, :, pos], axis=2)
        
        median_ind_dgr_cen = np.median(ind_dgr_cen, axis=0)
        median_ind_std_eig_cen = np.median(ind_std_eig_cen, axis=0)
        median_ind_weighted_eig_cen = np.median(ind_weighted_eig_cen, axis=0)
        ind_cen_dict = {#  'Industry Median Degree Centrality': median_ind_dgr_cen,
                    'Industry Median Standard Eigenvector Centrality': median_ind_std_eig_cen,
                    'Industry Median Weighted Eigenvector Centrality': median_ind_weighted_eig_cen}
        ind_keys = ind_cen_dict.keys()
        
        fig6, axes6 = plt.subplots(nrows=len(ind_keys), ncols=1, figsize=(12, 10))
        fig6.tight_layout(pad=3, h_pad=4, w_pad=3)
        ind_rank = range(1, industry_num + 1)
        for m in range(0, len(ind_keys)):
            for k in range(0, K):
                cen = ind_cen_dict[ind_keys[m]][k]
                ranked_node = np.argsort(-cen)
                axes6[m].plot(cen[ranked_node], marker='^', label=labels[k] + " Systemic Risk")
                for i in ind_rank:
                    axes6[m].text(i-1 + 0.1, cen[ranked_node[i-1]], industry_type_dict[industry_type[ranked_node[i-1]]],
                                  va='bottom', ha='left', wrap=True)
            axes6[m].set_xticks(range(0, industry_num))
            axes6[m].set_xticklabels(ind_rank)
            axes6[m].set_xlim([-1, industry_num - 1])
            axes6[m].legend()
            axes6[m].set_xlabel('Ranking')
            axes6[m].set_title(ind_keys[m])
        fig6.savefig(OUTPUT_FIGURE_PATH + "/Industries' Centrality Rank.png")