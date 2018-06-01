# -*- coding: utf-8 -*-
# Author : xiequanbin
# Date:  18-02-10
# Email:  xquanbin072095@gmail.com


import time
import numpy as np
import networkx as nx
from scipy import sparse
from scipy.special import gammaln


def metropolis_hastings(data, delta, tau, rho, steps):
    # set a random seed
    np.random.seed(12345)

    data = centralization(data)
    # set parameters
    (T, p) = data.shape
    beta = 2. / (p - 1)
    phi = tau * rho * (np.ones([p, p]) - np.eye(p)) + tau * np.eye(p)
    n_edges = 0
    total_edges = p * (p - 1) / 2.

    # calculate the initial log post probability
    current_log_post = -p * T / 2. * np.log(2 * np.pi) + p * (gammaln((delta + T) / 2.) - gammaln(delta / 2.))
    for i in range(0, p):
        current_log_post = current_log_post + delta / 2. * np.log(phi[i, i] / 2.)
        current_log_post = current_log_post - (T + delta) / 2. * np.log((phi[i, i] + np.sum(data[:, i] * data[:, i])) / 2.)
    current_log_post = current_log_post + total_edges * np.log(1 - beta)

    # start MCMC
    G = nx.Graph()
    G.add_nodes_from(list(range(0, p)))
    for s in range(0, steps):
        cliques = list(nx.find_cliques(G))

        add = 0
        delete = 0
        decider = np.random.rand()
        if n_edges == 0 or (n_edges < total_edges and decider < 0.5):
            add = 1
            node1, node2 = rand_edge_add(G, n_edges, total_edges)
            print 'node chosen to add: {} and {}'.format(node1, node2)
        else:
            delete = 1
            node1, node2 = rand_edge_delete(G, n_edges)
            print 'node chosen to delete: {} and {}'.format(node1, node2)

        clique_size_is_2 = 0
        can_add = 0
        can_delete = 0

        if add:
            can_delete = 0
            if nx.node_connected_component(G, node1) != nx.node_connected_component(G, node2):
                can_add = 1
                clique_size_is_2 = 1

            if not clique_size_is_2:
                # if two nodes belong to the same connected component, add edge between node1 and node2
                # according to Theorem2 in the paper: Decomposable graphical Gaussian modeldetermination, GIUDICI
                can_add = add_logic(cliques, p, node1, node2)

                # get the new clique and the size of it after add edge
                new_clique, new_clique_size = get_new_clique_aft_add(G, node1, node2)
                if new_clique_size == 2:
                    clique_size_is_2 = 1

        if delete:
            can_add = 0
            # if an edge is only contained in exactly one clique, then we can delete it
            if delete_logic(cliques, node1, node2):
                can_delete = 1
                # get the  clique consist of the two nodes and the size of it before delete edge
                new_clique, new_clique_size = get_clique_bfr_delete(cliques, node1, node2)
                if new_clique_size == 2:
                    clique_size_is_2 = 1

        # calculate the change in log post probability after add or delete edge
        if (can_add or can_delete) and clique_size_is_2:
            post_ratio = post_update(data, T, phi, delta, beta, node1, node2)
            if add:
                delta_log_post = post_ratio
                post_ratio = np.exp(post_ratio) / (n_edges + 1.) * (total_edges - n_edges)

            if delete:
                delta_log_post = - post_ratio
                post_ratio = np.exp(delta_log_post) / (total_edges - n_edges + 1.) * n_edges

            if n_edges == 0 or n_edges == total_edges:
                post_ratio = post_ratio / 2.

        elif(can_add or can_delete) and not clique_size_is_2:
            post_ratio = post_update_all(new_clique, data, T, phi, delta, beta, node1, node2)

            if add:
                delta_log_post = post_ratio
                post_ratio = np.exp(post_ratio) / (n_edges + 1.) * (total_edges - n_edges)

            if delete:
                delta_log_post = - post_ratio
                post_ratio = np.exp(delta_log_post) / (total_edges - n_edges + 1.) * n_edges

            if n_edges == 0 or n_edges == total_edges:
                post_ratio = post_ratio / 2.

        deciding = np.random.rand()
        # decide to add or delete an edge at last by comparing post_ratio with a random number
        if (can_add and post_ratio > deciding) or (can_delete and post_ratio > deciding):
            if can_add:
                G.add_edge(node1, node2)
                n_edges = n_edges + 1
                current_log_post = current_log_post + delta_log_post
                print "success to add edge"

            if can_delete:
                G.remove_edge(node1, node2)
                n_edges = n_edges - 1
                current_log_post = current_log_post + delta_log_post
                print "success to delete edge"

        else:
            delta_log_post = 0

        print 'edges: {}'.format(G.edges), "n_edges:{}".format(n_edges)
        print "{} iteration(s) finished!".format(s+1)

    return G


def centralization(data):
    adj_data = data - data.mean(axis=0)
    return adj_data


def rand_edge_add(G, n_edges, total_edges):
    p = G.number_of_nodes()
    unconnected_edges_num = 0
    node1 = 0
    if total_edges > n_edges:
        thr = np.random.randint(1, 2 * (total_edges - n_edges))
        while unconnected_edges_num + p - 1 - G.degree[node1] < thr:
            unconnected_edges_num += p - 1 - G.degree[node1]
            node1 += 1

        neighbors = sorted(G[node1].keys())
        node2 = 0
        while unconnected_edges_num < thr:
            if node2 != node1:
                if (len(neighbors) != 0) and (neighbors[0] == node2):
                    neighbors.remove(node2)
                else:
                    unconnected_edges_num += 1

            if unconnected_edges_num < thr:
                node2 += 1
    else:
        raise ValueError('Current edges should not be larger than maximum edges, '
                         'current edge:{}, maximum edge:{}'.format(n_edges, total_edges))
    return node1, node2


def rand_edge_delete(G, n_edges):
    connected_edges_num = 0
    node1 = 0
    thr = np.random.randint(1, 2 * n_edges)
    while connected_edges_num + G.degree[node1] < thr:
        connected_edges_num += G.degree[node1]
        node1 += 1

    neighbors = sorted(G[node1].keys())
    node2 = neighbors[thr - connected_edges_num - 1]

    return node1, node2


def add_logic(cliques, nodes_num, node1, node2):
    can_add = 0
    R = []
    T = []
    index1 = []
    index2 = []

    for i, c in enumerate(cliques):
        if node1 in c:
            R.append([cc for cc in c if node1 != cc])
            index1.append(i)
        if node2 in c:
            T.append([cc for cc in c if node2 != cc])
            index2.append(i)

    ns = np.ones([nodes_num, 1])
    junction_tree = cliques_to_jtree(cliques, nodes_num, ns)
    clique_G = nx.from_numpy_matrix(junction_tree)

    for i in range(0, len(index1)):
        for j in range(0, len(index2)):
            seperators = []
            shortest_path = nx.shortest_path(clique_G, source=index1[i], target=index2[j])
            for k in range(1, len(shortest_path)):
                seperators.append(set(cliques[shortest_path[k-1]]) & set(cliques[shortest_path[k]]))

            S = set(R[i]) & set(T[j])
            if len(S) != 0 and S in seperators:
                can_add = 1
                break

        if can_add:
            break

    return can_add


def cliques_to_jtree(cliques, nodes_num, ns):
    """
        A junction tree is a tree that satisfies the jtree property, which says:
    for each pair of cliques U,V with intersection S, all cliques on the path between U and V contain S.
    (This ensures that local propagation leads to global consistency.)
    For details, see Jensen and Jensen, "Optimal Junction Trees", UAI 94.
    :param cliques: /
    :param nodes_num: total number of nodes in a graph
    :param ns: ns[i] = number of values node i can take on
    :return junction_tree: junction_tree[i,j] = 1 if cliques i and j are connected
    """
    cliques_num = len(cliques)
    weight = np.zeros([cliques_num, 1])

    sparse_clique = sparse.lil_matrix((cliques_num, nodes_num))
    for i in range(0, cliques_num):
        sparse_clique[i, cliques[i]] = 1
        weight[i] = np.prod(ns[cliques[i]])

    primary_cost = sparse_clique.dot(sparse_clique.T).toarray()
    primary_cost = primary_cost - np.diag(np.diag(primary_cost))

    w = np.tile(weight, (1, cliques_num))
    secondary_cost = w + w.T
    secondary_cost = secondary_cost - np.diag(np.diag(secondary_cost))

    junction_tree = minimum_spanning_tree(-primary_cost, secondary_cost)   # Using -primary_cost gives maximum spanning tree
    # The root is arbitrary, but since the first pass is towards the root,
    # we would like this to correspond to going forward in time in a DBN.
    root = cliques_num

    return junction_tree


def minimum_spanning_tree(primary_cost, secondary_cost):
    # primary_cost = primary_cost.toarray()
    n = len(primary_cost)#.shape[0]
    cliques_adj = np.zeros([n, n])

    closest_clique = np.zeros(n, dtype=int)   # closest__clique[i] = ii if clique i's closest clique is ii
    used_clique = np.zeros(n)  # used_clique[i] = 1 if clique i used else 0
    used_clique[0] = 1
    primary_cost[primary_cost == 0] = np.inf
    secondary_cost[secondary_cost == 0] = np.inf
    low_cost1 = primary_cost[0, :]
    low_cost2 = secondary_cost[0, :]

    for i in range(1, n):
        ks = np.where(low_cost1 == low_cost1.min())[0]
        k = ks[np.argmin(low_cost2[ks])]
        cliques_adj[k, closest_clique[k]] = 1
        cliques_adj[closest_clique[k], k] = 1
        low_cost1[k] = np.inf
        low_cost2[k] = np.inf
        used_clique[k] = 1
        unused_clique = np.where(used_clique == 0)[0]
        for u in unused_clique:
            if primary_cost[k, u] < low_cost1[u]:
                low_cost1[u] = primary_cost[k, u]
                low_cost2[u] = secondary_cost[k, u]
                closest_clique[u] = k

    return cliques_adj


def delete_logic(cliques, node1, node2):
    can_delete = 0
    count = 0
    for c in cliques:
        if node1 in c and node2 in c:
            count += 1

    if count == 1:
        can_delete = 1

    return can_delete


def get_new_clique_aft_add(G, node1, node2):
    G.add_edge(node1, node2)
    cliques = nx.find_cliques(G)
    for c in cliques:
        if node1 in c and node2 in c:
            new_cliques = c
            new_cliques_size = len(new_cliques)
            break
    G.remove_edge(node1, node2)

    return new_cliques, new_cliques_size


def get_clique_bfr_delete(cliques, node1, node2):
    for c in cliques:
        if node1 in c and node2 in c:
            new_clique = c
            new_clique_size = len(new_clique)
            break

    return new_clique, new_clique_size


def post_update(data, data_rows_num, phi, delta, beta, node1, node2):
    det_phi_Sa = phi[node1, node1]
    det_phi_Sb = phi[node2, node2]
    det_phi_Sab = det_phi_Sa * det_phi_Sb - phi[node1, node2] * phi[node2, node1]

    det_phi_Sy_Sa = det_phi_Sa + np.sum(data[:, node1] * data[:, node1])
    det_phi_Sy_Sb = det_phi_Sb + np.sum(data[:, node2] * data[:, node2])
    det_phi_Sy_Sab = det_phi_Sy_Sa * det_phi_Sy_Sb - (phi[node1, node2] + np.sum(data[:, node1] * data[:, node2])) * \
                     (phi[node2, node1] + np.sum(data[:, node1] * data[:, node2]))

    post_ratio = (delta + 1) * np.log(det_phi_Sab)
    post_ratio -= delta * (np.log(det_phi_Sa) + np.log(det_phi_Sb))
    post_ratio -= (delta + data_rows_num + 1) * np.log(det_phi_Sy_Sab)
    post_ratio += (delta + data_rows_num) * (np.log(det_phi_Sy_Sa) + np.log(det_phi_Sy_Sb))
    post_ratio = post_ratio/2. + gammaln(delta/2.) + gammaln((delta + data_rows_num + 1)/2.)
    post_ratio = post_ratio - gammaln((delta + 1)/2.) - gammaln((delta + data_rows_num)/2.)
    post_ratio = post_ratio + np.log(beta) - np.log(1 - beta)

    return post_ratio


def post_update_all(new_clique, data, data_rows_num, phi, delta, beta, node1, node2):
    new_clique_size = len(new_clique)
    for ni in range(0, new_clique_size):
        if node1 == new_clique[ni]:
            new_clique[ni] = new_clique[-2]
            new_clique[-2] = node1
            break

    for ni in range(0, new_clique_size):
        if node2 == new_clique[ni]:
            new_clique[ni] = new_clique[-1]
            new_clique[-1] = node2
            break

    Sy = np.dot(data.T, data)
    phi_Sy = phi + Sy
    phi = phi[new_clique, :][:, new_clique]
    phi_Sy = phi_Sy[new_clique, :][:, new_clique]

    chol_L_phi = np.linalg.cholesky(phi)
    chol_diag_phi = np.diag(chol_L_phi)
    chol_L_phi_Sy = np.linalg.cholesky(phi_Sy)
    chol_diag_phi_Sy = np.diag(chol_L_phi_Sy)

    det_phi_DD = chol_diag_phi[-2]**2 * chol_diag_phi[-1]**2
    det_phi_ii = chol_diag_phi[-2]**2
    det_phi_jj = chol_diag_phi[-2]**2 + chol_L_phi[-1, -2]**2

    det_phi_Sy_DD = chol_diag_phi_Sy[-2]**2 * chol_diag_phi_Sy[-1]**2
    det_phi_Sy_ii = chol_diag_phi_Sy[-2]**2
    det_phi_Sy_jj = chol_diag_phi_Sy[-1]**2 + chol_L_phi_Sy[-1, -2]**2

    post_ratio = (delta + new_clique_size - 1)/2. * np.log(det_phi_DD)
    post_ratio -= (delta + new_clique_size - 2)/2. * (np.log(det_phi_ii) + np.log(det_phi_jj))
    post_ratio += (delta + new_clique_size + data_rows_num - 2)/2. * (np.log(det_phi_Sy_ii) + np.log(det_phi_Sy_jj))
    post_ratio -= (delta + new_clique_size + data_rows_num -1)/2. * np.log(det_phi_Sy_DD)

    post_ratio += gammaln((delta + new_clique_size -2)/2.)
    post_ratio += gammaln((delta + new_clique_size + data_rows_num - 1)/2.)
    post_ratio -= gammaln((delta + new_clique_size - 1)/2.)
    post_ratio -= gammaln((delta + new_clique_size + data_rows_num - 2)/2.)

    post_ratio = post_ratio + np.log(beta) - np.log(1 - beta)

    return post_ratio


def get_perfect_ordering_of_cliques(G, cliques):
    p = len(G.node)

    # At first, find a perfect elimination ordering of vertices using maximum cardinality search algorithm
    card = np.zeros(p)
    sorted_nodes = []
    unlabeled_nodes = range(0, p)
    while len(unlabeled_nodes) > 0:
        temp = np.where(card == card[unlabeled_nodes].max())[0]
        select_node = list(set(temp) - set(sorted_nodes))
        select_node = select_node[0]

        neighbors = G[select_node].keys()
        if neighbors:
            for nb in neighbors:
                card[nb] += 1

        sorted_nodes.append(select_node)
        unlabeled_nodes = list(set(range(0, p)) - set(sorted_nodes))

    # For decomposable graphs, the ordering of vertices established defines an ordering of cliques,
    # where the cliques are ordered by the highest numbered vertex contained in each. (jones etc, 2005)
    clique_num = len(cliques)
    scores = np.zeros(clique_num)   # scores[i] reports the cliques[i]'s rank score
    for ci in range(0, clique_num):
        i_score_list = [sorted_nodes.index(nd) for nd in cliques[ci]]
        scores[ci] = np.max(i_score_list)

    perfect_ordering = np.argsort(scores)   # the No.(i+1) cliques in a perfect ordering is perfect_ordering[i]
    sorted_cliques = []
    for i in range(0, clique_num):
        sorted_cliques.append(cliques[perfect_ordering[i]])

    return sorted_cliques


if __name__ == "__main__":

    # load test data
    data_list = []
    with open('./input/data_15.txt', 'r') as f:
        lines = f.readlines()
        for line in lines:
            sub_data = [float(i) for i in line.strip().split()]
            data_list.append(sub_data)
    data = np.array(data_list)

    # set params for metropolis hastings algorithm
    delta = 3
    tau = 0.0004
    rho = 0
    steps = 1000

    # get the graph after steps iteration
    start_time = time.time()
    G = metropolis_hastings(data, delta, tau, rho, steps)
    end_time = time.time()
    print "time cost: {}s".format(round(end_time - start_time), 2)