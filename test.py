# -*- coding: utf-8 -*-
# Author : xiequanbin
# Date:  18-02-10
# Email:  xquanbin072095@gmail.com


import numpy as np
import networkx as nx
from scipy import sparse
from scipy.special import gammaln



def metropolis_hastings(data, delta, tau, rho, steps):
    decider_list = []
    with open('./input/decider.txt', 'r') as f:
        lines = f.readlines()
        for line in lines:
            sub_data = [float(i) for i in line.strip().split()]
            decider_list += sub_data

    deciding_list = []
    with open('./input/deciding.txt', 'r') as f:
        lines = f.readlines()
        for line in lines:
            sub_data = [float(i) for i in line.strip().split()]
            deciding_list += sub_data

    add_int_list = []
    with open('./input/add_int.txt', 'r') as f:
        lines = f.readlines()
        for line in lines:
            sub_data = [float(i) for i in line.strip().split()]
            add_int_list += sub_data

    delete_int_list = []
    with open('./input/delete_int.txt', 'r') as f:
        lines = f.readlines()
        for line in lines:
            sub_data = [float(i) for i in line.strip().split()]
            delete_int_list += sub_data

    # set a random seed
    # np.random.seed(12345)

    data = centralization(data)
    # set parameters
    (T, p) = data.shape
    adj = np.zeros([p])
    beta = 2. / (p - 1)
    n_edges = 0
    total_edges = p * (p - 1) / 2.
    phi = tau * rho * (np.ones([p, p]) - np.eye(p)) + tau * np.eye(p)

    # calculate the initial log post probability
    current_log_post = -p * T / 2. * np.log(2 * np.pi) + p * (gammaln((delta + T) / 2.) - gammaln(delta / 2.))
    for i in range(0, p):
        current_log_post = current_log_post + delta / 2. * np.log(phi[i, i] / 2.)
        current_log_post = current_log_post - (T + delta) / 2. * np.log((phi[i, i] + np.sum(data[:, i] * data[:, i])) / 2.)
    current_log_post = current_log_post + total_edges * np.log(1 - beta)

    # start MCMC
    G = nx.Graph()
    G.add_nodes_from(list(range(0, p)))
    for k in range(0, steps):
        # conn_comp_list = list(nx.connected_components(G))
        cliques = list(nx.find_cliques(G))

        add = 0
        delete = 0
        # decider = np.random.rand()
        decider = decider_list[k]
        if n_edges == 0 or (n_edges < total_edges and decider < 0.5):
            add = 1
            node1, node2 = rand_edge_add(G, n_edges, total_edges,add_int_list[0])
            add_int_list.remove(add_int_list[0])

            print 'choose add node: {} and {}'.format(node1, node2)
        else:
            delete = 1
            node1, node2 = rand_edge_delete(G, n_edges,delete_int_list[0])
            delete_int_list.remove(delete_int_list[0])

            print 'choose delete node: {} and {}'.format(node1, node2)

        cliqueSizeIs2 = 0
        canAdd = 0
        canDelete = 0

        if add == 1:
            canDelete = 0
            if (nx.node_connected_component(G, node1) != nx.node_connected_component(G, node2)):
                canAdd = 1
                cliqueSizeIs2 = 1

            if cliqueSizeIs2 == 0:
                # if two nodes belong to the same connected component, add edge between node1 and node2
                # according to Theorem2 in the paper: Decomposable graphical Gaussian modeldetermination, GIUDICI
                canAdd = add_logic(cliques, p, node1, node2)

                # get the new clique and the size of it after add edge
                new_clique, new_clique_size = getNewCliqueAftAdd(G, node1, node2)
                if new_clique_size == 2:
                    cliqueSizeIs2 = 1

        if delete == 1:
            canAdd = 0
            # if an edge is only contained in exactly one clique, then we can delete it
            if delete_logic(cliques, node1, node2):
                canDelete = 1
                # get the  clique consist of the two nodes and the size of it before delete edge
                new_clique, new_clique_size = getCliqueBfrDelete(cliques, node1, node2)
                if new_clique_size == 2:
                    cliqueSizeIs2 = 1

        # calculate the change in log post probability after add or delete edge
        if ((canAdd == 1 or canDelete == 1) and cliqueSizeIs2 == 1):
            post_ratio = post_update(data, T, phi, delta, beta, node1, node2)
            if add == 1:
                delta_log_post = post_ratio
                post_ratio = np.exp(post_ratio) / (n_edges + 1.) * (total_edges - n_edges)

            if delete == 1:
                delta_log_post = -post_ratio
                post_ratio = np.exp(-post_ratio) / (total_edges - n_edges + 1.) * n_edges

            if n_edges == 0 or n_edges == total_edges:
                post_ratio = post_ratio / 2.

        elif(canAdd == 1 or canDelete == 1) and cliqueSizeIs2 != 1:
            post_ratio = post_update_all(new_clique, data, T, phi, delta, beta, node1, node2)

            if add == 1:
                delta_log_post = post_ratio
                post_ratio = np.exp(post_ratio) / (n_edges + 1.) * (total_edges - n_edges)

            if delete == 1:
                delta_log_post = -post_ratio
                post_ratio = np.exp(-post_ratio) / (total_edges - n_edges + 1.) * n_edges

            if n_edges == 0 or n_edges == total_edges:
                post_ratio = post_ratio / 2.

        # deciding = np.random.rand()
        deciding = deciding_list[k]
        # decide to add or delete an edge at last by comparing post_ratio with a random number
        if (canAdd == 1 and post_ratio > deciding) or (canDelete == 1 and post_ratio > deciding):
            if canAdd == 1:
                G.add_edge(node1, node2)
                n_edges = n_edges + 1
                current_log_post = current_log_post + delta_log_post
                print "success to add edge"

            if canDelete == 1:
                G.remove_edge(node1, node2)
                n_edges = n_edges - 1
                current_log_post = current_log_post + delta_log_post
                print "success to delete edge"

        else:
            delta_log_post = 0

        print 'edges: {}'.format(G.edges), "n_edges:{}".format(n_edges)
        print "{} iteration(s) finished!".format(k+1)

    return G



def centralization(data):
    data = data - data.mean(axis=0)
    return data


def rand_edge_add(G, n_edges, total_edges,s):
    p = G.number_of_nodes()
    unconnected_edges_num = 0
    node1 = 0
    if total_edges > n_edges:
        # thr = np.random.randint(1, 2 * (total_edges - n_edges))
        thr =int(s)
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


def rand_edge_delete(G, n_edges,s):
    connected_edges_num = 0
    node1 = 0
    # thr = np.random.randint(1, 2 * n_edges)
    thr = int(s)
    while connected_edges_num + G.degree[node1] < thr:
        connected_edges_num += G.degree[node1]
        node1 += 1

    neighbors = sorted(G[node1].keys())
    node2 = neighbors[thr - connected_edges_num - 1]

    return node1, node2


def add_logic(cliques, nodes_num, node1, node2):

    canAdd = 0
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
    jtree = cliques_to_jtree(cliques, nodes_num, ns)
    cliqueG = nx.from_numpy_matrix(jtree)

    for i in range(0, len(index1)):
        for j in range(0, len(index2)):
            seperators = []
            shortest_Path = nx.shortest_path(cliqueG, source=index1[i], target=index2[j])
            for k in range(1, len(shortest_Path)):
                seperators.append(set(cliques[shortest_Path[k-1]]) & set(cliques[shortest_Path[k]]))

            S = set(R[i]) & set(T[j])
            if len(S) != 0 and S in seperators:
                canAdd = 1
                break

        if canAdd:
            break

    return canAdd



def cliques_to_jtree(cliques, nodes_num, ns):
    """
        A junction tree is a tree that satisfies the jtree property, which says:
    for each pair of cliques U,V with intersection S, all cliques on the path between U and V contain S.
    (This ensures that local propagation leads to global consistency.)
    For details, see Jensen and Jensen, "Optimal Junction Trees", UAI 94.
    :param cliques: /
    :param nodes_num: total number of nodes in a graph
    :param ns: ns[i] = number of values node i can take on
    :return jtree: jtree[i,j] = 1 if cliques i and j are connected
    """
    cliques_num = len(cliques)
    weight = np.zeros([cliques_num, 1])

    sparse_clique = sparse.lil_matrix((cliques_num, nodes_num))
    for i in range(0, cliques_num):
        sparse_clique[i, cliques[i]] = 1
        weight[i] = np.prod(ns[cliques[i]])

    primary_cost = sparse_clique.dot(sparse_clique.T).toarray()
    primary_cost = primary_cost - np.diag(np.diag(primary_cost))

    W = np.tile(weight, (1, cliques_num))
    secondary_cost = W + W.T
    secondary_cost = secondary_cost - np.diag(np.diag(secondary_cost))

    jtree = minimum_spanning_tree( -primary_cost, secondary_cost)   # Using -primary_cost gives maximum spanning tree
    # The root is arbitrary, but since the first pass is towards the root,
    # we would like this to correspond to going forward in time in a DBN.
    root = cliques_num

    return jtree


def minimum_spanning_tree(primary_cost, secondary_cost):
    # primary_cost = primary_cost.toarray()
    n = len(primary_cost)#.shape[0]
    cliquesAdj = np.zeros([n, n])

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
        cliquesAdj[k, closest_clique[k]] = 1
        cliquesAdj[closest_clique[k], k] = 1
        low_cost1[k] = np.inf
        low_cost2[k] = np.inf
        used_clique[k] = 1
        NU = np.where(used_clique == 0)[0]
        for j in NU:
            if primary_cost[k, j] < low_cost1[j]:
                low_cost1[j] = primary_cost[k, j]
                low_cost2[j] = secondary_cost[k, j]
                closest_clique[j] = k

    return cliquesAdj


def delete_logic(cliques, node1, node2):
    canDelete = 0
    count = 0
    for c in cliques:
        if node1 in c and node2 in c:
            count += 1

    if count == 1:
        canDelete = 1

    return canDelete


def getNewCliqueAftAdd(G, node1, node2):
    G.add_edge(node1, node2)
    cliques = nx.find_cliques(G)
    for c in cliques:
        if node1 in c and node2 in c:
            new_cliques = c
            new_cliques_size = len(new_cliques)
            break
    G.remove_edge(node1, node2)

    return new_cliques, new_cliques_size



def getCliqueBfrDelete(cliques, node1, node2):
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




if __name__ == "__main__":

    # test data from zdata.txt
    data_list = []
    with open('./input/zdata.txt', 'r') as f:
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
    G = metropolis_hastings(data, delta, tau, rho, steps)