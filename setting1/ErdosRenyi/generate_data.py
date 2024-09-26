import numpy as np
import networkx as nx
import pickle
import matplotlib.pyplot as plt
import random
import pdb
from scipy.special import expit  # Sigmoid function


seed = 42
np.random.seed(seed)
random.seed(seed)

def generate_erdos_renyi_dag(n, p):
    G = nx.DiGraph()
    G.add_nodes_from(range(n))

    # Iterate over all possible pairs of nodes and add directed edges with probability p
    for i in range(n):
        for j in range(i + 1, n):
            if np.random.rand() < p:
                G.add_edge(i, j)
            elif np.random.rand() < p:
                G.add_edge(j, i)

    # Ensure the graph is acyclic by removing edges that create cycles
    try:
        cycles = list(nx.find_cycle(G, orientation="original"))
        while cycles:
            G.remove_edge(*cycles[0][:2])
            cycles = list(nx.find_cycle(G, orientation="original"))
    except nx.NetworkXNoCycle:
        pass
    
    return G

def path_exists_not_via(adj, a, b, c):
    """
    adj: adjacency matrix of the graph
    a: source node
    b(c): destination node
    c(b): return true if there is a path not going via c(b)
    """
    n = len(adj)
    frontier = [a]
    visited = [0 for _ in range(n)]
    while frontier:
        node = frontier[0]
        visited[node] = 1
        del frontier[0]
        if node == b:
            return True
        children = [ch for ch in range(n) if adj[node][ch]==1 and ch!=c]
        for ch in children:
            if not visited[ch]:
                frontier.append(ch)
    return False

def check_ground_truth_confounding(G, n):
    cnfmatrix = [[0 for i in range(n)] for j in range(n)]
    adjmatrix = [[0 for i in range(n)] for j in range(n)]

    for edge in G.edges:
        adjmatrix[edge[0]][edge[1]] = 1

    for k in range(n):
        for i in range(n):
            for j in range(n):
                if i!=j and i!=k and j!=k:
                    if path_exists_not_via(adjmatrix, k, i, j) and path_exists_not_via(adjmatrix, k, j, i):
                        cnfmatrix[i][j] = 1

    return cnfmatrix

def generate_data(G, n_samples):
    nodes = list(nx.topological_sort(G))
    data = {node: np.zeros(n_samples) for node in nodes}
    for node in nodes:
        parents = list(G.predecessors(node))
        if not parents:
            # No parents: generate random binary variable with a slight imbalance
            data[node] = np.random.binomial(1, 0.6, n_samples)
        else:
            # With parents: generate binary variable based on complex interactions of parents
            parent_data = np.column_stack([data[parent] for parent in parents])
            
            # Complex interaction: a mixture of nonlinear and interaction terms with different weights
            additive_term = np.sum(parent_data, axis=1)
            multiplicative_term = np.prod(parent_data, axis=1)
            interaction_term = np.prod(np.sin(parent_data * np.pi / 2), axis=1)
            
            # Weighting the terms differently
            logits = 0.3 * additive_term + 0.5 * multiplicative_term + 0.2 * interaction_term + np.random.normal(0, 0.2, n_samples)
            
            # Introduce context-dependent noise
            if node % 2 == 0:
                logits += np.random.normal(0, 0.1, n_samples)
            else:
                logits -= np.random.normal(0, 0.1, n_samples)
                
            prob = expit(logits)  # Logistic function to convert logits to probabilities
            data[node] = np.random.binomial(1, prob)
    
    return np.column_stack([data[node] for node in nodes])

def generate_interventional_data(G, n_samples, intervention_node, intervention_value):
    nodes = list(nx.topological_sort(G))
    data = {node: np.zeros(n_samples) for node in nodes}
    
    for node in nodes:
        if node == intervention_node:
            data[node] = np.full(n_samples, intervention_value)
        else:
            parents = list(G.predecessors(node))
            if not parents:
                # No parents: generate random binary variable with a slight imbalance
                data[node] = np.random.binomial(1, 0.6, n_samples)
            else:
                # With parents: generate binary variable based on complex interactions of parents
                parent_data = np.column_stack([data[parent] for parent in parents])
                
                # Complex interaction: a mixture of nonlinear and interaction terms with different weights
                additive_term = np.sum(parent_data, axis=1)
                multiplicative_term = np.prod(parent_data, axis=1)
                interaction_term = np.prod(np.sin(parent_data * np.pi / 2), axis=1)
                
                # Weighting the terms differently
                logits = 0.3 * additive_term + 0.5 * multiplicative_term + 0.2 * interaction_term + np.random.normal(0, 0.2, n_samples)
                
                # Introduce context-dependent noise
                if node % 2 == 0:
                    logits += np.random.normal(0, 0.1, n_samples)
                else:
                    logits -= np.random.normal(0, 0.1, n_samples)
                    
                prob = expit(logits)  # Logistic function to convert logits to probabilities
                data[node] = np.random.binomial(1, prob)
        
        return np.column_stack([data[node] for node in nodes])
 
# Parameters
graph_size = [10,15,20,25]
edge_prob = 0.3
num_samples = [100,200,300,400,500]
# num_samples = [1000,2000,3000,4000,5000]
for N in graph_size:
    interventions = {i: [0,1] for i in range(N)}
    G = generate_erdos_renyi_dag(N, edge_prob)
    gt_cnf_matrix = check_ground_truth_confounding(G, N)
    with open('data/'+str(N)+'_graph.gpickle', 'wb') as f:
        pickle.dump(G, f)
    np.save('data/'+str(N)+'_gt_cnf.npy',gt_cnf_matrix)
    
    for sample_size in num_samples:
        obserational_data = generate_data(G, sample_size)
        contexts = {}
        for i in range(N):
            for intervention in interventions[i]:
                contexts[str(i)+'_'+str(intervention)] = generate_interventional_data(G, sample_size, i, intervention)
        
        np.save('data/'+str(N)+'_'+str(sample_size)+'_obs_data.npy', obserational_data)
        np.save('data/'+str(N)+'_'+str(sample_size)+'_contexts.npy', contexts)

