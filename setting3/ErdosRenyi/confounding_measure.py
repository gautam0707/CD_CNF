import numpy as np
import random
import pdb
from sklearn.metrics import mutual_info_score
import pickle

seed = 42
np.random.seed(seed)
random.seed(seed)

graph_size = [10,15,20,25]
num_samples = [100,200,300,400,500]
# num_samples = [1000,2000,3000,4000,5000]

def check_path(G, i, j):
    frontier = [i]
    visited = [0 for i in range(G.number_of_nodes())]
    while frontier:
        current = frontier[0]
        if current == j:
            return True
        visited[current] = 1
        del frontier[0]
        next = G.successors(current)
        for ch in next:
            if visited[ch] == 0:
                frontier.append(ch)
    return False

for n in graph_size:
    micnf = []
    pairs = []
    for i in range(n):
        for j in range(i+1,n):
            pairs.append((i,j))
    G = None
    with open('data/'+str(n)+'_graph.gpickle', 'rb') as f:
        G = pickle.load(f)

    for sample_size in num_samples:
        estimated_cnf_matrix = [[0 for i in range(n)] for j in range(n)]
        D_obs = np.load('data/'+str(n)+'_'+str(sample_size)+'_obs_data.npy')
        contexts = np.load('data/'+str(n)+'_'+str(sample_size)+'_contexts.npy', allow_pickle=True).item()
        for pair in pairs:
            cnt = 0
            x, y = pair[0], pair[1]
            interventions = [0,1]
            # pdb.set_trace()
            Ex = D_obs[:,x]
            Ey = D_obs[:,y]
            
            for intervention in interventions:
                D_intx = contexts[str(x)+'_'+str(intervention)][:,x]
                D_inty = contexts[str(y)+'_'+str(intervention)][:,y]

                if check_path(G, x, y):
                    D_inty = contexts[str(x)+'_'+str(intervention)][:,y]
                elif check_path(G, y, x):
                    D_intx = contexts[str(y)+'_'+str(intervention)][:,x]
                
                Ex = np.concatenate([Ex, D_intx])
                Ey = np.concatenate([Ey, D_inty])

            # measure of confounding
            res = 1-np.exp(-mutual_info_score(Ex, Ey))
            # print(res, end=' ')
            estimated_cnf_matrix[x][y] = res>0.1
        np.save('results/'+str(n)+'_'+str(sample_size)+'estimated_cnf_matrix.npy', estimated_cnf_matrix)