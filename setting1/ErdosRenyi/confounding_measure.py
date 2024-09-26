import numpy as np
import random
import pdb

seed = 42
np.random.seed(seed)
random.seed(seed)

graph_size = [10,15,20,25]
num_samples = [100,200,300,400,500]
# num_samples = [1000,2000,3000,4000,5000]

# IYX = KL( P(Y|X) || P(Y|do(X)) ) 
# IXY = KL( P(X|Y) || P(X|do(Y)) ) 

def calculate_probability_distribution(data):
    data = data.astype(int).flatten()
    # pdb.set_trace()
    value_counts = np.bincount(data, minlength=2)
    probability_distribution = value_counts / len(data)
    return probability_distribution

def kl_divergence(p, q, p1, p2):
    assert len(p) == len(q), "Distributions must have the same length"
    epsilon = 1e-10
    kl_div = np.sum(np.array([p1, p2]) * np.log((p + epsilon) / (q + epsilon)))
    return kl_div

for n in graph_size:
    micnf = []
    pairs = []
    for i in range(n):
        for j in range(i+1,n):
            pairs.append((i,j))
    IXY = 0
    IYX = 0
    for sample_size in num_samples:
        estimated_cnf_matrix = [[0 for i in range(n)] for j in range(n)]
        D_obs = np.load('data/'+str(n)+'_'+str(sample_size)+'_obs_data.npy')
        contexts = np.load('data/'+str(n)+'_'+str(sample_size)+'_contexts.npy', allow_pickle=True).item()
        for pair in pairs:
            cnt = 0
            x, y = pair[0], pair[1]
            interventions = [0,1]
            # pdb.set_trace()
            for intervention in interventions:
                condition_x = D_obs[:,x]==intervention
                condition_y = D_obs[:,y]==intervention

                data_y_givenx = D_obs[condition_x][:,y]
                data_x_giveny = D_obs[condition_y][:,x] 

                data_y_givenx = data_y_givenx.reshape(len(data_y_givenx), 1)
                data_x_giveny = data_x_giveny.reshape(len(data_x_giveny), 1)

                D_intx = contexts[str(x)+'_'+str(intervention)]
                D_inty = contexts[str(y)+'_'+str(intervention)]

                # pdb.set_trace()

                data_y_dox = D_intx[:,y][:len(data_y_givenx)].reshape(len(data_y_givenx), 1)
                data_x_doy = D_inty[:,x][:len(data_x_giveny)].reshape(len(data_x_giveny), 1)
                
                joints11 = sum(((D_obs[condition_x][:,x]==intervention) & (D_obs[condition_x][:,y]==intervention))*1)/len(data_y_givenx)
                joints12 = sum(((D_obs[condition_x][:,x]==intervention) & (D_obs[condition_x][:,y]==1-intervention))*1)/len(data_y_givenx)
                joints21 = sum(((D_obs[condition_y][:,x]==intervention) & (D_obs[condition_y][:,y]==intervention))*1)/len(data_x_giveny)
                joints22 = sum(((D_obs[condition_y][:,x]==1-intervention) & (D_obs[condition_y][:,y]==intervention))*1)/len(data_x_giveny)

                cnt+=1
                # pdb.set_trace()

                cpdxy = calculate_probability_distribution(data_x_giveny)
                cpdxdoy = calculate_probability_distribution(data_x_doy)

                cpdyx = calculate_probability_distribution(data_y_givenx)
                cpdydox = calculate_probability_distribution(data_y_dox)

                if intervention==0:
                    IYX += kl_divergence(cpdyx, cpdydox, joints11, joints12)
                    IXY += kl_divergence(cpdxy, cpdxdoy, joints21, joints22)
                else:
                    IYX += kl_divergence(cpdyx, cpdydox, joints12, joints11)
                    IXY += kl_divergence(cpdxy, cpdxdoy, joints22, joints21)
            # measure of confounding
            res = 1-np.exp(-min(IXY/cnt , IYX/cnt))
            # print(res,end=' ')
            estimated_cnf_matrix[x][y] = res>0.2
        np.save('results/'+str(n)+'_'+str(sample_size)+'estimated_cnf_matrix.npy', estimated_cnf_matrix)