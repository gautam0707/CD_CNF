import numpy as np
from generate_data import generate_interventional_data_x, generate_interventional_data_y, generate_observational_data
from config import Config
import pandas as pd
import pdb

cfg=Config()
np.random.seed(np.random.randint(1,1e7))
# IYX = KL( P(Y|X) || P(Y|do(X)) ) 
# IXY = KL( P(X|Y) || P(X|do(Y)) ) 

import numpy as np

def calculate_probability_distribution(data):
    data = data.astype(int).flatten()
    value_counts = np.bincount(data)
    probability_distribution = value_counts / len(data)
    return probability_distribution

def kl_divergence(p, q, p1, p2):
    assert len(p) == len(q), "Distributions must have the same length"
    epsilon = 1e-10
    kl_div = np.sum(np.array([p1, p2]) * np.log((p + epsilon) / (q + epsilon)))
    return kl_div

for n in range(len(cfg.N)):
    micnf = []
    for _ in range(5): # 5 runs of measure
        IYX = 0
        IXY = 0
        D_obs = generate_observational_data(cfg.N[n])

        cnt = 0
        for val in [0,1]:  
            condition_x = D_obs[:,1]==val
            condition_y = D_obs[:,2]==val
            
            data_y_givenx = D_obs[condition_x][:,2]
            data_x_giveny = D_obs[condition_y][:,1]    
            
            D_int_x = generate_interventional_data_x(len(data_y_givenx), val)
            D_int_y = generate_interventional_data_y(len(data_x_giveny), val)

            data_y_dox = D_int_x[:,2].reshape(len(data_y_givenx), 1)
            data_x_doy = D_int_y[:,1].reshape(len(data_x_giveny), 1)

            joints11 = sum(((D_obs[condition_x][:,1]==val) & (D_obs[condition_x][:,2]==val))*1)/len(data_y_givenx)
            joints12 = sum(((D_obs[condition_x][:,1]==val) & (D_obs[condition_x][:,2]==1-val))*1)/len(data_y_givenx)
            joints21 = sum(((D_obs[condition_y][:,1]==val) & (D_obs[condition_y][:,2]==val))*1)/len(data_x_giveny)
            joints22 = sum(((D_obs[condition_y][:,1]==1-val) & (D_obs[condition_y][:,2]==val))*1)/len(data_x_giveny)

            cnt+=1
            cpdxy = calculate_probability_distribution(data_x_giveny)
            cpdxdoy = calculate_probability_distribution(data_x_doy)

            cpdyx = calculate_probability_distribution(data_y_givenx)
            cpdydox = calculate_probability_distribution(data_y_dox)
            pdb.set_trace()
            if val==0:
                IYX += kl_divergence(cpdyx, cpdydox, joints11, joints12)
                IXY += kl_divergence(cpdxy, cpdxdoy, joints21, joints22)
            else:
                IYX += kl_divergence(cpdyx, cpdydox, joints12, joints11)
                IXY += kl_divergence(cpdxy, cpdxdoy, joints22, joints21)
        # measure of confounding
        res = 1-np.exp(-min(IXY/cnt , IYX/cnt))
        micnf.append(res)
    print(micnf)
    np.save(str(n)+'indep', micnf)