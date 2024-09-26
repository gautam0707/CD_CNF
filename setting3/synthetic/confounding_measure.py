import numpy as np
from generate_data import generate_observational_data_icm
from config import Config
import pandas as pd
from sklearn.metrics import mutual_info_score

cfg=Config()
np.random.seed(np.random.randint(1,1e7))

for n in range(len(cfg.N)):
    mi = []
    for _ in range(5):
        IYX = 0
        IXY = 0
        D_obs = generate_observational_data_icm(cfg.N[n])

        Ex = D_obs[:,1]
        Eyx = D_obs[:,2]
        
        # measure of confounding
        res = 1-np.exp(-mutual_info_score(Ex, Eyx))
        mi.append(res)
    print(mi)
    np.save(str(n)+'indep', mi)