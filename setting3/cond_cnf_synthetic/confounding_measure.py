import numpy as np
from generate_data import generate_observational_data_icm, generate_interventional_x_data_icm
from config import Config
import pandas as pd
from sklearn.metrics import mutual_info_score
from pyitlib import discrete_random_variable as drv

cfg=Config()
np.random.seed(np.random.randint(1,1e7))

import numpy as np

# for n in range(len(cfg.N)):
#     uncond = []
#     cond_o = []
#     cond_z = []
#     cond_zo = []
#     for _ in range(5):
#         D_obs = generate_observational_data_icm(cfg.N[n])
#         D_int = generate_interventional_x_data_icm(cfg.N[n])
#         O = D_obs[:,1]
#         Z = D_obs[:,0]
#         ZO = D_obs[:,:2]
#         Ex = D_obs[:,2]
#         Ey = D_int[:,3]
#         # import pdb
#         # pdb.set_trace()
#         # measure of confounding
#         res = 1-np.exp(-mutual_info_score(Ex, Ey))
#         uncond.append(res)
        
#         cmi=drv.information_mutual_conditional(Ex,Ey,O)
#         res = 1-np.exp(-cmi)
#         cond_o.append(res)

#         cmi=drv.information_mutual_conditional(Ex,Ey,Z)
#         res = 1-np.exp(-cmi)
#         cond_z.append(res)

#         # cmi=drv.information_mutual_conditional(Ex,Ey,ZO)
#         # res = 1-np.exp(-cmi)
#         # cond_zo.append(res)


#     print("uncond", uncond)
#     print("cond_o", cond_o)
#     print("cond_z", cond_z)
#     # print("cond_zo", cond_zo)

#     np.save('uncond/'+str(n), uncond)
#     np.save('cond_o/'+str(n), cond_o)
#     np.save('cond_z/'+str(n), cond_z)
#     # np.save('cond_zo/'+str(n), cond_zo)
 
for n in range(len(cfg.N)):
    uncond = []
    cond_z = []
    for _ in range(5):
        D_obs = generate_observational_data_icm(cfg.N[n])
        D_int = generate_interventional_x_data_icm(cfg.N[n])

        Z = D_obs[:,0]
        Ex = D_obs[:,1]
        Ey = D_int[:,2]
        
        # measure of confounding
        res = 1-np.exp(-mutual_info_score(Ex, Ey))
        uncond.append(res)

        cmi=drv.information_mutual_conditional(Ex,Ey,Z)
        res = 1-np.exp(-cmi)
        cond_z.append(res)

    print("uncond", uncond)
    print("cond_z", cond_z)

    np.save('uncond_1/'+str(n), uncond)
    np.save('cond_z_1/'+str(n), cond_z)
 
