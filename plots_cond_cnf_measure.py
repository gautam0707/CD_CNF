import matplotlib.pyplot as plt
import numpy as np
import matplotlib

font = {'size'   : 32}
matplotlib.rc('font', **font)

fig, axs = plt.subplots(1, 2, figsize=(24, 6))

for i in range(2):
    path='setting2/cond_cnf_synthetic/'

    uncond = [np.mean(np.load(path+'uncond/'+str(a)+'.npy')) for a in range(5)]
    uncond_errs = [np.std(np.load(path+'uncond/'+str(a)+'.npy')) for a in range(5)]

    cond_o = [np.mean(np.load(path+'cond_o/'+str(a)+'.npy')) for a in range(5)]
    cond_o_errs = [np.std(np.load(path+'cond_o/'+str(a)+'.npy')) for a in range(5)]

    cond_z = [np.mean(np.load(path+'cond_z/'+str(a)+'.npy')) for a in range(5)]
    cond_z_errs = [np.std(np.load(path+'cond_z/'+str(a)+'.npy')) for a in range(5)]

    uncond_1 = [np.mean(np.load(path+'uncond_1/'+str(a)+'.npy')) for a in range(5)]
    uncond_1_errs = [np.std(np.load(path+'uncond_1/'+str(a)+'.npy')) for a in range(5)]

    cond_z_1 = [np.mean(np.load(path+'cond_z_1/'+str(a)+'.npy')) for a in range(5)]
    cond_z_1_errs = [np.std(np.load(path+'cond_z_1/'+str(a)+'.npy')) for a in range(5)]

    xs = [1,2,3,4,5]
    lw=5
    if i==0:
        axs[i].errorbar(xs, uncond, yerr=uncond_errs, label=r'$CNF-2(X_i,X_j|\emptyset)$',linewidth=lw)
        axs[i].errorbar(xs, cond_o, yerr=cond_o_errs, label=r'$CNF-2(X_i,X_j|Z_1)$',linewidth=lw)
        axs[i].errorbar(xs, cond_z, yerr=cond_z_errs, label=r'$CNF-2(X_i,X_j|Z_2)$',linewidth=lw)

    else:
        axs[i].errorbar(xs, uncond_1, yerr=uncond_1_errs, label=r'$CNF-2(X_i,X_j|\emptyset)$',linewidth=lw)
        axs[i].errorbar(xs, cond_z_1, yerr=cond_z_1_errs, label=r'$CNF-2(X_i,X_j|Z)$',linewidth=lw)
    
    axs[i].set_xticks([1,2,3,4,5],['1000', '2000','3000','4000','5000'])
    axs[i].legend(framealpha=0,loc='best', bbox_to_anchor=(0.45, 0.1, 0.5, 0.5))
    axs[i].set_xlabel('Number of Samples', fontsize=40)
    if i==0:
        axs[i].set_title(r'$\mathcal{G}_5$',fontsize=40)
        axs[i].set_ylabel(r'$CNF-2$')
        axs[i].legend(framealpha=0,loc='lower center', bbox_to_anchor=(0.35, 0.02, 0.25, 0.25))

    else:
        axs[i].set_title(r'$\mathcal{G}_6$',fontsize=40)
        axs[i].set_ylabel(r'$CNF-2$')


plt.savefig('cond_cnf_measure.png',dpi=128,bbox_inches='tight')
plt.savefig('cond_cnf_measure.pdf',format='pdf',bbox_inches='tight')