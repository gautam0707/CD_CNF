import matplotlib.pyplot as plt
import numpy as np
import matplotlib

font = {'size'   : 32}
matplotlib.rc('font', **font)

fig, axs = plt.subplots(1, 3, figsize=(32, 6))
for i in range(3):
    path=''
    if i==0:
        path='setting1/synthetic/'
    elif i==1:
        path='setting2/synthetic/'
    else:
        path='setting3/synthetic/'

    data1 = [np.mean(np.load(path+str(a)+'indep.npy')) for a in range(5)]
    data1errs = [np.std(np.load(path+str(a)+'indep.npy')) for a in range(5)]

    data2 = [np.mean(np.load(path+str(a)+'xy.npy')) for a in range(5)]
    data2errs = [np.std(np.load(path+str(a)+'xy.npy')) for a in range(5)]

    data3 = [np.mean(np.load(path+str(a)+'zx_zy.npy')) for a in range(5)]
    data3errs = [np.std(np.load(path+str(a)+'zx_zy.npy')) for a in range(5)]

    data4 = [np.mean(np.load(path+str(a)+'zx_zy_xy.npy')) for a in range(5)]
    data4errs = [np.std(np.load(path+str(a)+'zx_zy_xy.npy')) for a in range(5)]

    xs = [1,2,3,4,5]
    lw=5
    axs[i].errorbar(xs, data4, yerr=data4errs, label=r'(Cnf) $\mathcal{G}_4$',linewidth=lw)
    axs[i].errorbar(xs, data3, yerr=data3errs, label=r'(Cnf) $\mathcal{G}_3$',linewidth=lw)
    axs[i].errorbar(xs, data2, yerr=data2errs, label=r'(Uncnf) $\mathcal{G}_2$',color='red',linewidth=lw)
    axs[i].errorbar(xs, data1, yerr=data1errs, label=r'(Uncnf) $\mathcal{G}_1$', color='black',linewidth=lw)

    axs[i].set_xticks([1,2,3,4,5],['1000', '2000','3000','4000','5000'])
    axs[i].legend(framealpha=0,loc='best', bbox_to_anchor=(0.45, 0.1, 0.5, 0.5))
    if i==0:
        axs[i].set_ylabel(r'$CNF-1(X_i, X_j)$')
        axs[i].set_title('Setting 1')
        axs[i].set_xlabel(r'Number of Samples', fontsize=40)
    elif i==1:
        axs[i].set_ylabel(r'$CNF-2(X_i, X_j)$')
        axs[i].set_title('Setting 2')
        axs[i].set_xlabel(r'Number of Samples', fontsize=40)
    else:
        axs[i].set_ylabel(r'$CNF-3(X_i, X_j)$')
        axs[i].set_title('Setting 3')
        axs[i].set_xlabel(r'Number of Samples', fontsize=40)


plt.savefig('cnf_measure.png',dpi=128,bbox_inches='tight')
plt.savefig('cnf_measure.pdf',format='pdf',bbox_inches='tight')