# import matplotlib.pyplot as plt
# import numpy as np
# import matplotlib

# font = {'size'   : 20}
# matplotlib.rc('font', **font)

# def adjacent_values(vals, q1, q3):
#     upper_adjacent_value = q3 + (q3 - q1) * 1.5
#     upper_adjacent_value = np.clip(upper_adjacent_value, q3, vals[-1])

#     lower_adjacent_value = q1 - (q3 - q1) * 1.5
#     lower_adjacent_value = np.clip(lower_adjacent_value, vals[0], q1)
#     return lower_adjacent_value, upper_adjacent_value


# def set_axis_style(ax, labels):
#     ax.set_xticks(np.arange(1, len(labels) + 1), labels=labels)
#     ax.set_xlim(0.25, len(labels) + 0.75)
#     ax.set_xlabel('Number of Contexts')


# # create test data
# np.random.seed(19680801)
# data1 = [np.load(str(a)+'no_z_to_xy.npy') for a in range(6)]
# data2 = [np.load(str(a)+'.npy') for a in range(6)]

# mixeddata = []

# for i in range(len(data1)):
#     mixeddata.append(data1[i])
#     mixeddata.append(data2[i])


# fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(9, 4), sharey=True)

# parts = ax.violinplot(
#         mixeddata, showmeans=False, showmedians=False,
#         showextrema=False)

# for pc in parts['bodies']:
#     pc.set_facecolor('aquamarine')
#     pc.set_edgecolor('aquamarine')
#     pc.set_alpha(1)

# quartile1, medians, quartile3 = np.percentile(mixeddata, [25, 50, 75], axis=1)
# whiskers = np.array([
#     adjacent_values(sorted_array, q1, q3)
#     for sorted_array, q1, q3 in zip(mixeddata, quartile1, quartile3)])
# whiskers_min, whiskers_max = whiskers[:, 0], whiskers[:, 1]

# inds = np.arange(1, len(medians) + 1)

# ax.scatter(inds, medians, marker='o', color='white', s=30, zorder=3)
# ax.vlines(inds, quartile1, quartile3, color='k', linestyle='-', lw=5)
# ax.vlines(inds, whiskers_min, whiskers_max, color='k', linestyle='-', lw=1)

# # set style for the axes
# labels = [500,600,700,800,900,1000]

# set_axis_style(ax, labels)
# ax.set_ylabel("$CNF-1(X_i,X_j)$")
# plt.subplots_adjust(bottom=0.15, wspace=0.05)

# plt.savefig('violins1.png',dpi=128,bbox_inches='tight')
# plt.savefig('violins1.pdf',format='pdf',bbox_inches='tight')

import matplotlib.pyplot as plt
import numpy as np
import matplotlib

font = {'size'   : 16}
matplotlib.rc('font', **font)
plt.figure(figsize=(11, 6))

# Create the violin plots
# Generate some random data
data1 = [np.mean(np.load(str(a)+'indep.npy')) for a in range(5)]
data1errs = [np.std(np.load(str(a)+'indep.npy')) for a in range(5)]

data2 = [np.mean(np.load(str(a)+'xy.npy')) for a in range(5)]
data2errs = [np.std(np.load(str(a)+'xy.npy')) for a in range(5)]

data3 = [np.mean(np.load(str(a)+'zx_zy.npy')) for a in range(5)]
data3errs = [np.std(np.load(str(a)+'zx_zy.npy')) for a in range(5)]

data4 = [np.mean(np.load(str(a)+'zx_zy_xy.npy')) for a in range(5)]
data4errs = [np.std(np.load(str(a)+'zx_zy_xy.npy')) for a in range(5)]


fig, ax = plt.subplots()
xs = [1,2,3,4,5]

ax.errorbar(xs, data3, yerr=data3errs, label=r'(Cnfounded) $Z\rightarrow X_i, Z\rightarrow X_j$')
ax.errorbar(xs, data4, yerr=data4errs, label=r'(Cnfounded) $Z\rightarrow X_i, Z\rightarrow X_j, X_i\rightarrow X_j$')
ax.errorbar(xs, data2, yerr=data2errs, label=r'(Unconfounded) $X_i\rightarrow X_j$',color='red')
ax.errorbar(xs, data1, yerr=data1errs, label=r'(Unconfounded) Isolated $Z, X_i, X_j$', color='black')

plt.xticks([1,2,3,4,5],[1000, 2000,3000,4000,5000])
plt.legend(framealpha=0,loc='best', bbox_to_anchor=(0.45, 0.1, 0.5, 0.5))
plt.ylabel(r'$CNF-1(X_i, X_j)$')
plt.xlabel('Number of Samples')
plt.savefig('errorbar1.png',dpi=128,bbox_inches='tight')
plt.savefig('errorbar1.pdf',format='pdf',bbox_inches='tight')