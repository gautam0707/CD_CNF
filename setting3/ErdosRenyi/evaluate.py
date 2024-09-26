import numpy as np
import random

seed = 42
np.random.seed(seed)
random.seed(seed)

graph_size = [10,15,20,25]
num_samples = [100,200,300,400,500]
# num_samples = [1000,2000,3000,4000,5000]

for N in graph_size:
    gt_cnf_matrix = np.load('data/'+str(N)+'_gt_cnf.npy')
    for sample_size in num_samples:
        TP = 0
        TN = 0
        FP = 0
        FN = 0
        estimated_cnf_matrix = np.load('results/'+str(N)+'_'+str(sample_size)+'estimated_cnf_matrix.npy')
        for i in range(N):
            for j in range(i+1,N):
                if gt_cnf_matrix[i][j] == 1:
                    if estimated_cnf_matrix[i][j]==1:
                        TP+=1
                    else:
                        FN+=1
                else:
                    if estimated_cnf_matrix[i][j]==1:
                        FP+=1
                    else:
                        TN+=1
        try:
            precision = TP/(TP+FP)
            recall = TP/(TP+FN)
            f1 = 2*precision*recall/(precision+recall)
            print("N: ", N, "Sample Size: ", sample_size, "Precision: ", round(precision,2) , "Recall: ", round(recall,2), "f1: ", round(f1,2))

        except Exception as e:
            continue



