from sklearn.metrics import mutual_info_score
labels_true = [0, 1, 1, 0, 1, 1]
labels_pred = [0, 0, 1, 1, 0, 1]
print(mutual_info_score(labels_true, labels_pred))