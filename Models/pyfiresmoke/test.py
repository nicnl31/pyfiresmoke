import numpy as np
from sklearn.metrics import roc_auc_score

if __name__ == "__main__":
    y_true = np.array([0,0,1,1,1,1])
    y_pred_probs = np.array([0.4, 0.6, 0.5, 0.5, 0.55, 0.6])

    print(roc_auc_score(y_true, y_pred_probs))
