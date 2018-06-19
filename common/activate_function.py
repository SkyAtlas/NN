import numpy as np

# from scip.specialy import expit
def sigmoid(X):
    return 1 / (1 + np.exp(-X))

