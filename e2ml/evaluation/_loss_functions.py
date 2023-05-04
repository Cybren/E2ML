import numpy as np
def zero_one_loss(y_true, y_pred):
    if(len(y_true) != len(y_pred)):
        print("arrays do not have the same length!")
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    return np.sum(np.abs(y_true-y_pred))/len(y_true)

def binary_cross_entropy_loss(y_true, y_pred):
    if(len(y_true) != len(y_pred)):
        print("arrays do not have the same length!")
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    return np.sum(y_true * np.log(y_pred) - (1 - y_true) * np.log(1 - y_pred))/y_true.shape[0]