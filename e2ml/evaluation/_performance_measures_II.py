import numpy as np
import matplotlib.pyplot as plt

def roc_curve(labels, x, scores):
    """
    Generate the Receiver Operating Characteristic (ROC) curve for a binary classification problem.

    Parameters
    ----------
    labels : array-like of shape (n_samples,)
        True class labels for each sample.
    x : int or str
        Positive class or class of interest.
    scores : array-like of shape (n_samples,)
        Scores or probabilities assigned to each sample.

    Returns
    -------
    roc_curve : ndarray of shape (n_thresholds, 2)
        Array containing the true positive rate (TPR) and false positive rate (FPR) pairs
        at different classification thresholds.

    """
# TODO 
    labels = np.array(labels)
    scores = np.array(scores)

    p_index = scores[labels==x]
    n_index = scores[labels!=x]

    temp = np.lexsort(np.vstack((scores, labels)))

    zs_idx = np.argsort(scores)
    zs = scores[zs_idx]
    ls = labels[zs_idx]

    roc_curve = np.zeros((len(scores)+2, 2))
    for i,t in enumerate(zs):
        pred = zs >= t
        tp = (ls == pred) & (ls == x)
        fp = (ls != pred) & (ls != x)

        tpr = sum(tp) / sum(labels == x)
        fpr = sum(fp) / sum(labels != x)

        roc_curve[i+1, :] = [tpr, fpr] 

    roc_curve[-1:, :] = [1,1]

    return roc_curve

def roc_auc(points):
    """
    Compute the Area Under the Receiver Operating Characteristic (ROC) curve.

    Parameters
    ----------
    points : array-like
        List of points representing the ROC curve.

    Returns
    -------
    auc : float
        Area Under the ROC curve.

    """
# TODO 



def draw_lift_chart(true_labels, pos, predicted):
    """
    Draw a Lift Chart based on the true labels, positive class, and predicted class labels.

    Parameters
    ----------
    true_labels : array-like
        True class labels for each sample.
    pos : int or str
        Positive class or class of interest.
    predicted : array-like
        Predicted class labels for each sample.

    Returns
    -------
    None

    """
# TODO 
    plt.figure(figsize=(3,3))
    plt.xlabel("Datasetsize")
    plt.ylabel("True Positives")

    tp = np.ones((len(true_labels), 2))
    tp[:,0] = np.cumsum(tp[:,0])
    for (l, p, i) in zip(true_labels, predicted, np.arange(0,len(true_labels))):
        tp[i,1] = (l==p) & (l == pos)

    tp[:,1] = np.cumsum(tp[:,1])
    plt.plot(tp[:,0], tp[:,1]) 

def kl_divergence(p, q):
    """
    Compute the Kullback-Leibler (KL) divergence between two probability distributions `p` and `q`.

    Parameters
    ----------
    p : array-like
        Probability distribution P.
    q : array-like
        Probability distribution Q.

    Returns
    -------
    kl_div : float
        KL divergence between P and Q.

    """
# TODO 
    p = np.array(p)
    q = np.array(q)

    return np.sum(p*np.log(p/q))
