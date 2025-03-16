from nmi import nmi
from accuracy import accuracy

def evaluatecluster(Idx, datalabels):
    """
    Evaluate the clustering result.
    
    This function computes the Normalized Mutual Information (NMI) and 
    clustering accuracy (ACC) based on the provided cluster labels (Idx) 
    and the ground truth labels (datalabels).

    It is assumed that the functions `nmi` and `accuracy` are available 
    and behave equivalently to their MATLAB counterparts.

    Parameters:
        Idx : array-like
            Cluster labels produced by the clustering algorithm.
        datalabels : array-like
            Ground truth labels.

    Returns:
        NMI : float
            Normalized Mutual Information.
        ACC : float
            Clustering accuracy as a fraction (e.g., 0.95 means 95% accuracy).
    """
    NMI = nmi(Idx, datalabels)
    ACC = accuracy(datalabels, Idx) / 100.0
    return NMI, ACC
