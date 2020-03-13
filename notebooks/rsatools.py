import numpy as np
import pandas as pd
import nibabel as nib
import os
from os import listdir
from os.path import isfile, join, exists
import matplotlib.pyplot as plt
from scipy.spatial.distance import pdist, squareform,cosine,euclidean,mahalanobis
from statsmodels.stats.multitest import multipletests
import scipy.io as sio
import scipy.stats as stats

#########################
# for validating matrices
#########################

def nancheck(Y):
    assert(np.sum(np.isnan(Y)) == 0)
    return Y

def validate_2D(Y):
    Y = np.array(Y)
    assert(Y.ndim==2)
    assert(Y.shape[0] > 1)
    assert(Y.shape[1] > 1)
    if np.sum(np.isnan(Y)) > 0:
        raise ValueError('input contains nans')
    else:
        return Y
    
def assert_validdm(Y):
    if np.ndim(Y) == 2:
        if Y.shape[0] == Y.shape[1]:
            if np.all(np.isclose(np.diagonal(Y),1)) or np.all(np.isclose(np.diagonal(Y),0)):
                if assert_symmetric(Y):
                    return True
    return False
        
def assert_symmetric(Y, rtol=1e-05, atol=1e-08):
    return np.allclose(Y, Y.T, rtol=rtol, atol=atol)

#########################
# for computing rdvs from matrices
#########################

def rdv(Y,dist='correlation'):
    return pdist(validate_2D(Y),dist)

def rsm2rdm(Y):
    return (nancheck(Y) - 1) * -1

#########################
# for computing rdv corrs
#########################

def rdvcorr(Y1, Y2, dist='correlation', corr = 'pearson'):
    
    # ensure inputs have no nans, have same dim
    Y1, Y2 = nancheck(Y1), nancheck(Y2)

    # case 0: inputs are mismatched (error)
    assert(Y1.shape[0] == Y2.shape[0])
    
    # case 1: inputs are data matrices
    if np.ndim(Y1) > 1 and np.ndim(Y2) > 1 and assert_validdm(Y1) is False and assert_validdm(Y2) is False:
        Y1, Y2 = rdv(Y1,dist), rdv(Y2,dist)
    
    # case 2: inputs are already rdvs/rsvs
    # do nothing
    
    # case 3: inputs are already rdms/rsms
    if assert_validdm(Y1) is True and assert_validdm(Y2) is True:
        Y1, Y2 = squareform(Y1,force='tovector'), squareform(Y2,force='tovector')
    
    # perform corr
    if corr == 'pearson':
        r_val = np.corrcoef(Y1,Y2)[1,0]
    elif corr == 'spearman':
        pass #todo
    elif corr == 'kendall':
        pass #todo
    elif corr == 'cosine':
        pass #todo
    else:
        raise ValueError('invalid correlation metric')
        
    return r_val

    
def rdvcorr_list(Y_target, Y_list, dist='correlation', corr = 'pearson'):
    
    if len(Y_list) < 1:
        raise ValueError('candidate list is empty')
        
    if np.logical_not(Y_target.shape[0] == Y_list[0].shape[0]):
        raise ValueError('inputs must be the same shape')
     
    corr_list = []
    
    for Y in Y_list:
        corr_list.append(rdvcorr(Y_target, Y, dist, corr))
        
    return corr_list
    