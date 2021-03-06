import numpy as np
import pandas as pd
import nibabel as nib
import os
from os import listdir
from os.path import isfile, join, exists
import matplotlib.pyplot as plt
from scipy.spatial.distance import pdist,squareform,cosine,euclidean,mahalanobis
import scipy.io as sio
import scipy.stats as stats
import seaborn as sns
from numpy.polynomial.polynomial import polyfit
from IPython.core.debugger import set_trace

#########################
# for validating matrices
#########################

def assert_nonans(Y):
    if sumnans(Y) > 0:
        raise ValueError('nans present in matrix')
    return Y

def assert_valid2D(Y):
    Y = np.array(Y)
    if Y.ndim is not 2 or Y.shape[0] < 2 or Y.shape[1] < 2 or sumnans(Y) > 0:
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

def rsm2rdm(Y,nan_ok = False):
    if nan_ok is True:
        return ((Y - 1) * -1)
    else:
        return (assert_nonans(Y) - 1) * -1

def rdv(Y,dist='correlation'):
    return pdist(assert_valid2D(Y),dist)

def rdm(Y,dist='correlation'):
    return squareform(pdist(assert_valid2D(Y),dist))

def rsv(Y,dist='correlation'):
    return rsm2rdm(pdist(assert_valid2D(Y),dist))

def rsm(Y,dist='correlation'):
    return rsm2rdm(squareform(pdist(assert_valid2D(Y),dist)))

def rdv_categ(Y, categ_idx, dist='correlation'):
    return pdist(collapse_categs(assert_valid2D(Y), categ_idx), dist)

def rsv_categ(Y, categ_idx, dist='correlation'):
    return rsm2rdm(pdist(collapse_categs(assert_valid2D(Y), categ_idx), dist))

def rdm_categ(Y, categ_idx, dist='correlation'):
    return squareform(pdist(collapse_categs(assert_valid2D(Y), categ_idx), dist))

def rsm_categ(Y, categ_idx, dist='correlation'):
    return rsm2rdm(squareform(pdist(collapse_categs(assert_valid2D(Y), categ_idx), dist)))

#########################
# for computing rdv corrs
#########################

def rdvcorr(Y1, Y2, dist='correlation', corr = 'pearson'):
    
    # ensure inputs have no nans, have same dim
    Y1, Y2 = assert_nonans(Y1), assert_nonans(Y2)
    
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
    
#########################
# misc matrix operations
#########################

def v2m(dv):
    if np.ndim(dv) is not 1:
        raise ValueError('input must be a vector')
    dm = squareform(dv)
    assert_validdm(dm)
    return dm

def m2v(dm):
    if np.ndim(dm) is not 2:
        raise ValueError('input must be a symmetrical distance matrix')
    dm = squareform(dm,force='tovector')
    assert_validdm(dm)
    return dm

def sumnans(Y):
    return np.sum(np.isnan(Y))

def collapse_categs(Y_item, categ_idx):
    Y_item = assert_valid2D(Y_item)
    cats = np.unique(categ_idx)
    n = len(cats)
    
    Y_categ = []
    
    for i in range(n):
        cat = cats[i]
        avg = np.mean(Y_item[categ_idx==cat,:],axis=0)
        Y_categ.append(avg)
        
    Y_categ = np.vstack(Y_categ)
   
    return Y_categ

def collapse_categs_4D(Y_item, categ_idx):
    cats = np.unique(categ_idx)
    n = len(cats)
    
    Y_categ = []
    
    for i in range(n):
        cat = cats[i]
        avg = np.mean(Y_item[categ_idx==cat],axis=0)
        Y_categ.append(avg)
    
    Y_categ = np.stack(Y_categ,axis=0)
    return Y_categ

def univar_mean(Y):
    Y = assert_valid2D(Y)
    return np.mean(Y,axis=1)

def stack_mean(Y_list):
    for Y in Y_list:
        _ = assert_valid2D(Y)
    ndim = np.ndim(Y_list[0])
    return np.mean(np.stack(Y_list,axis=ndim),axis=ndim)
    
def subsample_dm(Y_dist, incl_idx):
    
    try:
        if np.ndim(Y_dist) == 1:
            Y_dist = v2m(np.ravel(Y_dist))
            flag = True
        else:
            flag = False
    except:
        set_trace()
        
    n = len(incl_idx)
    Y_sub = np.empty((n,n))
    Y_sub[:] = np.nan
    
    r,c = 0,0
    
    for a in range(n):
        for b in range(n):
            Y_sub[a,b] = Y_dist[incl_idx[a],incl_idx[b]]
            
    if flag is True:
        Y_sub = m2v(Y_sub)
        
    return Y_sub

def remove_dm_nans(Y_dist):
    nan_rows = np.argwhere(np.isnan(Y_dist[0]))
    valid_rows = np.setdiff1d(np.arange(Y_dist.shape[0]),nan_rows)
    Y_dist_nonans = subsample_dm(Y_dist, valid_rows)
    return Y_dist_nonans

def nan_matrix(dims):
    Y = np.empty(dims)
    Y[:] = np.nan
    return Y

#########################
# plotting operations
#########################

def plot_matrix(Y, lib = 'mpl', cbar = True, cmap = 'viridis', tl = '', xl = '', yl = '', vmin = 0, vmax = 0, ticks = True, fontsize = None):
    
    assert(Y.ndim < 3)
    
    if lib is not 'mpl' and lib is not 'sns':
        raise ValueError('must specify mpl or sns library')
    
    minval,maxval = np.min(Y),np.max(Y)
    
    if vmin == 0: 
        vmin = minval
    if vmax == 0:
        vmax = maxval
    
    if lib == 'mpl':
        if Y.shape[0] == Y.shape[1]:     
            fig = plt.imshow(Y, cmap=cmap, clim=(vmin,vmax))
        else:
            fig = plt.imshow(Y, aspect='auto', cmap=cmap, clim=(vmin,vmax))
            
        if cbar is True:
            plt.colorbar()
        
        if ticks is False:
            plt.xticks([])
            plt.yticks([])
                
    elif lib == 'sns':
        
        if Y.shape[0] == Y.shape[1]:
            sq = True
        else:
            sq = False
            
        fig = sns.heatmap(Y, square=sq, cmap=cmap, cbar=cbar, vmin=vmin, vmax=vmax, xticklabels = ticks, yticklabels = ticks)
            
    plt.title(tl, fontsize = fontsize)
    plt.xlabel(xl, fontsize = fontsize)
    plt.ylabel(yl, fontsize = fontsize)
    
    return

def subplot_str_helper(label, n):
    if type(label) is str:
        label = [label]
    if len(label) == 1:
        label = label * n
    return label

def plot_matrices(Y_list, lib = 'mpl', cbar = True, cmap = 'viridis', tl = [''], xl = [''], yl = [''], vmin = 0, vmax = 0, ticks = True, fontsize = None):
    
    n = len(Y_list)
    if type(Y_list) is not list:
        raise ValueError('input must be a list of arrays')
    
    tl = subplot_str_helper(tl, n)
    xl = subplot_str_helper(xl, n)
    yl = subplot_str_helper(yl, n)
          
    c = 1
    for Y in Y_list:
        plt.subplot(1,n,c)
        assert(Y.ndim < 3)
        plot_matrix(Y, lib = lib, cbar = cbar, cmap = cmap, tl = tl[c-1], xl = xl[c-1], yl = yl[c-1], vmin = vmin, vmax = vmax, ticks = ticks, fontsize = fontsize)
        c += 1

        
# todo: clean up
def dm_corr_scatter(Y, corr = 'pearson', dotsize = 1, lib = 'mpl', cbar = True, cmap = 'viridis', tl = ['rdm1','rdm2','corr'], xl = ['imgs','imgs','rdv1'], yl = ['imgs','imgs','rdv2'], vmin = 0, vmax = 0, ticks = True, fontsize = None):
    
    if type(Y) is not list or len(Y) != 2:
        raise ValueError('input must be a list of two distance vectors/matrices')
    
    for i in range(2):
        if np.ndim(Y[i]) == 1 and assert_validdm(squareform(Y[i])) is True:
            Y[i] = squareform(Y[i])
    
        if assert_validdm(Y[i]) is False:
            raise ValueError('input %d is not valid distance matrix' % i)
    
    if Y[0].shape[0] != Y[1].shape[0]:
        raise ValueError('inputs have different dimensionality')
    
    for i in range(2):
        plt.subplot(1,3,i+1)
        plot_matrix(Y[i], lib = lib, cbar = cbar, cmap = cmap, tl = tl[i], xl = xl[i], yl = yl[i], vmin = vmin, vmax = vmax, ticks = ticks, fontsize = fontsize)
        
        # make this more efficient
        Y[i] = squareform(Y[i],force='tovector')
    
    #dotsize = 300 // len(Y[0])
    
    # add generic correlate function
    if corr == 'pearson':
        r = np.corrcoef(Y[0],Y[1])[0,1]
    else:
        raise ValueError('corr method not yet implemented')
    
    plt.subplot(1,3,3)
    plt.scatter(Y[0],Y[1], dotsize)
    
    b, m = polyfit(Y[0],Y[1], 1)

    plt.plot(Y[0], b + m * Y[0], '-', lw = 5, color = 'k')
    plt.title('RDM %s corr = %.3f' % (corr, r), fontsize=fontsize)
    plt.xlabel(xl[2],fontsize=fontsize)
    plt.ylabel(yl[2],fontsize=fontsize)
    plt.axis('equal')
    
    if ticks is False:
        plt.xticks([])
        plt.yticks([])
        
        