import numpy as np
import pandas as pd
import nibabel as nib
import os
from os import listdir
from os.path import isfile, join, exists
from scipy.spatial.distance import pdist, squareform, cosine, euclidean, mahalanobis
from statsmodels.stats.multitest import multipletests
import scipy.io as sio
import scipy.stats as stats

def floc_selectivity(Y, labels, FDR_p = 0.0001):
    
    assert(Y.ndim == 2)
    n_neurons_in_layer = Y.shape[1]
    
    # create output dictionary
    pref_dict = dict()
    pref_dict['twoway_Ns'] = dict() # record of # sig neurons from all 2-sample t-test comparisons between domain pairs
    pref_dict['domain_counts'] = [] # final N selective neurons for each domain when contrasting all domain pairs
    pref_dict['domain_props'] = [] # same, but proportions of layer neurons
    pref_dict['domain_idx'] = [] # indices of the selective neurons
    pref_dict['domain_unit_rankings'] = [] # neurons ranked by selectivity for all domains
    
    domain_idx = np.unique(labels)
    
    # iterate through domains 
    for domainA in domain_idx:
                
        pref_dict['twoway_Ns'][str(domainA)] = dict()
        
        # get data from curr domain
        Y_curr = Y[labels==domainA]
        
        dom_pref_rankings = []
        dom_flag = False

        # iterate through domains (again)
        for domainB in domain_idx:
            
            Y_test = Y[labels==domainB]
    
            # calculate t and p maps
            t,p = stats.ttest_ind(Y_curr,Y_test, axis=0)

            # deal with nans
            t[np.isnan(t)] = 0
            p[np.isnan(p)] = 1

            # determine which neurons remain significant after FDR correction
            # https://stats.stackexchange.com/questions/63441/what-are-the-practical-differences-between-the-benjamini-hochberg-1995-and-t
            FDR = multipletests(p, alpha=FDR_p, method='FDR_by', is_sorted=False, returnsorted=False)

            # sort indices according to the t map
            # sort the neuron indices according to the t map
            dom_pref_ranking = np.flip(np.argsort(t))

            # assert that no indices are repeated
            assert(len(np.unique(dom_pref_ranking)) == len(dom_pref_ranking))

            # calculate the size of the significant ROI
            dom_nsig = np.sum(np.logical_and(FDR[0] == True, t > 0))

            # skip if test was domain vs. same domain
            if domainA != domainB:

                # save nsig in pref dict for plotting
                pref_dict['twoway_Ns'][str(domainA)][str(domainB)] = dom_nsig

                # store neuron selectivity rankings
                dom_pref_rankings.append(dom_pref_ranking)
                
                # if the first comparison...
                if dom_flag is False:
                    dom_neurons = dom_pref_ranking[:dom_nsig] # create deepnet selective region
                    dom_flag = True
                else: # slim down the region
                    dom_neurons = pd.Index.intersection(pd.Index(dom_neurons), pd.Index(dom_pref_ranking[:dom_nsig]), sort = False)

        dom_neurons = dom_neurons.to_numpy()

        # calculate the size of the significant ROI
        dom_nsig = len(dom_neurons)

        # figure out which neurons are most selective across all categ pair comparisons...
        dom_ranking_score = np.zeros(n_neurons_in_layer)

        # "score" each index using the sort function
        for j in range(len(domain_idx)-1):
            dom_score = np.argsort(dom_pref_rankings[j])

            # accumulate scores
            dom_ranking_score = dom_ranking_score + dom_score

        # get the final selectivity indices - lowest score = most selective
        dom_ranking_score_final = np.argsort(dom_ranking_score)
        
        # log
        pref_dict['domain_counts'].append(len(dom_neurons))
        pref_dict['domain_props'].append(len(dom_neurons) / n_neurons_in_layer)
        pref_dict['domain_idx'].append(dom_neurons)
        pref_dict['domain_unit_rankings'].append(dom_ranking_score_final)
        
    return pref_dict