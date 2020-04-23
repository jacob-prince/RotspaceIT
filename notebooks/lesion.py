import numpy as np
#import matplotlib.pyplot as plt
import pandas as pd
import scipy.stats as stats
import os
from os.path import exists, join
import sys
from tqdm import tqdm
import time
import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data.dataset import Dataset
import torchvision.models as models
import torchvision.datasets as datasets 
from torchsummary import summary
from torch.autograd import Variable as V
from torch.nn import functional as F
import torch.nn  as nn
from PIL import Image
from IPython.core.debugger import set_trace
import scipy.io as sio
from time import sleep
import gc
import copy

#import importlib
import nnutils as utils
import floc_selectivity as fs
import rsatools as rsa

###############################################

# lesioning function: takes in a batch of activations and a mask of appropriate shape
# if apply=True, performs element-wise multiplication between acts and mask. 
# lesioning accomplished by setting some mask units equal to 0

def lesion(x,mask,apply):
    if apply is True:
        #print(f'applying mask of shape {mask.shape} to acts of shape {x.shape}')
        if len(mask.shape) == 4 and x.shape != mask.shape:
            if mask.shape[0] == x.shape[0] and mask.shape[-1] == x.shape[-1]:
                mask = torch.squeeze(mask)
            else:
                raise ValueError('mask and activation shapes are not equal')
        return x * mask
    else:
        return x
    
    
# function to transfer all modules from a given architecture to a lesioning model
# which, when initialized, will not have had modules defined

def transfer_modules(from_model, to_model):
    
    _, _, layers_fmt, modules = utils.get_layer_names(from_model)
    
    for i in range(len(modules)):
        setattr(to_model,layers_fmt[i].split('_')[1],modules[i])

    return to_model

# lesioning model: takes in source model, dict of masks (can be empty).
# custom forward function for applying lesioning via masking if apply=True
# forward function applies modules sequentially by accessing their names directly
# activations are saved in a dictionary post-masking

class LesionNet(nn.Module):
    
    def __init__(self, source_model, masks, num_classes = 1000): # default for imagenet clf
        super(LesionNet, self).__init__()
        
        self = transfer_modules(source_model, self) # transfer modules from source model
    
        # deal with masks
        self.masks = masks
        
        self.layers, _, self.layers_fmt, _ = utils.get_layer_names(self)
        
        #if self.masks['apply'] is False: # create empty masks for each layer if user didn't supply masks
        #    for layer in self.layers:
        #        self.masks[layer] = None
                                    
    def forward(self, x):
        
        activations = dict()
        
        fc_flag = False # for knowing when to flatten (won't work for models that "widen" e.g. autoencoders)
        
        # for each layer
        for i in range(len(self.layers)):
            
            layer = self.layers[i]            

            # apply that layer's forward attribute
            operation = getattr(self,layer)
            x = operation(x)
            
            # get the mask for that layer, and tile along the image dimension
            if self.masks['apply'] == True:
                if layer != self.layers[-1]:
                    mask = self.masks[layer].repeat(x.shape[0],1,1,1)
                else:
                    mask = self.masks[layer]
            else:
                mask = None
            
            # apply lesioning
            x = lesion(x, mask, self.masks['apply'])
            
            activations[layer] = x
            
            # flatten if necessary -> do it before the first linear
            if fc_flag is False and ('fc' in self.layers[i+1] or 'linear' in self.layers[i+1]):
                x = torch.flatten(x, 1)
                fc_flag = True
                
            # helpful print statement to verify that masking worked as expected
            #print(f'# units inactive:  {torch.sum(x==0)/x.shape[0]}     ({layer})')
                
        return x, activations

def get_device(empty_cache=True):
    if empty_cache is True:
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu") 
    return device


# def lesioning_experiment(model,
#                          lesioning_domain,
#                          imageset_dir_loc, 
#                          imageset_dir_exp, 
#                          activation_savedir):

def lesioning_validation_pass(model, 
                              device, 
                              val_dataset,
                              val_data_loader,
                              domain_masks = None):
    
    if isinstance(domain_masks,dict):
        domains = domain_masks.keys()
        domain_mean_acts = dict()
    
        for domain in domains:
            domain_mean_acts[domain] = dict()
            for lay in model.layers:
                domain_mean_acts[domain][lay] = []
        
    #criterion = nn.CrossEntropyLoss()
    
    true_labels = []
    pred_labels = []

    # switch model to evaluation mode
    model.eval()

#     acc1s = []
#     acc5s = []

    # calculate accuracy on validation set
#     n_correct = 0
    with torch.no_grad():
        with tqdm(total=100,desc='Computing validation accuracy') as pbar:
            for batch_idx, (imgs,labels) in enumerate(val_data_loader):
            
                #t_start = time.time()
                
                #try:
                #    print(t_end - t_start)
                #except:
                #    pass
                
                time.sleep(0.1)
                #print(batch_idx)
                
                #print(imgs.shape)
            
                if torch.cuda.is_available():
                    out, acts = model(imgs.to(device))
                    labels = labels.to(device)
                else:
                    out, acts = model(imgs)

                true_labels.append(labels.detach().cpu().numpy())
                preds = torch.max(out, 1)[1].view(labels.size())
                pred_labels.append(preds.detach().cpu().numpy())
                
                if model.masks['apply'] == False and isinstance(domain_masks,dict):
                    for domain in domains:
                        for lay in model.layers:
                            
                            mean_act = torch.mean(acts[lay][:,domain_masks[domain][lay] == 0],dim=1)
                            
                            domain_mean_acts[domain][lay].append(mean_act)
                            #print(domain,lay,mean_act.shape)
                            #set_trace()
                            
                pbar.update(100/len(val_data_loader))
                
                #t_end = time.time()

    #val_acc = 100. * n_correct / len(val_dataset.classes)

    #print(val_acc)

    categs = np.sort(np.unique(np.concatenate(true_labels))) #val_dataset.targets
    #print(categs)
    assert(len(categs) == 1000)
    #categs = np.unique(categ_idx)

    all_pred_labels = np.concatenate(pred_labels)
    all_true_labels = np.concatenate(true_labels)

    all_c_accs = np.zeros((len(categs),))

    for c in categs:
        idx = np.argwhere(all_true_labels == c) # which indices are from this category?
        all_c_accs[c] = np.nanmean(all_pred_labels[idx] == c) # what's the accuracy at those indices?

    all_layer_mean_acts = dict()
    
    if model.masks['apply'] == False and isinstance(domain_masks,dict):
    
        for domain in domains:

            all_layer_mean_acts[domain] = dict()

            for lay in model.layers:

                    c_acts = np.zeros((len(categs),))
                    all_mean_acts = torch.cat(domain_mean_acts[domain][lay]).cpu().numpy()
                    #print(all_mean_acts.shape)

                    for c in categs:
                        idx = np.argwhere(all_true_labels == c)
                        c_acts[c] = np.nanmean(all_mean_acts[idx])

                    all_layer_mean_acts[domain][lay] = c_acts

    return all_c_accs, all_layer_mean_acts


def set_lesioning_method(layer_mask_dict, method, target_lay, randomize, layer_names, device):
    
    output_dict = copy.deepcopy(layer_mask_dict)
    
    if method == 'sledgehammer':
        for lay in layer_names:
            n = int(torch.sum(output_dict[lay] == 0).cpu().numpy())
            print(f'{n} inactive units in layer {lay}')
            
            if randomize is True:
                output_dict[lay] = utils.random_shuffle_tensor(output_dict[lay]).to(device)
    
    elif method == 'single-layer':
        if target_lay in output_dict.keys(): # only proceed if valid layer was passed
            for lay in layer_names:
                if lay != target_lay: # remove lesioning from all layers except target layer
                    output_dict[lay] = torch.Tensor(np.ones(output_dict[lay].shape)).to(device)
                else:
                    if randomize is True:
                        # randomize values in mask
                        output_dict[lay] = utils.random_shuffle_tensor(output_dict[lay]).to(device)
                        
                n = int(torch.sum(output_dict[lay] == 0).cpu().numpy())
                print(f'{n} inactive units in layer {lay}')
        else:
            raise ValueError('layer not found in mask dict')
            
    elif method == 'cascade-forward':
        if target_lay in output_dict.keys():
            flag = True
            for lay in layer_names:
                
                if flag is True:
                    # either randomize or don't
                    if randomize is True:
                        # randomize values in mask
                        output_dict[lay] = utils.random_shuffle_tensor(output_dict[lay]).to(device)
                else: # remove lesions
                    output_dict[lay] = torch.Tensor(np.ones(output_dict[lay].shape)).to(device)
                
                if lay == target_lay:
                    # change flag
                    flag = False
                    
                n = int(torch.sum(output_dict[lay] == 0).cpu().numpy())
                print(f'{n} inactive units in layer {lay}')
        else:
            raise ValueError('layer not found in mask dict')
                
    elif method == 'cascade-backward':
        if target_lay in output_dict.keys():
            flag = False
            for lay in layer_names:
                
                if lay == target_lay:
                    # change flag
                    flag = True
                    
                if flag is True:
                    # either randomize or don't
                    if randomize is True:
                        # randomize values in mask
                        output_dict[lay] = utils.random_shuffle_tensor(output_dict[lay]).to(device)
                else: # remove lesions
                    output_dict[lay] = torch.Tensor(np.ones(output_dict[lay].shape)).to(device)
                 
                n = int(torch.sum(output_dict[lay] == 0).cpu().numpy())
                print(f'{n} inactive units in layer {lay}')
        else:
            raise ValueError('layer not found in mask dict')
                     
    else:
        raise NotImplementedError()
        
    return output_dict
    
    