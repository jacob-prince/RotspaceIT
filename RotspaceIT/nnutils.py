import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import scipy.stats as stats
import os
from os.path import join, exists
import sys
import torch
import torchvision
import torchvision.transforms as transforms
import torchvision.models as models
import torchvision.datasets as datasets
from torch.autograd import Variable as V
from torch.nn import functional as F
import torch.nn  as nn
from PIL import Image
from IPython.core.debugger import set_trace
import scipy.io as sio
from tqdm import tqdm
import time

sys.path.append('/home/jacobpri/git/RotspaceIT/')
from RotspaceIT import nethook as nethook

def get_layer_names(model):
    
    lay_names_torch = []
    lay_names_user = []
    lay_conv_strs = []
    lay_modules = []
    
    layers = list(model.named_modules())
    
    count = 1
    conv_count = 0

    for i in range(1,len(layers)):
        
        module_type = layers[i][1].__class__.__name__

        nonbasic_types = np.array(['Sequential','BasicBlock','Bottleneck','Fire',
                                 '_DenseBlock', '_DenseLayer', 'Transition', '_Transition','InvertedResidual','_InvertedResidual','ConvBNReLU','CORblock_Z','CORblock_S','CORblock'])

        conv_types = ['Conv','Linear']

        if np.logical_not(np.isin(module_type, nonbasic_types)):

            # get layer naming info
            lay_names_torch.append(layers[i][0])
            lay_names_user.append(str(count) + '_' + str(layers[i][1]).split('(')[0])

            # get conv tag
            if any(s in lay_names_user[-1] for s in conv_types) and 'downsample' not in lay_names_torch[-1]:
                conv_count += 1
            lay_conv_strs.append(str(conv_count))

            # update
            lay_names_user[-1] = lay_names_user[-1] + '_' + lay_conv_strs[-1]
            
            lay_modules.append(layers[i][1])

            count += 1
    
    lay_names_user_fmt = []
    
    c = 0
    for lay in lay_names_user:
        lay_type = lay.split('_')[1]
        
        if 'Conv' in lay_type:
            if 'downsample' in lay_names_torch[c]:
                fmt = 'downsample'
            elif 'skip' in lay_names_torch[c]:
                fmt = 'conv_skip'
            else:
                fmt = 'conv'
        elif 'Norm' in lay_type:
            if 'skip' in lay_names_torch[c]:
                fmt = 'norm_skip'
            else:
                fmt = 'norm'
        elif 'ReLU' in lay_type:
            fmt = 'relu'
        elif 'MaxPool' in lay_type:
            fmt = 'maxpool'
        elif 'AvgPool' in lay_type:
            fmt = 'avgpool'
        elif 'Linear' in lay_type:
            fmt = 'fc'
        elif 'Dropout' in lay_type:
            fmt = 'drop'
        elif 'Identity' in lay_type:
            fmt = 'identity'
        elif 'Flatten' in lay_type:
            fmt = 'flatten'
        elif 'Lesion' in lay_type:
            fmt = 'lesion'
        elif 'Sigmoid' in lay_type:
            fmt = 'sigmoid'
        else:
            print(lay_type)
            raise ValueError('fmt not implemented yet')
        c+=1
        
        lay_names_user_fmt.append(lay.split('_')[0] + '_' + fmt + lay.split('_')[2])

    #print(lay_names_user_fmt)
        
    return lay_names_torch, lay_names_user, lay_names_user_fmt, lay_modules

def convert_relu(parent):
    for child_name, child in parent.named_children():
        if isinstance(child, nn.ReLU):
            setattr(parent, child_name, nn.ReLU(inplace=False))
        elif len(list(child.children())) > 0:
            convert_relu(child)
            

def check_dir_for_layer(abs_path, layer_str, save_fmt):
    
    if exists(abs_path) is False:
        raise ValueError('must input existing directory')
        
    all_files = os.listdir(abs_path)
    strs_to_check = []
    
    for file in all_files:
        lay_str = file.split('_')[0] + '_' + file.split('_')[1]
        
        if save_fmt in file:
            strs_to_check.append(lay_str)
    
    if any(s in layer_str for s in strs_to_check):
        return True
    else:
        return False
    
def load_batched_activations(abs_path, layer_list, batch_size, reshape_to_2D = True):
    
    Y = []
    
    if not isinstance(layer_list, list):
        #print('converting layer list from string to list')
        layer_list = [layer_list]
    
    if exists(abs_path) is False:
        raise ValueError('must input existing directory')
      
    all_files = os.listdir(abs_path)
    
    # for each layer, assemble matrix and append
    for layer in layer_list:
        #print('layer:',layer)
        
        load_files = []
        batch_idx = []
        for file in all_files:
            if layer in file and str(batch_size) in file:
                load_files.append(file)
                batch_idx.append(int(file.split('-')[-1][:-4]))
        
        load_files = np.array(load_files)[np.argsort(batch_idx)]
        
        #print(loadfiles)
            
        Y_lay = []
        for file in load_files: # messy but fine
            
            #print('string %s found in path %s' % (layer, file))
            Y_batch = np.load(join(abs_path,file))
            #print(Y_batch.shape)
            dims = Y_batch.shape
                
            if np.sum(np.isnan(Y_batch)) > 0:
                raise ValueError('loaded batch contains nan(s)')

            if np.ndim(Y_batch) != 2 and reshape_to_2D is True:
                Y_batch = np.reshape(Y_batch, (dims[0], dims[1] * dims[2] * dims[3]))

            Y_lay.append(Y_batch)
        
        if len(layer_list) == 1:
            return np.vstack(Y_lay)
        else:
            Y.append(np.vstack(Y_lay))
           
    return Y

    

def load_batched_activations_old(abs_path, layer_list, batch_size, reshape_to_2D = True):
    
    Y = []
    
    if not isinstance(layer_list, list):
        #print('converting layer list from string to list')
        layer_list = [layer_list]
    
    if exists(abs_path) is False:
        raise ValueError('must input existing directory')
      
    pathlist_fn = join(abs_path,'absolute_act_filepaths_batchsize-%d.npy' % batch_size)
    
    if exists(pathlist_fn) is False:
        print(pathlist_fn)
        raise ValueError('dir is missing list of absolute filepaths')
        
    pathlist = np.load(pathlist_fn,allow_pickle=True)
        
    # get items that have the correct batch size
    valid_paths = []
    for path in pathlist:
        if 'batchsize-' + str(batch_size) in path:
            valid_paths.append(path)
    
    # for each layer, assemble matrix and append
    for layer in layer_list:
        #print('layer:',layer)
        Y_lay = []
        for path in valid_paths: # messy but fine
            if layer in path:
                #print('string %s found in path %s' % (layer, path))
                Y_batch = np.load(path)
                #print(Y_batch.shape)
                dims = Y_batch.shape
                
                if np.sum(np.isnan(Y_batch)) > 0:
                    raise ValueError('loaded batch contains nan(s)')
                    
                if np.ndim(Y_batch) != 2 and reshape_to_2D is True:
                    Y_batch = np.reshape(Y_batch, (dims[0], dims[1] * dims[2] * dims[3]))
               
                Y_lay.append(Y_batch)
        
        if len(layer_list) == 1:
            return np.vstack(Y_lay)
        else:
            Y.append(np.vstack(Y_lay))
           
    return Y

def extract_and_save_activations(model, device, data_loader, layers_to_retain, layers_to_retain_fmt, activation_savedir, save_as = '.npy', verbose = True, overwrite = True):
    absolute_act_filepaths = []
    total_MB_saved = 0
    batch_size = data_loader.batch_size

    # not training mode
    model.eval()

    # no gradient saving 
    with torch.no_grad():

        # dictionary comprehension
        features = {layer: None for layer in layers_to_retain}

        with tqdm(total=100,desc='Computing network activations') as pbar:

            # iterate through batches
            for batch_no, (imgs,labels) in enumerate(data_loader):

                batch_str = 'batch-' + str(batch_no)

                time.sleep(0.1)

                # nethook is saving features in a dictionary
                if torch.cuda.is_available():
                    out, acts = model(imgs.to(device))
                else:
                    out, acts = model(imgs)

                #set_trace()
                # counter for layer names
                idx = 0

                # accumulate activations by looping through layer names
                for layer_name in layers_to_retain:

                    # get name
                    layer_str = layers_to_retain_fmt[idx]

                    save_str = layer_str + '_batchsize-' + str(batch_size) + '_' + batch_str + save_as
                    save_fn = join(activation_savedir, save_str)

                    absolute_act_filepaths.append(save_fn)

                    # need to reshape for concatenation
                    X = acts[layer_name] # model.retained_layer(layer_name)

                    # move to CPU and detatch from computational grid
                    X = X.cpu().detach().numpy() 

                    # compute size in memory:
                    X_mem = np.divide(float(X.size * X.itemsize),1000000)

                    # save batch
                    if verbose is True:
                        print("\tfile to save: %s (size in memory: %0.3f MB)" % (save_str, X_mem))

                    if exists(save_fn) and overwrite is False:
                        if verbose is True:
                            print('skipping, file %s already exists' % save_str)
                    else:

                        total_MB_saved += X_mem

                        if save_as == '.npy':
                            np.save(save_fn, X)
                        elif save_as == '.mat':
                            sio.savemat(save_fn, {'X': X})
                        else:
                            raise ValueError('no file type specified')

#                     # concatenate if necessary
#                     if save_layer_rdvs is True:

#                         # chunk on to the dictionary
#                         if features[layer_name] is None:
#                             features[layer_name] = X
#                         else:                
#                             features[layer_name] = np.concatenate((features[layer_name], X), axis = 0)

                    idx += 1 

                pbar.update(100/len(data_loader))

    path_save_str = 'absolute_act_filepaths_batchsize-' + str(batch_size) + save_as
    names_save_str = 'lay_names_user_fmt_batchsize-' + str(batch_size) + save_as

    path_save_fn = join(activation_savedir, path_save_str)
    names_save_fn = join(activation_savedir, names_save_str)

    if verbose is True:
        print(path_save_fn)
        print(names_save_fn)

    if save_as == '.npy':
        np.save(path_save_fn, absolute_act_filepaths)
        np.save(names_save_fn, layers_to_retain_fmt)
    elif save_as == '.mat':
        sio.savemat(path_save_fn, {'absolute_act_filepaths': absolute_act_filepaths})
        sio.savemat(names_save_fn, {'lay_names_user_fmt': layers_to_retain_fmt})
    else:
        raise ValueError('no file type specified')

    if verbose is True:
        print('total memory saved: %0.3f MB' % total_MB_saved)
    
    return


def extract_and_save_activations_nethook(model, device, data_loader, layers_to_retain, layers_to_retain_fmt, activation_savedir, save_as = '.npy', verbose = True, overwrite = True):
    absolute_act_filepaths = []
    total_MB_saved = 0
    batch_size = data_loader.batch_size

    # not training mode
    model.eval()

    # no gradient saving 
    with torch.no_grad():

        # dictionary comprehension
        features = {layer: None for layer in layers_to_retain}

        with tqdm(total=100,desc='Computing network activations') as pbar:

            # iterate through batches
            for batch_no, (imgs,labels) in enumerate(data_loader):

                batch_str = 'batch-' + str(batch_no)

                time.sleep(0.1)

                # nethook is saving features in a dictionary
                if torch.cuda.is_available():
                    out = model(imgs.to(device))
                else:
                    out = model(imgs)

                # counter for layer names
                idx = 0

                # accumulate activations by looping through layer names
                for layer_name in layers_to_retain:

                    # get name
                    layer_str = layers_to_retain_fmt[idx]

                    save_str = layer_str + '_batchsize-' + str(batch_size) + '_' + batch_str + save_as
                    save_fn = join(activation_savedir, save_str)

                    absolute_act_filepaths.append(save_fn)

                    # need to reshape for concatenation
                    X = model.retained_layer(layer_name)

                    # move to CPU and detatch from computational grid
                    X = X.cpu().detach().numpy() 

                    # compute size in memory:
                    X_mem = np.divide(float(X.size * X.itemsize),1000000)

                    # save batch
                    if verbose is True:
                        print("\tfile to save: %s (size in memory: %0.3f MB)" % (save_str, X_mem))

                    if exists(save_fn) and overwrite is False:
                        print('skipping, file %s already exists' % save_str)
                    else:

                        total_MB_saved += X_mem

                        if save_as == '.npy':
                            np.save(save_fn, X)
                        elif save_as == '.mat':
                            sio.savemat(save_fn, {'X': X})
                        else:
                            raise ValueError('no file type specified')

                    # concatenate if necessary
#                     if save_layer_rdvs is True:

#                         # chunk on to the dictionary
#                         if features[layer_name] is None:
#                             features[layer_name] = X
#                         else:                
#                             features[layer_name] = np.concatenate((features[layer_name], X), axis = 0)

                    idx += 1 

                pbar.update(100/len(data_loader))

    path_save_str = 'absolute_act_filepaths_batchsize-' + str(batch_size) + save_as
    names_save_str = 'lay_names_user_fmt_batchsize-' + str(batch_size) + save_as

    path_save_fn = join(activation_savedir, path_save_str)
    names_save_fn = join(activation_savedir, names_save_str)

    print(path_save_fn)
    print(names_save_fn)

    if save_as == '.npy':
        np.save(path_save_fn, absolute_act_filepaths)
        np.save(names_save_fn, layers_to_retain_fmt)
    elif save_as == '.mat':
        sio.savemat(path_save_fn, {'absolute_act_filepaths': absolute_act_filepaths})
        sio.savemat(names_save_fn, {'lay_names_user_fmt': layers_to_retain_fmt})
    else:
        raise ValueError('no file type specified')

    print('total memory saved: %0.3f MB' % total_MB_saved)
    
    return

def reproducible_results(seed=1):
    torch.random.manual_seed(seed)
    np.random.seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    
    
def transfer_weights_biases(from_model, to_model):
    
    from_layers,_,_ = utils.get_layer_names(from_model)
    to_layers,_,_ = utils.get_layer_names(to_model)
    
    for i in range(len(from_layers)):
        if 'conv' in to_layers[i] or 'linear' in to_layers[i]:

            target = getattr(to_model,to_layers[i])
            field = from_layers[i]
            
            if '.' in field:
                block = field.split('.')[0]
                idx = int(field.split('.')[1])

                attr = getattr(from_model,block)[idx]
                
                target.weight = attr.weight
                target.bias = attr.bias
                
                setattr(to_model,to_layers[i],target)
                print(f'transferred params from layer {from_layers[i]} to layer {to_layers[i]}')

    return to_model


def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, dim=1, largest=True, sorted=True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
        res.append(correct_k.mul_(1.0 / batch_size).item())
    return res

def random_shuffle_tensor(X):
    idx = torch.randperm(X.nelement())
    out = X.view(-1)[idx].view(X.size())
    return out
    