import numpy as np
import matplotlib.pyplot as plt
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
from torch.utils.data import Subset
from torch.utils.data.dataset import Dataset
from torch.utils.data.sampler import SubsetRandomSampler
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

import importlib
import nnutils as utils
import floc_selectivity as fs
import rsatools as rsa
import lesion as lsn

if torch.cuda.is_available():
    torch.cuda.empty_cache()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu") 

##############

# current implementation works for alexnet and vgg16
# close to working for resnet18 but they implement downsampling in a complicated way
arch = 'alexnet'
trained_on = 'object'
imageset = 'imagefiles-fullset'
img_dim = 224
batch_size = 50
FDR_threshold = 0.05
val_batch_size = 250
subset_by = 1

lesion_domain = sys.argv[1] #'Faces'
lesioning_method = sys.argv[2] #'sledgehammer' # sledgehammer, cascade-forward, cascade-backward, single-layer
target_layer = sys.argv[3] #'relu3'
randomize_lesions = sys.argv[4]

if randomize_lesions == 'True' or randomize_lesions is True:
    randomize_lesions = True
else:
    randomize_lesions = False
    
print(randomize_lesions)
print(type(randomize_lesions))
    
overwrite = True
save_as = '.npy'
save_layer_rdvs = False
rdv_dist = 'correlation'
verbose = False
draw_plots = False

homedir = '/home/jacobpri/git/RotspaceIT/'
network = arch + '-' + trained_on
FDR_str = str(FDR_threshold).replace('.','_')
activation_savedir = join(homedir,'data','d02_modeling','activations',network,'localizer',imageset,'dim'+str(img_dim))
os.makedirs(activation_savedir,exist_ok=True)

imageset_dir = join(homedir,'imagesets','localizer',imageset)
assert(exists(imageset_dir))

val_imageset_dir = '/lab_data/tarrlab/common/datasets/ILSVRC/Data/CLS-LOC/val/'
assert(exists(val_imageset_dir))

floc_savedir = join(homedir,'data','d02_modeling','selectivity',network, 'localizer',imageset,'dim'+str(img_dim))
os.makedirs(floc_savedir,exist_ok=True)

utils.reproducible_results(365)

if lesioning_method == 'sledgehammer':
    target_str = ''
else:
    target_str = '_target-' + target_layer

lesion_resultsdir = join(homedir,'data','d02_modeling','lesioning',network,imageset)

results_str = f'LesionResults_dim-{img_dim}_FDR-{FDR_str}_subset-{subset_by}_method-{lesioning_method}_random-{str(randomize_lesions)}{target_str}_domain-{lesion_domain}'

os.makedirs(lesion_resultsdir,exist_ok=True)

print(results_str)

###################################

if exists(join(lesion_resultsdir,f'{results_str}.npy')) and overwrite is False:
    print('result already exists. skipping.')
else:

    lesioning_domains = [lesion_domain]#,'Scenes','Objects','Bodies','Scrambled']#,'Scenes','Objects','Bodies','Scrambled']

    weight_dir =  join(homedir,'data','d02_modeling','weights')
    
    if arch == 'alexnet':
        src_model = models.alexnet()
        if trained_on == 'object':
            checkpoint = torch.load(os.path.join(weight_dir,'alexnet_imagenet_final.pth.tar'),map_location='cpu')
            checkpoint['state_dict'] = {str.replace(k,'module.',''): v for k,v in checkpoint['state_dict'].items()}
            src_model.load_state_dict(checkpoint['state_dict'])
    if arch == 'vgg16':
        if trained_on == 'object':
            src_model = models.vgg16(pretrained=True)

    # convert relus from in-place to not in-place so preceding layer acts are preserved
    utils.convert_relu(src_model) 

    # create empty masks object
    masks = dict()
    masks['apply'] = False

    # initialize lesioning model using modules from source network
    model = lsn.LesionNet(src_model, masks)

    if torch.cuda.is_available():
        model.to(device)
        print(device)

    #print(model)
    
    ####################

    dataset = datasets.ImageFolder(root = imageset_dir)

    # normalize images using parameters from the training image set
    data_transform = transforms.Compose([       
     transforms.Resize(img_dim),                   
     transforms.CenterCrop((img_dim,img_dim)),         
     transforms.ToTensor(),                    
     transforms.Normalize(                      
     mean=[0.485, 0.456, 0.406],                
     std=[0.229, 0.224, 0.225]                  
     )])

    # reload the dataset, this time applying the transform
    dataset =  datasets.ImageFolder(root = imageset_dir, transform = data_transform)

    # data loader object is required for passing images through the network - choose batch size and num workers here
    data_loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=0,
        shuffle=False
    )

    ######################

    utils.extract_and_save_activations(model,
                                       device,
                                       data_loader, 
                                       model.layers, 
                                       model.layers_fmt, 
                                       activation_savedir,
                                       overwrite = overwrite,
                                       verbose = verbose)

    ########################

    # print some info -> verify correct # imgs, etc
    categ_idx = np.array(dataset.targets)
    floc_domains = dataset.classes ### be careful with this line
    
    print(floc_domains)
    
    #set_trace()
    
    FDR_str = str(FDR_threshold).replace('.','_')
    floc_str = f'layer_pref_dicts_FDR_{FDR_str}.npy'
    floc_fullsave_fn = join(floc_savedir,floc_str)

    ## run deepnet floc experiment, returning a dict with each layer's pref dict

    if exists(floc_fullsave_fn):
        print('preference dict already exists. loading...')
        pref_dicts = np.load(floc_fullsave_fn,allow_pickle=True).item()

    else:
        pref_dicts = dict()

        with tqdm(total=100,desc='Conducting deepnet floc experiments') as pbar:
            for layer in model.layers_fmt:

                time.sleep(0.1) 
                
                print(layer)

                Y = utils.load_batched_activations(activation_savedir, layer, batch_size, reshape_to_2D = True)
                # get selective units. FDR threshold is 0.05
                pref_dicts[layer] = fs.floc_selectivity(Y, categ_idx, FDR_threshold)

                pbar.update(100/len(model.layers_fmt))

        np.save(floc_fullsave_fn, pref_dicts)

    #########################

    domain_categ_sel_masks = dict()

    for lesioning_domain in lesioning_domains:

        print(lesioning_domain)

        lesioning_idx = [i for i in range(len(floc_domains)) if floc_domains[i] == lesioning_domain][0]

        categ_sel_masks = dict()

        for lay in range(len(model.layers)):
            mask = pref_dicts[model.layers_fmt[lay]]['domain_sel_masks'][lesioning_idx] # index 1 for faces
            Y = utils.load_batched_activations(activation_savedir, [model.layers_fmt[lay]], batch_size, reshape_to_2D = False)
            dims = Y.shape[1:]
            mask = np.reshape(mask,dims)
            categ_sel_masks[model.layers[lay]] = torch.Tensor(mask).to(device)

        # set apply parameter to true
        categ_sel_masks['apply'] = False
        categ_sel_masks['fc8'] = torch.Tensor(np.ones(categ_sel_masks['fc8'].shape)).to(device)

        #### right here: change the masks based on the lesioning method ####
        domain_categ_sel_masks[lesioning_domain] = lsn.set_lesioning_method(categ_sel_masks,
                                                                        lesioning_method,
                                                                        target_layer,
                                                                        randomize_lesions,
                                                                        model.layers, 
                                                                        device)

    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    masks = dict()
    masks['apply'] = False

    model = lsn.LesionNet(src_model, masks) # inputs: source arch, masks

    utils.convert_relu(model)

    if torch.cuda.is_available():
        model.to(device)

    #!nvidia-smi

    ###############

    results = dict()
    results['no_lesion'] = dict()

    # reload the dataset, this time applying the transform
    val_dataset =  datasets.ImageFolder(root = val_imageset_dir, transform = data_transform)
    
    val_subset_idx = list(range(0, len(val_dataset), subset_by))
    val_dataset = Subset(val_dataset, val_subset_idx)

    # data loader object is required for passing images through the network - choose batch size and num workers here
    val_data_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=val_batch_size,
        num_workers=12,
        shuffle=False
    )

    all_c_accs, all_layer_mean_acts = lsn.lesioning_validation_pass(model, 
                                                                    device, 
                                                                    val_dataset,
                                                                    val_data_loader,
                                                                    domain_categ_sel_masks)

    results['no_lesion']['accuracies'] = all_c_accs
    results['no_lesion']['selective_unit_acts'] = all_layer_mean_acts # in selective subsets only

    #####################

    for lesioning_domain in lesioning_domains:

        results[lesioning_domain] = dict()

        categ_sel_masks = domain_categ_sel_masks[lesioning_domain]
        categ_sel_masks['apply'] = True

        lesion_model = lsn.LesionNet(src_model,categ_sel_masks) # inputs: source arch, masks

        utils.convert_relu(lesion_model)

        if torch.cuda.is_available():
            lesion_model.to(device)

        all_c_accs, all_layer_mean_acts = lsn.lesioning_validation_pass(lesion_model, 
                                                                        device, 
                                                                        val_dataset,
                                                                        val_data_loader)

        results[lesioning_domain]['accuracies'] = all_c_accs
        results[lesioning_domain]['selective_unit_acts'] = all_layer_mean_acts # in selective subsets only 

        ##############

        costs = results['no_lesion']['accuracies'] - results[lesioning_domain]['accuracies']
        results[lesioning_domain]['costs'] = costs
        results[lesioning_domain]['mean_cost'] = np.nanmean(costs)

#         plt.figure(figsize=(17,5))
#         plt.subplot(131)
#         plt.scatter(results['no_lesion']['accuracies'],results[lesioning_domain]['accuracies'],5)
#         mean_cost = np.mean(costs)
#         plt.axis('square')
#         plt.plot(np.arange(0,1,0.01),np.arange(0,1,0.01),'g--')
#         plt.xlabel('acc without lesions',fontsize=14)
#         plt.ylabel('acc with lesions',fontsize=14)
#         plt.title(f'cost of {lesioning_domain} lesioning\ncateg mean = {round(mean_cost,3)}',fontsize=14);

        corrs = []
        for i in range(len(model.layers)):
            lay = model.layers[i]
            x = results['no_lesion']['selective_unit_acts'][lesioning_domain][lay]
            y = results['no_lesion']['accuracies'] - results[lesioning_domain]['accuracies']
            corr = np.corrcoef(x,y)[1,0]
            corrs.append(round(corr,4))

#         plt.subplot(133)
#         plt.plot(np.arange(len(model.layers)),corrs,'mo-')
#         plt.xticks(np.arange(len(model.layers)), model.layers, rotation = 90);
#         plt.grid(True)
#         plt.title(f'read-out effect of {lesioning_domain} lesioning\n layer summary',fontsize=14)
#         plt.ylabel('pearson r',fontsize=14)
#         plt.xlabel('layer',fontsize=14)
#         plt.ylim([0,0.7])

        corrs = np.array(corrs)
        results[lesioning_domain]['readout_effect_corrs'] = corrs

        nan_idx = np.isnan(corrs)
        corrs[nan_idx] = 0
        max_idx = np.argmax(corrs)

        x = results['no_lesion']['selective_unit_acts'][lesioning_domain][model.layers[max_idx]]
        y = results['no_lesion']['accuracies'] - results[lesioning_domain]['accuracies']

        results[lesioning_domain]['topcorr_layer_acts'] = x

#         plt.subplot(132)
#         plt.scatter(x,y,5)
#         plt.plot(np.unique(x), np.poly1d(np.polyfit(x, y, 1))(np.unique(x)),'r')
#         plt.title(f'read-out effect: top corr layer\n{model.layers[max_idx]}: r = {round(np.corrcoef(x,y)[1,0],3)}',
#                  fontsize=14)
#         plt.xlabel('mean categ. activation',fontsize=14)
#         plt.ylabel('categ. cost of lesioning',fontsize=14)
#         plt.show()

#         figfn = '1_CategCost_ReadoutEffect_Summaries.png'
#         plt.savefig(join(lesion_resultsdir,figfn),format='png')

        rankings = np.flip(np.argsort(costs))

        results[lesioning_domain]['cost_rankings'] = rankings

#         plot_rankings = copy.deepcopy(rankings)
#         img_rankings = copy.deepcopy(rankings)
#         for r in range(len(rankings)):
#             img_rankings[r] = rankings[r] * 50

#         data_transform_ = transforms.Compose([       
#          transforms.Resize(112),                   
#          transforms.CenterCrop((112,112))])

#         val_dataset_ =  datasets.ImageFolder(root = val_imageset_dir, transform = data_transform_)

#         p = 1000
#         plt.figure(figsize=(80,50))
#         c = 1
#         for i in range(p):
#             plt.subplot(25,40,c)
#             plt.imshow(val_dataset_[img_rankings[i]][0])
#             plt.xticks([])
#             plt.yticks([])
#             plt.axis('tight')
#             ax = plt.gca()
#             ax.get_xaxis().set_visible(False)
#             ax.get_yaxis().set_visible(False)
#             c += 1

#         figfn = '2_CategCostRanking_Montage.png'
#         plt.savefig(join(lesion_resultsdir,figfn),format='png')
        
        ########## SAVE RESULTS DICT ##########
        resfn = f'{results_str}.npy'
        np.save(join(lesion_resultsdir,resfn), results)
        
        