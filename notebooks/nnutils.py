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
import nethook as nethook
import scipy.io as sio
from tqdm import tqdm
import time

def get_layer_names(model):
    
    lay_names_torch = []
    lay_names_user = []
    lay_conv_strs = []
    
    layers = list(model.named_modules())
    
    count = 1
    conv_count = 0

    for i in range(1,len(layers)):
        
        module_type = layers[i][1].__class__.__name__

        nonbasic_types = np.array(['Sequential','BasicBlock','Bottleneck','Fire',
                                 '_DenseBlock', '_DenseLayer', 'Transition', '_Transition','InvertedResidual','_InvertedResidual','ConvBNReLU'])

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

            count += 1
    
    lay_names_user_fmt = []
    
    for lay in lay_names_user:
        lay_type = lay.split('_')[1]
        
        if 'Conv' in lay_type:
            fmt = 'conv'
        elif 'Norm' in lay_type:
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
        else:
            print(lay_type)
            raise ValueError('fmt not implemented yet')
        
        lay_names_user_fmt.append(lay.split('_')[0] + '_' + fmt + lay.split('_')[2])

    print(lay_names_user_fmt)
        
    return lay_names_torch, lay_names_user, lay_names_user_fmt
