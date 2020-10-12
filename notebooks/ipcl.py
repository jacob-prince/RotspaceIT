from __future__ import print_function

import os
import torch
import torch.nn as nn
from functools import partial
from pathlib import Path
from torch.hub import load_state_dict_from_url

__all__ = ['ipcl_alexnet_gn_u128', 'ipcl_alexnet_gn_u128_random']

url_root = 'https://visionlab-pretrainedmodels.s3.amazonaws.com/project_instancenet/ipcl/'

default_model_dir = '/home/jacobpri/git/RotspaceIT/models' #os.path.join(os.environ['XDG_CACHE_HOME'], 'torch', 'checkpoints')

def convert_relu_layers(parent):
    for child_name, child in parent.named_children():
        if isinstance(child, nn.ReLU):
            setattr(parent, child_name, nn.ReLU(inplace=False))
        elif len(list(child.children())) > 0:
            convert_relu_layers(child)
            
class AlexNet(nn.Module):
    ''' Original AlexNet architecture, except the conv channels are not split into groups:
            https://papers.nips.cc/paper/4824-imagenet-classification-with-deep-convolutional-neural-networks.pdf
    
        Note that this version differs from the PyTorch TorchVision version, which implements:
            https://arxiv.org/abs/1404.5997                
        
        Optionally use GroupNorm, BatchNorm, or LocalResponseNorm layers (aka ChannelNorm):
            https://arxiv.org/pdf/1803.08494.pdf
                    
    '''
    def __init__(self, in_channels=3, num_classes=128, group_norm=True, batch_norm=False, 
                 channel_norm=False, l2norm=True):
        super(AlexNet, self).__init__()
        
        n = (group_norm,batch_norm,channel_norm)
        assert sum(n) == 1, f"Exactly one of group_norm, batch_norm, channel_norm should be true, got {n}"
        
        if group_norm:
            norm = lambda nc: nn.GroupNorm(num_groups=32, num_channels=nc)
        elif batch_norm:
            norm = lambda nc: nn.BatchNorm2d(nc)
        elif channel_norm:
            norm = lambda nc: nn.LocalResponseNorm(5)

        self._l2norm = l2norm
        
        self.conv_block_1 = nn.Sequential(
            nn.Conv2d(in_channels, 96, 11, 4, 2, bias=False),
            norm(96),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(3, 2),
        )
        self.conv_block_2 = nn.Sequential(
            nn.Conv2d(96, 256, 5, 1, 2, bias=False),
            norm(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(3, 2),
        )
        self.conv_block_3 = nn.Sequential(
            nn.Conv2d(256, 384, 3, 1, 1, bias=False),
            norm(384),
            nn.ReLU(inplace=True),
        )
        self.conv_block_4 = nn.Sequential(
            nn.Conv2d(384, 384, 3, 1, 1, bias=False),
            norm(384),
            nn.ReLU(inplace=True),
        )
        self.conv_block_5 = nn.Sequential(
            nn.Conv2d(384, 256, 3, 1, 1, bias=False),
            norm(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(3, 2),
        )
        self.avgpool = nn.AdaptiveAvgPool2d((6, 6))
        self.fc6 = nn.Sequential(
            nn.Linear(256 * 6 * 6, 4096),
            nn.BatchNorm1d(4096),
            nn.ReLU(inplace=True),
        )
        self.fc7 = nn.Sequential(
            nn.Linear(4096, 4096),
            nn.BatchNorm1d(4096),
            nn.ReLU(inplace=True),
        )
        self.fc8 = nn.Sequential(
            nn.Linear(4096, num_classes)
        )
        if self._l2norm: self.l2norm = Normalize(2)

    def forward(self, x):
        x = self.conv_block_1(x)
        x = self.conv_block_2(x)
        x = self.conv_block_3(x)
        x = self.conv_block_4(x)
        x = self.conv_block_5(x)
        x = self.avgpool(x)
        x = x.view(x.shape[0], -1)
        x = self.fc6(x)
        x = self.fc7(x)
        x = self.fc8(x)
        if self._l2norm: x = self.l2norm(x)
        return x

    def compute_feat(self, x, layer):
        if layer <= 0:
            return x
        x = self.conv_block_1(x)
        if layer == 1:
            return x
        x = self.conv_block_2(x)
        if layer == 2:
            return x
        x = self.conv_block_3(x)
        if layer == 3:
            return x
        x = self.conv_block_4(x)
        if layer == 4:
            return x
        x = self.conv_block_5(x)
        if layer == 5:
            return x
        x = self.avgpool(x)
        x = x.view(x.shape[0], -1)
        x = self.fc6(x)
        if layer == 6:
            return x
        x = self.fc7(x)
        if layer == 7:
            return x
        x = self.fc8(x)
        if layer == 8:
            return x
        if self._l2norm: x = self.l2norm(x)
        return x

class Normalize(nn.Module):

    def __init__(self, power=2):
        super(Normalize, self).__init__()
        self.power = power

    def forward(self, x):
        norm = x.pow(self.power).sum(1, keepdim=True).pow(1. / self.power)
        out = x.div(norm)
        return out

def alexnet(**kwargs):
    """Constructs an AlexNet model with GroupNorm2d Layers.
    """
    model = AlexNet(group_norm=True, **kwargs)
    return model

def alexnet_gn(**kwargs):
    """Constructs an AlexNet model with GroupNorm2d Layers.
    """
    model = AlexNet(group_norm=True, **kwargs)

    return model

def alexnet_bn(**kwargs):
    """Constructs an AlexNet model with BatchNorm2d Layers.
    """
    model = AlexNet(group_norm=False, batch_norm=True, **kwargs)

    return model

def alexnet_cn(**kwargs):
    """Constructs an AlexNet model with LocalResponseNorm/ChannelNorm Layers.
    """
    model = AlexNet(group_norm=False, channel_norm=True, **kwargs)

    return model

# --------------------------------
#  model builder
# --------------------------------

def build_model(model_class, weight_dir, weights_url=None, supervised=True, low_dim=1000, in_channels=3, note=''):
    
    model_dir = default_model_dir if weights_url is None else weight_dir
    l2norm = not supervised

    if in_channels is not None:
        model = model_class(in_channels=in_channels, num_classes=low_dim, l2norm=l2norm)
    else:
        model = model_class(num_classes=low_dim, l2norm=l2norm)
        
    if weights_url:
        print(f"... loading checkpoint: {Path(weights_url).name}")
        checkpoint = load_state_dict_from_url(weights_url, model_dir=model_dir, map_location=torch.device('cpu'))
        state_dict = {str.replace(k,'module.',''): v for k,v in checkpoint['state_dict'].items()}
        model.load_state_dict(state_dict)
        print("... state loaded.")
    
    if note:
        print(f"... NOTE: {note}")
    
    convert_relu_layers(model)
    
    return model

# --------------------------------
#  pretrained models
# --------------------------------

def ipcl_alexnet_gn_u128(weight_dir=None):
    weight_file = '06_instance_imagenet_AlexNet_n5_lr03_pct40_t07_div1000_e100_bs128_bm20_gn_rep2_final_weights_only.pth.tar'
    return build_model(
        alexnet_gn,
        weight_dir,
        weights_url = url_root + weight_file,
        supervised = False,
        low_dim = 128)

def ipcl_alexnet_gn_u128_random(weight_dir=None):
    return build_model(
        alexnet_gn,
        weight_dir,
        weights_url = None,
        supervised = False,
        low_dim = 128)

if __name__ == '__main__':

    import torch
    model = alexnet().cuda()
    data = torch.rand(10, 3, 224, 224).cuda()

    for i in range(10):
        out = model.compute_feat(data, i)
        print(i, out.shape)