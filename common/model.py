# Copyright (c) 2018-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

import torch.nn as nn
import torch

'''Wrapper around PyTorch Module, which is base class for all neural network modules. 
Models should always subclass this class. Modules can also contain other Modules, allowing to nest them in
tree structure. You can assign submodules as regular attributes. Submodules assigned in this way will be registered,
and will have their parameters converted too when you call .cuda(), etc.
All neural network components for PyTorch should inherit from nn.Module and override the forward() method. Inheriting from nn.Module 
provides functionality to your component (e.g. makes it keep track of its trainable parameters, you can swap it between 
CPU and GPU with the .to(device) method, where device can be a CPU device torch.device("cpu") or CUDA device torch.device("cuda:0")'''
class TemporalModel(nn.Module):
    """
    Reference 3D pose estimation model with temporal convolutions.
    This implementation can be used for all use-cases.

    Arguments:
        num_joints_in -- number of input joints (e.g. 17 for Human3.6M)
        in_features -- number of input features for each joint (typically 2 for 2D input)
        num_joints_out -- number of output joints (can be different than input)
        filter_widths -- list of convolution widths, which also determines the # of blocks and receptive field
        causal -- use causal convolutions instead of symmetric convolutions (for real-time applications)
        dropout -- dropout probability
        channels -- number of convolution channels
        dense -- use regular dense convolutions instead of dilated convolutions (ablation experiment)
    """
    def __init__(self, num_joints_in, in_features, num_joints_out, filter_widths, causal=False, dropout=0.25, channels=1024, dense=False):
        #super init on nn.Module
        super().__init__()
        
        #Validate input
        for fw in filter_widths:
            assert fw % 2 != 0, 'Only odd filter widths are supported'
        
        self.num_joints_in = num_joints_in
        self.in_features = in_features
        self.num_joints_out = num_joints_out
        self.filter_widths = filter_widths
        
        self.drop = nn.Dropout(dropout)
        self.relu = nn.ReLU(inplace=True)
        
        self.pad = [ filter_widths[0] // 2 ]


        self.expand_bn = nn.BatchNorm1d(channels, momentum=0.1)
        self.shrink = nn.Conv1d(channels, num_joints_out*3, 1)

        #self.conv1 = nn.Conv1d(num_joints_in*in_features, channels, filter_widths[0], bias=False)


        self.expand_conv = nn.Conv1d(num_joints_in*in_features, channels, filter_widths[0], bias=False)
        
        layers_conv = []
        layers_bn = []
        
        self.causal_shift = [ (filter_widths[0]) // 2 if causal else 0 ]
        next_dilation = filter_widths[0]

        self.features = nn.ModuleDict({})

        #self.features.add_module('conv0', nn.Conv1d(channels, channels, filter_widths[i] if not dense else (2*self.pad[-1] + 1), dilation=next_dilation if not dense else 1, bias=False))

        #iterate from 1 thru 4
        for i in range(1, len(filter_widths)):
            self.pad.append((filter_widths[i] - 1) * next_dilation // 2)
            self.causal_shift.append((filter_widths[i]//2 * next_dilation) if causal else 0)


            '''
            self.features.add_module('conv'+str(i), nn.Conv1d(channels, channels, filter_widths[i] if not dense else (2*self.pad[-1] + 1), dilation=next_dilation if not dense else 1, bias=False))
            self.features.add_module('bn'+str(i), nn.BatchNorm1d(channels, momentum=0.1))



            self.features.add_module('conv'+str(i)+'2', nn.Conv1d(channels, channels, 1, dilation=1, bias=False))
            self.features.add_module('bn'+str(i)+'2', nn.BatchNorm1d(channels, momentum=0.1))'''


            
            
            layers_conv.append(nn.Conv1d(channels, channels, filter_widths[i] if not dense else (2*self.pad[-1] + 1), dilation=next_dilation if not dense else 1, bias=False))
            layers_bn.append(nn.BatchNorm1d(channels, momentum=0.1))

            layers_conv.append(nn.Conv1d(channels, channels, 1, dilation=1, bias=False))
            layers_bn.append(nn.BatchNorm1d(channels, momentum=0.1))

            '''
            setattr(self, "conv" + str(i), nn.Conv1d(channels, channels, filter_widths[i] if not dense else (2*self.pad[-1] + 1), dilation=next_dilation if not dense else 1, bias=False))
            setattr(self, "bn" + str(i), nn.BatchNorm1d(channels, momentum=0.1))

            setattr(self, "conv" + str(i) + "2", nn.Conv1d(channels, channels, 1, dilation=1, bias=False))
            setattr(self, "bn" + str(i) + "2", nn.BatchNorm1d(channels, momentum=0.1))'''
            

            '''
            layers_conv["conv" + str(i)] = nn.Conv1d(channels, channels, filter_widths[i] if not dense else (2*self.pad[-1] + 1), dilation=next_dilation if not dense else 1, bias=False)
            layers_bn ["bn" + str(i)] = nn.BatchNorm1d(channels, momentum=0.1)


            layers_conv["conv" + str(i) + "2"] = nn.Conv1d(channels, channels, 1, dilation=1, bias=False)
            layers_bn["conv" + str(i) + "2"] = nn.BatchNorm1d(channels, momentum=0.1)'''
            

            next_dilation *= filter_widths[i]


        
        self.conv_layer_names = list(layers_conv)
        self.bn_layer_names = list(layers_bn)

        
        self.layers_conv = nn.ModuleList(layers_conv)    #ALT IS MODULEDICT
        #print(layers_conv)
        self.layers_bn = nn.ModuleList(layers_bn)
        

    def set_bn_momentum(self, momentum):
        self.expand_bn.momentum = momentum
        for bn in self.layers_bn:
            bn.momentum = momentum
            
    def receptive_field(self):
        """
        Return the total receptive field of this model as # of frames.
        """
        frames = 0
        for f in self.pad:
            frames += f
        return 1 + 2*frames
    
    def total_causal_shift(self):
        """
        Return the asymmetric offset for sequence padding.
        The returned value is typically 0 if causal convolutions are disabled,
        otherwise it is half the receptive field.
        """
        frames = self.causal_shift[0]
        next_dilation = self.filter_widths[0]
        for i in range(1, len(self.filter_widths)):
            frames += self.causal_shift[i] * next_dilation
            next_dilation *= self.filter_widths[i]
        return frames


    #run forward feed on network
    def _forward_blocks(self, x):
        x = self.drop(self.relu(self.expand_bn(self.expand_conv(x))))


        for i in range(len(self.pad) - 1):

            pad = self.pad[i+1]
            shift = self.causal_shift[i+1]

            res = x[:, :, pad + shift : x.shape[2] - pad + shift]


            '''
            for j, mod in enumerate(self.layers_conv):
                if (j == 2*i):
                    desired_conv_mod_1.append(list(mod))
                elif (j == 2*i + 1):
                    desired_conv_mod_2.append(list(mod))
            for j, mod in enumerate(self.layers_bn):
                if (j == 2*i):
                    desired_bn_mod_1.append(list(mod))
                elif (j == 2*i + 1):
                    desired_bn_mod_2.append(list(mod))'''

                
            for j, mod_conv in enumerate(self.layers_conv):
                for k, mod_bn in enumerate(self.layers_bn):
                    if j==2*i and k==2*i:
                        x = self.drop(self.relu(mod_bn(mod_conv(x))))
                    elif j==2*i +1 and k==2*i + 1:
                        x = res + self.drop(self.relu(mod_bn(mod_conv(x))))



            #x = self.drop(self.relu(self.layers_bn[2*i](self.layers_conv[2*i](x))))
            #x = res + self.drop(self.relu(self.layers_bn[2*i + 1](self.layers_conv[2*i + 1](x))))


            
            #x = self.drop(self.relu(self.__getattr__("bn" + str(i+1) ) (self.__getattr__("conv" + str(i+1) )(x))))
            #x = res + self.drop(self.relu(self.__getattr__("bn" + str(i+1) + "2" ) (self.__getattr__("conv" + str(i+1) + "2")(x))))
        
        x = self.shrink(x)
        return x

        
    def forward(self, x):
        assert len(x.shape) == 4
        assert x.shape[-2] == self.num_joints_in
        assert x.shape[-1] == self.in_features
        
        x.contiguous(memory_format=torch.channels_last)

        sz = x.shape[:3]
        x = x.view(x.shape[0], x.shape[1], -1)
        x = x.permute(0, 2, 1)
        
        x = self._forward_blocks(x)
        
        x = x.permute(0, 2, 1)
        x = x.view(sz[0], -1, self.num_joints_out, 3)

        
        return x


def get_model():
    m = TemporalModel(17, 2, 17, [3, 3, 3, 3, 3], causal=False, dropout=0.25)
    m.eval()
    return m

#fuse some layers to save mem
def get_layers_to_fuse(model):
    '''
    layers = model.conv_layer_names + model.bn_layer_names #model.bn_layer_names
    print(layers)'''

    layers=[]
    #generate keys
    for i in range(4):
        layers.append("conv" + str(i+1))
        layers.append("bn" + str(i+1))
        layers.append('relu')

        layers.append("conv" + str(i+1) + "2")

        layers.append("bn" + str(i+1) + "2")

        layers.append('relu')

    #print([layers])
    return [layers]



#format model for use on mobile
def trace_model():
    m = get_model()
    layers = get_layers_to_fuse(m)

    return m, layers
    


class TemporalModelOptimized1f(TemporalModel):
    """
    3D pose estimation model optimized for single-frame batching, i.e.
    where batches have input length = receptive field, and output length = 1.
    This scenario is only used for training when stride == 1.
    
    This implementation replaces dilated convolutions with strided convolutions
    to avoid generating unused intermediate results. The weights are interchangeable
    with the reference implementation.
    """
    
    def __init__(self, num_joints_in, in_features, num_joints_out,
                 filter_widths, causal=False, dropout=0.25, channels=1024):
        """
        Initialize this model.
        
        Arguments:
        num_joints_in -- number of input joints (e.g. 17 for Human3.6M)
        in_features -- number of input features for each joint (typically 2 for 2D input)
        num_joints_out -- number of output joints (can be different than input)
        filter_widths -- list of convolution widths, which also determines the # of blocks and receptive field
        causal -- use causal convolutions instead of symmetric convolutions (for real-time applications)
        dropout -- dropout probability
        channels -- number of convolution channels
        """
        super().__init__(num_joints_in, in_features, num_joints_out, filter_widths, causal, dropout, channels)
        
        self.expand_conv = nn.Conv1d(num_joints_in*in_features, channels, filter_widths[0], stride=filter_widths[0], bias=False)
        
        layers_conv = []
        layers_bn = []
        
        self.causal_shift = [ (filter_widths[0] // 2) if causal else 0 ]
        next_dilation = filter_widths[0]
        for i in range(1, len(filter_widths)):
            self.pad.append((filter_widths[i] - 1)*next_dilation // 2)
            self.causal_shift.append((filter_widths[i]//2) if causal else 0)
            
            layers_conv.append(nn.Conv1d(channels, channels, filter_widths[i], stride=filter_widths[i], bias=False))
            layers_bn.append(nn.BatchNorm1d(channels, momentum=0.1))
            layers_conv.append(nn.Conv1d(channels, channels, 1, dilation=1, bias=False))
            layers_bn.append(nn.BatchNorm1d(channels, momentum=0.1))
            next_dilation *= filter_widths[i]
            
        self.layers_conv = nn.ModuleList(layers_conv)
        self.layers_bn = nn.ModuleList(layers_bn)
        
    def _forward_blocks(self, x):
        x = self.drop(self.relu(self.expand_bn(self.expand_conv(x))))
        
        for i in range(len(self.pad) - 1):
            res = x[:, :, self.causal_shift[i+1] + self.filter_widths[i+1]//2 :: self.filter_widths[i+1]]
            
            x = self.drop(self.relu(self.layers_bn[2*i](self.layers_conv[2*i](x))))
            x = res + self.drop(self.relu(self.layers_bn[2*i + 1](self.layers_conv[2*i + 1](x))))
        
        x = self.shrink(x)
        return x