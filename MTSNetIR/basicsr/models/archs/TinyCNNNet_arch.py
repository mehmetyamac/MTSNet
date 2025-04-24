
import torch
from torch import nn
from torch.nn import functional as F
import math
from typing import List, Union, Optional
from basicsr.models.archs.MTSNet_arch import *


# Author: Mehmet Yamac
# Email: mehmet.yamac@tuni.fi
# Affiliation: Tampere University
# Copyright Permission: Granted for use in academic and research settings only.
# Any commercial use or distribution is prohibited without express written consent of the author.

class TinyCNNNet(nn.Module):
    def __init__(self, L, N, residual=False):
        super(TinyCNNNet, self).__init__()
        
        self.residual = residual
        #print('residual:')
        #print(self.residual)
        
        # First layer
        self.conv1 = nn.Conv2d(3, N, 3, padding=1)
        self.relu1 = nn.ReLU()
        
        # Middle layers
        self.layers = nn.ModuleList()
        for i in range(L-2):
            self.layers.append(nn.Conv2d(N, N, 3, padding=1))
            self.layers.append(nn.ReLU())
        
        # Last layer
        self.conv_last = nn.Conv2d(N, 3, 3, padding=1)
        
    def forward(self, x):
        # First layer
        out = self.conv1(x)
        out = self.relu1(out)
        
        # Middle layers
        for layer in self.layers:
            out = layer(out)
        
        # Last layer
        out = self.conv_last(out)
        if self.residual:
            out = out + x
        
        return out

    def calculate_gmac(self, H, W):
        """
        Calculate the total GMACs for the TinyCNNNet.

        Parameters:
        - H (int): Height of the input feature map.
        - W (int): Width of the input feature map.

        Returns:
        - float: Total GMACs for the network.
        """
        total_macs = 0
        
        # First layer
        C_in, C_out, K = 3, self.conv1.out_channels, 3
        H_out = H
        W_out = W
        total_macs += calculate_conv_Mac(C_in, C_out, H_out, W_out, K)
        
        # Middle layers
        for layer in self.layers:
            if isinstance(layer, nn.Conv2d):
                C_in = layer.in_channels
                C_out = layer.out_channels
                K = layer.kernel_size[0]
                total_macs += calculate_conv_Mac(C_in, C_out, H_out, W_out, K)
        
        # Last layer
        C_in, C_out, K = self.conv_last.in_channels, self.conv_last.out_channels, 3
        total_macs += calculate_conv_Mac(C_in, C_out, H_out, W_out, K)
        
        # Convert to GMACs
        gmacs = total_macs / 1e9
        return gmacs

    

if __name__ == '__main__':
   
    # Example usage
    H = 256    # Height of the input feature map
    W = 256    # Width of the input feature map
    L = 5      # Number of layers
    N = 96     # Number of channels in the middle layers
    residual = True

    net = TinyCNNNet(L, N, residual)
    gmacs = net.calculate_gmac(H, W)
    print(f"Total GMACs: {gmacs} GMAC")

    # Calculate # of parameters
    print(f"Number of trainable parameters: {count_parameters(net)}")

    

    #import torch
    #from torch.profiler import profile, record_function, ProfilerActivity

    #with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA], profile_memory=True, record_shapes=True) as prof:
    #    with record_function("model_inference"):
    #        mts_net(input_tensor)

    #print(prof.key_averages().table(sort_by="cpu_memory_usage", row_limit=10))