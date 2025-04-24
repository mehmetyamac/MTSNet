import torch
from torch import nn
from torch.nn import functional as F
from basicsr.models.archs.MTSNet_arch import *
from typing import List, Union, Optional


# Author: Mehmet Yamac
# Email: mehmet.yamac@tuni.fi
# Affiliation: Tampere University
# Copyright Permission: Granted for use in academic and research settings only.
# Any commercial use or distribution is prohibited without express written consent of the author.

class TinyTensorNet(nn.Module):
    def __init__(self, L, N, window_scales=[8,16,32,64],T=3,residual=True):
        super(TinyTensorNet, self).__init__()
        
        # Define maximum patch size
        self.p = window_scales[-1]
        self.residual = residual
        print('residual:')
        print(self.residual)
        # First layer
        self.ten1=MultiScale(in_channels=3, 
                             out_channels=N,
                             T=T,
                             window_scales=window_scales,
                             out_window_scales=window_scales)
        
        self.relu1 = nn.ReLU()
        
        # Middle layers
        self.layers = nn.ModuleList()
        for i in range(L-2):
            self.layers.append(MultiScale(
            in_channels=N,
            out_channels=N,
            T=T,
            window_scales=window_scales,
            out_window_scales=window_scales))
            self.layers.append(nn.ReLU())
        
        # Last layer
        #self.ten_last = nn.Conv2d(N, 3, 3, padding=1)
        
        # Last layer
        self.ten_last = MultiScale(
            in_channels=N,
            out_channels=3,
            T=T,
            window_scales=window_scales,
            out_window_scales=window_scales)
        
    def forward(self, x):
        # Calculate the padding amounts along height and width dimensions
        sx, sy = x.shape[2:]
        px = (self.p - (sx % self.p)) % self.p
        py = (self.p - (sy % self.p)) % self.p
        # Apply padding using PyTorch's nn.functional.pad() function
        x = torch.nn.functional.pad(x, (0, py, 0, px), mode='reflect')
        # First layer
        out = self.ten1(x)
        out = self.relu1(out)
        
        # Middle layers
        for layer in self.layers:
            out = layer(out)
        
        # Last layer
        out = self.ten_last(out)
        if self.residual:
            out=out+x
        out = out[:,:,:sx,:sy]
        return out

    
    def calculate_total_mac(self, N, H, W):
        total_macs = 0

        total_macs += self.ten1.calculate_total_mac(H,W)

        # Calculate MACs for each MTSL layer
        for mts in self.layers:
            if isinstance(mts, nn.ReLU):
                continue
            #print(mts)
            total_macs += mts.calculate_total_mac(H, W)
            
        total_macs += calculate_conv_Mac(N, 3, H, W, 3, 1)#self.ten_last.calculate_total_mac(H,W)
        
        # Calculate MACs for the conv2d layer
        #conv2d_macs = calculate_conv_Mac(self.out_channels, self.out_channels, H, W, self.kernel_size, 1)

        # Add conv2d MACs to the total
        #total_macs += conv2d_macs

        return total_macs
    

if __name__ == '__main__':
   
    # Parameters for the MTSBlock
    N = 52
    L = 5

    mts_net = TinyTensorNet(N=N, L=L)

    H = W = 256  # Input dimensions

    # Create a random tensor with the specified dimensions
    input_tensor = torch.rand(1, 3, 256, 256)

    output_tensor= mts_net(input_tensor)
    print(output_tensor.shape)


    # Calculate GMACs
    total_flops = mts_net.calculate_total_mac(H, W)
    print(f"Total GMAC for MTSNet with H=W=256: {total_flops / 1e9}")

    # Calculate # of parameters
    print(f"Number of trainable parameters: {count_parameters(mts_net)}")
