import numpy as np
#import matplotlib.pyplot as plt
import torch
from torch import nn
from torch.nn import functional as F
import math
from typing import List, Union, Optional
from torch.autograd import Function
import torch.jit
import torch.jit as jit
import torchvision.ops as ops

#This is the implementation of Multi-scale Tensor Summation (MTS) factorization operation

# Author: Mehmet Yamac
# Email: mehmet.yamac@tuni.fi
# Affiliation: Tampere University
# Copyright Permission: Granted for use in academic and research settings only.
# Any commercial use or distribution is prohibited without express written consent of the author.

def calculate_mac(H, W, w, h, C_in, C_out, T, w_out, h_out):
    """
    Calculate MAC (Multiply-Accumulate operations) of a MTS (Multiscale Tensorial Summation Factorization)
    operation based on given dimensions, channel counts, and a summation factor T.

    Parameters:
    - H: Height of the input feature map.
    - W: Width of the input feature map.
    - w: Width parameter for operation.
    - h: Height parameter for operation.
    - C_in: Number of input channels.
    - C_out: Number of output channels.
    - T: A scaling factor that multiplies the total MACs.
    - w_out: Width of the output windows.
    - h_out: Height of the output windows.

    Returns:
    - MACs: The calculated Multiply-Accumulate operations, scaled by T.

    Example usage:
    # H = 256    # Height of the input feature map
    # W = 256    # Width of the input feature map
    # w = 64     # Window Width
    # h = 64     # Window Height
    # C_in = 48  # Number of input channels
    # C_out = 48 # Number of output channels
    # T = 3      # Summation factor
    # w_out = 32 # Width of the output feature map
    # h_out = 32 # Height of the output feature map

    # macs = calculate_mac(H, W, w, h, C_in, C_out, T, w_out, h_out)/1e9
    # print(f"MACs: {macs} GMAC")
    """
    # Calculate the terms based on the provided formulas
    term1 = ((H / h) * (W / w) * (w * w_out) * h) * C_in
    term2 = ((H / h) * (W / w) * (h * h_out) * w_out) * C_in
    term3 = int(H * (h_out / h)) * int(W * (w_out / w)) * C_in * C_out

    # Sum the terms for the total MACs
    total_macs = (term1 + term2 + term3) * T

    return total_macs

def calculate_conv_Mac(C_in, C_out, H, W, K, groups=1):
    """
    Calculate the MAC operations for a convolutional layer.

    Parameters:
    - C_in (int): Number of input channels.
    - C_out (int): Number of output channels.
    - H (int): Height of the output feature map (H_out), which may differ from the input height (H_in).
    - W (int): Width of the output feature map (W_out), which may differ from the input width (W_in).
    - K (int): Kernel size (assuming a square kernel for simplicity).
    - groups (int): Number of groups for grouped convolutions (1 for standard convolutions).

    Returns:
    - int: The number of MAC operations for the convolutional layer.

    Note:
    The output dimensions (H_out and W_out) can be calculated from the input dimensions (H_in and W_in),
    stride, padding, and dilation using the following formulas:
    H_out = floor((H_in + 2*padding - dilation*(K-1) - 1) / stride + 1)
    W_out = floor((W_in + 2*padding - dilation*(K-1) - 1) / stride + 1)
    """
    # Author: Mehmet Yamac
    # Email: mehmet.yamac@tuni.fi
    # Affiliation: Tampere University, 2024
    # Copyright Permission: Granted for use in academic and research settings only.
    # Any commercial use or distribution is prohibited without express written consent of the author.

    return C_in * C_out * H * W * (K ** 2) / groups

def window_partition(x, window_size):
    """
    Args:
        x: (B, C, H, W)
        window_size (int): window size
    Returns:
        windows: (num_windows*B, C, window_size, window_size)
    """
    B, C, H, W = x.shape
    x = x.view(B, C, H // window_size, window_size, W // window_size, window_size)
    windows = x.permute(0, 2, 4, 1, 3, 5).contiguous().view(-1, C, window_size, window_size)
    return windows


def window_reverse(windows, window_size, H, W):
    """
    Args:
        windows: (num_windows*B, C, window_size, window_size)
        window_size (int): Window size
        H (int): Height of image
        W (int): Width of image
    Returns:
        x: (B, C, H, W)
    """
    B, C = windows.shape[0] // (H // window_size * W // window_size), windows.shape[1]
    x = windows.view(B, H // window_size, W // window_size, C, window_size, window_size)
    x = x.permute(0, 3, 1, 4, 2, 5).contiguous().view(B, C, H, W)
    return x

#Custom Backpropogation, if memory is limited
# Explicit calculation of each tensor multiplication and its backwards

class TensorMatrixMultiplyFunction(Function):
    @staticmethod
    def forward(ctx, x, *matrices):
        ctx.save_for_backward(x, *matrices)

        N = len(matrices)
        outputs = x
        for i in range(N):
            # Calculate the permutation indices to bring the i-th dimension (excluding batch) to the last for matmul
            if i >0:
                permutation = torch.roll(torch.arange(N), 1) + 1
                outputs = outputs.permute(0, *permutation)

            # Perform matrix multiplication with the i-th matrix
            outputs = torch.matmul(outputs, matrices[i])

        # Permute back to the original dimensions
        inv_permute = torch.roll(torch.arange(N), -i) + 1
        outputs = outputs.permute(0, *inv_permute)

        return outputs

    @staticmethod
    def backward(ctx, grad_output):
        x, *matrices = ctx.saved_tensors
        N = len(matrices)
        grad_x = grad_output.clone()
        grad_matrices = [None] * N  # Initialize gradients for matrices as None
        outputs = [None] * N  # Initialize outputs for matrices as None

        outputs[0] = x

        # Forward
        for i in range(N - 1):
            if i > 0:
                permutation = torch.roll(torch.arange(N), 1) + 1
                outputs[i + 1] = outputs[i].permute(0, *permutation)
            else:
                outputs[i + 1] = outputs[i]

            outputs[i + 1] = torch.matmul(outputs[i + 1].float(), matrices[i].float())

        # Backward
        for i in range(N - 1, -1, -1):
            inv_permute = torch.roll(torch.arange(N), -1) + 1
            grad_x = grad_x.permute(0, *inv_permute)

            permute = torch.roll(torch.arange(N), +1) + 1
            if i != 0:
                outputs[i] = outputs[i].permute(0, *permute)

            # Ensure that the operation is conducted in the expected precision
            #if grad_x.dtype != matrices[i].dtype:
            #    grad_x = grad_x.to(matrices[i].dtype)
            #    outputs[i] = outputs[i].to(matrices[i].dtype)


            b, x1, x2, x3 = outputs[i].shape
            x_flattened = outputs[i].contiguous().view(b * x1 * x2, x3)

            b, x1, x2, x3 = grad_x.shape
            grad_x_flattened = grad_x.contiguous().view(b * x1 * x2, x3)

            


            grad_matrices[i] = torch.matmul(x_flattened.t().float(), grad_x_flattened.float())

            grad_x = torch.matmul(grad_x.float(), matrices[i].t().float())

        return (grad_x,) + tuple(grad_matrices)


class TensorMatrixMultiply(nn.Module):
    def __init__(self, in_shape: List[int], out_shape: Union[int, List[int]], init_temperature: float,
                 tensors: Optional[List[torch.Tensor]] = None, use_identity_init: bool = False):
        super(TensorMatrixMultiply, self).__init__()

        self.in_shape = list(in_shape)
        self.N = len(in_shape)
        self.out_shape = out_shape if isinstance(out_shape, list) else [out_shape] * self.N
        self.init_temperature = init_temperature
        self.use_identity_init = use_identity_init

        if tensors is None:
            self.init_tensors()
        else:
            self.T = nn.ParameterList([nn.Parameter(tensor) for tensor in tensors[::-1]])

    def init_tensors(self):
        scale_factor = self.init_temperature ** (1 / self.N)
        if self.use_identity_init:
            self.T = nn.ParameterList()
            for i in range(self.N):
                if self.in_shape[i] != self.out_shape[i]:
                    raise ValueError(f"Cannot initialize with identity matrix due to dimension mismatch at index {i}: in_shape {self.in_shape[i]} vs out_shape {self.out_shape[i]}")
                #identity_matrix = torch.eye(self.in_shape[i])
                identity_matrix = torch.eye(self.in_shape[i]) / scale_factor
                #print(self.init_temperature)
                self.T.append(nn.Parameter(identity_matrix))
            self.T = self.T[::-1]
        else:
            self.T = nn.ParameterList([
                nn.Parameter(torch.randn(self.in_shape[i], self.out_shape[i]) / (scale_factor * torch.sqrt(
                    torch.tensor(self.out_shape[i], dtype=torch.float))))
                for i in range(self.N)][::-1])


    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Call the custom autograd function
        return TensorMatrixMultiplyFunction.apply(x, *self.T)


class TensorMultiplication_einsum(jit.ScriptModule):
    def __init__(self, in_shape: List[int], out_shape: List[int], init_temperature: float, use_identity_init: bool = False, 
                tensors: Optional[List[torch.Tensor]] = None):
        super(TensorMultiplication_einsum, self).__init__()
        self.in_shape = in_shape
        self.out_shape = out_shape
        self.N = len(in_shape)
        self.init_temperature = init_temperature
        self.use_identity_init = use_identity_init

        if tensors is None:
            self.init_tensors()
        else:
            self.T = nn.ParameterList([nn.Parameter(tensor) for tensor in tensors[::-1]])

    def init_tensors(self):
        scale_factor = self.init_temperature ** (1 / self.N)
        if self.use_identity_init:
            self.T = nn.ParameterList()
            for i in range(self.N):
                if self.in_shape[i] != self.out_shape[i]:
                    raise ValueError(f"Cannot initialize with identity matrix due to dimension mismatch at index {i}: in_shape {self.in_shape[i]} vs out_shape {self.out_shape[i]}")
                #identity_matrix = torch.eye(self.in_shape[i])
                identity_matrix = torch.eye(self.in_shape[i]) / scale_factor
                #print(self.init_temperature)
                self.T.append(nn.Parameter(identity_matrix))
            self.T = self.T[::-1]
        else:
            self.T = nn.ParameterList([
                nn.Parameter(torch.randn(self.in_shape[i], self.out_shape[i]) / (scale_factor * torch.sqrt(
                    torch.tensor(self.out_shape[i], dtype=torch.float))))
                for i in range(self.N)][::-1])
        
    @jit.script_method
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = torch.einsum('bchw,wW,hH,cC->bCHW', x, self.T[0], self.T[1], self.T[2])
        return out


class TensorSum(nn.Module):
    def __init__(self, in_shape: List[int], out_shape: Union[int, List[int]], init_temperature: float, T: int,
                 auto_differentiate: bool = True, use_identity_init: bool = False, use_einsum: bool = True):
        super().__init__()
        self.in_shape = list(in_shape)
        self.out_shape = out_shape if not isinstance(out_shape, int) else [out_shape] * len(self.in_shape)
        self.T = T

        if auto_differentiate:
            if use_einsum:
                self.TMMs = nn.ModuleList(
                    [TensorMultiplication_einsum(in_shape, out_shape, init_temperature=init_temperature,
                                      use_identity_init=use_identity_init) for _ in range(self.T)])
        else:
            self.TMMs = nn.ModuleList(
                [TensorMatrixMultiply(in_shape, out_shape, init_temperature=init_temperature,
                                      use_identity_init=use_identity_init) for _ in range(self.T)])

    def forward(self, x: torch.Tensor):
        futures = [jit.fork(self.TMMs[i].forward, x) for i in range(self.T)]
        outputs = [jit.wait(fut) for fut in futures]

        result = outputs[0].clone()
        for output in outputs[1:]:
            result.add_(output)
        return result


class TensorSumEfficient(nn.Module):
    def __init__(self, in_shape: List[int], out_shape: Union[int, List[int]], init_temperature: float,
                 tensors: Optional[List[torch.Tensor]] = None, T: int = 3, drop_block_p: float = 0. ,block_ratio: int=4, max_drop: float =0.8, 
                 use_identity_init=False):
        super().__init__()
        self.in_shape = list(in_shape)
        self.N = len(in_shape)
        self.out_shape = out_shape if not isinstance(out_shape, int) else [out_shape] * self.N
        self.init_temperature = init_temperature
        self.T = T
        self.matrices = tensors
        self.use_identity_init = use_identity_init
        block_size = int(in_shape[-1] / block_ratio) + 1
        drop_block_p = drop_block_p * self.init_temperature

        # Ensure drop_block_p does not exceed max_drop
        if drop_block_p > max_drop:
            drop_block_p = max_drop
        self.dropblock = ops.DropBlock2d(p=drop_block_p, block_size=block_size) if drop_block_p > 0. else nn.Identity()
        


        if tensors is None:
            self.init_tensors()
        else:
            self.matrices= nn.ParameterList([nn.Parameter(tensor) for tensor in tensors[::-1]])

    def init_tensors(self):
        scale_factor = self.init_temperature ** (1 / self.N)
        if self.use_identity_init:
            self.matrices = nn.ParameterList()
            for i in range(self.N):
                if self.in_shape[i] != self.out_shape[i]:
                    raise ValueError(f"Cannot initialize with identity matrix due to dimension mismatch at index {i}: in_shape {self.in_shape[i]} vs out_shape {self.out_shape[i]}")
                #identity_matrix = torch.eye(self.in_shape[i])
                identity_matrix = torch.eye(self.in_shape[i]) / scale_factor
                #print(self.init_temperature)
                self.matrices.append(nn.Parameter(identity_matrix))
            self.matrices = self.matrices[::-1]
        else:
            self.matrices = nn.ParameterList([
                nn.Parameter(torch.randn(self.T, self.in_shape[i], self.out_shape[i]) / (scale_factor * torch.sqrt(
                    torch.tensor(self.out_shape[i], dtype=torch.float))))
                for i in range(self.N)][::-1])

    def forward(self, x: torch.Tensor):
        B, C, H, W = x.shape
        x = self.dropblock(x)
        x = x.contiguous().view(B, 1, C, H, W)

        for i in range(self.N):
            if i > 0:
                permutation = torch.roll(torch.arange(self.N), 1) + 2
                x = x.permute(0, 1, *permutation)

            B, TT, C, H, W = x.shape
            x = x.contiguous().view(B, TT, C * H, W)
            x = torch.matmul(x, self.matrices[i])
            _, _, W_new = self.matrices[i].shape
            x = x.view(B, self.T, C, H, W_new)

        # Sum over the TT dimension and squeeze to have B, C, H, W
        x = x.sum(dim=1).squeeze(1)
        # Permute back to the original dimensions
        inv_permute = torch.roll(torch.arange(self.N), -i) + 1
        x = x.permute(0, *inv_permute)

        return x



class TensorSumEfficient_ein(nn.Module):
    def __init__(self, in_shape: List[int], out_shape: Union[int, List[int]], init_temperature: float,
                 tensors: Optional[List[torch.Tensor]] = None, T: int = 3, use_identity_init=False):
        super().__init__()
        self.in_shape = list(in_shape)
        self.N = len(in_shape)
        self.out_shape = out_shape if not isinstance(out_shape, int) else [out_shape] * self.N
        self.init_temperature = init_temperature
        self.T = T
        self.matrices = tensors
        self.use_identity_init =use_identity_init


        if tensors is None:
            self.init_tensors()
        else:
            self.matrices= nn.ParameterList([nn.Parameter(tensor) for tensor in tensors[::-1]])

    def init_tensors(self):
        scale_factor = self.init_temperature ** (1 / self.N)
        if self.use_identity_init:
            self.matrices = nn.ParameterList()
            for i in range(self.N):
                if self.in_shape[i] != self.out_shape[i]:
                    raise ValueError(f"Cannot initialize with identity matrix due to dimension mismatch at index {i}: in_shape {self.in_shape[i]} vs out_shape {self.out_shape[i]}")
                #identity_matrix = torch.eye(self.in_shape[i])
                identity_matrix = torch.eye(self.in_shape[i]) / scale_factor
                #print(self.init_temperature)
                self.matrices.append(nn.Parameter(identity_matrix))
            self.matrices = self.matrices[::-1]
        else:
            self.matrices = nn.ParameterList([
                nn.Parameter(torch.randn(self.T, self.in_shape[i], self.out_shape[i]) / (scale_factor * torch.sqrt(
                    torch.tensor(self.out_shape[i], dtype=torch.float))))
                for i in range(self.N)][::-1])

    def forward(self, x: torch.Tensor):
        B, C, H, W = x.shape
        #x = x.contiguous().unsqueeze(1).expand(B, self.T, C, H, W)
        x_expanded = x.unsqueeze(1).expand(B, self.T, C, H, W)

        # Use the provided matrices for transformation

        # Perform the mode multiplications using einsum
        out = torch.einsum('btchw,twW,thH,tcC->btCHW', x_expanded, self.matrices[0], self.matrices[1], self.matrices[2])

        # Sum over the T dimension
        #out = out.sum(dim=1)

        # Perform the mode multiplications using einsum
        #out = torch.einsum('btchw,twW,thH,tcC->btCHW', x, self.matrices[0], self.matrices[1], self.matrices[2])
       
        # Sum over the TT dimension and squeeze to have B, C, H, W
        x = out.sum(dim=1)

        return x

class FeatureProcessingBlock(nn.Module):
    """
    Performs a GTS (Generalized Tensor Summation Factorization) Operation, for a fixed window size
    and summation factor T
    Ex Usage:
    block = FeatureProcessingBlock()
    macs = block.calculate_mac(256,256)
    print(f"MACs: {macs/1e9} GMAC")

    # if tensor_sum_type = 'standard', use TensorSum with either autograd of custom backward
    # if tensor_sum_type = "efficient" and "use_einsum=True" use TensorSumEfficient_ein, 
    # if use_einsum=False, use TensorSumEfficient
    """
    def __init__(self, in_channels=48, out_channels=48, T=3, window_size=8, out_window_size=8,
                 init_temperature=1, input_shape=None, output_shape=None, use_identity_init=False, 
                 auto_differentiate=True, use_einsum=True, tensor_sum_type="standard", drop_block_p = 0.):
        super().__init__()

        self.in_channels = in_channels
        self.T = T
        self.window_size = window_size
        self.out_window_size = out_window_size
        self.out_channels = out_channels
        self.c_r = (out_window_size / window_size)
        self.input_shape = [self.in_channels, self.window_size,
                            self.window_size] if input_shape is None else input_shape
        self.output_shape = [self.out_channels, self.out_window_size,
                             self.out_window_size] if output_shape is None else output_shape

        if tensor_sum_type == "standard":
            # This block uses the standard TensorSum operation, possibly with custom differentiation
            self.multilinear_operation = TensorSum(self.input_shape, self.output_shape, init_temperature, T,
                                                   use_identity_init=use_identity_init, auto_differentiate=auto_differentiate)
        elif tensor_sum_type == "efficient" and use_einsum:
            # This block uses an efficient tensor summation method with einsum optimization
            self.multilinear_operation = TensorSumEfficient_ein(in_shape=self.input_shape, out_shape=self.output_shape, init_temperature=init_temperature, 
                                                                T=T,use_identity_init=use_identity_init)
        elif tensor_sum_type == "efficient" and not use_einsum:
            # This block uses an efficient tensor summation method without einsum optimization
            self.multilinear_operation = TensorSumEfficient(in_shape=self.input_shape, out_shape=self.output_shape, init_temperature=init_temperature, 
                                                                T=T,use_identity_init=use_identity_init, drop_block_p= drop_block_p)
        else:
            raise ValueError(f"Unsupported tensor_sum_type: {tensor_sum_type}")


    def forward(self, x):
        b, c, h, w = x.shape
        # Partition the input tensor into non-overlapping windows of size w
        x = window_partition(x, self.window_size)
        # Multilinear operation
        x = self.multilinear_operation(x)
        # Reverse the partitioning operation
        x = window_reverse(x, self.out_window_size, math.ceil(h * self.c_r), math.ceil(w * self.c_r))
        return x

    def calculate_mac(self, H, W):
        """
        Calculate the MAC (Multiply-Accumulate operations) for this block based on input dimensions H and W.
        Note: This function should be called with the dynamic shape of an actual input tensor for accuracy.
        """
        h = w = self.window_size  # Using the block's window_size for h and w
        h_out = w_out = self.out_window_size
        # MAC calculation does not directly factor in batch size
        total_macs = calculate_mac(H, W, h, w, self.in_channels, self.out_channels, self.T, w_out, h_out)
        return total_macs


class MultiScale(nn.Module):
    """
            Performs a MTS (Multi Scale Tensor Summation Factorization) Operation,
            # Example usage:
            H = 256  # Height of the input feature map
            W = 256  # Width of the input feature map

            # Instantiate MultiScale with required parameters
            # Note: Provide parameters if the class does not have suitable defaults
            multi_scale = MultiScale(in_channels=30, out_channels=30, T=3, window_scales=[16, 32, 64, 128], out_window_scales=[16, 32, 64, 128])

            # Calculate total MACs for the MultiScale operation, assuming H and W are constant across scales for simplicity
            total_macs = multi_scale.calculate_total_mac(H, W)

            # Print the total MACs, converted to GMACs for readability
            print(f"Total MACs: {total_macs/1e9} GMAC")

        """

    def __init__(self, in_channels=30, out_channels=30, T=3, window_scales=[16, 32, 64, 128],
                 out_window_scales=None, input_shape=None, output_shape=None, use_identity_init: bool = False, 
                 auto_differentiate: bool = True, use_einsum: bool = False, tensor_sum_type="efficient", drop_block_p=0.):
        super(MultiScale, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.T = T
        self.num_scales = len(window_scales)

        # If out_window_scales is None, use window_scales for out_window_scales
        if out_window_scales is None:
            out_window_scales = window_scales

        # Set initial temperature based on whether identity initialization is used
        if use_identity_init:
            init_temperature = torch.tensor(self.T * self.num_scales, dtype=torch.float)
        else:
            init_temperature = torch.sqrt(torch.tensor(self.T * self.num_scales, dtype=torch.float))

        self.MOs = nn.ModuleList([
            FeatureProcessingBlock(in_channels=self.in_channels, out_channels=self.out_channels, T=self.T,
                                   window_size=window_scales[k], out_window_size=out_window_scales[k],
                                   input_shape=input_shape, output_shape=output_shape,
                                   init_temperature=init_temperature,
                                   use_identity_init=use_identity_init, tensor_sum_type= tensor_sum_type, use_einsum=use_einsum, 
                                   auto_differentiate= auto_differentiate, drop_block_p=drop_block_p)
            for k in range(self.num_scales)
        ])

    def forward(self, x: torch.Tensor):
        futures = [jit.fork(self.MOs[i].forward, x) for i in range(self.num_scales)]
        outputs = [jit.wait(fut) for fut in futures]
        result = outputs[0].clone()
        for output in outputs[1:]:
            result.add_(output)
        return result

    #def forward(self, x: torch.Tensor):
    #    y = self.MOs[0](x)
    #    for i in range(1, self.num_scales):
    #        y = y+self.MOs[i](x)
    #    return y

    def calculate_total_mac(self, H, W):
        total_macs = 0
        for block in self.MOs:
            # Assuming H and W are constant for simplicity. If they differ per block, adjust accordingly.
            total_macs += block.calculate_mac(H, W)
        return total_macs

class FeedForward(nn.Module):
    def __init__(self, dim, ffn_expansion_factor=1, group_ratio=1, bias=False, FFactivation='Relu'):
        super(FeedForward, self).__init__()

        hidden_features = int(dim * ffn_expansion_factor)
        self.dim=dim
        self.ffn_expansion_factor=ffn_expansion_factor
        self.group_ratio = group_ratio

        # Initialize the activation function based on the FFactivation argument
        if FFactivation == 'Relu':
            self.activation = nn.ReLU()
        elif FFactivation == 'Gelu':
            self.activation = nn.GELU()
        elif FFactivation is None:
            self.activation = None
        else:
            raise ValueError("Invalid FFactivation. Choose 'Relu', 'Gelu', or None.")

        self.project_in = nn.Conv2d(dim, hidden_features * 2, kernel_size=1, bias=bias)

        self.dwconv = nn.Conv2d(hidden_features * 2, hidden_features * 2, kernel_size=3, stride=1,
                                padding=1, groups=(hidden_features * 2) // group_ratio, bias=bias)

        self.project_out = nn.Conv2d(hidden_features, dim, kernel_size=1, bias=bias)

    def forward(self, x):
        x = self.project_in(x)
        x1, x2 = self.dwconv(x).chunk(2, dim=1)

        # Apply activation if it's not None
        if self.activation is not None:
            x1 = self.activation(x1)

        x = x1 * x2    # Simple gated
        x = self.project_out(x)
        return x

    def calculate_mac(self, H, W):
        hidden_features = int(self.dim * self.ffn_expansion_factor)
        macs = 0
        macs += calculate_conv_Mac(self.dim, hidden_features * 2, H, W, 1)  # project_in
        macs += calculate_conv_Mac(hidden_features * 2, hidden_features * 2, H, W, 3, hidden_features * 2 // self.group_ratio)  # dwconv
        macs += calculate_conv_Mac(hidden_features, self.dim, H, W, 1)  # project_out
        return macs


class MHG(nn.Module):
    def __init__(self, in_channels=64, groups=1, num_heads=1, dwconv_usage=False):
        super(MHG, self).__init__()

        # Initialize parameters
        self.in_channels = in_channels
        self.groups = groups
        self.num_heads = num_heads
        self.out_channels = in_channels * 2 * num_heads  # Total output channels
        self.dwconv_usage = dwconv_usage

        # Create the 2D convolutional layer with kernel size 1
        self.conv = nn.Conv2d(in_channels=self.in_channels,
                              out_channels=self.out_channels,
                              kernel_size=1,
                              groups=self.groups)

        if self.dwconv_usage==True:
            self.preprocessing = self.dwconv = nn.Conv2d(self.out_channels, self.out_channels, kernel_size=3, stride=1,
                                padding=1, groups=self.out_channels, bias=False)
        else:
            self.preprocessing = nn.Identity()

    def forward(self, x):
        # Apply the convolution
        output = self.conv(x) # This is pointwise-and-groupwise convolution
        #Apply self preprocessing if it is preferred
        output = self.preprocessing(output) 

        # Split the output into num_heads * 2 tensors, each with in_channels channels
        outputs_split = []
        for i in range(self.num_heads * 2):
            start_channel = i * self.in_channels
            end_channel = start_channel + self.in_channels
            outputs_split.append(output[:, start_channel:end_channel, :, :])

        # Perform element-wise multiplication for each pair and sum the results
        multiplication_results = []
        for i in range(0, len(outputs_split), 2):
            multiplied = outputs_split[i] * outputs_split[i + 1]
            multiplication_results.append(multiplied)

        # Sum the results of the multiplications
        final_result = torch.sum(torch.stack(multiplication_results), dim=0)
        return final_result

    def calculate_mac(self, H, W):
        if self.dwconv_usage==True:
            GMAC = calculate_conv_Mac(self.out_channels, self.out_channels, H, W, 3, self.out_channels)
            GMAC = GMAC + calculate_conv_Mac(self.in_channels, self.in_channels * 2 * self.num_heads, H, W, 1, self.groups)
        else:
            GMAC = calculate_conv_Mac(self.in_channels, self.in_channels * 2 * self.num_heads, H, W, 1, self.groups)

        return GMAC

class LayerNormFunction(torch.autograd.Function):

    @staticmethod
    def forward(ctx, x, weight, bias, eps):
        ctx.eps = eps
        N, C, H, W = x.size()
        mu = x.mean(1, keepdim=True)
        var = (x - mu).pow(2).mean(1, keepdim=True)
        y = (x - mu) / (var + eps).sqrt()
        ctx.save_for_backward(y, var, weight)
        y = weight.view(1, C, 1, 1) * y + bias.view(1, C, 1, 1)
        return y

    @staticmethod
    def backward(ctx, grad_output):
        eps = ctx.eps

        N, C, H, W = grad_output.size()
        y, var, weight = ctx.saved_variables
        g = grad_output * weight.view(1, C, 1, 1)
        mean_g = g.mean(dim=1, keepdim=True)

        mean_gy = (g * y).mean(dim=1, keepdim=True)
        gx = 1. / torch.sqrt(var + eps) * (g - y * mean_gy - mean_g)
        return gx, (grad_output * y).sum(dim=3).sum(dim=2).sum(dim=0), grad_output.sum(dim=3).sum(dim=2).sum(
            dim=0), None


class LayerNorm2d(nn.Module):

    def __init__(self, channels, eps=1e-6):
        super(LayerNorm2d, self).__init__()
        self.register_parameter('weight', nn.Parameter(torch.ones(channels)))
        self.register_parameter('bias', nn.Parameter(torch.zeros(channels)))
        self.eps = eps

    def forward(self, x):
        return LayerNormFunction.apply(x, self.weight, self.bias, self.eps)

class MTSL(nn.Module):
    """
    # Example usage:

    # Initialize MTSL with example parameters
    mtsl = MTSL(in_channels=48,
                out_channels=48,
                groups=3,
                num_heads=5,
                T=3,
                window_scales=[8, 16, 32, 64],
                ffn_expansion_factor=2,
                group_ratio=1,
                bias=False,
                FFactivation='Relu',
                post_mts_function='MHG',
                alpha=0.1,
                beta=0.1)

    # Example input dimensions H and W (height and width of the input feature map)
    H, W = 128, 128

    # Calculate and print the total FLOPs
    total_flops = mtsl.calculate_total_mac(H, W)
    print(f"Total FLOPs: {total_flops/1e9}")
    """
    def __init__(self, in_channels, out_channels, groups, num_heads, T, window_scales, ffn_expansion_factor,
                 group_ratio, bias, FFactivation, post_mts_function='MHG', alpha=0.1, beta=0.1, dwconv_usage=False, drop_block_p=0.):
        super(MTSL, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.ffn_expansion_factor = ffn_expansion_factor
        self.group_ratio = group_ratio  # Fixed to use the provided parameter instead of hardcoding to 1
        self.num_heads = num_heads
        self.groups = groups

        # Initialize components
        self.layer_norm1 = LayerNorm2d(in_channels)  # Assuming LayerNorm2d is defined and appropriate
        self.multi_scale_transform = MultiScale(in_channels, out_channels, T, window_scales, drop_block_p=drop_block_p)
        self.layer_norm2 = LayerNorm2d(out_channels)  # Assuming LayerNorm2d is defined and appropriate
        self.feed_forward_network = FeedForward(out_channels, ffn_expansion_factor, group_ratio, bias, FFactivation)

        # Initialize post MTS function
        self.post_mts_function = self._init_post_mts_function(post_mts_function, in_channels, groups, num_heads, dwconv_usage=dwconv_usage)

        self.alpha = nn.Parameter(torch.zeros((1, in_channels, 1, 1)) + alpha, requires_grad=True)
        self.beta = nn.Parameter(torch.zeros((1, in_channels, 1, 1)) + beta, requires_grad=True)

    def _init_post_mts_function(self, post_mts_function, in_channels, groups, num_heads, dwconv_usage):
        if post_mts_function == 'MHG':
            return MHG(in_channels, groups, num_heads, dwconv_usage=dwconv_usage)
        elif post_mts_function == 'ReLU':
            return nn.ReLU()
        elif post_mts_function == 'GELU':
            return nn.GELU()
        elif post_mts_function == 'LeakyReLU':
            return nn.LeakyReLU()
        else:
            raise ValueError("Invalid post_mts_function. Choose from 'MHG', 'ReLU', 'GELU', or 'LeakyReLU'.")

    def forward(self, x):
        res1 = x
        x = self.multi_scale_transform(self.layer_norm1(x))
        x = self.post_mts_function(x) * self.alpha + res1
        res2 = x
        x = self.feed_forward_network(self.layer_norm2(x)) * self.beta + res2
        return x

    def calculate_total_mac(self, H, W):
        total_macs = 0
        total_macs += self.multi_scale_transform.calculate_total_mac(H, W)

        #FeedForward and MHG
        total_macs += self.feed_forward_network.calculate_mac(H, W)

        if isinstance(self.post_mts_function, MHG):
            total_macs += self.post_mts_function.calculate_mac(H, W)
        # No MACs calculation for ReLU, GELU, and LeakyReLU since they are negligible

        #Layer Norm1
        total_macs += H*W*self.in_channels

        #Layer Norm2
        total_macs += H * W * self.out_channels

        return total_macs


class MTSBlock(nn.Module):
    """
    import torch
    import torch.nn as nn

    # Assuming MTSBlock and all its dependencies (MTSL, FeedForward, MHG, etc.) are defined correctly

    # Parameters for the MTSBlock
    in_channels = 48
    out_channels = 48
    groups = 3
    num_heads = 5
    T = 3  # Assuming T is given as 3 for this example
    window_scales = [8, 16, 32, 64]
    ffn_expansion_factor = 2
    group_ratio = 1
    bias = False
    FFactivation = None  # Assuming this implies no activation function is used
    post_mts_function = 'MHG'
    L = 4
    alpha = 1
    beta = 1
    kernel_size = 3
    padding = 1

    # Instantiate MTSBlock
    mts_block = MTSBlock(in_channels=in_channels,
                         out_channels=out_channels,
                         groups=groups,
                         num_heads=num_heads,
                         T=T,
                         window_scales=window_scales,
                         ffn_expansion_factor=ffn_expansion_factor,
                         group_ratio=group_ratio,
                         bias=bias,
                         FFactivation=FFactivation,
                         post_mts_function=post_mts_function,
                         L=L,
                         alpha=alpha,
                         beta=beta,
                         kernel_size=kernel_size,
                         padding=padding)

    # Example input dimensions
    H, W = 256, 256  # Example dimensions, adjust as needed

    # # Example input dimensions
    # H, W = 256, 256  # Example dimensions, adjust as needed
    #
    # # Calculate FLOPs
    # total_flops = mts_block.calculate_total_mac(H, W)
    # print(f"Total FLOPs for MTSBlock: {total_flops/1e9}")
    """
    def __init__(self, in_channels, out_channels, groups, num_heads, T, window_scales, ffn_expansion_factor,
                 group_ratio, bias, FFactivation, post_mts_function, L, alpha, beta, kernel_size=3, padding=1, dwconv_usage=False, drop_block_p= 0.):
        super(MTSBlock, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.ffn_expansion_factor = ffn_expansion_factor
        self.group_ratio = 1
        self.num_heads = num_heads
        self.groups = groups
        self.kernel_size = kernel_size

        # Convert numpy array to list if not already a list
        if isinstance(window_scales, np.ndarray):
            window_scales = window_scales.tolist()

        # Initialize the ModuleList for MTSL layers
        self.mtsl_layers = nn.ModuleList()
        for i in range(L):
            # Copy window_scales to avoid modifying the original list outside this scope
            current_window_scales = window_scales.copy()

            # Remove the last element of window_scales for each layer after the first
            # Ensure that at least one element remains in the list
            if i > 0 and len(current_window_scales) > 1:
                current_window_scales.pop()  # Remove the last element

            # Create an MTSL layer with the current window_scales
            mtsl_layer = MTSL(in_channels, out_channels, groups, num_heads, T, current_window_scales,
                              ffn_expansion_factor, group_ratio, bias, FFactivation,
                              post_mts_function, alpha, beta, dwconv_usage, drop_block_p)

            # Add the newly created layer to the module list
            self.mtsl_layers.append(mtsl_layer)

        # MTSL layers
        #self.mtsl_layers = nn.ModuleList(
        #    [MTSL(in_channels, out_channels, groups, num_heads, T, window_scales, ffn_expansion_factor,
        #          group_ratio, bias, FFactivation, post_mts_function, alpha, beta, dwconv_usage) for _ in range(L)])

        # Standard 2D convolutional layer
        self.conv2d = nn.Conv2d(out_channels, out_channels, kernel_size=kernel_size, padding=padding, bias=bias)

    def forward(self, x):
        # Save the original input for the residual connection
        original_input = x

        # Apply MTSL layers in sequence
        for mtsl in self.mtsl_layers:
            x = mtsl(x)

        # Apply the standard 2D convolutional layer
        x = self.conv2d(x)

        # Add the original input (residual connection)
        x += original_input

        return x

    def calculate_total_mac(self, H, W):
        total_macs = 0

        # Calculate MACs for each MTSL layer
        for mtsl in self.mtsl_layers:
            total_macs += mtsl.calculate_total_mac(H, W)

        # Calculate MACs for the conv2d layer
        conv2d_macs = calculate_conv_Mac(self.out_channels, self.out_channels, H, W, self.kernel_size, 1)

        # Add conv2d MACs to the total
        total_macs += conv2d_macs

        return total_macs


class MTSNet(nn.Module):
    def __init__(self, N, NB, in_channels=3, groups=3, num_heads=5, T=3,
                 window_scales=[8, 16, 32, 64], ffn_expansion_factor=1,
                 group_ratio=1, bias=False, FFactivation='Relu', post_mts_function='MHG', L=2, alpha=0.1, beta=0.1,
                 embed_style='Hybrid', embed_dim=9, em_window_scales=[8, 16, 32, 64], kernel_size=3, padding=1, 
                 alpha_decay=0.1, beta_decay=1, reduction_list = [], dwconv_usage=False, drop_block_p=0.):
        super(MTSNet, self).__init__()

        # if all(isinstance(item, list) for item in window_scales):
        #    window_scales = [tuple(item) for item in window_scales]
        self.N = N
        self.in_channels = in_channels
        self.out_channels = in_channels
        self.ffn_expansion_factor = ffn_expansion_factor
        self.group_ratio = 1
        self.num_heads = num_heads
        self.groups = groups
        self.kernel_size = kernel_size
        self.px, self.py = window_scales[-1], window_scales[-1]  # find_max_patch_size(window_scales)
        self.em = embed_style

        if in_channels == 3:
            rgb_mean = (0.4488, 0.4371, 0.4040)
            self.register_buffer('mean', torch.Tensor(rgb_mean).view(1, 3, 1, 1))
        else:
            self.register_buffer('mean', torch.zeros(1, 1, 1, 1))

        # First layer
        self.ten1 = MultiScale( in_channels= in_channels, out_channels= N, T= embed_dim, 
        window_scales= em_window_scales, 
        tensor_sum_type="efficient", drop_block_p=drop_block_p)

        if self.em == 'Hybrid':
            self.conv1_1 = nn.Conv2d(2 * N, N, kernel_size=1, padding=0, bias=False)
            self.conv_em = nn.Conv2d(3, N, 3, padding=1, bias=False)

        # Generate the alpha_decay_v list
        alpha_decay_v = [alpha_decay ** i for i in range(NB)]
        # Generate the beta_decay_v list
        beta_decay_v = [beta_decay ** i for i in range(NB)]

        original_window_scales = window_scales
        # NB variants of window_scales with progressive removal
        window_scales_variants = []
        current_window_scales = list(original_window_scales)

        for i in range(NB):
            if i in reduction_list:
                # Remove the last element if not already empty
                if current_window_scales:
                    current_window_scales = current_window_scales[:-1]
            # Append the current state of window_scales to the variants list
            window_scales_variants.append(list(current_window_scales))

            # Ensure there's an error if any variant becomes empty
        for variant in window_scales_variants:
            if not variant:
                raise ValueError(
                    "A window_scales variant is empty after removals. Please check your reduction_list or initial window_scales.")


        self.mts_blocks = nn.ModuleList([
            MTSBlock(N, N, groups, num_heads, T,
                     window_scales_variants[i],
                     ffn_expansion_factor, group_ratio, bias, FFactivation,
                     post_mts_function, L, alpha * alpha_decay_v[i], beta * beta_decay_v[i], dwconv_usage=dwconv_usage, 
                     drop_block_p=drop_block_p)
            for i in range(NB)
        ])

        # Final 2D CNN layer to produce 3-channel RGB output
        self.final_conv = nn.Conv2d(N, in_channels, kernel_size=kernel_size, padding=padding)

    def forward(self, x):
        # Calculate the padding amounts along height and width dimensions
        sx, sy = x.shape[2:]
        px = (self.px - (sx % self.px)) % self.px
        py = (self.py - (sy % self.py)) % self.py
        # Apply padding using PyTorch's nn.functional.pad() function
        x = torch.nn.functional.pad(x, (0, py, 0, px), mode='reflect')
        x = (x - self.mean)
        firstx = x
        # First layer
        x = self.ten1(x)
        # out = self.relu1(out)
        if self.em == 'Hybrid':
            x = self.conv1_1(torch.cat((x, self.conv_em(firstx)), dim=1))

        # Pass through each MTSBlock
        for mts_block in self.mts_blocks:
            x = mts_block(x)

        # Final convolution to get RGB output
        x = self.final_conv(x)

        # residual
        x = x + firstx

        x = x +self.mean
        x = x[:, :, :sx, :sy]

        return x

    def calculate_MAC(self, H, W):
        total_MAC = 0

        # Calculate GMAC for MultiScale layer
        total_MAC += self.ten1.calculate_total_mac(H, W)

        # Calculate GMAC for conv1_1 and conv_em, if applicable
        if self.em == 'Hybrid':
            total_MAC += calculate_conv_Mac(2 * self.N, self.N, H, W, 1)  # conv1_1
            total_MAC += calculate_conv_Mac(3, self.N, H, W, 3)  # conv_em

        # Calculate GMAC for each MTSBlock
        for mts_block in self.mts_blocks:
            total_MAC += mts_block.calculate_total_mac(H, W)

        # Calculate GMAC for the final_conv layer
        total_MAC += calculate_conv_Mac(self.N, self.in_channels, H, W, self.kernel_size)



        return total_MAC

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

if __name__ == '__main__':
    import torch
    import torch.nn as nn
    import torch.nn.functional as F

    import torch
    import torch.nn as nn



    # Parameters for the MTSBlock
    N = 56
    groups = 4
    num_heads = 4
    T = 3  # T is given as 3 for this example
    window_scales = [8,16, 32, 64]
    ffn_expansion_factor = 2.2
    group_ratio = 1
    bias = False
    FFactivation = None  # This implies no activation function is used
    post_mts_function = 'MHG'
    L = 4
    alpha = 1
    beta = 1
    kernel_size = 3
    padding = 1

    mts_net = MTSNet(N=56, NB=2, in_channels=3, groups=groups, num_heads=num_heads, T=T, window_scales=window_scales,
                     ffn_expansion_factor=ffn_expansion_factor, group_ratio=group_ratio, bias=bias,
                     FFactivation=FFactivation, post_mts_function=post_mts_function, L=L, alpha=alpha, beta=beta,
                     embed_style='Hybrid', embed_dim=9, em_window_scales=window_scales, kernel_size=kernel_size,
                     padding=padding, reduction_list=[], dwconv_usage=True, drop_block_p=0.)

    H = W = 256  # Input dimensions

    # Calculate GMACs
    total_flops = mts_net.calculate_MAC(H, W)
    print(f"Total GMAC for MTSNet with H=W=256: {total_flops / 1e9}")

    # Create a random tensor with the specified dimensions
    input_tensor = torch.rand(1, 3, 256, 256)

    output_tensor= mts_net(input_tensor)
    print(output_tensor.shape)

    # Calculate # of parameters
    print(f"Number of trainable parameters: {count_parameters(mts_net)}")

    

    #import torch
    #from torch.profiler import profile, record_function, ProfilerActivity

    #with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA], profile_memory=True, record_shapes=True) as prof:
    #    with record_function("model_inference"):
    #        mts_net(input_tensor)

    #print(prof.key_averages().table(sort_by="cpu_memory_usage", row_limit=10))
