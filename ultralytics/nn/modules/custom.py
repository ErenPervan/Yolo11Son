# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license
"""Custom modules for pothole detection: DSConv, SimAM, and GELU-based convolutions."""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


__all__ = (
    "DSConv",
    "DySnakeConv",
    "SimAM",
    "ConvGELU",
    "C3k2_DSConv",
    "C2f_DSConv",
)


def autopad(k, p=None, d=1):
    """Pad to 'same' shape outputs."""
    if d > 1:
        k = d * (k - 1) + 1 if isinstance(k, int) else [d * (x - 1) + 1 for x in k]
    if p is None:
        p = k // 2 if isinstance(k, int) else [x // 2 for x in k]
    return p


# ============================================================================
# Dynamic Snake Convolution (DSConv)
# Paper: "Dynamic Snake Convolution based on Topological Geometric Constraints 
#         for Tubular Structure Segmentation"
# Reference: https://arxiv.org/abs/2307.08388
# ============================================================================

class DSConv(nn.Module):
    """Dynamic Snake Convolution for capturing irregular/curved boundaries.
    
    This convolution adapts its sampling locations to follow snake-like patterns,
    making it ideal for detecting objects with irregular edges like potholes.
    
    Attributes:
        in_ch (int): Number of input channels.
        out_ch (int): Number of output channels.
        kernel_size (int): Size of the convolution kernel.
        extend_scope (float): Scope for extending the deformation field.
        morph (int): 0 for x-direction, 1 for y-direction snake.
    """
    
    def __init__(self, in_ch, out_ch, kernel_size=3, extend_scope=1.0, morph=0, if_offset=True):
        """Initialize DSConv layer.
        
        Args:
            in_ch (int): Number of input channels.
            out_ch (int): Number of output channels.
            kernel_size (int): Size of the convolution kernel.
            extend_scope (float): Scope for extending deformation.
            morph (int): 0 for x-axis snake, 1 for y-axis snake.
            if_offset (bool): Whether to use learnable offsets.
        """
        super().__init__()
        self.kernel_size = kernel_size
        self.extend_scope = extend_scope
        self.morph = morph
        self.if_offset = if_offset
        self.in_ch = in_ch
        self.out_ch = out_ch
        
        # Learnable offset for deformable positions
        self.offset_conv = nn.Conv2d(in_ch, 2 * kernel_size, 3, padding=1, bias=True)
        
        # Dynamic snake convolution
        self.dsc_conv = nn.Conv2d(
            in_ch, 
            out_ch, 
            kernel_size=(kernel_size, 1) if morph == 0 else (1, kernel_size),
            stride=1,
            padding=((kernel_size - 1) // 2, 0) if morph == 0 else (0, (kernel_size - 1) // 2),
            bias=False
        )
        
        self.bn = nn.BatchNorm2d(out_ch)
        self.act = nn.GELU()  # Using GELU activation
        
        # Initialize offset conv
        nn.init.constant_(self.offset_conv.weight, 0.)
        nn.init.constant_(self.offset_conv.bias, 0.)
        
    def forward(self, x):
        """Forward pass with dynamic snake convolution.
        
        Args:
            x (torch.Tensor): Input tensor of shape (B, C, H, W).
            
        Returns:
            (torch.Tensor): Output tensor.
        """
        if self.if_offset:
            offset = self.offset_conv(x)
            offset = torch.tanh(offset) * self.extend_scope
            x_deformed = self._deform_input(x, offset)
        else:
            x_deformed = x
            
        out = self.dsc_conv(x_deformed)
        out = self.bn(out)
        out = self.act(out)
        return out
    
    def _deform_input(self, x, offset):
        """Apply deformation to input based on learned offsets.
        
        Args:
            x (torch.Tensor): Input tensor.
            offset (torch.Tensor): Offset tensor.
            
        Returns:
            (torch.Tensor): Deformed input tensor.
        """
        B, C, H, W = x.shape
        
        # Create base grid
        grid_y, grid_x = torch.meshgrid(
            torch.linspace(-1, 1, H, device=x.device),
            torch.linspace(-1, 1, W, device=x.device),
            indexing='ij'
        )
        
        # Apply offsets based on morph direction
        if self.morph == 0:  # x-direction snake
            offset_x = offset[:, :self.kernel_size, :, :].mean(dim=1, keepdim=True)
            offset_y = offset[:, self.kernel_size:, :, :].mean(dim=1, keepdim=True)
        else:  # y-direction snake  
            offset_x = offset[:, self.kernel_size:, :, :].mean(dim=1, keepdim=True)
            offset_y = offset[:, :self.kernel_size, :, :].mean(dim=1, keepdim=True)
        
        # Normalize offsets
        offset_x = offset_x / (W / 2)
        offset_y = offset_y / (H / 2)
        
        # Apply offsets to grid
        grid_x = grid_x.unsqueeze(0).unsqueeze(0).expand(B, 1, H, W) + offset_x
        grid_y = grid_y.unsqueeze(0).unsqueeze(0).expand(B, 1, H, W) + offset_y
        
        # Stack grid
        grid = torch.cat([grid_x, grid_y], dim=1).permute(0, 2, 3, 1)
        
        # Sample from input using grid
        x_deformed = F.grid_sample(x, grid, mode='bilinear', padding_mode='border', align_corners=True)
        
        return x_deformed


class DySnakeConv(nn.Module):
    """Dynamic Snake Convolution block with multi-directional processing.
    
    Combines x-axis and y-axis snake convolutions with a standard convolution
    for comprehensive feature extraction of irregular boundaries.
    """
    
    def __init__(self, c1, c2, k=3, s=1):
        """Initialize DySnakeConv block.
        
        Args:
            c1 (int): Number of input channels.
            c2 (int): Number of output channels.
            k (int): Kernel size.
            s (int): Stride (not used, for API compatibility).
        """
        super().__init__()
        c_ = c2 // 3  # channel for each branch
        
        # Standard convolution branch
        self.conv_0 = nn.Sequential(
            nn.Conv2d(c1, c_, k, padding=autopad(k), bias=False),
            nn.BatchNorm2d(c_),
            nn.GELU()
        )
        
        # X-direction snake convolution
        self.conv_x = DSConv(c1, c_, kernel_size=k, morph=0, extend_scope=1.0)
        
        # Y-direction snake convolution  
        self.conv_y = DSConv(c1, c_, kernel_size=k, morph=1, extend_scope=1.0)
        
        # Fusion convolution
        self.conv_fusion = nn.Sequential(
            nn.Conv2d(c_ * 3, c2, 1, bias=False),
            nn.BatchNorm2d(c2),
            nn.GELU()
        )
        
    def forward(self, x):
        """Forward pass through multi-directional snake convolution.
        
        Args:
            x (torch.Tensor): Input tensor.
            
        Returns:
            (torch.Tensor): Output tensor with fused features.
        """
        y_0 = self.conv_0(x)
        y_x = self.conv_x(x)
        y_y = self.conv_y(x)
        
        # Concatenate all branches
        y = torch.cat([y_0, y_x, y_y], dim=1)
        
        return self.conv_fusion(y)


# ============================================================================
# Simple Attention Module (SimAM)
# Paper: "SimAM: A Simple, Parameter-Free Attention Module for CNNs"
# Reference: https://proceedings.mlr.press/v139/yang21o
# ============================================================================

class SimAM(nn.Module):
    """Simple Attention Module - Parameter-free attention mechanism.
    
    SimAM computes 3D attention weights without adding any learnable parameters.
    It evaluates the importance of each neuron based on an energy function,
    enabling the model to focus on important features like pothole regions.
    
    Attributes:
        e_lambda (float): Regularization parameter to prevent division by zero.
    """
    
    def __init__(self, e_lambda=1e-4):
        """Initialize SimAM attention module.
        
        Args:
            e_lambda (float): Small constant for numerical stability.
        """
        super().__init__()
        self.e_lambda = e_lambda
        
    def forward(self, x):
        """Apply SimAM attention to input tensor.
        
        The attention is computed based on the energy function:
        E = 4 * (sigma^2 + lambda) where sigma^2 is the variance.
        
        Args:
            x (torch.Tensor): Input tensor of shape (B, C, H, W).
            
        Returns:
            (torch.Tensor): Attention-weighted output tensor.
        """
        B, C, H, W = x.shape
        n = H * W - 1
        
        # Calculate mean and variance
        x_minus_mu_square = (x - x.mean(dim=[2, 3], keepdim=True)).pow(2)
        
        # Compute attention weights based on energy function
        # Higher energy = less important, so we use 1/energy as weight
        y = x_minus_mu_square / (4 * (x_minus_mu_square.sum(dim=[2, 3], keepdim=True) / n + self.e_lambda)) + 0.5
        
        # Apply sigmoid to get attention weights in [0, 1]
        attention = torch.sigmoid(y)
        
        return x * attention


# ============================================================================
# GELU-based Convolution Modules
# ============================================================================

class ConvGELU(nn.Module):
    """Standard convolution module with GELU activation instead of SiLU.
    
    Attributes:
        conv (nn.Conv2d): Convolutional layer.
        bn (nn.BatchNorm2d): Batch normalization layer.
        act (nn.GELU): GELU activation function.
    """
    
    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, d=1, act=True):
        """Initialize ConvGELU layer.
        
        Args:
            c1 (int): Number of input channels.
            c2 (int): Number of output channels.
            k (int): Kernel size.
            s (int): Stride.
            p (int, optional): Padding.
            g (int): Groups.
            d (int): Dilation.
            act (bool | nn.Module): Activation function.
        """
        super().__init__()
        self.conv = nn.Conv2d(c1, c2, k, s, autopad(k, p, d), groups=g, dilation=d, bias=False)
        self.bn = nn.BatchNorm2d(c2)
        self.act = nn.GELU() if act is True else act if isinstance(act, nn.Module) else nn.Identity()
        
    def forward(self, x):
        """Apply convolution, batch normalization and GELU activation.
        
        Args:
            x (torch.Tensor): Input tensor.
            
        Returns:
            (torch.Tensor): Output tensor.
        """
        return self.act(self.bn(self.conv(x)))
    
    def forward_fuse(self, x):
        """Apply convolution and activation without batch normalization.
        
        Args:
            x (torch.Tensor): Input tensor.
            
        Returns:
            (torch.Tensor): Output tensor.
        """
        return self.act(self.conv(x))


# ============================================================================
# Enhanced C3k2 and C2f blocks with DSConv
# ============================================================================

class Bottleneck_DSConv(nn.Module):
    """Bottleneck block with Dynamic Snake Convolution.
    
    Standard bottleneck with DSConv for better irregular boundary detection.
    """
    
    def __init__(self, c1, c2, shortcut=True, g=1, k=(3, 3), e=0.5):
        """Initialize Bottleneck with DSConv.
        
        Args:
            c1 (int): Number of input channels.
            c2 (int): Number of output channels.
            shortcut (bool): Whether to use residual connection.
            g (int): Groups for convolution.
            k (tuple): Kernel sizes for both convolutions.
            e (float): Expansion ratio.
        """
        super().__init__()
        c_ = int(c2 * e)
        self.cv1 = ConvGELU(c1, c_, k[0], 1)
        self.cv2 = DySnakeConv(c_, c2, k[1], 1)
        self.add = shortcut and c1 == c2
        
    def forward(self, x):
        """Forward pass through bottleneck with DSConv.
        
        Args:
            x (torch.Tensor): Input tensor.
            
        Returns:
            (torch.Tensor): Output tensor.
        """
        return x + self.cv2(self.cv1(x)) if self.add else self.cv2(self.cv1(x))


class C3k2_DSConv(nn.Module):
    """C3k2 block with Dynamic Snake Convolution for pothole detection.
    
    Enhanced C3k2 that uses DSConv bottlenecks for better boundary detection
    of irregular shapes like potholes.
    """
    
    def __init__(self, c1, c2, n=1, c3k=False, e=0.5, g=1, shortcut=True):
        """Initialize C3k2_DSConv block.
        
        Args:
            c1 (int): Number of input channels.
            c2 (int): Number of output channels.
            n (int): Number of bottleneck blocks.
            c3k (bool): Whether to use c3k variant.
            e (float): Expansion ratio.
            g (int): Groups.
            shortcut (bool): Whether to use shortcut connections.
        """
        super().__init__()
        self.c = int(c2 * e)
        self.cv1 = ConvGELU(c1, 2 * self.c, 1, 1)
        self.cv2 = ConvGELU((2 + n) * self.c, c2, 1)
        
        # Use DSConv bottlenecks
        self.m = nn.ModuleList(
            Bottleneck_DSConv(self.c, self.c, shortcut, g, k=(3, 3), e=1.0) 
            for _ in range(n)
        )
        
    def forward(self, x):
        """Forward pass through C3k2_DSConv block.
        
        Args:
            x (torch.Tensor): Input tensor.
            
        Returns:
            (torch.Tensor): Output tensor.
        """
        y = list(self.cv1(x).chunk(2, 1))
        y.extend(m(y[-1]) for m in self.m)
        return self.cv2(torch.cat(y, 1))


class C2f_DSConv(nn.Module):
    """C2f block with Dynamic Snake Convolution.
    
    Enhanced C2f that incorporates DSConv for better irregular shape detection.
    """
    
    def __init__(self, c1, c2, n=1, shortcut=False, g=1, e=0.5):
        """Initialize C2f_DSConv block.
        
        Args:
            c1 (int): Number of input channels.
            c2 (int): Number of output channels.
            n (int): Number of bottleneck blocks.
            shortcut (bool): Whether to use shortcut connections.
            g (int): Groups.
            e (float): Expansion ratio.
        """
        super().__init__()
        self.c = int(c2 * e)
        self.cv1 = ConvGELU(c1, 2 * self.c, 1, 1)
        self.cv2 = ConvGELU((2 + n) * self.c, c2, 1)
        
        self.m = nn.ModuleList(
            Bottleneck_DSConv(self.c, self.c, shortcut, g, k=(3, 3), e=1.0)
            for _ in range(n)
        )
        
    def forward(self, x):
        """Forward pass through C2f_DSConv block.
        
        Args:
            x (torch.Tensor): Input tensor.
            
        Returns:
            (torch.Tensor): Output tensor.
        """
        y = list(self.cv1(x).chunk(2, 1))
        y.extend(m(y[-1]) for m in self.m)
        return self.cv2(torch.cat(y, 1))


# ============================================================================
# SimAM-enhanced blocks
# ============================================================================

class C3k2_SimAM(nn.Module):
    """C3k2 block with SimAM attention for enhanced feature focus.
    
    Uses SimAM attention after bottleneck operations to help the model
    focus on important features like pothole regions.
    """
    
    def __init__(self, c1, c2, n=1, c3k=False, e=0.5, g=1, shortcut=True):
        """Initialize C3k2_SimAM block.
        
        Args:
            c1 (int): Number of input channels.
            c2 (int): Number of output channels.
            n (int): Number of bottleneck blocks.
            c3k (bool): Whether to use c3k variant.
            e (float): Expansion ratio.
            g (int): Groups.
            shortcut (bool): Whether to use shortcut connections.
        """
        super().__init__()
        self.c = int(c2 * e)
        self.cv1 = ConvGELU(c1, 2 * self.c, 1, 1)
        self.cv2 = ConvGELU((2 + n) * self.c, c2, 1)
        self.simam = SimAM()
        
        # Standard bottleneck with GELU
        self.m = nn.ModuleList(
            Bottleneck_GELU(self.c, self.c, shortcut, g, k=(3, 3), e=1.0) 
            for _ in range(n)
        )
        
    def forward(self, x):
        """Forward pass through C3k2_SimAM block.
        
        Args:
            x (torch.Tensor): Input tensor.
            
        Returns:
            (torch.Tensor): Output tensor with SimAM attention.
        """
        y = list(self.cv1(x).chunk(2, 1))
        y.extend(m(y[-1]) for m in self.m)
        out = self.cv2(torch.cat(y, 1))
        return self.simam(out)


class Bottleneck_GELU(nn.Module):
    """Standard bottleneck block with GELU activation."""
    
    def __init__(self, c1, c2, shortcut=True, g=1, k=(3, 3), e=0.5):
        """Initialize Bottleneck with GELU.
        
        Args:
            c1 (int): Number of input channels.
            c2 (int): Number of output channels.
            shortcut (bool): Whether to use residual connection.
            g (int): Groups.
            k (tuple): Kernel sizes.
            e (float): Expansion ratio.
        """
        super().__init__()
        c_ = int(c2 * e)
        self.cv1 = ConvGELU(c1, c_, k[0], 1)
        self.cv2 = ConvGELU(c_, c2, k[1], 1, g=g)
        self.add = shortcut and c1 == c2
        
    def forward(self, x):
        """Forward pass through GELU bottleneck.
        
        Args:
            x (torch.Tensor): Input tensor.
            
        Returns:
            (torch.Tensor): Output tensor.
        """
        return x + self.cv2(self.cv1(x)) if self.add else self.cv2(self.cv1(x))


# ============================================================================
# Combined DSConv + SimAM blocks (Best of both worlds)
# ============================================================================

class C3k2_DSConv_SimAM(nn.Module):
    """C3k2 block combining DSConv and SimAM for optimal pothole detection.
    
    This block uses:
    - DSConv for better irregular boundary detection
    - SimAM for enhanced attention on important features
    - GELU for stable training
    """
    
    def __init__(self, c1, c2, n=1, c3k=False, e=0.5, g=1, shortcut=True):
        """Initialize C3k2_DSConv_SimAM block.
        
        Args:
            c1 (int): Number of input channels.
            c2 (int): Number of output channels.
            n (int): Number of bottleneck blocks.
            c3k (bool): Whether to use c3k variant.
            e (float): Expansion ratio.
            g (int): Groups.
            shortcut (bool): Whether to use shortcut connections.
        """
        super().__init__()
        self.c = int(c2 * e)
        self.cv1 = ConvGELU(c1, 2 * self.c, 1, 1)
        self.cv2 = ConvGELU((2 + n) * self.c, c2, 1)
        self.simam = SimAM()
        
        # Use DSConv bottlenecks
        self.m = nn.ModuleList(
            Bottleneck_DSConv(self.c, self.c, shortcut, g, k=(3, 3), e=1.0) 
            for _ in range(n)
        )
        
    def forward(self, x):
        """Forward pass through combined DSConv + SimAM block.
        
        Args:
            x (torch.Tensor): Input tensor.
            
        Returns:
            (torch.Tensor): Output tensor with DSConv and SimAM enhancements.
        """
        y = list(self.cv1(x).chunk(2, 1))
        y.extend(m(y[-1]) for m in self.m)
        out = self.cv2(torch.cat(y, 1))
        return self.simam(out)


class SPPF_SimAM(nn.Module):
    """Spatial Pyramid Pooling - Fast with SimAM attention.
    
    Enhanced SPPF module that applies SimAM attention for better
    feature focus in multi-scale pooling.
    """
    
    def __init__(self, c1, c2, k=5):
        """Initialize SPPF_SimAM module.
        
        Args:
            c1 (int): Number of input channels.
            c2 (int): Number of output channels.
            k (int): Kernel size for max pooling.
        """
        super().__init__()
        c_ = c1 // 2
        self.cv1 = ConvGELU(c1, c_, 1, 1)
        self.cv2 = ConvGELU(c_ * 4, c2, 1, 1)
        self.m = nn.MaxPool2d(kernel_size=k, stride=1, padding=k // 2)
        self.simam = SimAM()
        
    def forward(self, x):
        """Forward pass through SPPF with SimAM.
        
        Args:
            x (torch.Tensor): Input tensor.
            
        Returns:
            (torch.Tensor): Output tensor with multi-scale features and attention.
        """
        x = self.cv1(x)
        y1 = self.m(x)
        y2 = self.m(y1)
        y3 = self.m(y2)
        out = self.cv2(torch.cat([x, y1, y2, y3], 1))
        return self.simam(out)

