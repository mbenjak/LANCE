# Software Name: LANCE
# SPDX-FileCopyrightText: Copyright (c) 2026 Martin Benjak
# SPDX-License-Identifier: BSD 3-Clause "New"
#
# This software is distributed under the BSD-3-Clause license.
#
# Authors: see CONTRIBUTORS.md


"""Fixed point implementation of the MultilayerInterpolator to avoid floating point drift."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from enc.bitstream.bicubic_params import get_bicubic_params_int


class BicubicParametrizedInt(nn.Module):
    """Integer version of BicubicParametrized for fixed-point computation.
    
    This class implements bicubic interpolation with Mitchell-Netravali filter
    in fixed-point arithmetic to avoid floating point drift between encoder and decoder.
    """
    
    def __init__(self, power: int, fpfm: int, pure_int: bool, latent_size: list):
        """
        Args:
            power: Power of 2 for the scaling factor (factor = 2^power)
            fpfm: Fixed-point multiplication factor for integer computation
            pure_int: Whether to use pure integer computation (int32) or float with integer values
        """
        super(BicubicParametrizedInt, self).__init__()
        self.power = power
        factor = 2 ** power
        self.factor = factor
        self.taps = 4
        self.fpfm = fpfm
        self.pure_int = pure_int
        self.latent_size = latent_size
        
        if factor != 1:
            self.k_size = self.taps * factor if factor > 1 else self.taps
            self.register_buffer('kernel', get_bicubic_params_int(power), persistent=False)
            #self.register_buffer('pos', self.get_bicubic_distance_int(), persistent=False)
        else:
            self.k_size = None
            self.kernel = None

    def get_bicubic_distance_int(self):
        """Get the distance values for bicubic interpolation kernel sampling positions.
        
        Returns positions scaled by fpfm for reproducible integer arithmetic.
        """
        factor = self.factor
        
        if factor > 1:
            # Upsampling case
            upsampling_factor = int(factor)
            kernel_size = self.taps * upsampling_factor
            half_size = kernel_size // 2
            
            # Compute positions in fixed-point: pos = (i / factor + 0.5 / factor) * fpfm
            # = (i * fpfm + fpfm // 2) / factor
            positions = []
            for i in range(half_size):
                # pos_fpfm = (i + 0.5) * fpfm / factor
                # Using integer arithmetic: (i * fpfm + fpfm // 2) / factor
                numerator = i * self.fpfm + self.fpfm // 2 + upsampling_factor // 2
                pos_fpfm = numerator // upsampling_factor
                positions.append(pos_fpfm)
                
                # Verify position is less than 2.0 * fpfm
                assert pos_fpfm < 2 * self.fpfm, f"Position {pos_fpfm / self.fpfm} must be less than 2.0"
            
            return torch.tensor(positions, dtype=torch.int64)
        else:
            # Downsampling case (factor < 1)
            taps = self.taps
            half_size = taps // 2
            
            # Compute positions in fixed-point: pos = (i + 0.5) * fpfm
            positions = []
            for i in range(half_size):
                pos_fpfm = i * self.fpfm + self.fpfm // 2
                positions.append(pos_fpfm)
                
                # Verify position is less than 2.0 * fpfm
                assert pos_fpfm < 2 * self.fpfm, f"Position {pos_fpfm / self.fpfm} must be less than 2.0"
            
            return torch.tensor(positions, dtype=torch.int64)

    def upsample_int(self, x):
        """Upsample using integer arithmetic.
        
        Args:
            x: Input tensor (in fixed-point, scaled by fpfm)
            
        Returns:
            Upsampled tensor (in fixed-point, scaled by fpfm)
        """
        k = self.k_size
        P0 = k // self.factor
        C = self.factor * P0 + (k - self.factor) // 2
        _, _, lh, lw = self.latent_size
        
        #kernel = self.mitchell_netravali_int(param_int)
        #weight = torch.cat([torch.flip(kernel, dims=[-1]), kernel], dim=-1)
        weight = self.kernel
        weight = weight.view(1, -1)
        
        # transposed convolutuion does not support integer types, so we implement upsampling
        # manually using polyphase filtering

        polyphase_weight = [weight[0, i::self.factor].flip(-1) for i in range(self.factor)]

        B, C, H, W = x.size()
        H_out = H * self.factor
        W_out = W * self.factor

        # Horizontal filtering
        y_h = torch.zeros(B, C, H, W_out, dtype=x.dtype, device=x.device)
        for i in range(self.factor):
            if i < self.factor // 2:
                x_pad = F.pad(x, (2, 1, 0, 0), mode="replicate")
            else:
                x_pad = F.pad(x, (1, 2, 0, 0), mode="replicate")
            w = polyphase_weight[(i + self.factor // 2) % self.factor].view((1, 1, 1, 4))
            y_h[:, :, :, i::self.factor] = F.conv2d(x_pad, w, stride=1)

        # Renormalize after convolution
        if self.pure_int:
            y_h = y_h.to(torch.int64)
            y_h = y_h + torch.sign(y_h) * self.fpfm // 2
            neg_result = -((-y_h) // self.fpfm)
            pos_result = y_h // self.fpfm
            y_h = torch.where(y_h < 0, neg_result, pos_result)
        else:
            y_h = y_h + torch.sign(y_h) * self.fpfm / 2
            neg_result = -((-y_h) / self.fpfm)
            pos_result = y_h / self.fpfm
            y_h = torch.where(y_h < 0, neg_result, pos_result).to(torch.int64).to(torch.float)
        
        # Vertical filtering
        y = torch.zeros(B, C, H_out, W_out, dtype=x.dtype, device=x.device)
        for i in range(self.factor):
            if i < self.factor // 2:
                y_h_pad = F.pad(y_h, (0, 0 , 2, 1), mode="replicate")
            else:
                y_h_pad = F.pad(y_h, (0, 0, 1, 2), mode="replicate")        
            w = polyphase_weight[(i + self.factor // 2) % self.factor].view((1, 1, 4, 1)) 
            y[:, :, i::self.factor, :] = F.conv2d(y_h_pad, w, stride=1)

        # Renormalize after convolution
        if self.pure_int:
            y = y.to(torch.int64)
            y = y + torch.sign(y) * self.fpfm // 2
            neg_result = -((-y) // self.fpfm)
            pos_result = y // self.fpfm
            y = torch.where(y < 0, neg_result, pos_result)
        else:
            y = y + torch.sign(y) * self.fpfm / 2
            neg_result = -((-y) / self.fpfm)
            pos_result = y / self.fpfm
            y = torch.where(y < 0, neg_result, pos_result).to(torch.int64).to(torch.float)
        
        return y[:, :, :lh, :lw]

    def downsample_int(self, x):
        """Downsample using integer arithmetic.
        
        Args:
            x: Input tensor (in fixed-point, scaled by fpfm)
            
        Returns:
            Downsampled tensor (in fixed-point, scaled by fpfm)
        """
        k = self.k_size
        P0 = int((self.taps - 1/self.factor) // 2)

        _, _, lh, lw = self.latent_size
        _, _, h, w = [int(t) for t in x.shape]

        if h * self.factor != lh:
            padh = int(lh / self.factor - h)
        else:
            padh = 0

        if w * self.factor != lw:
            padw = int(lw / self.factor - w)
        else:
            padw = 0
        
        #kernel = self.mitchell_netravali_int(param_int)
        #weight = torch.cat([torch.flip(kernel, dims=[-1]), kernel], dim=-1)
        weight = self.kernel
        weight = weight.view(1, -1)
        
        # Horizontal filtering
        x_prepad = F.pad(x, (0, padw, 0, padh), mode="replicate")
        if P0 >= 0:
            x_pad = F.pad(x_prepad, (P0, P0, 0, 0), mode="replicate")
        else:
            x_pad = x_prepad[:, :, :, -P0:x_prepad.size(3)+P0]
        
        y = F.conv2d(x_pad, weight.view((1, 1, 1, k)), stride=(1, int(1/self.factor)))
        
        # Renormalize after convolution
        if self.pure_int:
            y = y.to(torch.int64)
            y = y + torch.sign(y) * self.fpfm // 2
            neg_result = -((-y) // self.fpfm)
            pos_result = y // self.fpfm
            y = torch.where(y < 0, neg_result, pos_result)
        else:
            y = y + torch.sign(y) * self.fpfm / 2
            neg_result = -((-y) / self.fpfm)
            pos_result = y / self.fpfm
            y = torch.where(y < 0, neg_result, pos_result).to(torch.int64).to(torch.float)
        
        # Vertical filtering
        if P0 >= 0:
            x_pad = F.pad(y, (0, 0, P0, P0), mode="replicate")
        else:
            x_pad = y[:, :, -P0:y.size(2)+P0, :]
        
        y = F.conv2d(x_pad, weight.view((1, 1, k, 1)), stride=(int(1/self.factor), 1))
        
        # Renormalize after convolution
        if self.pure_int:
            y = y.to(torch.int64)
            y = y + torch.sign(y) * self.fpfm // 2
            neg_result = -((-y) // self.fpfm)
            pos_result = y // self.fpfm
            y = torch.where(y < 0, neg_result, pos_result)
        else:
            y = y + torch.sign(y) * self.fpfm / 2
            neg_result = -((-y) / self.fpfm)
            pos_result = y / self.fpfm
            y = torch.where(y < 0, neg_result, pos_result).to(torch.int64).to(torch.float)
        
        return y

    def forward(self, x):
        """Forward pass with integer arithmetic.
        
        Args:
            x: Input tensor (in fixed-point, scaled by fpfm)
            param_int: Integer parameters [B, C] scaled by fpfm
            
        Returns:
            Interpolated tensor (in fixed-point, scaled by fpfm)
        """
        if self.factor == 1:
            return x
        elif self.factor > 1:
            return self.upsample_int(x)
        else:
            return self.downsample_int(x)


class MultilayerInterpolatorInt(nn.Module):
    """Integer version of MultilayerInterpolator for fixed-point computation.
    
    This class manages multiple BicubicParametrizedInt instances to interpolate
    a feature map to different scales in a multi-layer pyramid. Only supports
    param="shared" mode (shared parameters across all scales).
    """
    
    def __init__(self, num_layers: int, original_layer: int, fpfm:int=2**24, ext_fpfm:int=2**24, pure_int:bool=True, latent_sizes:list=[]):
        """
        Args:
            num_layers: Total number of layers in the pyramid
            original_layer: Index of the original resolution layer (0-indexed)
            fpfm: Fixed-point multiplication factor for integer computation
            pure_int: Whether to use pure integer computation (int32) or float with integer values
        """
        super(MultilayerInterpolatorInt, self).__init__()
        assert original_layer <= num_layers, "Original layer must be less than or equal to number of layers"
        assert ext_fpfm <= fpfm, "ext_fpfm must be less than or equal to fpfm"

        self.num_layers = num_layers
        self.original_layer = original_layer
        self.fpfm = fpfm
        self.ext_fpfm = ext_fpfm
        self.pure_int = pure_int
        
        # Create interpolators for each layer
        # Only bicubic mode is supported (no area downsampling)
        self.interpolators = nn.ModuleList([
            BicubicParametrizedInt(power=original_layer - layer, fpfm=fpfm, pure_int=pure_int, latent_size=latent_sizes[layer])
            for layer in range(num_layers)
        ])

    def forward(self, x):
        """Forward pass to generate multi-scale feature pyramid.
        
        Args:
            x: Input tensor at original_layer resolution (in fixed-point, scaled by fpfm)
            
        Returns:
            List of tensors at different scales (all in fixed-point, scaled by fpfm)
        """
        # Input is already scaled by fpfm, no need to scale again
        if self.pure_int and x.dtype != torch.int64:
            x = (x * self.fpfm).round().to(torch.int64)
        pyramid = [
            self.interpolators[i](x) 
            for i in range(self.num_layers)
        ]

        for i in range(self.num_layers):
            x = pyramid[i]
            prec = self.fpfm // self.ext_fpfm
            if self.pure_int:
                x = x + torch.sign(x) * prec // 2
                neg_result = -((-x) // prec)
                pos_result = x // prec
                x = torch.where(x < 0, neg_result, pos_result)
            else:
                x = x + torch.sign(x) * prec / 2
                neg_result = -((-x) / prec)
                pos_result = x / prec
                x = torch.where(x < 0, neg_result, pos_result).to(torch.int64).to(torch.float)
            pyramid[i] = x.float() / self.ext_fpfm

        return pyramid