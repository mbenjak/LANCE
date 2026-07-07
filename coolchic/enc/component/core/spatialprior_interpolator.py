# Software Name: LANCE
# SPDX-FileCopyrightText: Copyright (c) 2026 Martin Benjak
# SPDX-License-Identifier: BSD 3-Clause "New"
#
# This software is distributed under the BSD-3-Clause license.
#
# Authors: see CONTRIBUTORS.md

import torch
import torch.nn as nn
import torch.nn.functional as F
from enc.bitstream.spatialprior_interpolatorint import BicubicParametrizedInt
from enc.bitstream.bicubic_params import get_bicubic_params_float

class BicubicParametrized(nn.Module):
    def __init__(self, power, latent_size):
        super(BicubicParametrized, self).__init__()
        self.power = power
        factor = 2 ** power
        self.factor = factor
        self.latent_size = latent_size
        self.taps = 4
        if factor != 1:
            self.k_size = self.taps * factor if factor > 1 else self.taps
            self.register_buffer('kernel', get_bicubic_params_float(power), persistent=False)
        else:
            self.k_size = None
            self.pos = None
            self.param = None
            self.kernel = None

    def upsample(self, x):
        k = self.k_size
        P0 = k // self.factor
        C = self.factor * P0 + ( k - self.factor) // 2
        _, _, lh, lw = self.latent_size

        weight = self.kernel
        weight = weight.view(1, -1)

        if self.training:  # training using non-separable (more stable)
            kernel_2d = (torch.kron(weight, weight.T).view((1, 1, k, k)))

            x_pad = F.pad(x, (P0, P0, P0, P0), mode="replicate")
            yc = F.conv_transpose2d(x_pad, kernel_2d, stride=self.factor)

            H, W = yc.size()[-2:]
            y = yc[
                :,
                :,
                C : H - C,
                C : W - C,
            ]

        else:  # testing through separable (less complex)
            # horizontal filtering
            x_pad = F.pad(x, (P0, P0, 0, 0), mode="replicate")
            yc = F.conv_transpose2d(x_pad, weight.view((1, 1, 1, k)), stride=(1, self.factor))
            W = yc.size()[-1]
            y = yc[
                :,
                :,
                :,
                C : W - C,
            ]

            # vertical filtering
            x_pad = F.pad(y, (0, 0, P0, P0), mode="replicate")
            yc = F.conv_transpose2d(x_pad, weight.view((1, 1, k, 1)), stride=(self.factor, 1))
            H = yc.size()[-2]
            y = yc[:, :, C : H - C, :]
        return y[:, :, :lh, :lw]
    
    def downsample(self, x):
        k = self.k_size
        P0 = int((self.taps - 1/self.factor) // 2)

        weight = self.kernel
        weight = weight.view(1, -1) 
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

        if padh > 0 or padw > 0:
            x_prepad = F.pad(x, (0, padw, 0, padh), mode="replicate")
        else:
            x_prepad = x
        if self.training:  # training using non-separable (more stable)
            kernel_2d = (torch.kron(weight, weight.T).view((1, 1, k, k)))
            if P0 >= 0:
                x_pad = F.pad(x_prepad, (P0, P0, P0, P0), mode="replicate")
            else:
                x_pad = x_prepad[:, :, -P0:x_prepad.size(2)+P0, -P0:x_prepad.size(3)+P0]
            y = F.conv2d(x_pad, kernel_2d, stride=int(1/self.factor))
        else:  # testing through separable (less complex)
            # horizontal filtering
            if P0 >= 0:
                x_pad = F.pad(x_prepad, (P0, P0, 0, 0), mode="replicate")
            else:
                x_pad = x_prepad[:, :, :, -P0:x_prepad.size(3)+P0]
            y = F.conv2d(x_pad, weight.view((1, 1, 1, k)), stride=(1, int(1/self.factor)))

            # vertical filtering
            if P0 >= 0:
                x_pad = F.pad(y, (0, 0, P0, P0), mode="replicate")
            else:
                x_pad = y[:, :, -P0:y.size(2)+P0, :]
            y = F.conv2d(x_pad, weight.view((1, 1, k, 1)), stride=(int(1/self.factor), 1))
        return y

    def forward(self, x):
        if self.factor == 1:
            return x
        elif self.factor > 1:
            return self.upsample(x)
        else:
            return self.downsample(x)

class MultilayerInterpolator(nn.Module):
    def __init__(self, num_layers, original_layer, latent_sizes=[]):
        super(MultilayerInterpolator, self).__init__()
        assert original_layer <= num_layers, "Original layer must be less than or equal to number of layers"

        self.num_layers = num_layers
        self.original_layer = original_layer
        self.interpolators = nn.ModuleList([
            BicubicParametrized(power=original_layer - layer, latent_size=latent_sizes[layer])
            for layer in range(num_layers)
        ])

    def forward(self, x):
        return [
            self.interpolators[i](x) 
            for i in range(self.num_layers)
        ]