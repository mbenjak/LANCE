# Software Name: LANCE
# SPDX-FileCopyrightText: Copyright (c) 2026 Martin Benjak
# SPDX-License-Identifier: BSD 3-Clause "New"
#
# This software is distributed under the BSD-3-Clause license.
#
# Authors: see CONTRIBUTORS.md

import torch
import torch.nn.functional as F
import torch.nn as nn
from torch import Tensor
from collections import OrderedDict
from typing import Tuple, Optional
import math
from enc.component.core.quantizer import (
    POSSIBLE_QUANTIZATION_NOISE_TYPE,
    POSSIBLE_QUANTIZER_TYPE,
    quantize,
)
from enc.component.core.arm import _laplace_cdf
from enc.component.core.spatialprior_interpolator import MultilayerInterpolator

class SpatialPrior(nn.Module):
    def __init__(self, spatial_prior_n_channels: int = 32, img_size: Optional[Tuple[int, int]] = None, 
                 spatial_prior_map_downsampling: int = 0, spatial_prior_map_interpolation_mode: str = "bicubic",
                 arm_mod_layer_context: str = "concat", n_latents: int = 7,
                 spatial_prior_arm_conditions: list = [], latent_sizes: list = []) -> None:
        super(SpatialPrior, self).__init__()
        self.n_latents = n_latents
        self.spatial_prior_n_channels = spatial_prior_n_channels
        self.spatial_prior_map_downsampling = spatial_prior_map_downsampling
        self.spatial_prior_map_interpolation_mode = spatial_prior_map_interpolation_mode
        self.arm_mod_layer_context = arm_mod_layer_context
        self.spatial_prior_arm_conditions = spatial_prior_arm_conditions
        self.latent_sizes = latent_sizes
        
        if spatial_prior_map_interpolation_mode in ["multi", "bicubic"]:
            self.bic_interpolator = MultilayerInterpolator(num_layers=n_latents, 
                                                           original_layer=spatial_prior_map_downsampling, 
                                                           latent_sizes=latent_sizes)
        else:
            self.bic_interpolator = None
            
        if self.bic_interpolator is not None:
            for param in self.bic_interpolator.parameters():
                param.requires_grad = False

        self.set_image_size(img_size)

        layer_signals = []
        for i in range(n_latents):
            h_max, w_max = [int(math.ceil(x / (2**i))) for x in img_size]
            layer_channel = torch.ones(h_max * w_max, 1) * int(i * 256 // n_latents)
            layer_signals.append(layer_channel)
        layer_signals = torch.cat(layer_signals, dim=0) / 256.0  # Normalize to [0, 1]
        self.register_buffer("layer_signals", layer_signals, persistent=False)
    
    def set_image_size(self, img_size: Tuple[int, int]) -> None:
        """Set the image size and initialize the spatial prior map accordingly.
        
        Args:
            img_size: Height and width (H, W) of the frame to be coded
        """
        self.img_size = img_size
        h_max, w_max = [int(math.ceil(x / (2**self.spatial_prior_map_downsampling))) for x in img_size]
        self.spatial_prior_map = nn.Parameter(torch.empty(1, self.spatial_prior_n_channels, h_max, w_max), requires_grad=True)
        self._initialize_map()
                    
                    
    def _initialize_map(self) -> None:
        """Initialize the learnable spatial prior map with small random values."""
        if self.spatial_prior_map is not None:
            with torch.no_grad():
                self.spatial_prior_map.normal_(0, 0.01)
    
    @staticmethod
    def med_predictor(a: Tensor, b: Tensor, c: Tensor, temp: float, training: bool) -> Tensor:
        """Median predictor function.
        
        Args:
            a: Right neighbor pixel values [1, C, H, W]
            b: Bottom neighbor pixel values [1, C, H, W] 
            c: Diagonal neighbor pixel values [1, C, H, W]
            
        Returns:
            Tensor: Predicted pixel values [1, C, H, W]
        """
        max_ab = torch.maximum(a, b)
        min_ab = torch.minimum(a, b)
        linear_pred = a + b - c
        if training:            
            c_geq_max = torch.sigmoid((c - max_ab) / temp)
            c_leq_min = torch.sigmoid((min_ab - c) / temp)         
            loco_pred = (c_geq_max * min_ab) + \
                        (c_leq_min * max_ab) + \
                        ((1 - c_geq_max - c_leq_min) * linear_pred)
        else:
            c_geq_max = c >= max_ab
            c_leq_min = c <= min_ab
            loco_pred = torch.where(c_geq_max, min_ab,
                            torch.where(c_leq_min, max_ab, linear_pred))
        return loco_pred

    def interpolate_map(self, latent_grid_index: int, quantized_spatial_prior_map: Tensor) -> Tensor:
        """Interpolates the spatial prior map to the size of the given latent layer.
        
        Args:
            latent_grid_index: Index of the latent grid
            quantized_spatial_prior_map: Quantized spatial prior map [1, C, H_map, W_map]
            
        Returns:
            Tensor of shape [N, spatial_prior_n_channels] with features for each latent position
        """
        h_grid, w_grid = [int(math.ceil(x / (2**latent_grid_index))) for x in self.img_size]
        scale_factor = 2**(self.spatial_prior_map_downsampling - latent_grid_index)
        
        interpolated_map = torch.nn.functional.interpolate(
            quantized_spatial_prior_map, 
            scale_factor=scale_factor,
            mode=self.spatial_prior_map_interpolation_mode,
        )  
        
        features = interpolated_map[:,:,:h_grid, :w_grid]
        
        return features
    
    def interpolate_pyramid(self, quantized_spatial_prior_map: Tensor) -> Tuple[Tensor, ...]:
        """Interpolates the spatial prior map at different scales for the pyramid.

        Args:
            n_latents: Number of latent levels.
            quantized_spatial_prior_map: Quantized spatial prior map [1, C, H, W].

        Returns:
            Tuple of interpolated maps at different scales.
        """
        if self.spatial_prior_map_interpolation_mode in ["multi", "bicubic"]:
            map_list = self.bic_interpolator(quantized_spatial_prior_map)
        else:
            map_list = [self.interpolate_map(i, quantized_spatial_prior_map) for i in range(self.n_latents)]
        
        flat_pyramid = torch.cat(
            [x.contiguous().view(self.spatial_prior_n_channels, -1).transpose(0, 1) for x in map_list],
            dim=0
        )
        if self.arm_mod_layer_context == "concat":
            flat_pyramid = torch.cat([flat_pyramid, self.layer_signals], dim=1)
        return flat_pyramid

    def forward(self, 
                encoder_gains: float,
                quantizer_noise_type: POSSIBLE_QUANTIZATION_NOISE_TYPE = "kumaraswamy",
                quantizer_type: POSSIBLE_QUANTIZER_TYPE = "softround",
                soft_round_temperature: Optional[Tensor] = torch.tensor(0.3),
                noise_parameter: Optional[Tensor] = torch.tensor(1.0),
                AC_MAX_VAL: int = -1,
                training: bool = True) -> Tuple[Tensor, Tensor]:
        
        quantized_map = quantize(
            self.spatial_prior_map * encoder_gains,
            quantizer_noise_type if training else "none",
            quantizer_type if training else "hardround",
            soft_round_temperature,
            noise_parameter,
        )
        
        # Clamp spatial prior map if we need to write a bitstream
        if AC_MAX_VAL != -1:
            quantized_map = torch.clamp(
                quantized_map, -AC_MAX_VAL, AC_MAX_VAL + 1
            )

        conditions = []
        if "loco" in self.spatial_prior_arm_conditions:
            padded_spatial_prior_map = torch.nn.functional.pad(quantized_map, (1, 0, 1, 0))
            a = padded_spatial_prior_map[:, :, :-1, 1:]  
            b = padded_spatial_prior_map[:, :, 1:, :-1]
            c = padded_spatial_prior_map[:, :, :-1, :-1]
            loco_pred = self.med_predictor(a, b, c, soft_round_temperature, training)
            if quantizer_type == "ste" and training:  
                with torch.no_grad():
                    loco_pred = loco_pred - self.med_predictor(a, b, c, soft_round_temperature, True) + self.med_predictor(a, b, c, soft_round_temperature, False)
            conditions.append(loco_pred)

        n_conditions = len(self.spatial_prior_arm_conditions) if self.spatial_prior_arm_conditions[0] != "" else 0
        if n_conditions > 0:
            _, _, h, w = quantized_map.shape
            condition = torch.stack(conditions, dim=-1).view(h * w, n_conditions)
        else:
            condition = None

        return quantized_map, condition
    
    def get_param(self) -> OrderedDict[str, Tensor]:
        """Return **a copy** of the weights and biases inside the module.

        Returns:
            A copy of all weights & biases in the layers.
        """
        param = OrderedDict({k: v.detach().clone() for k, v in self.named_parameters()})
        if self.spatial_prior_map is not None:
            param["spatial_prior_map"] = self.spatial_prior_map.detach().clone()
        return param

    def set_param(self, param: OrderedDict[str, Tensor]):
        """Replace the current parameters of the module with param.

        Args:
            param: Parameters to be set.
        """
        self.load_state_dict(param)

    def initialize(self) -> None:
        """Re-initialize **in place** the parameters of the upsampling."""
        self.contexts.data.fill_(1.0)
        self._initialize_map()
