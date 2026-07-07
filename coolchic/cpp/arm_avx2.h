/*
    Software Name: Cool-Chic / LANCE
    SPDX-FileCopyrightText: Copyright (c) 2023-2025 Orange
    SPDX-FileCopyrightText: Copyright (c) 2026 Martin Benjaks
    SPDX-License-Identifier: BSD 3-Clause "New"

    This software is distributed under the BSD-3-Clause license.
    Authors: see CONTRIBUTORS.md
*/

#ifndef ARM_AVX2_H
#define ARM_AVX2_H

void custom_conv_11_int32_avx2_8_X_X(weights_biases *kwtX_n_n, weights_biases *kbX_n, // kwtX_n_n[n_hidden_layers] -- kernel weights, transposed.
                                     weights_biases *kwOUT_n_2, weights_biases *kbOUT_2, // _n_2, weights not transposed.
                                     int32_t *context_indicies, int32_t n_contexts_param, int32_t n_hidden_layers_param,
                                     int32_t *src,
                                     int src_h, int src_w, int src_pad,
                                     int32_t *spatial_prior_resampled,
                                     int layer, int n_layer, bool layerid_context,
                                     BACContext &bac_context
                                     );
void custom_conv_11_int32_avx2_16_X_X(weights_biases *kwtX_n_n, weights_biases *kbX_n, // kwtX_n_n[n_hidden_layers] -- kernel weights, transposed.
                                      weights_biases *kwOUT_n_2, weights_biases *kbOUT_2, // _n_2, weights not transposed.
                                      int32_t *context_indicies, int32_t n_contexts_param, int32_t n_hidden_layers_param,
                                      int32_t *src,
                                      int src_h, int src_w, int src_pad,
                                      int32_t *spatial_prior_resampled,
                                      int layer, int n_layer, bool layerid_context,
                                      BACContext &bac_context
                                      );
void custom_conv_11_int32_avx2_24_X_X(weights_biases *kwtX_n_n, weights_biases *kbX_n, // kwtX_n_n[n_hidden_layers] -- kernel weights, transposed.
                                      weights_biases *kwOUT_n_2, weights_biases *kbOUT_2, // _n_2, weights not transposed.
                                      int32_t *context_indicies, int32_t n_contexts_param, int32_t n_hidden_layers_param,
                                      int32_t *src,
                                      int src_h, int src_w, int src_pad,
                                      int32_t *spatial_prior_resampled,
                                      int layer, int n_layer, bool layerid_context,
                                      BACContext &bac_context
                                      );
void custom_conv_11_int32_avx2_32_X_X(weights_biases *kwtX_n_n, weights_biases *kbX_n, // kwtX_n_n[n_hidden_layers] -- kernel weights, transposed.
                                      weights_biases *kwOUT_n_2, weights_biases *kbOUT_2, // _n_2, weights not transposed.
                                      int32_t *context_indicies, int32_t n_contexts_param, int32_t n_hidden_layers_param,
                                      int32_t *src,
                                      int src_h, int src_w, int src_pad,
                                      int32_t *spatial_prior_resampled,
                                      int layer, int n_layer, bool layerid_context,
                                      BACContext &bac_context
                                      );

#endif // ARM_AVX2_H
