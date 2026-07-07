/*
    Software Name: LANCE
    SPDX-FileCopyrightText: Copyright (c) 2026 Martin Benjak
    SPDX-License-Identifier: BSD 3-Clause "New"

    This software is distributed under the BSD-3-Clause license.
    Authors: see CONTRIBUTORS.md
*/

#ifndef SPARM_CPU_H
#define SPARM_CPU_H

#include "frame-memory.h"

void custom_sp_conv_11_int32_cpu_X_X_X(weights_biases *kwtX_n_n, weights_biases *kbX_n, // kwt0_16_16 -- kernel weights, transposed.
                                    weights_biases *kwOUT_n_2, weights_biases *kbOUT_2,
                                    int32_t *context_indicies, int32_t n_contexts, int n_hidden_layers,
                                    int32_t *SRC,
                                    int src_h, int src_w, int src_pad, int stride,
                                    BACContext &bac_context
                                    );

#endif // SPARM_CPU_H
