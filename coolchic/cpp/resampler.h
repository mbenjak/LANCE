/*
    Software Name: LANCE
    SPDX-FileCopyrightText: Copyright (c) 2026 Martin Benjak
    SPDX-License-Identifier: BSD 3-Clause "New"

    This software is distributed under the BSD-3-Clause license.
    Authors: see CONTRIBUTORS.md
*/

#ifndef RESAMPLER_H
#define RESAMPLER_H

#include "frame-memory.h"

void resample_spatial_prior(
    frame_memory<int32_t> &src,
    frame_memory<int32_t> &dst,
    int target_h,
    int target_w,
    int spatial_prior_downscale,
    int layer_number
);

#endif // RESAMPLER_H
