/*
    Software Name: LANCE
    SPDX-FileCopyrightText: Copyright (c) 2026 Martin Benjak
    SPDX-License-Identifier: BSD 3-Clause "New"

    This software is distributed under the BSD-3-Clause license.
    Authors: see CONTRIBUTORS.md
*/

#ifndef LOCO_H
#define LOCO_H

#include <algorithm>

inline 
int32_t loco_predictor(const int32_t a, const int32_t b, const int32_t c)
{
    if (c >= std::max(a, b))
        return std::min(a, b);
    else if (c <= std::min(a, b))
        return std::max(a, b);
    else
        return a + b - c;
}

#endif //LOCO_H