/*
    Software Name: LANCE
    SPDX-FileCopyrightText: Copyright (c) 2026 Martin Benjak
    SPDX-License-Identifier: BSD 3-Clause "New"

    This software is distributed under the BSD-3-Clause license.
    Authors: see CONTRIBUTORS.md
*/

#include "resampler.h"
#include <algorithm>
#include <cmath>

// Precomputed bicubic Mitchell-Netravali kernel values, scaled by 2^24
// These are generated using the Mitchell-Netravali filter with B=0, C=0.75
static const int32_t bicubic_kernel_down2[4] = {
    -1572864,  9961472,  9961472, -1572864
};

static const int32_t bicubic_kernel_up2[8] = {
    -589824, -1769472,  4390912, 14745600, 14745600,  4390912, -1769472,
    -589824
};

static const int32_t bicubic_kernel_up4[16] = {
    -172032, -1105920, -1843200, -1204224,  1925120,  7151616, 12574720,
    16228352, 16228352, 12574720,  7151616,  1925120, -1204224, -1843200,
    -1105920,  -172032
};

static const int32_t bicubic_kernel_up8[32] = {
    -46080,  -359424,  -844800, -1354752, -1741824, -1858560, -1557504,
    -691200,   879616,  3105792,  5749760,  8565760, 11308032, 13730816,
    15588352, 16634880, 16634880, 15588352, 13730816, 11308032,  8565760,
    5749760,  3105792,   879616,  -691200, -1557504, -1858560, -1741824,
    -1354752,  -844800,  -359424,   -46080
};

static const int32_t bicubic_kernel_up16[64] = {
    -11904,  -100224,  -259200,  -470400,  -715392,  -975744, -1233024,
    -1468800, -1664640, -1802112, -1862784, -1828224, -1680000, -1399680,
    -968832,  -369024,   417152,  1383552,  2500480,  3737216,  5063040,
    6447232,  7859072,  9267840, 10642816, 11953280, 13168512, 14257792,
    15190400, 15935616, 16462720, 16740992, 16740992, 16462720, 15935616,
    15190400, 14257792, 13168512, 11953280, 10642816,  9267840,  7859072,
    6447232,  5063040,  3737216,  2500480,  1383552,   417152,  -369024,
    -968832, -1399680, -1680000, -1828224, -1862784, -1802112, -1664640,
    -1468800, -1233024,  -975744,  -715392,  -470400,  -259200,  -100224,
    -11904
};

static const int32_t bicubic_kernel_up32[128] = {
    -3029,   -26347,   -70805,  -134059,  -213845,  -307819,  -413717,
    -529195,  -651989,  -779755,  -910229, -1041067, -1170005, -1294699,
    -1412885, -1522219, -1620437, -1705195, -1774229, -1825200, -1855824,
    -1863792, -1846800, -1802544, -1728720, -1623024, -1483152, -1306800,
    -1091664,  -835440,  -535824,  -190512,   202672,   642960,  1126640,
    1649872,  2208816,  2799632,  3418480,  4061520,  4724912,  5404816,
    6097392,  6798800,  7505200,  8212752,  8917616,  9615952, 10303920,
    10977680, 11633392, 12267216, 12875312, 13453840, 13998960, 14506832,
    14973616, 15395472, 15768560, 16089040, 16353072, 16556816, 16696432,
    16768080, 16768080, 16696432, 16556816, 16353072, 16089040, 15768560,
    15395472, 14973616, 14506832, 13998960, 13453840, 12875312, 12267216,
    11633392, 10977680, 10303920,  9615952,  8917616,  8212752,  7505200,
    6798800,  6097392,  5404816,  4724912,  4061520,  3418480,  2799632,
    2208816,  1649872,  1126640,   642960,   202672,  -190512,  -535824,
    -835440, -1091664, -1306800, -1483152, -1623024, -1728720, -1802544,
    -1846800, -1863792, -1855824, -1825200, -1774229, -1705195, -1620437,
    -1522219, -1412885, -1294699, -1170005, -1041067,  -910229,  -779755,
    -651989,  -529195,  -413717,  -307819,  -213845,  -134059,   -70805,
    -26347,    -3029
};

static const int32_t bicubic_kernel_up64[256] = {
    -757,    -6752,   -18453,   -35573,   -57835,   -84949,  -116608,
    -152555,  -192469,  -236096,  -283115,  -333269,  -386251,  -441771,
    -499552,  -559296,  -620736,  -683552,  -747477,  -812213,  -877483,
    -942997, -1008448, -1073579, -1138069, -1201664, -1264043, -1324949,
    -1384075, -1441131, -1495840, -1547904, -1597056, -1642976, -1685397,
    -1724021, -1758571, -1788747, -1814272, -1834853, -1850203, -1860032,
    -1864053, -1861973, -1853515, -1838379, -1816288, -1786949, -1750075,
    -1705376, -1652565, -1591349, -1521451, -1442571, -1354432, -1256741,
    -1149211, -1031552,  -903477,  -764693,  -614923,  -453867,  -281248,
    -96773,    99829,   308467,   528669,   759963,  1001861,  1253891,
    1515565,  1786411,  2065941,  2353683,  2649150,  2951867,  3261350,
    3577123,  3898702,  4225611,  4557366,  4893491,  5233502,  5576923,
    5923270,  6272066,  6622830,  6975082,  7328342,  7682130,  8035966,
    8389370,  8741862,  9092962,  9442190,  9789066, 10133110, 10473842,
    10810782, 11143450, 11471366, 11794051, 12111022, 12421803, 12725910,
    13022867, 13312190, 13593403, 13866022, 14129571, 14383566, 14627531,
    14860982, 15083443, 15294430, 15493467, 15680070, 15853763, 16014062,
    16160491, 16292566, 16409811, 16511742, 16597883, 16667750, 16720867,
    16756750, 16774923, 16774923, 16756750, 16720867, 16667750, 16597883,
    16511742, 16409811, 16292566, 16160491, 16014062, 15853763, 15680070,
    15493467, 15294430, 15083443, 14860982, 14627531, 14383566, 14129571,
    13866022, 13593403, 13312190, 13022867, 12725910, 12421803, 12111022,
    11794051, 11471366, 11143450, 10810782, 10473842, 10133110,  9789066,
    9442190,  9092962,  8741862,  8389370,  8035966,  7682130,  7328342,
    6975082,  6622830,  6272066,  5923270,  5576923,  5233502,  4893491,
    4557366,  4225611,  3898702,  3577123,  3261350,  2951867,  2649150,
    2353683,  2065941,  1786411,  1515565,  1253891,  1001861,   759963,
    528669,   308467,    99829,   -96773,  -281248,  -453867,  -614923,
    -764693,  -903477, -1031552, -1149211, -1256741, -1354432, -1442571,
    -1521451, -1591349, -1652565, -1705376, -1750075, -1786949, -1816288,
    -1838379, -1853515, -1861973, -1864053, -1860032, -1850203, -1834853,
    -1814272, -1788747, -1758571, -1724021, -1685397, -1642976, -1597056,
    -1547904, -1495840, -1441131, -1384075, -1324949, -1264043, -1201664,
    -1138069, -1073579, -1008448,  -942997,  -877483,  -812213,  -747477,
    -683552,  -620736,  -559296,  -499552,  -441771,  -386251,  -333269,
    -283115,  -236096,  -192469,  -152555,  -116608,   -84949,   -57835,
    -35573,   -18453,    -6752,     -757
};

// Fixed-point precision (all kernels scaled by 2^24)
static const int FPFM = 24;

// Helper to get kernel for given power
static const int32_t* get_kernel(int power, int &kernel_size) {
    switch (power) {
        case -1:
            kernel_size = 4;
            return bicubic_kernel_down2;
        case 1:
            kernel_size = 8;
            return bicubic_kernel_up2;
        case 2:
            kernel_size = 16;
            return bicubic_kernel_up4;
        case 3:
            kernel_size = 32;
            return bicubic_kernel_up8;
        case 4:
            kernel_size = 64;
            return bicubic_kernel_up16;
        case 5:
            kernel_size = 128;
            return bicubic_kernel_up32;
        case 6:
            kernel_size = 256;
            return bicubic_kernel_up64;
        default:
            kernel_size = 0;
            return nullptr;
    }
}

static void bicubic_upsample(
    frame_memory<int32_t> &src,
    frame_memory<int32_t> &dst,
    int power
)
{
    int kernel_size;
    const int32_t *kernel = get_kernel(power, kernel_size);
    
    int factor = 1 << power;
    int ks = kernel_size / factor;
    
    // Split kernel into polyphase components
    int32_t polyphase_kernels[factor][ks];
    for (int i = ks-1; i >= 0; i--)
        for (int phase = 0; phase < factor; phase++, kernel++)
            polyphase_kernels[phase][i] = *kernel;
    
    // Horizontal filtering
    frame_memory<int32_t> tmp;
    tmp.update_to(src.h, src.w * factor, 0, 1);
    int32_t *tmp_ptr = tmp.origin();
    int32_t *src_ptr = src.origin();
    for (int y = 0; y < src.h; y++, src_ptr+=2*src.pad)
    {
        for (int x = 0; x < src.w; x++, src_ptr++)
        {
            // Apply each polyphase filter for each output position
            for (int phase = 0; phase < factor; phase++, tmp_ptr++)
            {
                int64_t sum = 0;
                int start_offset = (phase < factor/2) ? -2 : -1;
                int phase_index = (phase + factor / 2) % factor;
                for (int i = 0; i < ks; i++) {
                    int src_pos = i + start_offset;
                    if (x + src_pos < 0)
                        src_pos = -x; // replicate left edge
                    else if (x + src_pos >= src.w)
                        src_pos = src.w - x - 1; // replicate right edge

                    sum += ((int64_t)src_ptr[src_pos] << (FPFM - 8)) * polyphase_kernels[phase_index][i];
                    //printf("x:%d y:%d i:%d src:%d kernel:%d\n", x, y, i, (int64_t)src_ptr[src_pos] << (FPFM - 8), polyphase_kernels[phase_index][i]);
                }
                //printf("y: %d, x:%d, x(t):%d result:%lld\n\n",y, x, x*factor+phase, sum);
                // Round and shift
                if (sum < 0)
                    *tmp_ptr = (int32_t)(-((-sum+(1 << (FPFM-1))) >> FPFM));
                else
                    *tmp_ptr = (int32_t)((sum+(1 << (FPFM-1))) >> FPFM);
                //printf("%d ", *tmp_ptr);
            }
        }
        //printf("\n\n");
    }
    
    // Vertical filtering
    src_ptr = tmp.origin();
    int32_t *dst_ptr;
    
    for (int y = 0; y < tmp.h; y++)
    {
        for (int x = 0; x < tmp.w; x++, src_ptr++)
        {
            dst_ptr = dst.origin() + y * factor * dst.stride + x;

            // Apply each polyphase filter for each output position
            for (int phase = 0; phase < factor; phase++, dst_ptr+=dst.stride)
            {
                int64_t sum = 0;
                int start_offset = (phase < factor/2) ? -2 : -1;
                int phase_index = (phase + factor / 2) % factor;
                for (int i = 0; i < ks; i++) {
                    int src_pos = i + start_offset;
                    if (y + src_pos < 0)
                        src_pos = -y; // replicate left edge
                    else if (y + src_pos >= tmp.h)
                        src_pos = tmp.h - y - 1; // replicate right edge

                    sum += (int64_t)src_ptr[src_pos*tmp.stride] * polyphase_kernels[phase_index][i];
                    //printf("x:%d y:%d i:%d src:%ld kernel:%d\n", x, y, i, (int64_t)src_ptr[src_pos*tmp.stride], polyphase_kernels[phase_index][i]);
                }
                //printf("y: %d, x:%d, x(t):%d result:%lld\n\n",y, x, x*factor+phase, sum);
                // Round and shift
                if (sum < 0)
                    sum = (int32_t)-((-sum+(1 << (FPFM-1))) >> FPFM);
                else
                    sum = (int32_t)((sum+(1 << (FPFM-1))) >> FPFM);

                if (sum < 0)
                    *dst_ptr = (int32_t)-((-sum+(1 << (FPFM-8-1))) >> (FPFM - 8));
                else
                    *dst_ptr = (int32_t)((sum+(1 << (FPFM-8-1))) >> (FPFM - 8));
                //printf("%d ", *dst_ptr);
            }
        }
        //printf("\n\n");
    }
}

// Bicubic downsampling using strided convolution
static void bicubic_downsample(
    frame_memory<int32_t> &src,
    frame_memory<int32_t> &dst,
    int power
)
{
    int kernel_size;
    const int32_t *kernel = get_kernel(-1, kernel_size);
    int stride = 1 << power;
    int pad = (4 - stride) / 2;

    if(pad < 0){
        printf("resampler.cpp::bicubic_downsample(): Downsampling ratios > 4 are not implemented, yet.");
        exit(1);
    }
    
    
    
    // Horizontal filtering
    frame_memory<int32_t> tmp;
    tmp.update_to(src.h, src.w >> power, 0, 1);
    int32_t *tmp_ptr = tmp.origin();
    int32_t *src_ptr = src.origin()-pad+2;

    
    for (int y = 0; y < src.h; y++, src_ptr += 2*src.pad)
    {
        for (int x =-pad+2; x < src.w+pad-1; x+=stride, src_ptr+=stride, tmp_ptr++)
        {
            int64_t sum = 0;
            for (int i = 0; i < kernel_size; i++) {
                int src_pos = i - 2;
                if (x + src_pos < 0)
                    src_pos = -x; // replicate left edge
                else if (x + src_pos >= src.w)
                    src_pos = src.w - x - 1; // replicate right edge
                sum += ((int64_t)src_ptr[src_pos] << (FPFM - 8)) * kernel[i];
                //printf("x:%d y:%d i:%d src:%d kernel:%d\n", x, y, i, (int32_t)src_ptr[src_pos] << (FPFM - 8), kernel[i]);
            }
            // Round and shift
            if (sum < 0)
                *tmp_ptr = (int32_t)(-((-sum+(1 << (FPFM-1))) >> FPFM));
            else
                *tmp_ptr = (int32_t)((sum+(1 << (FPFM-1))) >> FPFM);
        }
    }
    
    // Vertical filtering
    int32_t *dst_ptr = dst.origin();
    src_ptr = tmp.origin() + (-pad+2) * tmp.stride;
    
    for (int y = -pad+2; y < tmp.h+pad-1; y+=stride, src_ptr += tmp.stride*(stride-1))
    {
        for (int x =0; x < tmp.w; x++, src_ptr++, dst_ptr++)
        {
            int64_t sum = 0;
            for (int i = 0; i < kernel_size; i++) {
                int src_pos = i - 2;
                if (y + src_pos < 0)
                    src_pos = -y; // replicate top edge
                else if (y + src_pos >= tmp.h)
                    src_pos = tmp.h - y - 1; // replicate bottom edge
                sum += (int64_t)src_ptr[src_pos*tmp.stride] * kernel[i];
            }
            // Round and shift
            if (sum < 0)
                    sum = (int32_t)(-((-sum+(1 << (FPFM-1))) >> FPFM));
                else
                    sum = (int32_t)((sum+(1 << (FPFM-1))) >> FPFM);

            if (sum < 0)
                    *dst_ptr = (int32_t)-((-sum+(1 << (FPFM-8-1))) >> (FPFM - 8));
                else
                    *dst_ptr = (int32_t)((sum+(1 << (FPFM-8-1))) >> (FPFM - 8));
        }
    }

}

void resample_spatial_prior(
    frame_memory<int32_t> &src,
    frame_memory<int32_t> &dst,
    int target_h,
    int target_w,
    int spatial_prior_downscale,
    int layer_number
)
{
    dst.update_to(target_h, target_w, 0, 1);
    int power = spatial_prior_downscale - layer_number;
    
    // If power is 0, just copy
    if (power == 0 || (src.h == target_h && src.w == target_w))
    {
        
        int32_t *src_ptr = src.origin();
        int32_t *dst_ptr = dst.origin();
        
        for (int y = 0; y < target_h; y++)
        {
            for (int x = 0; x < target_w; x++)
            {
                dst_ptr[x] = src_ptr[x];
            }
            src_ptr += src.stride;
            dst_ptr += dst.stride;
        }
        return;
    }
    
    if (power > 0)
        bicubic_upsample(src, dst, power);
    else 
        bicubic_downsample(src, dst, -power);
}
