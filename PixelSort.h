#pragma once

enum PixelSortBy {
    PSB_Hue,
    PSB_Saturation,
    PSB_Luminance,
    PSB_R,
    PSB_G,
    PSB_B,
};

enum PixelSortPattern {
    PSP_Linear,
    PSP_Radial_Zoom,
    PSP_Radial_Spin,
    PSP_Polygon,
    PSP_Spiral,
    PSP_Sine,
    PSP_Triangle,
    PSP_Saw_Tooth,
    PSP_Optical_Flow,
};

struct Pixel {
    union {
        struct {
            float r, g, b, a;
        };
        float e[4];
    };
};

struct PixelSortPatternParm {
    PixelSortPattern pattern;
};

struct PixelSortPatternParmLinear {
    PixelSortPatternParm base;
    float angle;
};

struct PixelSortPatternParmRadialZoom {
    PixelSortPatternParm base;
    int center[2];
};

/*
 *
 * Other patterns to be added
 *
 */

void PixelSort(const Pixel *input, int width, int height, Pixel *output,
        PixelSortBy sort_by, float threshold_min, float threshold_max, bool reverse_sort_order,
        PixelSortPatternParm *pattern_parm, bool anti_aliasing);
