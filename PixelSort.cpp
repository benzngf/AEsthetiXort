#include <PixelSort.h>

void PixelSortGPU(const Pixel *input, int width, int height, Pixel *output,
        PixelSortBy sort_by, float threshold_min, float threshold_max, bool reverse_sort_order,
        PixelSortPatternParm *pattern_parm, bool anti_aliasing) {
    
    switch(pattern_parm->pattern) {
        case PSP_Linear:
            PixelSortGPULinear(input, width, height, output, sort_by, threshold_min, threshold_max, revers_sort_order,
                    (PixelSortPatternParmLinear *)pattern_parm, anti_aliasing);
        case PSP_Radial_Zoom:
            PixelSortGPURadialZoom(input, width, height, output, sort_by, threshold_min, threshold_max, revers_sort_order,
                    (PixelSortPatternParmRadialZoom *)pattern_parm, anti_aliasing);
        /*
        case PSP_Radial_Spin:
        case PSP_Polygon:
        case PSP_Spiral:
        case PSP_Sine:
        case PSP_Triangle:
        case PSP_Saw_Tooth:
        case PSP_Optical_Flow:
        */
        default:
    }
}

void PixelSort(const Pixel *input, int width, int height, Pixel *output,
        PixelSortBy sort_by, float threshold_min, float threshold_max, bool reverse_sort_order,
        PixelSortPatternParm *pattern_parm, bool anti_aliasing) {

    if (/* CUDA is available */) {
        PixelSortGPU(input, width, height, output, sort_by, threshold_min, threshold_max, revers_sort_order,
                pattern_parm, anti_aliasing);
    } else {
        PixelSortCPU(input, width, height, output, sort_by, threshold_min, threshold_max, revers_sort_order,
                pattern_parm, anti_aliasing);
    }
}

