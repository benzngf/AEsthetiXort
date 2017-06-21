#include "PixelSort.h"



void PixelSort(const Pixel *input, int width, int height, Pixel *output,
        PixelSortBy sort_by, float threshold_min, float threshold_max, bool reverse_sort_order,
        PixelSortPatternParm *pattern_parm, bool anti_aliasing) {
	
    if (true/* CUDA is available */) {
/*        PixelSortGPU(input, width, height, output, sort_by, threshold_min, threshold_max, reverse_sort_order,
                pattern_parm, anti_aliasing);*/
    } else {
       /* PixelSortCPU(input, width, height, output, sort_by, threshold_min, threshold_max, reverse_sort_order,
                pattern_parm, anti_aliasing);*/
    }
}

