#include "PixelSort.h"
#include <stdio.h>
#include <math.h>

__device__ __host__ int CeilDiv(int a, int b) { return (a-1)/b + 1; }
__device__ __host__ int CeilAlign(int a, int b) { return CeilDiv(a, b) * b; }

#ifdef DEBUG
#define debug_print(...) fprintf(stderr, __VA_ARGS__) 
#else
#define debug_print(...)
#endif

void PixelSortPatternParmLinear::FilledOffset(const float x, const float y, const float w, const float h, float *output) {
    float delta[2];

    delta[0] = cos(((PixelSortPatternParmLinear *)this)->angle);
    delta[1] = sin(((PixelSortPatternParmLinear *)this)->angle);
}

void PixelSortGPU(const Pixel *input, int width, int height, Pixel *output,
	PixelSortBy sort_by, float threshold_min, float threshold_max, bool reverse_sort_order,
	PixelSortPatternParm *pattern_parm, bool anti_aliasing, bool sort_alpha) {

    PixelSortPatternParm *pattern_parm_gpu = nullptr;
	switch (pattern_parm->pattern) {
        case PSP_Linear:
            debug_print("PSP_Linear (%d)\n", pattern_parm->pattern);
            debug_print("angle: %f\n", ((PixelSortPatternParmLinear *)pattern_parm)->angle);
            cudaMalloc(&pattern_parm_gpu, sizeof(PixelSortPatternParmLinear));
            cudaMemcpy(&pattern_parm_gpu, pattern_parm, sizeof(PixelSortPatternParmLinear), cudaMemcpyHostToDevice);
            break;
        case PSP_Radial_Spin:
            debug_print("PSP_Radial_Spin (%d)\n", pattern_parm->pattern);
            debug_print("center: (%f, %f)\n", 
                    ((PixelSortPatternParmRadialSpin *)pattern_parm)->center[0],
                    ((PixelSortPatternParmRadialSpin *)pattern_parm)->center[1]);
            debug_print("WHRatio: %f\n", ((PixelSortPatternParmRadialSpin *)pattern_parm)->WHRatio);
            debug_print("rotation: %f\n", ((PixelSortPatternParmRadialSpin *)pattern_parm)->rotation);
            cudaMalloc(&pattern_parm_gpu, sizeof(PixelSortPatternParmRadialSpin));
            cudaMemcpy(&pattern_parm_gpu, pattern_parm, sizeof(PixelSortPatternParmRadialSpin), cudaMemcpyHostToDevice);
            break;
        case PSP_Polygon:
            debug_print("PSP_Polygon (%d)\n", pattern_parm->pattern);
            debug_print("center: (%f, %f)\n", 
                    ((PixelSortPatternParmPolygon *)pattern_parm)->center[0],
                    ((PixelSortPatternParmPolygon *)pattern_parm)->center[1]);
            debug_print("numSides: %d\n", ((PixelSortPatternParmPolygon *)pattern_parm)->numSides);
            debug_print("WHRatio: %f\n", ((PixelSortPatternParmPolygon *)pattern_parm)->WHRatio);
            debug_print("rotation: %f\n", ((PixelSortPatternParmPolygon *)pattern_parm)->rotation);
            cudaMalloc(&pattern_parm_gpu, sizeof(PixelSortPatternParmPolygon));
            cudaMemcpy(&pattern_parm_gpu, pattern_parm, sizeof(PixelSortPatternParmPolygon), cudaMemcpyHostToDevice);
            break;
        case PSP_Spiral:
            debug_print("PSP_Spiral (%d)\n", pattern_parm->pattern);
            debug_print("center: (%f, %f)\n", 
                    ((PixelSortPatternParmSpiral *)pattern_parm)->center[0],
                    ((PixelSortPatternParmSpiral *)pattern_parm)->center[1]);
            debug_print("curveAngle: %f\n", ((PixelSortPatternParmSpiral *)pattern_parm)->curveAngle);
            debug_print("WHRatio: %f\n", ((PixelSortPatternParmSpiral *)pattern_parm)->WHRatio);
            debug_print("rotation: %f\n", ((PixelSortPatternParmSpiral *)pattern_parm)->rotation);
            cudaMalloc(&pattern_parm_gpu, sizeof(PixelSortPatternParmSpiral));
            cudaMemcpy(&pattern_parm_gpu, pattern_parm, sizeof(PixelSortPatternParmSpiral), cudaMemcpyHostToDevice);
            break;
        case PSP_Sine: 
        case PSP_Triangle: 
        case PSP_Saw_Tooth:
            debug_print("PSP_Sine (%d)\n", pattern_parm->pattern);
            debug_print("waveLength: %f\n", ((PixelSortPatternParmWave *)pattern_parm)->waveLength);
            debug_print("waveheight: %f\n", ((PixelSortPatternParmWave *)pattern_parm)->waveHeight);
            debug_print("rotation: %f\n", ((PixelSortPatternParmWave *)pattern_parm)->rotation);
            cudaMalloc(&pattern_parm_gpu, sizeof(PixelSortPatternParmWave));
            cudaMemcpy(&pattern_parm_gpu, pattern_parm, sizeof(PixelSortPatternParmWave), cudaMemcpyHostToDevice);
            break;
        case PSP_Optical_Flow:
            debug_print("PSP_Optical_Flow (%d)\n", pattern_parm->pattern);
            cudaMalloc(&pattern_parm_gpu, sizeof(PixelSortPatternParmOpFlow));
            cudaMemcpy(&pattern_parm_gpu, pattern_parm, sizeof(PixelSortPatternParmOpFlow), cudaMemcpyHostToDevice);
            break;

        default:
            break;
	}

    

    cudaMemcpy(output, input, width*height*sizeof(Pixel), cudaMemcpyDeviceToDevice);
    cudaFree(pattern_parm_gpu);
}
